### kinetic-network-inference ###
# description:  try collocation based solution
# author:       HPR
# based on: https://doi.org/10.1515/sagmb-2020-0025 and https://doi.org/10.1198/016214508000000797

# using Turing
# using ApproxBayes
using DiffEqParamEstim
using OrdinaryDiffEq
using DiffEqFlux
using Lux
using Optimisers
using Zygote
using StatsPlots
using Plots
using Plots.Measures
using LinearAlgebra
using Random
using RCall
using CSV
using BenchmarkTools
using DataFrames
using StatsBase
using Printf
using Distributions
using MCMCChainsStorage, HDF5, MCMCChains, JLD2
using PDFmerger: append_pdf!
using StaticArrays, SparseArrays
include("_odes.jl")
include("_plot_utils.jl")
include("_plot_utils_NN.jl")

Random.seed!(42)
print(Threads.nthreads())

protein_name = "IDH1_WT"
OUTNAME = "nn_v7_noblackbox+offrates+bias"
folderN = "results/collocation/"*protein_name*"/"*OUTNAME*"/"
mkpath(folderN)

eps = 1e-06

# ----- INPUT -----
R"load(paste0('results/graphs/IDH1_WT_v4-offrates.RData'))" # TODO: make protein_name variable
@rget DATA;

X = Array{Union{Missing,Float64}}(DATA[:S])
A = Matrix{Int64}(DATA[:A])
B = Matrix{Int64}(DATA[:B])
x0 = Array{Float64}(coalesce.(X[1,:,1], 1.0))
N = transpose(B-A)

species = DATA[:species]
reactions = DATA[:reactions]
tporig = DATA[:timepoints]
replicates = DATA[:replicates]
info = DATA[:REACTIONS]

s = length(species)
r = length(reactions)
nr = length(replicates)
paramNames = ["σ"; reactions]

tspan = [minimum(tporig), maximum(tporig)]
tp = tporig[tporig .> 0]
Xm = X[2:size(X)[1],:,:]
Xn0 = replace(X, missing => 0.) # no missing values

# account for missing values
kis = []
for ii in 1:nr
    Xv = vec(X[:,:,ii]')
    push!(kis, findall(!ismissing, Xv))
end

# --------------------------------
# ----- try data collocation -----
# --------------------------------
# kernel bandwidth
mt = length(tporig) # number of time points
h = mt^(-1/5)*mt^(-3/35)*log(mt)^(-1/16)

# ----- collocation -----
dus = []
us = []
for ii in 1:nr
    du, u = @time collocate_data(Xn0[:,:,ii]', tporig, TriweightKernel(), h+0.001) # from DiffEqFlux, DiffEqParamEstim packages
    push!(dus,du)
    push!(us,u)
end


# -----------------------
# ----- NN solution -----
# -----------------------
# ----- construct neural net -----
struct MALayer{F1} <: Lux.AbstractExplicitLayer
    dims::Int
    init_weight::F1
end

function MALayer(;dims::Int, init_weight=Lux.glorot_uniform)
    return MALayer{typeof(init_weight)}(dims, init_weight)
end

function Lux.initialparameters(rng::AbstractRNG, MALayer::MALayer)
    return (weight=MALayer.init_weight(rng, MALayer.dims))
end
Lux.initialstates(::AbstractRNG, ::MALayer) = NamedTuple()

function (MALayer::MALayer)(x::AbstractMatrix, ps, st::NamedTuple, A=A, N=N)

    p = ps[1:r]
    b = ps[r+1:r+s]
    m = exp.(A*log.(x .+ eps))
    du = N*Diagonal(p)*m .+ b
    return du, st
end

model = Chain(MALayer(;dims=r+s))
# model = Lux.Chain(Lux.Dense(s => 1), Lux.Dense(1 => s))

# ----- training functions -----
# rng, optimiser, etc
rng = MersenneTwister(42)
opt = Optimisers.Adam(0.05)
# Σ = 2
# opt = Optimisers.AdamW()
# opt = Optimisers.RMSProp(0.05)

# function ksdist(x::AbstractVector{T}, y::AbstractVector{S}) where {T <: Real, S <: Real}
#   #adapted from HypothesisTest.jl
#   n_x, n_y = length(x), length(y)
#   sort_idx = sortperm([x; y])
#   pdf_diffs = [ones(n_x)/n_x; -ones(n_y)/n_y][sort_idx]
#   cdf_diffs = cumsum(pdf_diffs)
#   δp = maximum(cdf_diffs)
#   δn = -minimum(cdf_diffs)
#   δ = max(δp, δn)
#   return δ
# end

function loss_function(model, ps, st, data)
    ki = data[3] # index with no missing values
    ypred, st = Lux.apply(model, data[1], ps, st)
    dist = sqrt(mean(abs2, vec(ypred)[ki] .- vec(data[2])[ki]))
    # dist = ksdist(vec(ypred)[ki], vec(data[2])[ki])
    # d = MvNormal(vec(data[2])[ki], Σ*I)
    # dist = pdf(d, vec(ypred)[ki])
    return dist, st, ()
end

function main(tstate::Lux.Experimental.TrainState, vjp, data, epochs)
    data = data .|> gpu_device()
    losses = []
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(
            vjp, loss_function, data, tstate)
        if epoch % 100 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
        tstate = Lux.Training.apply_gradients(tstate, grads)
        push!(losses,loss)
    end
    return tstate, losses
end


# ----- train and predict -----
# epochs = 100000
epochs = 1000
dev_cpu = cpu_device()
dev_gpu = gpu_device()

tstates = []
losses = []
ypreds = []

for ii in 1:nr
    # initialise and train
    tstate = Lux.Experimental.TrainState(rng, model, opt)
    vjp_rule = AutoZygote()

    tstate, loss = @time main(tstate, vjp_rule, (Xn0[:,:,ii]', dus[ii], kis[ii]), epochs)
    ypred = dev_cpu(Lux.apply(tstate.model, dev_gpu(Xn0[:,:,ii]'), tstate.parameters, tstate.states)[1])
    # XX = us[ii]
    # XX[XX .< 0] .= 0.0
    # tstate, loss = @time main(tstate, vjp_rule, (XX, dus[ii], kis[ii]), epochs)
    # ypred = dev_cpu(Lux.apply(tstate.model, dev_gpu(XX), tstate.parameters, tstate.states)[1])

    # save train state and predictions
    save_object(folderN*"tstate_rep"*string(ii)*".jld2", tstate)
    ypreddf = DataFrame(ypred, [:ypred0, :ypred1, :ypred2, :ypred3, :ypred4])
    ypreddf[!,:species] = species
    dudf = DataFrame(dus[ii], [:du0, :du1, :du2, :du3, :du4])
    CSV.write(folderN*"pred_rep"*string(ii)*".csv", hcat(ypreddf,dudf))

    # append to list
    push!(tstates,tstate)
    push!(losses,loss)
    push!(ypreds, ypred)
end

# -- no blackbox
diagnostics_and_save_NN(tstates, ypreds, losses, false)

# -- blackbox
# diagnostics_and_save_NN(tstates, ypreds, losses, false, true)
# heatmap(tstates[1].parameters[1][1])
# heatmap(tstates[1].parameters[2][1])

# ----- get output and save -----
info[!, :param_1] = tstates[1].parameters[1:r]
info[!, :param_2] = tstates[2].parameters[1:r]
info[!, :param_3] = tstates[3].parameters[1:r]
CSV.write(folderN*"parameters.csv", info)

bias = DataFrame([tstates[1].parameters[r+1:r+s],
            tstates[2].parameters[r+1:r+s],
            tstates[3].parameters[r+1:r+s],
            species],
        [:b_1, :b_2, :b_3, :species])
CSV.write(folderN*"bias.csv", bias)


# # -----------------------------
# # ----- ABC solution -----
# # -----------------------------
# ----- prior -----
α_sigma = 1
θ_sigma = 1
α_k = 0
θ_k = 5

d1 = StatsPlots.plot(Gamma(α_sigma,θ_sigma), legend=false, lc=:black, title = "prior σ\nα="*string(α_sigma)*", θ="*string(θ_sigma), dpi = 600)
d2 = StatsPlots.plot(Normal(α_k,θ_k), legend=false, lc=:black, title = "prior k\nα="*string(α_k)*", θ="*string(θ_k), xlims = (-25,25), dpi = 600)

d = StatsPlots.plot(d1,d2, layout = (1,2))
savefig(d, folderN*"prior.png")

p0 = rand(Normal(α_k,θ_k), r)
prior = [Normal(α_k, θ_k) for i in 1:r]

# ----- simulation -----
function simulation(params, constants, targetdata)
    loss = 0
    for ii in 1:nr
        ki = kis[ii]
        m = exp.(A*log.(Xn0[:,:,ii]' .+ eps))
        simdata = N*Diagonal(params)*m
        loss += ApproxBayes.ksdist(vec(simdata)[ki], vec(dus[ii])[ki])
    end
    loss, 1
end

setup = ABCSMC(simulation, #simulation function
  r, # number of parameters
  2.0, # target ϵ - 0.1
  ApproxBayes.Prior(prior), #Prior for each of the parameters
  maxiterations=10000000, # maxiterations
#   X1, # constants
  nparticles=1000, # nparticles
  α=0.3, # The αth quantile of population i is chosen as the ϵ for population i + 1
  ϵ1=10^5, # Starting ϵ for first ABC SMC populations
  convergence=0.05, # ABC SMC stops when ϵ in population i + 1 is within 0.05 of populations i
#   :uniformkernel, # Parameter perturbation kernel
)

# targetdata: nothing
smc = @time runabc(setup, nothing, verbose = true, progress = true, parallel = true)
diagnostics_and_save_ABC(smc)

print(smc.accratio)
print("done")


# # -----------------------------
# # ----- Bayesian solution -----
# # -----------------------------
# # ----- hyperparameters -----
# Niter = 100
# nChains = 4
# numParam = length(paramNames)

# # sample from prior to get p0
# p0 = rand(Gamma(α_k,θ_k), r)

# # ----- MA and likelihood -----
# function MA(x, ps, A=A, N=N)
#     m = exp.(A*log.(x .+ eps))
#     du = N*Diagonal(ps)*m
#     return du
# end


# @model function likelihood_du(dus, α_sigma=α_sigma, θ_sigma=θ_sigma, α_k=α_k, θ_k=θ_k)
    
#     # priors
#     Σ ~ Gamma(α_sigma, θ_sigma)
#     k ~ Product([Gamma(α_k, θ_k) for i in 1:r])

#     # simulate du
#     for ii in 1:nr
#         dup = vec(MA(Xn0[:,:,ii]', k))
#         dus[ii] ~ MvNormal(dup, Σ*I)
#     end

#     # calculate likelihood
#     return nothing
# end

# # du = vec(mapreduce(permutedims, vcat, dus))
# dusv = []
# for ii in 1:nr
#     push!(dusv, vec(dus[ii]))
# end
# modelb = likelihood_du(dusv)
# myChains = @time sample(modelb, NUTS(10, 0.65, adtype = AutoZygote()), MCMCThreads(), Niter, nChains; progress=true, save_state=true)
# diagnostics_and_save(myChains, problem)

