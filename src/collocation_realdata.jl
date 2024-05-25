### kinetic-network-inference ###
# description:  try collocation based solution
# author:       HPR
# based on: https://doi.org/10.1515/sagmb-2020-0025 and https://doi.org/10.1198/016214508000000797

using Turing
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
using MCMCChainsStorage, HDF5, MCMCChains
using PDFmerger: append_pdf!
using StaticArrays, SparseArrays
include("_odes.jl")
include("_plot_utils.jl")
include("_plot_utils_NN.jl")

Random.seed!(42)
print(Threads.nthreads())

protein_name = "IDH1_WT"
OUTNAME = "nn_v1"
folderN = "results/collocation/"*protein_name*"/"*OUTNAME*"/"
mkpath(folderN)

eps = 0.001

# ----- INPUT -----
R"load(paste0('results/graphs/IDH1_WT_v3.RData'))" # TODO: make protein_name variable
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

    m = exp.(A*log.(x .+ eps))
    du = N*Diagonal(ps)*m
    return du, st
end

model = Chain(MALayer(;dims=r))


# ----- training functions -----
# account for missing values
kis = []
for ii in 1:nr
    Xv = vec(X[:,:,ii]')
    push!(kis, findall(!ismissing, Xv))
end

# rng, optimiser, etc
rng = MersenneTwister(42)
opt = Optimisers.Adam(0.05)
# opt = Optimisers.AdamW()

function loss_function(model, ps, st, data)
    ki = data[3] # index with no missing values
    y_pred, st = Lux.apply(model, data[1], ps, st)
    mse_loss = sqrt(mean(abs2, vec(y_pred)[ki] .- vec(data[2])[ki]))
    # mse_loss = mean(abs2, vec(y_pred) .- vec(data[2]))
    return mse_loss, st, ()
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
epochs = 1000
dev_cpu = cpu_device()
dev_gpu = gpu_device()

tstates = []
losses = []
ypreds = []

for ii in 1:nr
    tstate = Lux.Experimental.TrainState(rng, model, opt)
    vjp_rule = AutoZygote()
    
    tstate, loss = @time main(tstate, vjp_rule, (Xn0[:,:,ii]', dus[ii], kis[ii]), epochs)
    ypred = dev_cpu(Lux.apply(tstate.model, dev_gpu(Xn0[:,:,ii]'), tstate.parameters, tstate.states)[1])

    push!(tstates,tstate)
    push!(losses,loss)
    push!(ypreds, ypred)
end

diagnostics_and_save_NN(tstates, ypreds, losses, false)

# ----- get output and save -----
info[!, :param_1] = tstates[1].parameters
info[!, :param_2] = tstates[2].parameters
info[!, :param_3] = tstates[3].parameters
CSV.write(folderN*"parameters.csv", info)

# # -----------------------------
# # ----- Bayesian solution -----
# # -----------------------------
# # ----- hyperparameters -----
# Niter = 100
# nChains = 4
# numParam = length(paramNames)

# # ----- prior -----
# α_sigma = 1
# θ_sigma = 1
# α_k = 3
# θ_k = 1

# d1 = StatsPlots.plot(Gamma(α_sigma,θ_sigma), legend=false, lc=:black, title = "prior σ\nα="*string(α_sigma)*", θ="*string(θ_sigma), dpi = 600)
# d2 = StatsPlots.plot(Gamma(α_k,θ_k), legend=false, lc=:black, title = "prior k\nα="*string(α_k)*", θ="*string(θ_k), xlims = (0,25), dpi = 600)

# d = StatsPlots.plot(d1,d2, layout = (1,2))
# savefig(d, folderN*"prior.png")

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

