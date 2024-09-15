### kinetic-network-inference ###
# description:  try collocation based solution
# author:       HPR
# based on: https://doi.org/10.1515/sagmb-2020-0025 and https://doi.org/10.1198/016214508000000797

using DiffEqParamEstim
using OrdinaryDiffEq
using DiffEqFlux
using Lux
using Optimisers
using BSplineKit
using FiniteDifferences
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
OUTNAME = "nn_new_v4_act"
folderN = "results/collocation/"*protein_name*"/"*OUTNAME*"/"
mkpath(folderN)

eps = 1e-06
h = 0.501


# ----- INPUT -----
R"load(paste0('results/graphs/IDH1_WT_v7-MA.RData'))" # TODO: make protein_name variable
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
paramNames = reactions
nP = length(paramNames)

tspan = [minimum(tporig), maximum(tporig)]
Xn0 = replace(X, missing => 0.) # no missing values
mt = length(tporig) # number of time points
Xm = X[2:size(X)[1],:,:]
tpm = tporig[tporig .> 0]

# account for missing values
kis = []
for ii in 1:nr
    Xv = vec(X[:,:,ii]')
    push!(kis, findall(!ismissing, Xv))
end


# ----------------------------
# ----- data collocation -----
# ----------------------------
# ----- get species where initial du is 0 -----
initialdus = []
m = exp.(A*log.(Xn0[:,:,1]'.+eps))
for i in 1:100
    psim = rand(Gamma(1,1), nP)
    dusim = N*Diagonal(psim)*m

    push!(initialdus, dusim[:,1])
end
initialdus = mapreduce(permutedims, vcat, initialdus)
boxplot(initialdus, legend=false)
μ_init = mapslices(x -> mean(skipmissing(x)), initialdus, dims=1)

# 2-hop distance from substrate
k = vec(abs.(μ_init) .< 0.01)
print(species[k])
print(length(k))
# 1-hop distance from substrate
k2 = vec(abs.(μ_init) .> 0.01)
print(species[k2])
print(length(k2))


# ----- get du via interpolation -----
fdm = central_fdm(5, 1)
tpoints = collect(range(0.0,tspan[2],50))

function iterpolation(si, ii, tpoints)
    intp = interpolate(tporig, vec(Xn0[:,si,ii]'), BSplineOrder(4))
    uintp = [intp(t) for t in tpoints]
    duintp = [fdm(intp, t) for t in tpoints]

    return uintp, duintp
end


du_intps_d = []
for ii in 1:nr
    uintps = []
    duintps = []
    for si in 1:s
        uintp_i, duintp_i = iterpolation(si, ii, tpoints)
        push!(uintps, uintp_i)
        push!(duintps, duintp_i)
    end

    u_intps = mapreduce(permutedims, vcat, uintps)
    du_intps = mapreduce(permutedims, vcat, duintps)

    du_intps[:,50] .= du_intps[:,49]
    du_intps[:,1] .= du_intps[:,2]
    du_intps[k,1] .= 0.0

    closest_indices = [findmin(abs.(tpoints .- tpo))[2] for tpo in tporig]
    push!(du_intps_d,du_intps[:,closest_indices])
end

# uintp_i_tmp, duintp_i_tmp = iterpolation(3, 1, tpoints)
# duintp_i_tmp[50] = duintp_i_tmp[49]
# duintp_i_tmp[1] = duintp_i_tmp[2]
# plot(tpoints, duintp_i_tmp)

# something is wrong in the way the derivative is plotted!
# species[3]
# plot!(tporig, du_intps_d[1][3,:])


# ----- collocation -----
dus = []
us = []
for ii in 1:nr
    du, u = @time collocate_data(Xn0[:,:,ii]', tporig, TriweightKernel(), h) # from DiffEqFlux, DiffEqParamEstim packages
    du[k,1] .= 0.0

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

    # p = ps[1:r]
    # b = ps[r+1:r+s]
    m = exp.(A*log.(x .+ eps))
    du = N*Diagonal(ps)*m #.+ b
    return du, st
end


# ----- training functions -----
function loss_function(model, ps, st, data)
    
    # -- informed model
    # ypred, st = Lux.apply(model, data[1], ps, st)
    # ki = data[4] # index with no missing values
    # mse_loss = sqrt(mean(abs2, vec(ypred)[ki] .- vec(data[2])[ki]))
    
    # -- black box model
    ppred, st = Lux.apply(model, data[1], ps, st)
    m = exp.(A*log.(data[2].+eps))
    ypred = N*Diagonal(ppred)*m
    loss = mean(abs2, vec(ypred) .- vec(data[3]))
    # loss = -1*cor(vec(ypred), vec(data[3]))

    return loss, st, ()
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
# -- informed model
# model = Chain(MALayer(;dims=nP))
# -- blackbox model
in_dim = 100
hidden_dim = 1000
model = Lux.Chain(Lux.Dense(in_dim => hidden_dim, relu), Lux.Dense(hidden_dim => nP, relu))

rng = MersenneTwister(42)
opt = Optimisers.Adam(0.0001)

# epochs = 100000
epochs = 1000


dev_cpu = cpu_device()
dev_gpu = gpu_device()

tstates = []
losses = []
ypreds = []
ppreds = []

for ii in 1:nr
    print("\n"*string(ii)*"\n")

    # initialise and train
    # p0s = rand(Gamma(1,1), in_dim)
    p0s = rand(in_dim)
    
    tstate = Lux.Experimental.TrainState(rng, model, opt)
    vjp_rule = AutoZygote()

    # -- informed model
    # tstate, loss = @time main(tstate, vjp_rule, (Xn0[:,:,ii]', du_intps_d[ii], kis[ii]), epochs)
    # ypred = dev_cpu(Lux.apply(tstate.model, dev_gpu(Xn0[:,:,ii]'), tstate.parameters, tstate.states)[1])
    # -- black box model
    # for blackbox model: calculate dx
    tstate, loss = @time main(tstate, vjp_rule, (p0s, Xn0[:,:,ii]', du_intps_d[ii], kis[ii]), epochs)
    ppred = dev_cpu(Lux.apply(tstate.model, dev_gpu(p0s), tstate.parameters, tstate.states)[1])
    m = exp.(A*log.(Xn0[:,:,ii]'.+eps))
    ypred = N*Diagonal(ppred)*m

    # save train state and predictions
    save_object(folderN*"tstate_rep"*string(ii)*".jld2", tstate)
    ypreddf = DataFrame(ypred, [:ypred0, :ypred1, :ypred2, :ypred3, :ypred4])
    ypreddf[!,:species] = species
    dudf = DataFrame(du_intps_d[ii], [:du0, :du1, :du2, :du3, :du4])
    CSV.write(folderN*"pred_rep"*string(ii)*".csv", hcat(ypreddf,dudf))

    # append to list
    push!(tstates,tstate)
    push!(losses,loss)
    push!(ypreds, ypred)
    push!(ppreds, ppred)
end

# -- no blackbox
diagnostics_and_save_NN(tstates, ypreds, losses, false, true)

# -- blackbox
# diagnostics_and_save_NN(tstates, ypreds, losses, false, true)
heatmap(tstates[1].parameters[1][1])
heatmap(tstates[1].parameters[2][1])

pinf = mapreduce(permutedims, vcat, ppreds)
pdist = boxplot(pinf, palette = :bamako, legend=false, title="parameters", xlabel="parameter #", ylabel="parameter value", size = default(:size) .* (10,10))
savefig(pdist, folderN*"parameters.pdf")

# ----- get output and save -----
# info[!, :param_1] = tstates[1].parameters[1:r]
# info[!, :param_2] = tstates[2].parameters[1:r]
# info[!, :param_3] = tstates[3].parameters[1:r]
# CSV.write(folderN*"parameters.csv", info)

# bias = DataFrame([tstates[1].parameters[r+1:r+s],
#             tstates[2].parameters[r+1:r+s],
#             tstates[3].parameters[r+1:r+s],
#             species],
#         [:b_1, :b_2, :b_3, :species])
# CSV.write(folderN*"bias.csv", bias)


# # # -----------------------------
# # # ----- ABC solution -----
# # # -----------------------------
# # ----- prior -----
# α_sigma = 1
# θ_sigma = 1
# α_k = 0
# θ_k = 5

# d1 = StatsPlots.plot(Gamma(α_sigma,θ_sigma), legend=false, lc=:black, title = "prior σ\nα="*string(α_sigma)*", θ="*string(θ_sigma), dpi = 600)
# d2 = StatsPlots.plot(Normal(α_k,θ_k), legend=false, lc=:black, title = "prior k\nα="*string(α_k)*", θ="*string(θ_k), xlims = (-25,25), dpi = 600)

# d = StatsPlots.plot(d1,d2, layout = (1,2))
# savefig(d, folderN*"prior.png")

# p0 = rand(Normal(α_k,θ_k), r)
# prior = [Normal(α_k, θ_k) for i in 1:r]

# # ----- simulation -----
# function simulation(params, constants, targetdata)
#     loss = 0
#     for ii in 1:nr
#         ki = kis[ii]
#         m = exp.(A*log.(Xn0[:,:,ii]' .+ eps))
#         simdata = N*Diagonal(params)*m
#         loss += ApproxBayes.ksdist(vec(simdata)[ki], vec(dus[ii])[ki])
#     end
#     loss, 1
# end

# setup = ABCSMC(simulation, #simulation function
#   r, # number of parameters
#   2.0, # target ϵ - 0.1
#   ApproxBayes.Prior(prior), #Prior for each of the parameters
#   maxiterations=10000000, # maxiterations
# #   X1, # constants
#   nparticles=1000, # nparticles
#   α=0.3, # The αth quantile of population i is chosen as the ϵ for population i + 1
#   ϵ1=10^5, # Starting ϵ for first ABC SMC populations
#   convergence=0.05, # ABC SMC stops when ϵ in population i + 1 is within 0.05 of populations i
# #   :uniformkernel, # Parameter perturbation kernel
# )

# # targetdata: nothing
# smc = @time runabc(setup, nothing, verbose = true, progress = true, parallel = true)
# diagnostics_and_save_ABC(smc)

# print(smc.accratio)
# print("done")


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

