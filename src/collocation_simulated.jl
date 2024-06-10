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
using BSplineKit
using FiniteDifferences
using Zygote, Sundials, LSODA
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

protein_name = "Ex4_20s_spline"
OUTNAME = "fit_MA_with_MA"
folderN = "results/collocation/"*protein_name*"/"*OUTNAME*"/"
mkpath(folderN)

eps = 1e-06
h = 0.501


# ----- INPUT -----
R"load(paste0('data/simulation/Ex4_20s_MA_DATA.RData'))"
@rget DATA_MA;
DATA = DATA_MA

X = Array{Union{Missing,Float64}}(DATA[:X])
A = Matrix{Int64}(DATA[:A])
B = Matrix{Int64}(DATA[:B])
x0 = Array{Float64}(DATA[:x0])
N = transpose(B-A)

species = DATA[:species]
reactions = DATA[:reactions]
info = DATA[:info]
# p0 = [info.rate_vmax; info.rate_km]
p0 = info.rate_ma
tporig = DATA[:tp]

s = length(species)
r = size(info)[1]
# paramNames = [info.rate_name_vmax; info.rate_name_km]
paramNames = info.rate_name_ma
nP = length(paramNames)

tspan = [minimum(tporig), maximum(tporig)]
tp = tporig[tporig .> 0]
Xm = Array{Float64}(X[2:size(X)[1],:])
tpm = tporig[tporig .> 0]



# ----- get species where initial du is 0 -----
initialdus = []
m = exp.(A*log.(X'.+eps))
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
# 1-hop distance from substrate
k2 = vec(abs.(μ_init) .> 0.01)
print(species[k2])

m = exp.(A*log.(X'.+eps))
ducorrect = N*Diagonal(p0)*m


# ----- get du via interpolation -----
fdm = central_fdm(5, 1)
tpoints = collect(range(0.0,4.0,50))

function iterpolation(si, tpoints)
    intp = interpolate(tporig, vec(X[:,si]'), BSplineOrder(4))
    uintp = [intp(t) for t in tpoints]
    duintp = [fdm(intp, t) for t in tpoints]

    return uintp, duintp
end

uintps = []
duintps = []
for si in 1:s
    uintp_i, duintp_i = iterpolation(si, tpoints)
    push!(uintps, uintp_i)
    push!(duintps, duintp_i)
end

u_intps = mapreduce(permutedims, vcat, uintps)
du_intps = mapreduce(permutedims, vcat, duintps)

du_intps[:,50] .= du_intps[:,49]
du_intps[:,1] .= du_intps[:,2]
du_intps[k,1] .= 0.0

closest_indices = [findmin(abs.(tpoints .- tpo))[2] for tpo in tporig]
du_intps_d = du_intps[:,closest_indices]

# errdu_d = mean(abs2.(vec(ducorrect[:,2:5] - du_intps_d)))
errdu_d = mean(abs2.(vec(ducorrect - du_intps_d)))
print("\nerror on u: "*string(round(errdu_d;digits=2))*"\n")



# ----- collocation -----
du, u = @time collocate_data(X', tporig, TriweightKernel(), h) # from DiffEqFlux, DiffEqParamEstim packages
du[k,1] .= 0.0

pl1 = plot(tporig, X, lc=:black, title="u", legend=false, xlabel = "digestion time [hrs]", ylabel = "signal (u)", dpi=600, margin=5mm)
plot!(tporig, u', lc=:red)
pl2 = plot(tporig, du', lc=:red, title = "du",legend=false, xlabel = "digestion time [hrs]", ylabel = "du", dpi=600, margin=5mm)
pl = plot(pl1,pl2, layout = (1,2))
savefig(pl, folderN*"estimated_abundance_chosen.png")




# -----------------------
# ----- NN solution -----
# -----------------------
# ----- construct neural net - mass action -----
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

    p = ps #.+ rand(Uniform(-1e-02, 1e-02), nP)

    m = exp.(A*log.(x.+eps))
    du = N*Diagonal(p)*m
    return du, st
end


# ----- construct neural net - Michaelis-Menten -----
struct MMLayer{F1} <: Lux.AbstractExplicitLayer
    dims::Int
    init_weight::F1
end

function MMLayer(;dims::Int, init_weight=Lux.glorot_uniform)
    return MMLayer{typeof(init_weight)}(dims, init_weight)
end

function Lux.initialparameters(rng::AbstractRNG, MMLayer::MMLayer)
    return (weight=MMLayer.init_weight(rng, MMLayer.dims))
end
Lux.initialstates(::AbstractRNG, ::MMLayer) = NamedTuple()

function (MMLayer::MMLayer)(x::AbstractMatrix, ps, st::NamedTuple, A=A, N=N)

    vmax = ps[1:r]
    Km = ps[r+1:nP]

    m = exp.(A*log.(x .+ eps))
    
    du = N*((Diagonal(vmax)*m) ./ (Km .+ m))
    return du, st
end



# ----- training functions -----
function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data[1], ps, st)
    mse_loss = sqrt(mean(abs2, y_pred .- data[2]))
    # sls_loss = sum(abs2, y_pred .- data[2])
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
model = Chain(MALayer(;dims=nP))
rng = MersenneTwister(42)
opt = Optimisers.Adam(0.0001)
epochs = 10000

dev_cpu = cpu_device()
dev_gpu = gpu_device()

tstates = []
losses = []
ypreds = []
for i in 1:10
    tstate = Lux.Experimental.TrainState(rng, model, opt)
    vjp_rule = AutoZygote()

    # tstate, loss = @time main(tstate, vjp_rule, (X', du), epochs)
    # ypred = dev_cpu(Lux.apply(tstate.model, dev_gpu(X'), tstate.parameters, tstate.states)[1])
    tstate, loss = @time main(tstate, vjp_rule, (X', du_intps_d), epochs)
    ypred = dev_cpu(Lux.apply(tstate.model, dev_gpu(X'), tstate.parameters, tstate.states)[1])

    # append to list
    push!(tstates,tstate)
    push!(losses,loss)
    push!(ypreds, ypred)
end

diagnostics_and_save_NN_sim_multi(tstates, ypreds, losses, true)

print("done")



# # -----------------------------
# # ----- Bayesian solution -----
# # -----------------------------
# # ----- hyperparameters -----
# Niter = 1000
# nChains = 4
# numParam = length(paramNames)

# # ----- prior -----
# α_sigma = 1
# θ_sigma = 1
# α_k = 1
# θ_k = 1

# d1 = StatsPlots.plot(Gamma(α_sigma,θ_sigma), legend=false, lc=:black, title = "prior σ\nα="*string(α_sigma)*", θ="*string(θ_sigma), dpi = 600)
# d2 = StatsPlots.plot(Gamma(α_k,θ_k), legend=false, lc=:black, title = "prior k\nα="*string(α_k)*", θ="*string(θ_k), xlims = (0,25), dpi = 600)
# Plots.vline!(d2, [p0], line=:dash, lc=:red)

# d = StatsPlots.plot(d1,d2, layout = (1,2))
# savefig(d, folderN*"prior.png")

# # ----- MA and likelihood -----
# function MA(x, ps, A=A, N=N)
#     m = exp.(A*log.(x .+ eps))
#     du = N*Diagonal(ps)*m
#     return du
# end


# @model function likelihood_du(X, duflat, α_sigma=α_sigma, θ_sigma=θ_sigma, α_k=α_k, θ_k=θ_k)
    
#     # priors
#     Σ ~ Gamma(α_sigma, θ_sigma)
#     k ~ Product([Gamma(α_k, θ_k) for i in 1:r])

#     # simulate du
#     dup = MA(X', k)
#     dup = Array{Float64}(vec(dup))
#     # calculate likelihood
#     # NOTE: length predicted equals iteration over time and not species!
#     duflat ~ MvNormal(dup, Σ*I)
#     return nothing
# end

# modelb = likelihood_du(Array{Float64}(X), Array{Float64}(vec(du)))
# myChains = @time sample(modelb, NUTS(10, 0.65, adtype = AutoZygote()), MCMCThreads(), Niter, nChains; progress=true, save_state=true)

# # plot(myChains)
# diagnostics_and_save_sim(myChains, problem)

