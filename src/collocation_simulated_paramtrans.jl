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

protein_name = "Ex4_20s"
OUTNAME = "paramtrans"
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
tpoints = collect(range(0.5,4.0,50))

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
# du_intps[:,1] .= du_intps[:,2]
# du_intps[k,1] .= 0.0

closest_indices = [findmin(abs.(tpoints .- tpo))[2] for tpo in tpm]
du_intps_d = du_intps[:,closest_indices]

pl1 = plot(tporig, X, lc=:black, title="u", legend=false, xlabel = "digestion time [hrs]", ylabel = "signal (u)", dpi=600, margin=5mm)
plot!(tpoints, u_intps', lc=:red)
pl2 = plot(tpm, du_intps_d', lc=:red, title = "du",legend=false, xlabel = "digestion time [hrs]", ylabel = "du", dpi=600, margin=5mm)
pl = plot(pl1,pl2, layout = (1,2))
savefig(pl, folderN*"estimated_abundance_chosen.png")


# ----- collocation -----
du, u = @time collocate_data(X', tporig, TriweightKernel(), h) # from DiffEqFlux, DiffEqParamEstim packages
du[k,1] .= 0.0


# ----- combine both -----
du_intps_d = hcat(du[:,1],du_intps_d)

errdu_d = mean(abs2.(vec(ducorrect - du_intps_d)))
print("\nerror on u: "*string(round(errdu_d;digits=2))*"\n")


# -----------------------
# ----- NN solution -----
# -----------------------
# # ----- construct neural net - mass action -----
# struct MALayer{F1} <: Lux.AbstractExplicitLayer
#     dims::Int
#     init_weight::F1
# end

# function MALayer(;dims::Int, init_weight=Lux.glorot_uniform)
#     return MALayer{typeof(init_weight)}(dims, init_weight)
# end

# function Lux.initialparameters(rng::AbstractRNG, MALayer::MALayer)
#     return (weight=MALayer.init_weight(rng, MALayer.dims))
# end
# Lux.initialstates(::AbstractRNG, ::MALayer) = NamedTuple()

# function (MALayer::MALayer)(x::AbstractMatrix, ps, st::NamedTuple, A=A, N=N)

#     m = exp.(A*log.(X' .+ eps))
#     du = N*Diagonal(x)*m
#     return du, st
# end



# ----- training functions -----
m = exp.(A*log.(X' .+ eps))
function loss_function(model, ps, st, data)
    p_pred, st = Lux.apply(model, data[1], ps, st)
    y_pred = N*Diagonal(p_pred)*m
    mse_loss = sqrt(mean(abs2, y_pred .- data[2]))
    # cor_loss = 1/cor(vec(y_pred[:,2:5]),vec(data[2][:,2:5]))
    # loss = cor_loss + mse_loss
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
# model = Chain(Lux.Dense(10,100,relu),Lux.Dense(100,nP,relu))
in_dim = 100
hidden_dim = 100
# model = Chain(Lux.Dense(in_dim,hidden_dim),Lux.Dense(hidden_dim,hidden_dim),Lux.Dense(hidden_dim,nP))
model = Chain(Lux.Dense(in_dim,hidden_dim),Lux.Dense(hidden_dim,nP))
rng = MersenneTwister(42)
opt = Optimisers.Adam(0.0001)
epochs = 10000

# p0s = rand(Gamma(1,1), 100)
p0s = rand(in_dim)

dev_cpu = cpu_device()
dev_gpu = gpu_device()

tstates = []
losses = []
ypreds = []
ppreds = []
for i in 1:10
    print("\n"*string(i)*"\n")

    tstate = Lux.Experimental.TrainState(rng, model, opt)
    vjp_rule = AutoZygote()
    
    Random.seed!(i)
    p0s = rand(in_dim)
    # tstate, loss = @time main(tstate, vjp_rule, (X', du), epochs)
    # ypred = dev_cpu(Lux.apply(tstate.model, dev_gpu(X'), tstate.parameters, tstate.states)[1])
    tstate, loss = @time main(tstate, vjp_rule, (p0s, du_intps_d), epochs)
    ppred = dev_cpu(Lux.apply(tstate.model, dev_gpu(p0s), tstate.parameters, tstate.states)[1])
    ypred = N*Diagonal(ppred)*m

    # append to list
    push!(tstates,tstate)
    push!(losses,loss)
    push!(ypreds, ypred)
    push!(ppreds, ppred)
end

diagnostics_and_save_NN_sim_multi(tstates, ypreds, ppreds, losses, true)

print("done")

