### kinetic-network-inference ###
# description:  try collocation based solution
# author:       HPR
# based on: https://doi.org/10.1515/sagmb-2020-0025 and https://doi.org/10.1198/016214508000000797

using DiffEqParamEstim
using OrdinaryDiffEq
using DiffEqFlux
using Lux
using Optimisers
using Zygote
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
include("_odes.jl")

Random.seed!(42)
print(Threads.nthreads())

protein_name = "simulated"
OUTNAME = "test_t"
folderN = "results/collocation/"*protein_name*"/"*OUTNAME*"/"
mkpath(folderN)

# ----- INPUT -----
R"load(paste0('data/simulation/Ex2_DATA.RData'))"
@rget DATA;

X = Array{Union{Missing,Float64}}(DATA[:X])
A = Matrix{Int64}(DATA[:A])
B = Matrix{Int64}(DATA[:B])
x0 = Array{Float64}(DATA[:x0])
N = transpose(B-A)

species = DATA[:species]
reactions = DATA[:reactions]
info = DATA[:info]
p0 = info.rate
tporig = DATA[:tp]

s = length(species)
r = length(reactions)
paramNames = ["σ"; info.rate_name]

tspan = [minimum(tporig), maximum(tporig)]
tp = tporig[tporig .> 0]
Xm = Array{Float64}(X[2:size(X)[1],:])
Xmt = Xm .+ 1

# ----- choose kernel -----
# TODO: try different bandwidths

mt = length(tp) # number of time points
h = mt^(-1/5)*mt^(-3/35)*log(mt)^(-1/16)

kernels = [EpanechnikovKernel(), UniformKernel(), TriangularKernel(), QuarticKernel(), TriweightKernel(), 
TricubeKernel(), DiffEqFlux.GaussianKernel(), DiffEqFlux.CosineKernel(), LogisticKernel(), SigmoidKernel(), SilvermanKernel()]

pp = []
for kernel in kernels

    ks = string(kernel)
    print(ks)

    # get estimates of du and u
    du, u = @time collocate_data(Xm', tp, kernel, h*1.1) # from DiffEqFlux, DiffEqParamEstim packages

    # plot
    pl1 = plot(tporig, X, lc=:black, title="u, "*ks, legend=false, xlabel = "digestion time [hrs]", ylabel = "signal (u)", dpi=600, margin=5mm)
    plot!(tp, u', lc=:red)
    pl2 = plot(du', lc=:red, title = "du, "*ks,legend=false, xlabel = "digestion time [hrs]", ylabel = "du", dpi=600, margin=5mm)
    pl = plot(pl1,pl2, layout = (1,2))

    push!(pp, pl)
end
ppp = plot(pp...; size = default(:size) .* (1,11), layout=(11,1), dpi = 600, margin=10mm)
savefig(ppp, folderN*"estimated_abundance.pdf")


# ----- collocation -----
du, u = @time collocate_data(Xm', tp, EpanechnikovKernel(), h*1.1) # from DiffEqFlux, DiffEqParamEstim packages

pl1 = plot(tporig, X, lc=:black, title="u", legend=false, xlabel = "digestion time [hrs]", ylabel = "log (signal+1) (u)", dpi=600, margin=5mm)
plot!(tp, u', lc=:red)
pl2 = plot(du', lc=:red, title = "du",legend=false, xlabel = "digestion time [hrs]", ylabel = "du", dpi=600, margin=5mm)
pl = plot(pl1,pl2, layout = (1,2))

# ----- construct neural net -----
# global m = zeros(r)
# # global J = zeros(s,s)
# # global Au = zeros(r,s)
# # global Npm = zeros(s,r)
# # global Np = zeros(s,r)
# global du = zeros(s)

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
    # mul!(m, A, log.(x))
    # pm = ps.weight .* exp.(m)
    # du = mul!(du, N, pm)

    m = exp.(A*log.(x))
    du = N*Diagonal(ps)*m
    return du, st
end

model = Chain(MALayer(;dims=r))



# ----- set up training -----
rng = MersenneTwister(42)
opt = Optimisers.Adam(0.01f0)

function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data[1], ps, st)
    mse_loss = sqrt(mean(abs2, y_pred .- data[2]))
    return mse_loss, st, ()
end

tstate = Lux.Experimental.TrainState(rng, model, opt)
vjp_rule = AutoZygote()

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

dev_cpu = cpu_device()
dev_gpu = gpu_device()


# ----- train and predict -----
tstate, loss = @time main(tstate, vjp_rule, (Xmt', du), 1000)
y_pred = dev_cpu(Lux.apply(tstate.model, dev_gpu(Xmt'), tstate.parameters, tstate.states)[1])


# ----- evaluate -----
plot(du', lc=:black, title = "du", legend=false, xlabel = "digestion time [hrs]", ylabel = "du", dpi=600)
plot!(y_pred', lc=:red)

scatter(p0, tstate.parameters, legend=false,
xlabel = "true parameter", ylabel = "predicted parameter",
xlim = (0,maximum(p0)), ylim = (0,maximum(p0)))

plot(loss, title = "training loss", ylim = (0,maximum(loss)*1.5),
xlabel = "epoch", ylabel = "loss", legend=false, lc=:black)


