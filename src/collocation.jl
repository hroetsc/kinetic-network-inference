### kinetic-network-inference ###
# description:  try collocation based solution
# author:       HPR
# based on: https://doi.org/10.1515/sagmb-2020-0025 and https://doi.org/10.1198/016214508000000797


using DifferentialEquations
using Zygote
using StatsPlots, Plots.Measures
using LinearAlgebra
using Random
using RCall
using CSV
using BenchmarkTools
using DataFrames
using StatsBase
using DiffEqParamEstim
using OrdinaryDiffEq, DiffEqFlux, Flux, Optim, Lux
using Optimisers
using Printf
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


# ----- choose kernel -----
# TODO: try different bandwidths
# TODO: set up neural ODE

m = length(tp) # number of time points
h = m^(-1/5)*m^(-3/35)*log(m)^(-1/16)

kernels = [EpanechnikovKernel(), UniformKernel(), TriangularKernel(), QuarticKernel(), TriweightKernel(), 
TricubeKernel(), DiffEqFlux.GaussianKernel(), DiffEqFlux.CosineKernel(), LogisticKernel(), SigmoidKernel(), SilvermanKernel()]

pp = []
for kernel in kernels

    ks = string(kernel)

    # get estimates of du and u
    du, u = @time collocate_data(transpose(Xm), tp, kernel, h*1.1) # from DiffEqFlux, DiffEqParamEstim packages

    # plot
    pl1 = plot(tporig, X, lc=:black, title="u, "*ks, legend=false, xlabel = "digestion time [hrs]", ylabel = "signal (u)", dpi=600, margin=5mm)
    plot!(tp, transpose(u), lc=:red)
    pl2 = plot(transpose(du), lc=:red, title = "du, "*ks,legend=false, xlabel = "digestion time [hrs]", ylabel = "du", dpi=600, margin=5mm)
    pl = plot(pl1,pl2, layout = (1,2))

    push!(pp, pl)
end
ppp = plot(pp...; size = default(:size) .* (1,11), layout=(11,1), dpi = 600, margin=10mm)
savefig(ppp, folderN*"estimated_abundance.pdf")



# ----- collocation -----
du, u = @time collocate_data(transpose(Xm), tp, EpanechnikovKernel(), h*1.1) # from DiffEqFlux, DiffEqParamEstim packages


# ----- construct neural net -----
model = Lux.Chain(Lux.Dense(m => m))

# ----- set up training -----
opt = Optimisers.Adam(0.03f0)

function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data[1], ps, st)
    mse_loss = mean(abs2, y_pred .- data[2])
    return mse_loss, st, ()
end

tstate = Lux.Experimental.TrainState(rng, model, opt)
vjp_rule = AutoZygote()

function main(tstate::Lux.Experimental.TrainState, vjp, data, epochs)
    data = data .|> gpu_device()
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(
            vjp, loss_function, data, tstate)
        if epoch % 50 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end

dev_cpu = cpu_device()
dev_gpu = gpu_device()


# ----- train and predict -----
tstate = main(tstate, vjp_rule, (Xm, du'), 5000)
y_pred = dev_cpu(Lux.apply(tstate.model, dev_gpu(Xm), tstate.parameters, tstate.states)[1])


# ----- evaluate -----
plot(du', lc=:black, legend=false)
plot!(y_pred, lc=:red)



