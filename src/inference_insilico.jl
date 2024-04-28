### kinetic-network-inference ###
# description:  run inference
# author:       HPR

using Turing
using DifferentialEquations
using Zygote, Enzyme
using SciMLSensitivity
using Sundials
using StatsPlots
using LinearAlgebra
using Symbolics
using ModelingToolkit
using SparseArrays
using Random
using RCall
using CSV
using Serialization
using BenchmarkTools
using PDFmerger: append_pdf!
using DataFrames
using MCMCChainsStorage, HDF5, MCMCChains
using StatsBase
include("_odes.jl")
include("_plot_utils.jl")

Random.seed!(42)
print(Threads.nthreads())

protein_name = "insilicoEX2"
OUTNAME = "v6_SMC"

folderN = "results/inference/"*protein_name*"/"*OUTNAME*"/"
mkpath(folderN)

# ----- INPUT -----
R"load(paste0('data/simulation/Ex2_DATA.RData'))"
@rget DATA;

X = Array{Union{Missing,Float64}}(DATA[:X])
A = Matrix{Int64}(DATA[:A])
B = Matrix{Int64}(DATA[:B])
x0 = Array{Float64}(DATA[:x0])

species = DATA[:species]
reactions = DATA[:reactions]
info = DATA[:info]
p = info.rate
tporig = DATA[:tp]

s = length(species)
r = length(reactions)
paramNames = ["σ"; info.rate_name]

t1 = 0.01
tinit = [0.0, t1]
tspan = [t1, maximum(tporig)]

tp = tporig[tporig .> 0]
Xm = X[2:size(X)[1],:]

N = transpose(B-A)


# ----- settings -----
numParam = length(paramNames)
Niter = 10000
nChains = 8
nRepeats = 10
# noWarmup = 25

α_sigma = 3
θ_sigma = 3
α_k = 1
θ_k = 1

# --- plot prior distributions
d1 = StatsPlots.plot(InverseGamma(α_sigma,θ_sigma), legend=false, lc=:black, title = "prior σ\nα="*string(α_sigma)*", θ="*string(θ_sigma), dpi = 600)
d2 = StatsPlots.plot(InverseGamma(α_k,θ_k), legend=false, lc=:black, title = "prior k\nα="*string(α_k)*", θ="*string(θ_k), xlims = (0,25), dpi = 600)
Plots.vline!(d2, [p], line=:dash, lc=:red)

d = StatsPlots.plot(d1,d2, layout = (1,2))
savefig(d, folderN*"prior.png")


# ----- likelihood function -----
# ----- mass action kinetics ODE
# --- Jacobian
du0 = copy(x0)
u0 = copy(x0)
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> massaction!(du, u, p, 0.0), du0, u0)
jacspy = spy(jac_sparsity,title="sparsity of Jacobian",markersize=1,colorbar=false, dpi = 600)
savefig(jacspy, folderN*"jacobian_sparsity.png")

# --- modelingtoolkit solution
# @mtkbuild sys = modelingtoolkitize(problem_jac)
# problem_mtk = ODEProblem(sys, [], tspan, jac=true, sparse=true)

# --- initial ODE simulation slow and then faster version
# initial time step
f0! = ODEFunction(massaction_init!, jac = jacobian!; jac_prototype = float.(jac_sparsity))
problem_jac0 = ODEProblem(f0!, x0, tinit, p)
x01 = @btime solve(problem_jac0, CVODE_Adams(linear_solver=:KLU), saveat=t1; p=p)

# all following time steps
f! = ODEFunction(massaction!, jac = jacobian!; jac_prototype = float.(jac_sparsity))
problem_jac = ODEProblem(f!, x01.u[2], tspan, p)
initial_sol = @btime solve(problem_jac, CVODE_Adams(linear_solver=:KLU), saveat=tp; p=p)

# combine solutions
combined_t = vcat(x01.t, initial_sol.t)
combined_u = mapreduce(permutedims, vcat, vcat(x01.u, initial_sol.u))

# --- plot
ini = plot(combined_t, combined_u, title = "initial solution", dpi = 600)
savefig(ini, folderN*"initial_solution.png")


# ----- calculate likelihood
@model function likelihood(Xm, problem0, problem, α_sigma=α_sigma, θ_sigma=θ_sigma, α_k=α_k, θ_k=θ_k)
    
    # priors
    Σ ~ InverseGamma(α_sigma, θ_sigma) # TODO: check for better distribution?
    k ~ Product([InverseGamma(α_k, θ_k) for i in 1:r])

    # simulate ODE
    x01 = solve(problem0, CVODE_Adams(linear_solver=:KLU), saveat=t1; u0=x0, p=p)
    predicted = solve(problem, CVODE_Adams(linear_solver=:KLU), saveat=tp; u0=x01.u[2], p=p)

    # calculate likelihood
    for (ii, pred) in enumerate(predicted)
        s = Xm[:,ii]
        s ~ MvNormal(vec(pred), Σ^2 * I)
    end

    return nothing
end


# ----- inference and diagnostic plots -----
# ----- inference
# define model
model = likelihood(Xm, problem_jac0, problem_jac)

# run inference
# FIXME: NUTS AD type --> it is messing with the solver
# benchmark = @benchmark sample(model, SMC(), MCMCThreads(), 1, 1; progress=true, save_state=true)
# myChains = sample(model, NUTS(50, 0.65, adtype=ADTypes.AutoZygote()), MCMCThreads(), 100, nChains; progress=true, save_state=true)

# benchmark = @benchmark sample(model, SMC(), MCMCThreads(), 1, 1; progress=true, save_state=true)
myChains = sample(model, SMC(), MCMCThreads(), Niter, nChains; progress=true, save_state=true)
diagnostics_and_save_sim(myChains)

# repeat
for NRP in 2:nRepeats

    chains_reloaded = h5open(folderN*"chain.h5", "r") do io
        MCMCChainsStorage.read(io, Chains)
    end
    
    myChains = sample(model, SMC(), MCMCThreads(), Niter*NRP, nChains; progress=true, save_state=true, resume_from=chains_reloaded)
    diagnostics_and_save_sim(myChains)

end

