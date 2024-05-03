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
using BenchmarkTools, Profile
using PDFmerger: append_pdf!
using DataFrames
using MCMCChainsStorage, HDF5, MCMCChains
using StatsBase
using StaticArrays
include("_odes.jl")
include("_plot_utils.jl")

Random.seed!(42)
print(Threads.nthreads())

protein_name = "insilicoEX2"
<<<<<<< HEAD
OUTNAME = "v11_SMC"
=======
OUTNAME = "v9_SMC"
>>>>>>> 2bbf7be3b037698a3ce8eb4f66c1268e65a7b7f3

folderN = "results/inference/"*protein_name*"/"*OUTNAME*"/"
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

# static (and mutable) arrays
Xm = SMatrix{length(tp),s}(Xm)
A = SMatrix{r,s}(A)
N = SMatrix{s,r}(N)
p0 = MVector{r}(p0)
x0 = MVector{s}(x0)


# ----- settings -----
numParam = length(paramNames)
Niter = 10000
nChains = 8
nRepeats = 10
# noWarmup = 25

α_sigma = 1
θ_sigma = 1
α_k = 0.01
θ_k = 5

# --- plot prior distributions
d1 = StatsPlots.plot(Gamma(α_sigma,θ_sigma), legend=false, lc=:black, title = "prior σ\nα="*string(α_sigma)*", θ="*string(θ_sigma), dpi = 600)
d2 = StatsPlots.plot(Uniform(α_k,θ_k), legend=false, lc=:black, title = "prior k\nα="*string(α_k)*", θ="*string(θ_k), xlims = (0,25), dpi = 600)
Plots.vline!(d2, [p0], line=:dash, lc=:red)

d = StatsPlots.plot(d1,d2, layout = (1,2))
savefig(d, folderN*"prior.png")


# ----- likelihood function -----
# ----- mass action kinetics ODE
# ----- initialise static arrays -----
NP = @MArray rand(s,r)
du = @MVector rand(s)

# --- Jacobian
du0 = copy(x0)
u0 = copy(x0)
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> massaction_stable(du, u, p0, 0.0), du0, u0)
jacspy = spy(jac_sparsity,title="sparsity of Jacobian",markersize=1,colorbar=false, dpi = 600)
savefig(jacspy, folderN*"jacobian_sparsity.png")

# --- different problems
problem_0 = ODEProblem(massaction_stable, x0, tspan, p0)
# problem_0s = ODEProblem(massaction_static, x0, tspan, p0)

f! = ODEFunction(massaction_stable, jac = jacobian_stable; jac_prototype = float.(jac_sparsity))
problem_jac = ODEProblem(f!, x0, tspan, p0)

# @mtkbuild sys = modelingtoolkitize(problem_jac)
# problem_mtk = ODEProblem(sys, [], tspan, jac=true, sparse=true)

# --- plot
sol = @btime solve(problem_0, TRBDF2(), saveat=tporig; p=p0)
ini = plot(sol, title = "initial solution", dpi = 600)
savefig(ini, folderN*"initial_solution.png")



# ----- calculate likelihood
@model function likelihood(Xv, problem, x0, α_sigma=α_sigma, θ_sigma=θ_sigma, α_k=α_k, θ_k=θ_k)
    
    # priors
    Σ ~ Gamma(α_sigma, θ_sigma)
    k ~ Product([Uniform(α_k, θ_k) for i in 1:r])

    # simulate ODE
    predicted = solve(problem, TRBDF2(), saveat=tp; u0=x0, p=k)
    # x01 = solve(problem0, CVODE_Adams(linear_solver=:KLU), saveat=t1; u0=x0, p=k)
    # predicted = solve(problem, CVODE_Adams(linear_solver=:KLU), saveat=tp; u0=x01.u[2], p=k)

    # calculate likelihood
    # NOTE: length predicted equals iteration over time and not species!
    if predicted.retcode == ReturnCode.Success
        pred = vec(mapreduce(permutedims, vcat, predicted.u))
        Xv ~ MvNormal(pred, Σ*I)
    else
        print("\nwhoopsidupsi")
        Turing.@addlogprob! -Inf
    end
    return nothing
end


# ----- inference and diagnostic plots -----
# ----- inference
# define model
# model = likelihood(vec(Xm), problem_mtk)
model_nuts = likelihood(vec(Xm), problem_0, x0)

# run inference
<<<<<<< HEAD
# benchmark = @benchmark sample(model, SMC(), MCMCThreads(), 10, 1; progress=true, save_state=true)
# myChains = @time sample(model, SMC(), MCMCThreads(), Niter, nChains; progress=true, save_state=true)
# myChains = @time sample(model_nuts, SMC(), MCMCThreads(), Niter, nChains; progress=true, save_state=true)
global myChains = @time sample(model_nuts, SMC(), MCMCThreads(), Niter, nChains; progress=true, save_state=true)
diagnostics_and_save_sim(myChains, problem_0)
=======
# FIXME: NUTS AD type --> it is messing with the solver
# benchmark = @benchmark sample(model, SMC(), MCMCThreads(), 1, 1; progress=true, save_state=true)
# myChains = sample(model, NUTS(50, 0.65, adtype=ADTypes.AutoZygote()), MCMCThreads(), 100, nChains; progress=true, save_state=true)

# benchmark = @benchmark sample(model, SMC(), MCMCThreads(), 1, 1; progress=true, save_state=true)
myChains = sample(model, NUTS(), MCMCThreads(), Niter, nChains; thinning=10, progress=true, save_state=true)
diagnostics_and_save_sim(myChains)
>>>>>>> 2bbf7be3b037698a3ce8eb4f66c1268e65a7b7f3

# repeat
for NRP in 2:nRepeats

    chains_reloaded = h5open(folderN*"chain.h5", "r") do io
        MCMCChainsStorage.read(io, Chains)
    end
    
    global myChains = @time sample(model_nuts, SMC(), MCMCThreads(), Niter*NRP, nChains; progress=true, save_state=true, resume_from=chains_reloaded)
    diagnostics_and_save_sim(myChains, problem_0)

end

