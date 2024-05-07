### kinetic-network-inference ###
# description:  run inference
# author:       HPR

using Turing
using DifferentialEquations
using Zygote, Enzyme
using SciMLSensitivity
using Sundials
using StatsPlots, Plots.Measures
using LinearAlgebra
using ModelingToolkit, Symbolics
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

protein_name = "IDH1_WT"
OUTNAME = "NUTS_v1"

folderN = "results/inference/"*protein_name*"/"*OUTNAME*"/"
mkpath(folderN)


# ----- INPUT -----
R"load(paste0('results/graphs/IDH1_WT_v3.RData'))" # TODO: make protein_name variable
@rget DATA;

X = Array{Union{Missing,Float64}}(DATA[:S]) .+ 1
A = Matrix{Int64}(DATA[:A])
B = Matrix{Int64}(DATA[:B])
x0 = Array{Float64}(coalesce.(X[1,:,1], 1.0))
N = transpose(B-A)

species = DATA[:species]
reactions = DATA[:reactions]
tporig = DATA[:timepoints]
replicates = DATA[:replicates]

s = length(species)
r = length(reactions)
nr = length(replicates)
paramNames = ["σ"; reactions]

tspan = [minimum(tporig), maximum(tporig)]
tp = tporig[tporig .> 0]
Xm = X[2:size(X)[1],:,:]

# global constants
global m = zeros(r)
global Au = zeros(r,s)
global Npm = zeros(s,r)
global Np = zeros(s,r)

# ----- settings -----
numParam = length(paramNames)
Niter = 10
nChains = 3
nRepeats = 10

α_sigma = 1
θ_sigma = 1
α_k = 3
θ_k = 1


# --- plot prior distributions
d1 = StatsPlots.plot(Gamma(α_sigma,θ_sigma), legend=false, lc=:black, title = "prior σ\nα="*string(α_sigma)*", θ="*string(θ_sigma), dpi = 600)
d2 = StatsPlots.plot(Gamma(α_k,θ_k), legend=false, lc=:black, title = "prior k\nα="*string(α_k)*", θ="*string(θ_k), xlims = (0,25), dpi = 600)

d = StatsPlots.plot(d1,d2, layout = (1,2))
savefig(d, folderN*"prior.png")

# sample from prior to get p0
p0 = rand(Gamma(α_k,θ_k), r)


# ----- likelihood function -----
# ----- mass action kinetics ODE
# --- Jacobian
du0 = copy(x0)
u0 = copy(x0)
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> massaction_stable(du, u, p0, 0.0), du0, u0)
jacspy = spy(jac_sparsity,title="sparsity of Jacobian",markersize=1,colorbar=false, dpi = 600)
savefig(jacspy, folderN*"jacobian_sparsity.png")


# # --- initial ODE simulation slow and then faster version
# # initial time step
# f0! = ODEFunction(massaction_init!, jac = jacobian!; jac_prototype = float.(jac_sparsity))
# problem_jac0 = ODEProblem(f0!, x0, tinit, p0)
# x01 = @btime solve(problem_jac0, CVODE_Adams(linear_solver=:KLU), saveat=t1; p=p0)

# # all following time steps
# f! = ODEFunction(massaction!, jac = jacobian!; jac_prototype = float.(jac_sparsity))
# problem_jac = ODEProblem(f!, x01.u[2], tspan, p0)
# initial_sol = @btime solve(problem_jac, CVODE_Adams(linear_solver=:KLU), saveat=tp; p=p0)

# # combine solutions
# combined_t = vcat(x01.t, initial_sol.t)
# combined_u = mapreduce(permutedims, vcat, vcat(x01.u, initial_sol.u))


# --- modelingtoolkit and JAC solution
# f! = ODEFunction(massaction_fast, jac = jacobian_fast; jac_prototype = float.(jac_sparsity))
# problem_jac = ODEProblem(f!, x0, tspan, p0)
# sol_jac = @btime solve(problem_jac, TRBDF2(), saveat=tporig; p=p0)

# @mtkbuild sys = modelingtoolkitize(problem_jac)
# problem_mtk = ODEProblem(sys, [], tspan, jac=true, sparse=true)

fhvy! = ODEFunction(massaction_stable, jac = jacobian_fast; jac_prototype = float.(jac_sparsity))
problem_jachvy = ODEProblem(fhvy!, x0, tspan, p0)
@mtkbuild sys = modelingtoolkitize(problem_jachvy)
problem_mtk = ODEProblem(sys, [], tspan, jac=true, sparse=true)
sol_mtk = @btime solve(problem_mtk, TRBDF2(), saveat=tporig; p=p0)


problem = problem_mtk


# --- "pure" ODE solution
# problem_0 = ODEProblem(massaction_fast, x0, tspan, p0)
# sol_0 = @btime solve(problem_0, TRBDF2(), saveat=tporig; p=p0)

# --- plot
ini = plot(sol_mtk, title = "initial solution", dpi = 600)
savefig(ini, folderN*"initial_solution.png")


# ----- calculate likelihood
@model function likelihood(Xv, problem, x0, ki=ki, α_sigma=α_sigma, θ_sigma=θ_sigma, α_k=α_k, θ_k=θ_k)
    
    # priors
    Σ ~ Gamma(α_sigma, θ_sigma)
    k ~ Product([Gamma(α_k, θ_k) for i in 1:r])

    # simulate ODE
    predicted = solve(problem, TRBDF2(), saveat=tp; u0=x0, p=k)

    # calculate likelihood
    if predicted.retcode == ReturnCode.Success
        pred = repeat(vec(mapreduce(permutedims, vcat, predicted.u)),nr)
        Xv ~ MvNormal(pred[ki], Σ*I)
    else
        print("\nwhoopsidupsi")
        Turing.@addlogprob! -Inf
    end

    return nothing
end


# ----- inference and diagnostic plots -----
# ----- inference
# define model
Xv = vec(Xm)
ki = findall(!ismissing, Xv)
Xvv = Array{Float64}(Xv[ki])
model = likelihood(Xvv, problem, x0)

# run inference
# myChains = @time sample(model,  MH(), MCMCThreads(), Niter, nChains; progress=true, save_state=true)
myChains = @time sample(model,  NUTS(1, 0.65, adtype=AutoZygote()), MCMCThreads(), Niter, nChains; progress=true, save_state=true)
diagnostics_and_save(myChains, problem)

# repeat
for NRP in 2:nRepeats

    chains_reloaded = h5open(folderN*"chain.h5", "r") do io
        MCMCChainsStorage.read(io, Chains)
    end    
    
    myChains = @time sample(model, NUTS(1, 0.65, adtype=AutoZygote()), MCMCThreads(), Niter*NRP, nChains; progress=true, save_state=true, resume_from=chains_reloaded)
    diagnostics_and_save(myChains, problem)

end

