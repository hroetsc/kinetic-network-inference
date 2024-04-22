### kinetic-network-inference ###
# description:  run inference
# author:       HPR

using Turing
using DifferentialEquations
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

Random.seed!(42)
epsilon = 1e-03

protein_name = "IDH1_WT"
OUTNAME = "test_9"

folderN = "results/inference/"*protein_name*"/"*OUTNAME*"/"
mkpath(folderN)

# ----- INPUT -----
R"load(paste0('results/graphs/IDH1_WT.RData'))" # TODO: make protein_name variable
@rget DATA;

X = Array{Union{Missing,Float64}}(DATA[:S] .+ 1)
A = Matrix{Int64}(DATA[:A])
B = Matrix{Int64}(DATA[:B])
x0 = Array{Float64}(X[1,:,1])

tp = Vector{Float64}(DATA[:timepoints])
replicates = DATA[:replicates]
species = DATA[:species]
reactions = DATA[:reactions]

s = length(species)
r = length(reactions)
paramNames = [reactions; "σ"]
tspan = (minimum(tp),maximum(tp))
nr = length(replicates)
N = transpose(B-A)


# ----- settings -----
numParam = length(paramNames)
Niter = 1
nChains = 1

# TODO: different prior for on and off rates
mini = 0
maxi = 100
midi = fill(0.5, r)
sigi = fill(0.5, r)

p0 = rand(r)

# ---- sample from prior distribution
# FIXME / TODO
# @model function prior(Σ, k)
#     Σ ~ InverseGamma(2, 3)
#     k ~ Product([truncated(Normal(mu,sigma), mini, maxi) for (mu,sigma) in zip(midi,sigi)])
#     return Σ, k
# end

# prior_sample = likelihood(missing, missing)
# prior_sample()


# ----- likelihood function -----
# ----- mass action kinetics ODE
function massaction!(du, u, p, t)
    # logx = replace!(log.(u), -Inf => 0.0)
    u = max.(u, 1.0)
    m = exp.(A*log.(u))
    du[1:s] = N*Diagonal(p)*m
    nothing
end

function jacobian!(J, u, p, t)
    # logx = replace!(log.(u), -Inf => 0.0)
    # M = Diagonal(exp.(A*log.(u .+ epsilon)))
    u = max.(u, 1.0)
    M = Diagonal(exp.(A*log.(u)))
    J[1:s,1:s] = N*Diagonal(p)*M*A*inv(Diagonal(u))
    nothing
end

du0 = copy(x0)
u0 = copy(x0)
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> massaction!(du, u, p0, 0.0), du0, u0)
jacspy = spy(jac_sparsity,title="sparsity of Jacobian",markersize=1,colorbar=false, dpi = 600)
savefig(jacspy, folderN*"jacobian_sparsity.png")

f! = ODEFunction(massaction!, jac = jacobian!; jac_prototype = float.(jac_sparsity))
problem_jac = ODEProblem(f!, x0, tspan, p0)

@mtkbuild sys = modelingtoolkitize(problem_jac)
problem_mtk = ODEProblem(sys, [], tspan, jac=true, sparse=true)


initial_sol_adams = @btime solve(problem_mtk, CVODE_Adams(linear_solver=:KLU), saveat=tp; p=p0)
plot(initial_sol_adams, title = "initial solution - CVODE_Adams")
# initial_sol_rodas = @btime solve(problem_mtk, Rodas5P(), saveat=tp; p=p0)
# plot(initial_sol_rodas, title = "initial solution - Rodas5P")
# initial_sol_tmp = @btime solve(problem_mtk, Rosenbrock23(), saveat=tp; p=p0)
# plot(initial_sol_tmp, title = "initial solution - TRBDF2")

# ----- calculate likelihood
@model function likelihood(X, problem)
    
    # priors
    Σ ~ InverseGamma(2, 3) # TODO: check for better distribution?
    k ~ Product([truncated(Normal(mu,sigma), mini, maxi) for (mu,sigma) in zip(midi,sigi)])
    print(length(k))

    # simulate ODE
    # predicted = solve(problem, Rodas5P(), progress = true, saveat=tp; p=k)
    predicted = solve(problem, CVODE_Adams(linear_solver=:KLU), saveat=tp; p=k)
    print("prediction done")
    # print(predicted.u)

    # calculate likelihood
    # iterate replicates
    for i in 1:nr
        # iterate products
        for (ii, pred) in enumerate(predicted)
            s = X[ii,:,i]
            k = findall(!ismissing, s) # remove missing vales
            s[k] ~ MvNormal(vec(pred)[k], Σ^2 * I)
        end
    end

    return nothing
end


# ----- inference and diagnostic plots -----
# ----- diagnostics
function diagnostics_and_save(chain)

    # save chain
    open(folderN*"chain.jls", "w") do io
        serialize(io, chain)
    end

    # summary stats
    chain_dsc = describe(chain)
    CSV.write(folderN*"summary_stats.csv", chain_dsc[:1])
    CSV.write(folderN*"quantiles.csv", chain_dsc[:2])

    # plot chain
    chain_pl = plot(chain, dpi = 600)
    savefig(chain_pl, folderN*"chain.pdf")

    # TODO: residual plots

end


# ----- inference
# define model
model = likelihood(X, problem_jac)

# run inference
myChains = @benchmark sample(model, NUTS(), MCMCThreads(), Niter, nChains; progress=true, save_state=true)

diagnostics_and_save(myChains)

# repeat
for N in 2:100

    open(folderN*"chain.jls", "r") do io
        chains_reloaded = deserialize(io)
    end
    
    myChains = sample(model, NUTS(), MCMCThreads(), N*Niter, nChains; progress=true, save_state=true, resume_from=chains_reloaded)
    diagnostics_and_save(myChains)

end

