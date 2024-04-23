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
using PDFmerger: append_pdf!
using DataFrames

Random.seed!(42)
print(Threads.nthreads())

protein_name = "insilicoEX2"
OUTNAME = "v2"

folderN = "results/inference/"*protein_name*"/"*OUTNAME*"/"
mkpath(folderN)

# ----- INPUT -----
R"load(paste0('../aQUIRE-network/results/matrices/DATA_Ex2.RData'))" # TODO: make protein_name variable
@rget DATA;

X = Array{Union{Missing,Float64}}([DATA[:S] .+ 1 (DATA[:S0].-DATA[:DeltaC0])[:,1] .+ 1])
A = Matrix{Int64}(DATA[:A])
B = Matrix{Int64}(DATA[:B])
x0 = Array{Float64}(X[1,:])

species = DATA[:species]
reactions = DATA[:reactions]
p = reactions.rate

tp = [0.0, 1.0, 2.0, 3.0, 4.0]
s = length(species)+1
r = length(p)
paramNames = [reactions.rate_name; "σ"]
tspan = (minimum(tp),maximum(tp))

N = transpose(B-A)


# ----- settings -----
numParam = length(paramNames)
Niter = 250000
nChains = 8
noWarmup = 250

# TODO: different prior for on and off rates
mini = 0
maxi = 100
midi = p
sigi = fill(1.0, r)



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
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> massaction!(du, u, p, 0.0), du0, u0)
jacspy = spy(jac_sparsity,title="sparsity of Jacobian",markersize=1,colorbar=false, dpi = 600)
savefig(jacspy, folderN*"jacobian_sparsity.png")

f! = ODEFunction(massaction!, jac = jacobian!; jac_prototype = float.(jac_sparsity))
problem_jac = ODEProblem(f!, x0, tspan, p)

# @mtkbuild sys = modelingtoolkitize(problem_jac)
# problem_mtk = ODEProblem(sys, [], tspan, jac=true, sparse=true)

# initial_sol_adams = @btime solve(problem_mtk, CVODE_Adams(linear_solver=:KLU), saveat=tp; p=p0)
# plot(initial_sol_adams, title = "initial solution - CVODE_Adams")
# initial_sol_rodas = @btime solve(problem_mtk, Rodas5P(), saveat=tp; p=p0)
# plot(initial_sol_rodas, title = "initial solution - Rodas5P")
# initial_sol_tmp = @btime solve(problem_mtk, Rosenbrock23(), saveat=tp; p=p0)
# plot(initial_sol_tmp, title = "initial solution - TRBDF2")

# ----- calculate likelihood
@model function likelihood(X, problem)
    
    # priors
    Σ ~ InverseGamma(2, 3) # TODO: check for better distribution?
    k ~ Product([truncated(Normal(mu,sigma), mini, maxi) for (mu,sigma) in zip(midi,sigi)])

    # simulate ODE
    # predicted = solve(problem, Rodas5P(), progress = true, saveat=tp; p=k)
    predicted = solve(problem, CVODE_Adams(linear_solver=:KLU), saveat=tp; p=k)
    # print(predicted.u)

    # calculate likelihood
    for (ii, pred) in enumerate(predicted)
        s = X[:,ii]
        s ~ MvNormal(vec(pred), Σ^2 * I)
    end

    return nothing
end


# ----- inference and diagnostic plots -----
# ----- diagnostics
function diagnostics_and_save(myChains)

    # save chain
    open(folderN*"chain.jls", "w") do io
        serialize(io, myChains)
    end

    # summary stats
    chain_dsc = describe(myChains)
    CSV.write(folderN*"summary_stats.csv", chain_dsc[:1])
    CSV.write(folderN*"quantiles.csv", chain_dsc[:2])

    # plot chain
    # rm(folderN*"chain.pdf")
    # touch(folderN*"chain.pdf")
    
    nm = DataFrame(chain_dsc[:1]).parameters
    chains = Array(myChains)
    for i in 1:size(chains)[2]
        p = plot(chains[:,i], title = nm[i] , dpi = 600)
        savefig(p, "tmp.pdf")
        append_pdf!(folderN*"chain.pdf", "tmp.pdf", create=true, cleanup=true)
    end

    # TODO: residual plots
    # TODO: plot sampler https://turinglang.org/dev/docs/using-turing/sampler-viz

end


# ----- inference
# define model
model = likelihood(X, problem_jac)

# run inference
# FIXME: NUTS AD type --> it is messing with the solver
benchmark = @benchmark sample(model, MH(), MCMCThreads(), 1, 1; progress=true, save_state=true)
# myChains = sample(model, MH(), MCMCThreads(), Niter, nChains; progress=true, save_state=true)

myChains = sample(model, NUTS(50, 0.65, adtype=AutoZygote()), MCMCThreads(), 100, nChains; progress=true, save_state=true)

diagnostics_and_save(myChains)

# repeat
for N in 2:100

    chains_reloaded = nothing
    chains_reloaded = open(folderN*"chain.jls", "r") do io
        deserialize(io)
    end
    
    myChains = sample(model, MH(), MCMCThreads(), N*Niter, nChains; progress=true, save_state=true, resume_from=chains_reloaded)
    diagnostics_and_save(myChains)

end


