### kinetic-network-inference ###
# description:  run inference
# author:       HPR

using LinearAlgebra, Turing, DifferentialEquations, SciMLSensitivity, LSODA, Enzyme, StatsPlots, Random, RCall, CSV, Serialization

Random.seed!(42)

protein_name = "IDH1_WT"
OUTNAME = "test_8"

folderN = "results/inference/"*protein_name*"/"*OUTNAME*"/"
mkpath(folderN)

# ----- INPUT -----
R"load(paste0('results/graphs/IDH1_WT.RData'))" # TODO: make protein_name variable
@rget DATA;

S = DATA[:S]
A = DATA[:A]
B = DATA[:B]
x0 = Array{Float64}(transpose(S[1,:,1]))

tp = DATA[:timepoints]
replicates = DATA[:replicates]
species = DATA[:species]
reactions = DATA[:reactions]

s = length(species)
r = length(reactions)
# paramNames = [reactions; "sigma"]
tspan = (minimum(tp),maximum(tp))
nr = length(replicates)


# ----- settings -----
# numParam = length(paramNames)
Niter = 1
nChains = 1

# TODO: different prior for on and off rates
mini = -1
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
    M = prod((u .^ A), dims=2)
    du[1:s] = transpose(B-A)*(p .* M)
    return nothing
end


# ----- calculate likelihood
@model function likelihood(X, problem)
    
    # priors
    Σ ~ InverseGamma(2, 3) # TODO: check for better distribution?
    k ~ Product([truncated(Normal(mu,sigma), mini, maxi) for (mu,sigma) in zip(midi,sigi)])

    # simulate ODE
    # NOTE: need to change the automatic differentiation method into something that can deal with non-pure Julia solver
    # predicted = solve(problem, AutoTsit5(Rosenbrock23()), save_everystep = false; p=k)
    predicted = solve(problem, lsoda(), save_everystep = false, sensealg = GaussAdjoint(); p=k)

    # extract only time points that are matching with (or are closest to) the real data
    closest_indices = [findmin(abs.(predicted.t .- time_point))[2] for time_point in tp]
    prdct = predicted.u[closest_indices]

    # calculate likelihood
    # iterate replicates
    for i in 1:nr
        # iterate products
        for (ii, pred) in enumerate(prdct)
            s = X[ii,:,i]
            k = findall(!ismissing, s) # remove missing vales
            s[k] ~ MvNormal(vec(pred)[k], Σ^2 * I)
        end
    end

    return nothing
end

# TODO: use @benchmark to find out which part is the slowest

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
problem = ODEProblem(massaction!, x0, tspan, p0)
model = likelihood(S, problem)

# run inference
myChains = sample(model, NUTS(), MCMCThreads(), Niter, nChains; progress=true, save_state=true)
diagnostics_and_save(myChains)

# repeat
for N in 2:100

    open(folderN*"chain.jls", "r") do io
        chains_reloaded = deserialize(io)
    end
    
    myChains = sample(model, NUTS(), MCMCThreads(), N*Niter, nChains; progress=true, save_state=true, resume_from=chains_reloaded)
    diagnostics_and_save(myChains)

end

