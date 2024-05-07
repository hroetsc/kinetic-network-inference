
using Turing
using DifferentialEquations
using Zygote, Enzyme
using SciMLSensitivity
using Sundials
using StatsPlots, Plots.Measures
using LinearAlgebra
using ModelingToolkit, Symbolics, Catalyst
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


# ----- INPUT -----
R"load(paste0('results/graphs/IDH1_WT_v3.RData'))" # TODO: make protein_name variable
@rget DATA;

X = Array{Union{Missing,Float64}}(DATA[:S]) .+ 1.0
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
global m = zeros(r)

α_k = 3
θ_k = 1
p0 = rand(Gamma(α_k,θ_k), r)

# ----- benchmarks for MA ODES -----
u = rand(s) .+ 1
u[sample(1:s, 500, replace=false)] .= 1.0
p = copy(p0)

# --- x^A
m0 = @btime prod((transpose(u) .^ A), dims=2) # transpose does not make difference
@btime (transpose(u) .^ A) # this is already half the compute time and almost all of the memory

m2 = @btime exp.(A*log.(u))

Au = zeros(r)
@btime exp.(mul!(Au, A, log.(u)))


# sanity check
describe(m0)
describe(m2)
all(round.(vec(m0); digits=2) == round.(m2; digits=2))


# --- du - classic *
du = @btime N*Diagonal(p)*m

# --- du - mul!
NP = zeros(s,1)
du = zeros(s,1)

@btime mul!(NP, N, p) # no diagonal!
@btime mul!(du, NP, m)
# much lower memory usage!


global m = zeros(r)
global NP = zeros(s,r)
du = zeros(s)

# version 1
@btime mul!(m, A, log.(u))
@btime mul!(NP, N, Diagonal(p))
@btime mul!(du, NP, exp.(m))

# version 2
@btime mul!(m, A, log.(u))
@btime mP = p.*exp.(m)
@btime mul!(du, N, mP)


problem_0 = ODEProblem(massaction_fast, x0, tspan, p0)
sol_0 = @btime solve(problem_0, TRBDF2(autodiff=false), saveat=tporig; p=p0)
plot(sol_0)

problem_0s = ODEProblem(massaction_stable, x0, tspan, p0)
sol_0s = @btime solve(problem_0s, TRBDF2(), saveat=tporig; p=p0)
plot(sol_0s)


# ----- benchmarks for Jacobian -----
global m = zeros(r)
J = zeros(s,s)
Au = zeros(r,s)
Npm = zeros(s)
Np = zeros(s,r)

@btime mul!(m, A, log.(u))
# Npm - version 2
@btime mul!(Np, N, Diagonal(p))
@btime mul!(Npm, Np, Diagonal(exp.(m)))

@btime mul!(Au, A, inv(Diagonal(u)))
@btime mul!(J,Npm,Au)

M = @btime Diagonal(exp.(A*log.(u)))
J2 = @btime N*Diagonal(p)*M*A*inv(Diagonal(u))

# apply
du0 = copy(x0)
u0 = copy(x0)
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> massaction_stable(du, u, p0, 0.0), du0, u0)

f! = ODEFunction(massaction_fast, jac = jacobian_fast; jac_prototype = float.(jac_sparsity))
problem_jac = ODEProblem(f!, x0, tspan, p0)
sol_jak = @btime solve(problem_jac, TRBDF2(), saveat=tporig; p=p0)
plot(sol_jak)

fs! = ODEFunction(massaction_stable, jac = jacobian_stable; jac_prototype = float.(jac_sparsity))
problem_jacs = ODEProblem(fs!, x0, tspan, p0)
sol_jaks = @btime solve(problem_jacs, TRBDF2(), saveat=tporig; p=p0)
plot(sol_jak)



# ----- correctness -----
global m = zeros(r)
global J = zeros(s,s)
global Au = zeros(r,s)
global Npm = zeros(s,r)
global Np = zeros(s,r)

Random.seed!(236)
u = rand(s) .+ 1
u[sample(1:s, 500, replace=false)] .= 1.0

# --- du
mul!(m, A, log.(u))
pm = p.*exp.(m)
mul!(du, N, pm)

ms = prod((transpose(u) .^ A), dims=2)
dus = N*Diagonal(p)*ms

describe(du)
describe(dus)
all(round.(vec(du); digits=2) == round.(vec(dus); digits=2))

# --- J
mul!(m, A, log.(u))
mul!(Np, N, Diagonal(p))
mul!(Npm, Np, Diagonal(exp.(m)))
mul!(J, Npm, A)
# mul!(Au, A, inv(Diagonal(u)))
mul!(J,J,inv(Diagonal(u)))
mul!(J,Npm,Au)

MS = Diagonal(vec(prod((transpose(u) .^ A), dims=2)))
NPM2 = N*Diagonal(p)*MS
JS = N*Diagonal(p)*MS*A*inv(Diagonal(u))


all(round.(vec(J); digits=1) == round.(vec(JS); digits=1))
all(round.(vec(Npm); digits=1) == round.(vec(NPM2); digits=1))


plot(spy(J), spy(JS), layout=(1,2))
plot(spy(Npm), spy(NPM2), layout=(1,2))