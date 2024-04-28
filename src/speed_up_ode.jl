using Turing, DifferentialEquations, SciMLSensitivity, LSODA, Sundials, StatsPlots, LinearAlgebra, Random, RCall, CSV, Serialization
# using StaticArrays
using Symbolics
using ModelingToolkit
using BenchmarkTools
using IncompleteLU
using SparseArrays

Random.seed!(42)
epsilon = 1e-03

# ----- INPUT -----
R"load(paste0('results/graphs/IDH1_WT.RData'))"
@rget DATA;

X = Array{Union{Missing,Float64}}(DATA[:S])
A = Matrix{Int64}(DATA[:A])
B = Matrix{Int64}(DATA[:B])
x0 = Array{Float64}(X[1,:,1]) # not transpose!!

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

p0 = rand(r)

# ----------------------------------
# ----- clean and fast version -----
# ----------------------------------
# ----- ODEs -----
function massaction!(du, u, p, t)
    m = exp.(A*log.(u .+ epsilon))
    du[1:s] = N*Diagonal(p)*m
    nothing
end

function jacobian!(J, u, p, t)
    M = Diagonal(exp.(A*log.(u .+ epsilon)))
    J[1:s,1:s] = N*Diagonal(p)*M*A*inv(Diagonal(u.+ epsilon))
    nothing
end

du0 = copy(x0)
u0 = copy(x0)
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> massaction!(du, u, p0, 0.0), du0, u0)
spy(jac_sparsity,title="sparsity of Jacobian",markersize=1,colorbar=false)

f! = ODEFunction(massaction!, jac = jacobian!; jac_prototype = float.(jac_sparsity))
problem_jac = ODEProblem(f!, x0, tspan, p0)

@mtkbuild sys = modelingtoolkitize(problem_jac)
problem_mtk = ODEProblem(sys, [], tspan, jac=true, sparse=true)

sol_cvodeadams_mtk = @btime solve(problem_mtk, CVODE_Adams(), saveat=tp; p=p0)
plot(sol_cvodeadams_mtk)

sol_cvodeadams_mtk_lapack = @btime solve(problem_mtk, CVODE_Adams(linear_solver = :LapackDense), saveat=tp; p=p0)
plot(sol_cvodeadams_mtk_lapack)




# --------------------------
# ----- other attempts -----
# --------------------------

# R"load(paste0('../aQUIRE-network/results/matrices/DATA_Ex2.RData'))"
# @rget DATA;

# A = Matrix{Int64}(DATA[:A])
# B = Matrix{Int64}(DATA[:B])
# X = Array{Union{Missing,Float64}}(DATA[:S])
# x0 = Array{Float64}(X[1,:]).+ epsilon
# N = transpose(B-A)


# u = [x0; 100] .+ epsilon
# p = DATA[:reactions].rate

# M = exp.(A*log.(u))

# @btime logx = replace!(log.(u), -Inf => 0)
# @btime logx = log.(u)

# ----- ODEs -----
function massaction!(du, u, p, t)
    m = exp.(A*log.(u .+ epsilon))
    du[1:s] = N*Diagonal(p)*m
    nothing
end

function jacobian!(J, u, p, t)
    M = Diagonal(exp.(A*log.(u .+ epsilon)))
    J[1:s,1:s] = N*Diagonal(p)*M*A*inv(Diagonal(u.+ epsilon))
    nothing
end

# --- analytical Jacobian
du0 = copy(x0)
u0 = copy(x0)
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> massaction!(du, u, p0, 0.0), du0, u0)
spy(jac_sparsity,title="sparsity of Jacobian",markersize=1,colorbar=false)

f! = ODEFunction(massaction!, jac = jacobian!; jac_prototype = float.(jac_sparsity))
problem_jac = ODEProblem(f!, x0, tspan, p0)

@mtkbuild sys = modelingtoolkitize(problem_jac)
problem_mtk = ODEProblem(sys, [], tspan, jac=true, sparse=true)

# no preconditioning
sol_cvodeadams_jac = @btime solve(problem_jac, CVODE_Adams(); p=p0)
sol_cvodeadams_mtk = @btime solve(problem_mtk, CVODE_Adams(), saveat=tp; p=p0)
plot(sol_cvodeadams_mtk)


# preconditioning
u0 = problem_mtk.u0
p = problem_mtk.p
M = Diagonal(exp.(A*log.(x0 .+ epsilon)))
const jaccache = N*Diagonal(p0)*M*A*inv(Diagonal(x0.+ epsilon))
const WW = sparse(I - 1.0 * jaccache)

prectmp = ilu(WW, τ = 50.0)
const preccache = Ref(prectmp)

function psetupilu(p, t, u, du, jok, jcurPtr, gamma)
    if jok
        problem_mtk.f.jac(jaccache, u, p, t)
        jcurPtr[] = true

        # W = I - gamma*J
        @. WW = -gamma * jaccache
        idxs = diagind(WW)
        @. @view(WW[idxs]) = @view(WW[idxs]) + 1

        # Build preconditioner on W
        preccache[] = ilu(WW, τ = 5.0)
    end
end

function precilu(z, r, p, t, y, fy, gamma, delta, lr)
    ldiv!(z, preccache[], r)
end

sol_pre = @btime solve(problem_jac, CVODE_Adams(prec = precilu, psetup = psetupilu, prec_side = 1), saveat = tp)
plot(sol_pre)


# --- analytical Jacobian + automatic sparsity detection
du0 = copy(x0)
u0 = copy(x0)
jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> massaction!(du, u, p, 0.0), du0, u0)
spy(jac_sparsity,title="sparsity of Jacobian",markersize=1,colorbar=false)

fsp! = ODEFunction(massaction!, jac = jacobian!; jac_prototype = float.(jac_sparsity))
problem_jacsp = ODEProblem(fsp!, x0, tspan, p0)
sol_cvodeadams_jacsp = @btime solve(problem_jacsp, CVODE_Adams(); p=p0)


# --- tmp
@btime solve(problem_jac, CVODE_Adams(linear_solver = :LapackDense), save_everystep = false; p=p0)
@btime solve(problem_jac, CVODE_Adams(linear_solver = :GMRES), save_everystep = false; p=p0)
@btime solve(problem_jac, CVODE_Adams(linear_solver = :GMRES, method = :Newton), save_everystep = false; p=p0)
@btime solve(problem_jac, CVODE_Adams(linear_solver = :BCG, method = :Newton), save_everystep = false; p=p0)
solve(problem_jac, CVODE_Adams(linear_solver = :GMRES, method = :Newton), save_everystep = false; p=p0)
plot(sol_cvodeadams_jac)


# --- autodifferentiation 
problem_auto = ODEProblem(massaction!, x0, tspan, p0)
sol_lsoda = @btime solve(problem_auto, lsoda(); p=p0)
plot(sol_lsoda)
sol_cvodeadams = @btime solve(problem_auto, CVODE_Adams(); p=p0)
plot(sol_cvodeadams)
@btime solve(problem_auto, KenCarp47(linsolve = KrylovJL_GMRES()), save_everystep = true)


# --- tmp
@btime solve(problem, lsoda(), save_everystep = false; p=p0)
@btime solve(problem, CVODE_Adams(), save_everystep = false; p=p0)
@btime solve(problem, CVODE_BDF(), save_everystep = false; p=p0)
@btime solve(problem, lsoda(), save_everystep = false, sensealg = ForwardSensitivity(); p=p0)

