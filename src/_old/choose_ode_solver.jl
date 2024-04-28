### kinetic-network-inference ###
# description:  choose best ODE solver
# author:       HPR

using Turing, DifferentialEquations, SciMLSensitivity, LSODA, Enzyme, StatsPlots, Random, RCall, CSV, Serialization

Random.seed!(42)

# ----- INPUT -----
R"load(paste0('results/graphs/IDH1_WT.RData'))"
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


# ----- ODEs -----
function massaction!(du, u, p, t)
    M = prod((u .^ A), dims=2)
    du[1:s] = transpose(B-A)*(p .* M)
    return nothing
end

p0 = rand(r)
problem = ODEProblem(massaction!, x0, tspan, p0)


# ----- timing -----
# non-stiff
t_euler = @time solve(problem, Euler(); p=p0, dt=0.1)
t_tsit5 = @time solve(problem, Tsit5(); p=p0)
t_trbdf2 = @time solve(problem, TRBDF2(); p=p0)
t_heun = @time solve(problem, Heun(); p=p0)
t_rk4 = @time solve(problem, RK4(); p=p0)

# stiff
t_impleuler = @time solve(problem, ImplicitEuler(); p=p0)
t_lsoda = @time solve(problem, AutoTsit5(Rosenbrock23()); p=p0)
t_trapezoid = @time solve(problem, Trapezoid(); p=p0)
# t_qndf = @time solve(problem, QNDF(); p=p0) # too slow!
t_fbdf = @time solve(problem, FBDF(); p=p0)
t_lsoda2 = @time solve(problem, lsoda(); p=p0)
t_cvodebdf = @time solve(problem, CVODE_BDF(); p=p0)
t_kencarp4 = @time solve(problem, KenCarp4(); p=p0)
t_rosenbrock23 = @time solve(problem, Rosenbrock23(); p=p0)


@time solve(problem, lsoda(); p=p0)
@time solve(problem, lsoda(), save_everystep = false, sensealg = ForwardSensitivity(); p=p0)


# ----- summary -----
describe(t_lsoda2.u)

# ----- plots -----
plot(t_euler, title = "euler")
plot(t_tsit5, title = "tsit5")
plot(t_trbdf2, title = "trbdf2")
plot(t_heun, title = "heun")
plot(t_rk4, title = "rk4")

plot(t_impleuler, title = "impleuler")
plot(t_lsoda, title = "lsoda")
plot(t_trapezoid, title = "trapezoid")
plot(t_fbdf, title = "fbdf")
plot(t_lsoda2, title = "lsoda2")
plot(t_cvodebdf, title = "cvodebdf")

