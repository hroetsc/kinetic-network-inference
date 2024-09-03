### kinetic-network-inference ###
# description:  evaluate polynomial smoothing estimators for X and dX/dt
# author:       HPR

using DiffEqParamEstim
using OrdinaryDiffEq
using DiffEqFlux
using Lux
using Optimisers
using Zygote
using StatsPlots
using Plots
using Plots.Measures
using LinearAlgebra
using Random
using RCall
using CSV
using BenchmarkTools
using DataFrames
using StatsBase
using Distributions
using PDFmerger: append_pdf!
include("_odes.jl")
include("_plot_utils.jl")
include("_plot_utils_NN.jl")

Random.seed!(42)
print(Threads.nthreads())

protein_name = "Ex4_15s"
OUTNAME = "polynomial_smooting"
folderN = "results/collocation/"*protein_name*"/"*OUTNAME*"/"
mkpath(folderN)

eps = 1e-06

# ----- INPUT -----
R"load(paste0('data/simulation/Ex4_15s_MA_DATA.RData'))"
@rget DATA_MA;
DATA = DATA_MA

X = Array{Union{Missing,Float64}}(DATA[:X])
A = Matrix{Int64}(DATA[:A])
B = Matrix{Int64}(DATA[:B])
x0 = Array{Float64}(DATA[:x0])
N = transpose(B-A)

species = DATA[:species]
reactions = DATA[:reactions]
info = DATA[:info]
p0 = info.rate_ma
tporig = DATA[:tp]

s = length(species)
r = size(info)[1]
paramNames = info.rate_name_ma
nP = length(paramNames)

tspan = [minimum(tporig), maximum(tporig)]
tps = collect(range(0.0,tspan[2],50))
Xm = Array{Float64}(X[2:size(X)[1],:])
tpm = tporig[tporig .> 0]



# ----- get du with real parameters -----
m = exp.(A*log.(X'.+eps))
ducorrect = N*Diagonal(p0)*m

problemx = ODEProblem(massaction_stable, x0, tspan, p0)
intex = solve(problemx, TRBDF2(), saveat=tps)
integx = mapreduce(permutedims, vcat, intex.u)


# ----- get species where initial du is 0 -----
initialdus = []
m = exp.(A*log.(X'.+eps))
for i in 1:100
    psim = rand(Gamma(1,1), nP)
    dusim = N*Diagonal(psim)*m

    push!(initialdus, dusim[:,1])
end
initialdus = mapreduce(permutedims, vcat, initialdus)
boxplot(initialdus, legend=false)
μ_init = mapslices(x -> mean(skipmissing(x)), initialdus, dims=1)

# 2-hop distance from substrate
k = vec(abs.(μ_init) .< 0.01)
print(species[k])
# 1-hop distance from substrate
k2 = vec(abs.(μ_init) .> 0.01)
print(species[k2])



# ----------------------------
# ----- data collocation -----
# ----------------------------
# ----- choose kernel -----
mt = length(tporig) # number of time points
h = mt^(-1/5)*mt^(-3/35)*log(mt)^(-1/16)

kernels = [EpanechnikovKernel(), UniformKernel(), TriangularKernel(), QuarticKernel(), TriweightKernel(), 
TricubeKernel(), DiffEqFlux.GaussianKernel(), DiffEqFlux.CosineKernel(), LogisticKernel(), SigmoidKernel(), SilvermanKernel()]


for kernel in kernels

    ks = string(kernel)
    print(ks)

    # get estimates of du and u
    du, u = @time collocate_data(X', tporig, kernel, h+0.001) # from DiffEqFlux, DiffEqParamEstim packages
    # set initial du of 2-hop neighbours to 0
    du[k,1] .= 0.0

    # calculate error
    erru_all = mean(abs2.(vec(X' .- u)))
    errdu_all = mean(abs2.(vec(ducorrect .- du)))
    
    print("\nerror on u: "*string(round(erru_all;digits=2))*"\n")
    print("\nerror on du: "*string(round(errdu_all;digits=2))*"\n")

    # plot for each species
    rm(folderN*"kernel_"*ks*".pdf", force=true, recursive=true)
    chunkSize = 28
    counter = 1

    while counter <= s
        pp = []
        if counter+chunkSize-1 > s
            en = s
        else
            en = counter+chunkSize-1
        end
        print(en)

        for i in counter:en
            erru = mean(abs2.(vec(X[:,i] .- u[i,:])))
            errdu = mean(abs2.(vec(ducorrect[i,:] .- du[i,:])))

            # plot u
            plu = plot(tporig, u[i,:], lc=:black, title = species[i]*", σ="*string(round(erru;digits=2)), label="kernel est", xlabel = "digestion time [hrs]", ylabel = "u", dpi=600, margin=5mm)
            plot!(tporig, X[:,i], lc=:green, label="ground truth")
            plot!(tps, integx[:,i], lc=:blue, label="real par")

            # plot du
            pldu = plot(tporig, du[i,:], lc=:black, title = species[i]*", σ="*string(round(errdu;digits=2)), label="kernel est", xlabel = "digestion time [hrs]", ylabel = "u", dpi=600, margin=5mm)
            plot!(tporig, ducorrect[i,:], lc=:blue, label="real par")
            
            # combine
            push!(pp, plot(plu,pldu, layout = (1,2)))
        end

        pplot = plot(pp...; size = default(:size) .* (2,14), layout=(14,2), dpi = 600, margin=10mm)
        savefig(pplot, folderN*"tmp.pdf")
        append_pdf!(folderN*"kernel_"*ks*".pdf", folderN*"tmp.pdf", create=true, cleanup=true)
        counter += chunkSize
    end

end

# TriweightKernel is still the best


# ----- try different bandwidths -----
bandwidths = collect(range(0.501,2,200))
erru_bw = []
errdu_bw = []

for h in bandwidths
    du, u = @time collocate_data(X', tporig, TriweightKernel(), h)
    du[k,1] .= 0.0

    erru_all = mean(abs2.(vec(X' .- u)))
    errdu_all = mean(abs2.(vec(ducorrect .- du)))

    print(h)
    print("\nerror on u: "*string(round(erru_all;digits=2))*"\n")
    print("error on du: "*string(round(errdu_all;digits=2))*"\n")

    push!(erru_bw, erru_all)
    push!(errdu_bw, errdu_all)
end

plot(bandwidths, erru_bw, title="error on u", xlabel="bandwidth", ylabel="error", legend=false)
plot(bandwidths, errdu_bw, title="error on du", xlabel="bandwidth", ylabel="error", legend=false)


selected_bandwidths = [0.501, 0.6, 0.7, 1.0]
for h in selected_bandwidths

    hs = string(h)
    print(h)

    # get estimates of du and u
    du, u = @time collocate_data(X', tporig, TriweightKernel(), h) # from DiffEqFlux, DiffEqParamEstim packages
    du[k,1] .= 0.0
    
    # calculate error
    erru_all = mean(abs2.(vec(X' .- u)))
    errdu_all = mean(abs2.(vec(ducorrect .- du)))
    
    print("\nerror on u: "*string(round(erru_all;digits=2))*"\n")
    print("\nerror on du: "*string(round(errdu_all;digits=2))*"\n")

    # plot for each species
    rm(folderN*"kernel_Triweight_bw-"*hs*".pdf", force=true, recursive=true)
    chunkSize = 28
    counter = 1

    while counter <= s
        pp = []
        if counter+chunkSize-1 > s
            en = s
        else
            en = counter+chunkSize-1
        end
        print(en)

        for i in counter:en
            erru = mean(abs2.(vec(X[:,i] .- u[i,:])))
            errdu = mean(abs2.(vec(ducorrect[i,:] .- du[i,:])))

            # plot u
            plu = plot(tporig, u[i,:], lc=:black, title = species[i]*", σ="*string(round(erru;digits=2)), label="kernel est", xlabel = "digestion time [hrs]", ylabel = "u", dpi=600, margin=5mm)
            plot!(tporig, X[:,i], lc=:green, label="ground truth")
            plot!(tps, integx[:,i], lc=:blue, label="real par")

            # plot du
            pldu = plot(tporig, du[i,:], lc=:black, title = species[i]*", σ="*string(round(errdu;digits=2)), label="kernel est", xlabel = "digestion time [hrs]", ylabel = "u", dpi=600, margin=5mm)
            plot!(tporig, ducorrect[i,:], lc=:blue, label="real par")
            
            # combine
            push!(pp, plot(plu,pldu, layout = (1,2)))
        end

        pplot = plot(pp...; size = default(:size) .* (2,14), layout=(14,2), dpi = 600, margin=10mm)
        savefig(pplot, folderN*"tmp.pdf")
        append_pdf!(folderN*"kernel_Triweight_bw-"*hs*".pdf", folderN*"tmp.pdf", create=true, cleanup=true)
        counter += chunkSize
    end

end




# ------- attempt 2 ------
using BSplineKit
using FiniteDifferences
fdm = central_fdm(5, 1)

# tpoints = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
tpoints = collect(range(0.5,4.0,50))

function iterpolation(si, tpoints)
    intp = interpolate(tporig, vec(X[:,si]'), BSplineOrder(4))
    # extp = extrapolate(intp[1])
    # plot(tporig, vec(X[:,s]'))
    # plot!(intp)
    uintp = [intp(t) for t in tpoints]
    duintp = [fdm(intp, t) for t in tpoints]

    return uintp, duintp
end


uintps = []
duintps = []
for si in 1:s
    uintp_i, duintp_i = iterpolation(si, tpoints)
    push!(uintps, uintp_i)
    push!(duintps, duintp_i)
end

u_intps = mapreduce(permutedims, vcat, uintps)
du_intps = mapreduce(permutedims, vcat, duintps)

du_intps[:,50] .= du_intps[:,49]
# du_intps[:,1] .= du_intps[:,2]
# du_intps[k,1] .= 0.0


closest_indices = [findmin(abs.(tpoints .- tpo))[2] for tpo in tpm]
du_intps_d = du_intps[:,closest_indices]

errdu_d = mean(abs2.(vec(ducorrect[:,2:5] - du_intps_d)))
print("\nerror on u: "*string(round(errdu_d;digits=2))*"\n")


# --- plot all
h = 0.501
# get estimates of du and u
du, u = @time collocate_data(X', tporig, TriweightKernel(), 0.54) # from DiffEqFlux, DiffEqParamEstim packages
du[k,1] .= 0.0

# plot for each species
rm(folderN*"kernel+numericDiff.pdf", force=true, recursive=true)
chunkSize = 28
counter = 1

while counter <= s
    pp = []
    if counter+chunkSize-1 > s
        en = s
    else
        en = counter+chunkSize-1
    end
    print(en)

    for i in counter:en
        # plot u
        plu = plot(tporig, u[i,:], lc=:black, title = species[i], label="kernel est", xlabel = "digestion time [hrs]", ylabel = "u", dpi=600, margin=5mm)
        plot!(tporig, X[:,i], lc=:green, label="ground truth")
        plot!(tps, integx[:,i], lc=:blue, label="real par")
        plot!(tpoints, u_intps[i,:], lc=:orange, label="interp")

        # plot du
        pldu = plot(tporig, du[i,:], lc=:black, title = species[i], label="kernel est", xlabel = "digestion time [hrs]", ylabel = "u", dpi=600, margin=5mm)
        plot!(tporig, ducorrect[i,:], lc=:blue, label="real par")
        plot!(tpoints, du_intps[i,:], lc=:orange, label="interp")
        
        # combine
        push!(pp, plot(plu,pldu, layout = (1,2)))
    end

    pplot = plot(pp...; size = default(:size) .* (2,14), layout=(14,2), dpi = 600, margin=10mm)
    savefig(pplot, folderN*"tmp.pdf")
    append_pdf!(folderN*"kernel+numericDiff.pdf", folderN*"tmp.pdf", create=true, cleanup=true)
    counter += chunkSize
end


# TODO: fit on this??
# TODO: discard 0hrs from loss??
