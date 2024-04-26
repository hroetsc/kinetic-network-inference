### kinetic-network-inference ###
# description:  plotting functions for Julia
# author:       HPR


include("_odes.jl")
myquantile(A, p; dims, kwargs...) = mapslices(x->quantile(x, p; kwargs...), A; dims)
myquantileslice(A, p; dims=nothing, kwargs...) = mapslices(x -> quantile(skipmissing(x), p; kwargs...), A; dims=dims)

# ----- diagnostics -----
function diagnostics_and_save_sim(myChains)

    # save chain
    h5open(folderN*"chain.h5", "w") do io
        MCMCChainsStorage.write(io, myChains)
    end

    # summary stats
    summaries, quantiles = describe(myChains)
    CSV.write(folderN*"summary_stats.csv", summaries)
    CSV.write(folderN*"quantiles.csv", quantiles)
    
    # chain plots
    plot_chains_sim(myChains)

    # residual plots
    plot_kinetics(myChains)

    # TODO: plot sampler https://turinglang.org/dev/docs/using-turing/sampler-viz
    # TODO: convergence criterion

end


# ----- plot chain (simulated) -----
function plot_chains_sim(myChains)

    # plot
    rm(folderN*"chain.pdf", force=true, recursive=true)
    for i in 1:length(paramNames)
        # density plot
        pl1 = density(myChains[:,i,:], title = "marginal posterior "*paramNames[i], dpi = 600, legend = false,
        linewidth = 0.5, palette=:acton10, xlabel = "parameter value", ylabel = "density")
        if i > 1
            Plots.vline!(pl1, [p[i-1]], line=:dash, lc=:black)
        end
        
        # chain plot
        pl2 = plot(myChains[:,i,:], title = "chain "*paramNames[i], dpi = 600, legend = false,
        linewidth = 0.5, palette=:acton10, xlabel = "iteration", ylabel = "parameter value")
        if i > 1
            Plots.hline!(pl2, [p[i-1]], line=:dash, lc=:black)
        end

        pl = plot(pl1,pl2, layout = (1,2))
        savefig(pl, "tmp.pdf")
        append_pdf!(folderN*"chain.pdf", "tmp.pdf", create=true, cleanup=true)
    end

end


# ----- plot simulated concentrations -----
function plot_kinetics(myChains, burnin=0.7, steps=50)

    # more fine-grained time steps
    tps = collect(range(0.0,tspan[2],steps))

    # sample from posterior
    chains = Array(myChains)[:,2:numParam]
    NI = Int(size(chains)[1]/nChains)
    chains = reshape(chains, NI, nChains, numParam-1)

    burned = Int(NI*burnin)
    k = sample(burned:NI, Int(burned/2))
    
    # simulate ODE for each particle
    problem = ODEProblem(f0!, x0, [0.0, maximum(tp)], p)
    simulated = []
    for j in k
        for jj in 1:nChains
            out = solve(problem, CVODE_Adams(linear_solver=:KLU), saveat=tps; p=vec(chains[j,jj,:])).u
            if length(out) != steps
                out = fill(missing, (steps,s))
            else
                out = mapreduce(permutedims, vcat, out)
            end
            push!(simulated, out)
        end
    end
    simulated = reshape(transpose(mapreduce(permutedims, vcat, simulated)), steps, s, length(k)*nChains)

    # get mean and quantiles
    μ = transpose(reshape(mapslices(x -> mean(skipmissing(x)), simulated, dims=3), steps, s))
    q25 = transpose(reshape(mapslices(x -> quantile(skipmissing(x), 0.25), simulated, dims=3), steps, s))
    q75 = transpose(reshape(mapslices(x -> quantile(skipmissing(x), 0.75), simulated, dims=3), steps, s))
    
    # plot
    rm(folderN*"residuals.pdf", force=true, recursive=true)
    for t in 1:s
        ribbon = (q25[t,:], q75[t,:])
        
        pl = plot(tps, μ[t,:], ribbon=ribbon, fillalpha=0.3, lc=:purple, fc=:purple,
        title = species[t]*"\niterations="*string(NI), xlab = "time (hrs)", ylab = "concentration", label = "predicted", dpi = 600)
        plot!(tporig, X[:,t], label = "actual", lc=:black)
        scatter!(tporig, X[:,t], label = "actual", seriestype=:scatter, mc=:black)

        savefig(pl, "tmp.pdf")
        append_pdf!(folderN*"residuals.pdf", "tmp.pdf", create=true, cleanup=true)
    end


end



