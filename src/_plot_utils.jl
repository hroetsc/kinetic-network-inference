### kinetic-network-inference ###
# description:  plotting functions for Julia
# author:       HPR


include("_odes.jl")
myquantile(A, p; dims, kwargs...) = mapslices(x->quantile(x, p; kwargs...), A; dims)
myquantileslice(A, p; dims=nothing, kwargs...) = mapslices(x -> quantile(skipmissing(x), p; kwargs...), A; dims=dims)

# ----- diagnostics -----
# real data
function diagnostics_and_save(myChains, problem)

    # save chain
    print("saving chain and summary stats\n")
    h5open(folderN*"chain.h5", "w") do io
        MCMCChainsStorage.write(io, myChains)
    end

    # summary stats
    summaries, quantiles = describe(myChains)
    atcor = autocor(myChains)
    CSV.write(folderN*"summary_stats.csv", summaries)
    CSV.write(folderN*"quantiles.csv", quantiles)
    CSV.write(folderN*"autocorrelation.csv", atcor)
    
    # chain plots
    plot_chains(myChains)
    # residual plots
    plot_kinetics(myChains, problem)

end


# simulated data
function diagnostics_and_save_sim(myChains, problem)

    # save chain
    print("saving chain and summary stats\n")
    h5open(folderN*"chain.h5", "w") do io
        MCMCChainsStorage.write(io, myChains)
    end

    # summary stats
    summaries, quantiles = describe(myChains)
    atcor = autocor(myChains)
    CSV.write(folderN*"summary_stats.csv", summaries)
    CSV.write(folderN*"quantiles.csv", quantiles)
    CSV.write(folderN*"autocorrelation.csv", atcor)
    
    # chain plots
    plot_chains_sim(myChains)
    # residual plots
    plot_kinetics_sim(myChains, problem)

    # TODO: plot sampler https://turinglang.org/dev/docs/using-turing/sampler-viz
    # TODO: convergence criterion

end


# ----- plot chain (simulated) -----
# real data
function plot_chains(myChains, burnin=0.7)

    print("plotting chain...\n")
    pal = palette(:acton10, nChains+1)
    
    # sample from posterior
    chns = Array(myChains)
    NI = Int(size(chns)[1]/nChains)
    chns = reshape(transpose(chns), numParam, NI, nChains)
    
    burned = Int(NI*burnin)
    k = sort(sample(burned:NI, Int(ceil((NI-burned)/2)), replace=false))

    # plot
    rm(folderN*"chain.pdf", force=true, recursive=true)
    chunkSize = 80
    counter = 1
    while counter <= numParam
        pp = []
        if counter+chunkSize-1 > numParam
            en = numParam
        else
            en = counter+chunkSize-1
        end
        print(en)

        for i in counter:en
            # box plot
            pl1 = boxplot(chns[i,k,:], title = "marginal posterior "*paramNames[i]*", burnin="*string(burnin), dpi = 600, legend = false,
            linewidth = 0.5, palette=pal, xlabel = "", ylabel = "", margin=20mm)
            # chain plot
            pl2 = plot(myChains[:,i,:], title = "chain "*paramNames[i], dpi = 600, legend = false,
            linewidth = 0.5, palette=pal, xlabel = "", ylabel = "", margin=20mm)
            # combine
            push!(pp, plot(pl1,pl2, layout = (1,2)))
        end
        ch = plot(pp...; size = default(:size) .* (4,20), layout=(20,4), dpi = 600, margin=10mm)
        savefig(ch, folderN*"tmp.pdf")
        append_pdf!(folderN*"chain.pdf", folderN*"tmp.pdf", create=true, cleanup=true)

        counter += chunkSize
    end
    

end


# simulated data
function plot_chains_sim(myChains, burnin=0.7)
    
    print("plotting chain...\n")
    pal = palette(:acton10, nChains+1)
    
    # sample from posterior
    chns = Array(myChains)
    NI = Int(size(chns)[1]/nChains)
    chns = reshape(transpose(chns), numParam, NI, nChains)
    
    burned = Int(NI*burnin)
    k = sort(sample(burned:NI, Int(ceil((NI-burned)/2)), replace=false))

    # plot
    rm(folderN*"chain.pdf", force=true, recursive=true)
    for i in 1:length(paramNames)
        # box plot
        pl1 = boxplot(chns[i,k,:], title = "marginal posterior "*paramNames[i], dpi = 600, legend = false,
        linewidth = 0.5, palette=pal, xlabel = "chain, burnin="*string(burnin), ylabel = "parameter value")
        if i > 1
            Plots.hline!(pl1, [p0[i-1]], line=:dash, lc=:black)
        end
        
        # chain plot
        pl2 = plot(myChains[:,i,:], title = "chain "*paramNames[i], dpi = 600, legend = false,
        linewidth = 0.5, palette=pal, xlabel = "iteration (no burnin)", ylabel = "parameter value")
        if i > 1
            Plots.hline!(pl2, [p0[i-1]], line=:dash, lc=:black)
        end

        pl = plot(pl1,pl2, layout = (1,2))
        savefig(pl, "tmp.pdf")
        append_pdf!(folderN*"chain.pdf", "tmp.pdf", create=true, cleanup=true)
    end

end


# ----- plot simulated concentrations -----
function plot_kinetics(myChains, problem, burnin=0.7, steps=10)

    print("plotting kinetics....\n")
    cols = palette(:starrynight,5)

    # more fine-grained time steps
    tps = collect(range(0.0,tspan[2],steps))

    # sample from posterior
    chains = Array(myChains)[:,2:numParam]
    NI = Int(size(chains)[1]/nChains)
    chains = reshape(chains, NI, nChains, numParam-1)

    burned = Int(NI*burnin)
    k = sort(sample(burned:NI, Int(ceil((NI-burned)/2)), replace=false))
    
    # simulate ODE for each particle
    simulated = []
    for j in k
        for jj in 1:nChains
            # FIXME: make faster!
            out = solve(problem, TRBDF2(), saveat=tps; p=vec(chains[j,jj,:])).u
            if length(out) != steps
                out = fill(missing, (steps,s))
            else
                out = mapreduce(permutedims, vcat, out)
            end
            push!(simulated, out)
        end
    end
    simulated = reshape(transpose(mapreduce(permutedims, vcat, simulated)), steps, s, length(k)*nChains)

    # --- get mean and quantiles
    μ = transpose(reshape(mapslices(x -> mean(skipmissing(x)), simulated, dims=3), steps, s))
    q25 = transpose(reshape(mapslices(x -> quantile(skipmissing(x), 0.25), simulated, dims=3), steps, s))
    q75 = transpose(reshape(mapslices(x -> quantile(skipmissing(x), 0.75), simulated, dims=3), steps, s))
    
    # --- plot
    rm(folderN*"residuals.pdf", force=true, recursive=true)
    chunkSize = 80
    counter = 1
    while counter <= s
        pp = []
        if counter+chunkSize-1 > s
            en = s
        else
            en = counter+chunkSize-1
        end
        print(en)

        for t in counter:en
            ribbon = (q75[i,:] - q25[i,:]) ./ 2
        
            pl = plot(tps, μ[t,:], ribbon=ribbon, fillalpha=0.3, lc=:purple, fc=:purple,
            title = species[t]*"\niterations="*string(NI), xlab = "", ylab = "", dpi = 600, legend=false)
            for tt in 1:nr
                ttt = map(!,ismissing.(X[:,t,tt]))
                plot!(tporig[ttt], X[ttt,t,tt], lc=cols[tt])
                scatter!(tporig, X[:,t,tt], seriestype=:scatter, mc=cols[tt], markersize=1)
            end
            push!(pp, pl)
        end
        kn = plot(pp...; size = default(:size) .* (4,20), layout=(20,4), dpi = 600, margin=20mm)
        savefig(kn, folderN*"tmp.pdf")
        append_pdf!(folderN*"residuals.pdf", folderN*"tmp.pdf", create=true, cleanup=true)

        counter += chunkSize
    end

end


function plot_kinetics_sim(myChains, problem, burnin=0.7, steps=50)

    print("plotting kinetics....\n")

    # more fine-grained time steps
    tps = collect(range(0.0,tspan[2],steps))

    # sample from posterior
    chains = Array(myChains)[:,2:numParam]
    NI = Int(ceil(size(chains)[1]/nChains))
    chains = reshape(chains, NI, nChains, numParam-1)

    burned = Int(NI*burnin)
    k = sort(sample(burned:NI, Int((NI-burned)/2), replace=false))

    # simulate ODE for each particle
    simulated = []
    for j in k
        for jj in 1:nChains
            # FIXME: make faster!
            prm = MVector{r}(vec(chains[j,jj,:]))
            out = solve(problem, TRBDF2(), saveat=tps; p=prm).u
            if length(out) != steps
                out = fill(missing, (steps,s))
            else
                out = Array(mapreduce(permutedims, vcat, out))
            end
            push!(simulated, out)
        end
    end
    simulated = reshape(transpose(mapreduce(permutedims, vcat, simulated)), steps, s, length(k)*nChains)

    # --- get mean and quantiles
    μ = transpose(reshape(mapslices(x -> mean(skipmissing(x)), simulated, dims=3), steps, s))
    q25 = transpose(reshape(mapslices(x -> quantile(skipmissing(x), 0.25), simulated, dims=3), steps, s))
    q75 = transpose(reshape(mapslices(x -> quantile(skipmissing(x), 0.75), simulated, dims=3), steps, s))
    
    # plot
    rm(folderN*"residuals.pdf", force=true, recursive=true)
    for t in 1:s
        ribbon = (q75[i,:] - q25[i,:]) ./ 2
        
        pl = plot(tps, μ[t,:], ribbon=ribbon, fillalpha=0.3, lc=:purple, fc=:purple,
        title = species[t]*"\niterations="*string(NI), xlab = "time (hrs)", ylab = "concentration", label = "predicted", dpi = 600)
        plot!(tporig, X[:,t], label = "actual", lc=:black)
        scatter!(tporig, X[:,t], label = "actual", seriestype=:scatter, mc=:black)
        
        savefig(pl, "tmp.pdf")
        append_pdf!(folderN*"residuals.pdf", "tmp.pdf", create=true, cleanup=true)
    end


end



