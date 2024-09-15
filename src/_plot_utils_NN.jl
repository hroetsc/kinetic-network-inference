include("_odes.jl")
include("_plot_utils.jl")

myquantile(A, p; dims, kwargs...) = mapslices(x->quantile(x, p; kwargs...), A; dims)
myquantileslice(A, p; dims=nothing, kwargs...) = mapslices(x -> quantile(skipmissing(x), p; kwargs...), A; dims=dims)

function get_my_quantiles(Z, mtt, dim, param=false)
    if !param
        μ = transpose(reshape(mapslices(x -> mean(skipmissing(x)), Z, dims=dim), mtt, s))
        q25 = transpose(reshape(mapslices(x -> isempty(skipmissing(x)) ? NaN : quantile(skipmissing(x), 0.25), Z, dims=dim), mtt, s))
        q75 = transpose(reshape(mapslices(x -> isempty(skipmissing(x)) ? NaN : quantile(skipmissing(x), 0.75), Z, dims=dim), mtt, s))
    else
        μ = transpose(reshape(mapslices(x -> mean(skipmissing(x)), Z, dims=dim), nP))
        q25 = transpose(reshape(mapslices(x -> isempty(skipmissing(x)) ? NaN : quantile(skipmissing(x), 0.05), Z, dims=dim), nP))
        q75 = transpose(reshape(mapslices(x -> isempty(skipmissing(x)) ? NaN : quantile(skipmissing(x), 0.95), Z, dims=dim), nP))
    end
    return μ,q25,q75
end


# ----- simulated data -----
function diagnostics_and_save_NN_sim(tstate, y_pred, ma=false, steps=50)

    pinf = tstate.parameters

    # ----- plot metrics -----
    sc = scatter(p0, pinf, legend=false, title="parameters", mc=:black,
    # series_annotations = text.(paramNames,p0),
    xlabel = "true parameter", ylabel = "predicted parameter", xlim = (minimum(pinf)-0.1,maximum(pinf)+0.1), ylim = (minimum(pinf)-0.1,maximum(pinf)+0.1), dpi=600)
    Plots.abline!(sc, 1, 0, line=:dash, lc=:black)

    ls = plot(loss, title = "training loss", ylim = (0,maximum(loss)*1.5),
    xlabel = "epoch", ylabel = "loss", legend=false, lc=:black, dpi=600)

    pl1 = plot(sc, ls, layout=(1,2))
    savefig(pl1, folderN*"training_metrics.png")

    # ---- simulate ODE with parameters -----
    # more fine-grained time steps
    if ma
        # inferred parameters
        tps = collect(range(0.0,tspan[2],steps))
        problem = ODEProblem(massaction_stable, x0, tspan, pinf)
        integ = solve(problem, TRBDF2(), saveat=tps)
        integu = mapreduce(permutedims, vcat, integ.u)

        # real parameters
        problemx = ODEProblem(massaction_stable, x0, tspan, p0)
        intex = solve(problemx, TRBDF2(), saveat=tps)
        integx = mapreduce(permutedims, vcat, intex.u)

    end

    # ----- plot -----
    rm(folderN*"simulated.pdf", force=true, recursive=true)
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
            if ma
                plot!(tps, integu[:,i], lc=:red, label="predicted")
                plot!(tps, integx[:,i], lc=:blue, label="real par")
            end

            # plot du
            pldu = plot(tporig, du[i,:], lc=:black, title = species[i], label="kernel est", xlabel = "digestion time [hrs]", ylabel = "du", dpi=600, margin=5mm)
            plot!(tporig, y_pred[i,:], lc=:red, label="predicted")

            # combine
            push!(pp, plot(plu,pldu, layout = (1,2)))
        end

        pplot = plot(pp...; size = default(:size) .* (2,14), layout=(14,2), dpi = 600, margin=10mm)
        savefig(pplot, folderN*"tmp.pdf")
        append_pdf!(folderN*"simulated.pdf", folderN*"tmp.pdf", create=true, cleanup=true)
        counter += chunkSize
    end

end


# ----- simulated data - repeats -----
function diagnostics_and_save_NN_sim_multi(tstates, ypreds, ppreds, losses, ma=false, steps=50)

    du = du_intps_d
    nr = length(tstates)
    mt = length(tporig)
    tp_ypred = tporig

    pal = palette(:bamako, nr+1)

    # ----- loss
    print("loss....\n")

    ls = plot(losses[1], title="training loss", #ylim = (0,maximum(losses[1])*1.5),
    xlabel="epoch", ylabel="loss", label="rep_1", lc=pal[1], dpi=600)
    for ii in 2:nr
        plot!(losses[ii], col=pal[ii], label="rep_"*string(ii))
    end
    savefig(ls, folderN*"loss.png")


    # ----- parameters -----
    # pinfs = []
    # for ii in 1:nr
    #     push!(pinfs, tstates[ii].parameters)
    # end
    # pinf = mapreduce(permutedims, vcat, pinfs)
    pinf = mapreduce(permutedims, vcat, ppreds)
    μ_pinf, q25_pinf, q75_pinf = get_my_quantiles(pinf, nothing, 1, true)
    ribbon_pinf = (q75_pinf .- q25_pinf) ./ 2

    print("\nnegative parameters:\n")
    print(paramNames[vec(μ_pinf) .< 0])
    print("\n")

    # ----- plot metrics -----
    sc = scatter(p0, μ_pinf', yerror = ribbon_pinf,
    legend=false, title="parameters", mc=:black,
    # series_annotations = text.(paramNames,p0),
    xlabel = "true parameter", ylabel = "predicted parameter", xlim = (minimum(pinf)-0.1,maximum(pinf)+0.1), ylim = (minimum(pinf)-0.1,maximum(pinf)+0.1), dpi=600)
    Plots.abline!(sc, 1, 0, line=:dash, lc=:black)

    pl1 = plot(sc, ls, layout=(1,2))
    savefig(pl1, folderN*"training_metrics.png")


    # ---- simulate ODE with parameters -----
    # more fine-grained time steps
    if ma
        # inferred parameters
        tps = collect(range(0.0,tspan[2],steps))
        problem = ODEProblem(massaction_stable, x0, tspan, vec(μ_pinf))
        integ = solve(problem, TRBDF2(), saveat=tps)
        integu = mapreduce(permutedims, vcat, integ.u)

        # real parameters
        problemx = ODEProblem(massaction_stable, x0, tspan, p0)
        intex = solve(problemx, TRBDF2(), saveat=tps)
        integx = mapreduce(permutedims, vcat, intex.u)

    end

    # ----- plot -----
    # du with correct parameters
    if ma
        m = exp.(A*log.(X'.+eps))
        ducorrect = N*Diagonal(p0)*m
    end
    
    # predicted du
    YPREDU = reshape(mapreduce(permutedims, vcat, ypreds), length(tp_ypred), nr, s)
    YPREDU_μ, YPREDU_q25, YPREDU_q75 = get_my_quantiles(YPREDU, length(tp_ypred), 2)
    
    # actual plot
    rm(folderN*"simulated.pdf", force=true, recursive=true)
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
            ribbon_ypredu = (YPREDU_q75[i,:] - YPREDU_q25[i,:]) ./ 2

            # plot u
            plu = plot(tporig, u[i,:], lc=:black, title = species[i], label="kernel est", xlabel = "digestion time [hrs]", ylabel = "u", dpi=600, margin=5mm)
            plot!(tporig, X[:,i], lc=:green, label="ground truth")
            if ma
                plot!(tps, integu[:,i], lc=:red, label="predicted")
                plot!(tps, integx[:,i], lc=:blue, label="real par")
            end

            # plot du
            pldu = plot(tp_ypred, du[i,:], lc=:orange, title = species[i], label="interp", xlabel = "digestion time [hrs]", ylabel = "du", dpi=600, margin=5mm)
            plot!(tp_ypred, YPREDU_μ[i,:], ribbon=ribbon_ypredu, fillalpha=0.3, lc=:red, label="predicted")
            if ma
                plot!(tporig, ducorrect[i,:], lc=:blue, label="real par")
            end

            # combine
            push!(pp, plot(plu,pldu, layout = (1,2)))
        end

        pplot = plot(pp...; size = default(:size) .* (2,14), layout=(14,2), dpi = 600, margin=10mm)
        savefig(pplot, folderN*"tmp.pdf")
        append_pdf!(folderN*"simulated.pdf", folderN*"tmp.pdf", create=true, cleanup=true)
        counter += chunkSize
    end

end



# ----- real data -----
function diagnostics_and_save_NN(tstates, ypreds, losses, loss_only=false, blackbox=false, steps=10)

    pal = palette(:bamako, nr+1)
    dus = du_intps_d

    # ----- loss
    print("loss....\n")

    ls = plot(losses[1], title="training loss", #ylim = (0,maximum(losses[1])*1.5),
    xlabel="epoch", ylabel="loss", label="rep_1", lc=pal[1], dpi=600)
    for ii in 2:nr
        plot!(losses[ii], col=pal[ii], label="rep_"*string(ii))
    end
    savefig(ls, folderN*"loss.png")

    # ----- parameters
    if !blackbox
        print("parameters....\n")
        pinfs = []
        for ii in 1:nr
            push!(pinfs, tstates[ii].parameters)
        end
        pinf = mapreduce(permutedims, vcat, pinfs)
        pdist = boxplot(pinf, palette = :bamako, legend=false,
        title="parameters", xlabel="parameter #", ylabel="parameter value", size = default(:size) .* (10,10))
        savefig(pdist, folderN*"parameters.pdf")
    end

    if !loss_only

        # FIXME: too slow
        # if !blackbox
            # # ---- simulate ODE with parameters -----
            # # more fine-grained time steps
            # tps = collect(range(0.0,tspan[2],steps))
            # problem = ODEProblem(massaction_stable, x0, tspan, pinfs[1])
            # 
            # integu = []
            # for ii in 1:nr
            #     integ = @time solve(problem, TRBDF2(), saveat=tps; p=pinfs[ii])
            #     push!(integu, mapreduce(permutedims, vcat, integ.u))
            # end
        # end
        

        # ----- plot u and du -----
        print("kinetics....\n")
        UU = reshape(mapreduce(permutedims, vcat, us), mt, nr, s)
        DUU = reshape(mapreduce(permutedims, vcat, dus), mt, nr, s)
        YPREDU = reshape(mapreduce(permutedims, vcat, ypreds), mt, nr, s)

        UU_μ, UU_q25, UU_q75 = get_my_quantiles(UU, mt, 2)
        X_μ, X_q25, X_q75 = get_my_quantiles(X, mt, 3)
        DUU_μ, DUU_q25, DUU_q75 = get_my_quantiles(DUU, mt, 2)
        YPREDU_μ, YPREDU_q25, YPREDU_q75 = get_my_quantiles(YPREDU, mt, 2)
        
        rm(folderN*"simulated.pdf", force=true, recursive=true)
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
                ribbon_uu = (UU_q75[i,:] - UU_q25[i,:]) ./ 2
                ribbon_x = (X_q75[i,:] - X_q25[i,:]) ./ 2
                ribbon_duu = (DUU_q75[i,:] - DUU_q25[i,:]) ./ 2
                ribbon_ypredu = (YPREDU_q75[i,:] - YPREDU_q25[i,:]) ./ 2

                # plot u
                plu = plot(tporig, UU_μ[i,:], ribbon=ribbon_uu, fillalpha=0.3, lc=:black, fc=:black,
                title = species[i], label="kernel est", xlabel = "digestion time [hrs]", ylabel = "u", dpi=600, margin=5mm)
                # plot!(tps, integu[:,i], lc=:red, label="predicted")
                plot!(tporig, X_μ[i,:], ribbon=ribbon_x, fillalpha=0.3, lc=:green, fc=:green, label="ground truth")

                # plot du
                pldu = plot(tporig, DUU_μ[i,:], ribbon=ribbon_duu, fillalpha=0.3, lc=:black, fc=:black,
                title = species[i], label="kernel est", xlabel = "digestion time [hrs]", ylabel = "du", dpi=600, margin=5mm)
                plot!(tporig, YPREDU_μ[i,:], ribbon=ribbon_ypredu, fillalpha=0.3, lc=:red, fc=:red, label="predicted")

                # combine
                push!(pp, plot(plu,pldu, layout = (1,2)))
            end

            pplot = plot(pp...; size = default(:size) .* (2,14), layout=(14,2), dpi = 600, margin=10mm)
            savefig(pplot, folderN*"tmp.pdf")
            append_pdf!(folderN*"simulated.pdf", folderN*"tmp.pdf", create=true, cleanup=true)
            counter += chunkSize
        end
    end
    
    print("\n")

end




# ----- real data - ABC-SMC -----
function plot_chains_abc(parameters)

    print("plotting chain...\n")
    pal = palette(:acton10,nr+1)
    numParam = r

    # plot
    rm(folderN*"parameters.pdf", force=true, recursive=true)
    chunkSize = 28
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
            pl1 = histogram(parameters[:,i], title = "marginal posterior "*reactions[i], dpi = 600, legend = false,
            linewidth = 0.5, palette=pal, xlabel = "", ylabel = "", margin=5mm)
            # chain plot
            pl2 = plot(parameters[:,i], title = "particle "*reactions[i], dpi = 600, legend = false,
            linewidth = 0.5, palette=pal, xlabel = "", ylabel = "", margin=5mm)
            # combine
            push!(pp, plot(pl1,pl2, layout = (1,2)))
        end
        ch = plot(pp...; size = default(:size) .* (2,14), layout=(14,2), dpi = 600, margin=10mm)
        savefig(ch, folderN*"tmp.pdf")
        append_pdf!(folderN*"parameters.pdf", folderN*"tmp.pdf", create=true, cleanup=true)

        counter += chunkSize
    end
    

end


function plot_kinetics_abc(parameters, steps=10)
    
    # FIXME: too slow!
    # # simulate kinetics
    # tps = collect(range(0.0,tspan[2],steps))
    # problem = ODEProblem(massaction_stable, x0, tspan, parameters[1,:])
    
    # predict du
    predicted = []
    for j in 1:size(parameters)[1]
        for ii in 1:nr
            m = exp.(A*log.(Xn0[:,:,ii]' .+ eps))
            out = N*Diagonal(parameters[j,:])*m
            push!(predicted, out)
        end
    end
    predicted = reshape(transpose(mapreduce(permutedims, vcat, predicted)), s, mt, :)

    print("kinetics....\n")
    UU = reshape(mapreduce(permutedims, vcat, us), mt, nr, s)
    DUU = reshape(mapreduce(permutedims, vcat, dus), mt, nr, s)

    UU_μ, UU_q25, UU_q75 = get_my_quantiles(UU, mt, 2)
    X_μ, X_q25, X_q75 = get_my_quantiles(X, mt, 3)
    DUU_μ, DUU_q25, DUU_q75 = get_my_quantiles(DUU, mt, 2)
    YPREDU_μ, YPREDU_q25, YPREDU_q75 = get_my_quantiles(predicted, mt, 3)
    
    rm(folderN*"simulated.pdf", force=true, recursive=true)
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
            ribbon_uu = (UU_q75[i,:] - UU_q25[i,:]) ./ 2
            ribbon_x = (X_q75[i,:] - X_q25[i,:]) ./ 2
            ribbon_duu = (DUU_q75[i,:] - DUU_q25[i,:]) ./ 2
            ribbon_ypredu = (YPREDU_q75[i,:] - YPREDU_q25[i,:]) ./ 2

            # plot u
            plu = plot(tporig, UU_μ[i,:], ribbon=ribbon_uu, fillalpha=0.3, lc=:black, fc=:black,
            title = species[i], label="kernel est", xlabel = "digestion time [hrs]", ylabel = "u", dpi=600, margin=5mm)
            # plot!(tps, integu[:,i], lc=:red, label="predicted")
            plot!(tporig, X_μ[i,:], ribbon=ribbon_x, fillalpha=0.3, lc=:green, fc=:green, label="ground truth")

            # plot du
            pldu = plot(DUU_μ[i,:], ribbon=ribbon_duu, fillalpha=0.3, lc=:black, fc=:black,
            title = species[i], label="kernel est", xlabel = "digestion time [hrs]", ylabel = "du", dpi=600, margin=5mm)
            plot!(YPREDU_μ[i,:], ribbon=ribbon_ypredu, fillalpha=0.3, lc=:red, fc=:red, label="predicted")

            # combine
            push!(pp, plot(plu,pldu, layout = (1,2)))
        end
        
        pplot = plot(pp...; size = default(:size) .* (2,14), layout=(14,2), dpi = 600, margin=10mm)
        savefig(pplot, folderN*"tmp.pdf")
        append_pdf!(folderN*"simulated.pdf", folderN*"tmp.pdf", create=true, cleanup=true)
        counter += chunkSize
    end

    print("\n")
    return YPREDU_μ, DUU_μ
end




function diagnostics_and_save_ABC(smc, steps=10)

    # chain plots
    plot_chains_abc(smc.parameters)
    # residual plots
    YPREDU_μ, DUU_μ = plot_kinetics_abc(smc.parameters, steps)

    # save stuff
    param = DataFrame(smc.parameters', :auto)
    param[!,:rate] = reactions
    CSV.write(folderN*"parameters.csv", param)

    ypreddf = DataFrame(YPREDU_μ, [:ypred0, :ypred1, :ypred2, :ypred3, :ypred4])
    ypreddf[!,:species] = species
    dudf = DataFrame(DUU_μ, [:du0, :du1, :du2, :du3, :du4])
    CSV.write(folderN*"predicted.csv", hcat(ypreddf,dudf))

end

