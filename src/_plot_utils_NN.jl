include("_odes.jl")

myquantile(A, p; dims, kwargs...) = mapslices(x->quantile(x, p; kwargs...), A; dims)
myquantileslice(A, p; dims=nothing, kwargs...) = mapslices(x -> quantile(skipmissing(x), p; kwargs...), A; dims=dims)

function get_my_quantiles(Z, mtt, dim)
    μ = transpose(reshape(mapslices(x -> mean(skipmissing(x)), Z, dims=dim), mtt, s))
    q25 = transpose(reshape(mapslices(x -> isempty(skipmissing(x)) ? NaN : quantile(skipmissing(x), 0.25), Z, dims=dim), mtt, s))
    q75 = transpose(reshape(mapslices(x -> isempty(skipmissing(x)) ? NaN : quantile(skipmissing(x), 0.75), Z, dims=dim), mtt, s))
    return μ,q25,q75
end

# ----- simulated data -----
function diagnostics_and_save_NN_sim(tstate, y_pred, steps=50)

    pinf = tstate.parameters

    # ----- plot metrics -----
    sc = scatter(p0, pinf, legend=false, title="parameters", mc=:black,
    xlabel = "true parameter", ylabel = "predicted parameter", xlim = (0,maximum(p0)+0.1), ylim = (0,maximum(p0)+0.1), dpi=600,
    series_annotations = text.(reactions,p0))
    Plots.abline!(sc, 1, 0, line=:dash, lc=:black)

    ls = plot(loss, title = "training loss", ylim = (0,maximum(loss)*1.5),
    xlabel = "epoch", ylabel = "loss", legend=false, lc=:black, dpi=600)

    pl1 = plot(sc, ls, layout=(1,2))
    savefig(pl1, folderN*"training_metrics.png")

    # ---- simulate ODE with parameters -----
    # more fine-grained time steps
    tps = collect(range(0.0,tspan[2],steps))
    problem = ODEProblem(massaction_stable, x0, tspan, pinf)
    integ = solve(problem, TRBDF2(), saveat=tps)
    integu = mapreduce(permutedims, vcat, integ.u)

    # ----- plot -----
    pp = []
    for i in 1:s
        
        # plot u
        plu = plot(tporig, u[i,:], lc=:black, title = species[i], label="kernel est", xlabel = "digestion time [hrs]", ylabel = "u", dpi=600, margin=5mm)
        plot!(tps, integu[:,i], lc=:red, label="predicted")
        plot!(tporig, X[:,i], lc=:green, label="ground truth")

        # plot du
        pldu = plot(du[i,:], lc=:black, title = species[i], label="kernel est", xlabel = "digestion time [hrs]", ylabel = "u", dpi=600, margin=5mm)
        plot!(y_pred[i,:], lc=:red, label="predicted")

        # combine
        push!(pp, plot(plu,pldu, layout = (1,2)))
    end
    pplot = plot(pp...; size = default(:size) .* (2,Int(ceil(s/2))), layout=(Int(ceil(s/2)),2), dpi = 600, margin=10mm)
    savefig(pplot, folderN*"residuals_kernel.pdf")

end



# ----- real data -----
function diagnostics_and_save_NN(tstates, ypreds, losses, loss_only=false, steps=10)

    pal = palette(:bamako, nr+1)

    # ----- loss
    print("loss....\n")

    ls = plot(losses[1], title="training loss", #ylim = (0,maximum(losses[1])*1.5),
    xlabel="epoch", ylabel="loss", label="rep_1", lc=pal[1], dpi=600)
    for ii in 2:nr
        plot!(losses[ii], col=pal[ii], label="rep_"*string(ii))
    end
    savefig(ls, folderN*"loss.png")

    # ----- parameters
    print("parameters....\n")
    pinfs = []
    for ii in 1:nr
        push!(pinfs, tstates[ii].parameters)
    end
    pinf = mapreduce(permutedims, vcat, pinfs)
    pdist = boxplot(pinf, palette = :bamako, legend=false,
    title="parameters", xlabel="parameter #", ylabel="parameter value", size = default(:size) .* (10,10))
    savefig(pdist, folderN*"parameters.pdf")

    if !loss_only
        # ---- simulate ODE with parameters -----
        # more fine-grained time steps
        tps = collect(range(0.0,tspan[2],steps))
        problem = ODEProblem(massaction_stable, x0, tspan, pinfs[1])
        
        # FIXME: too slow
        # integu = []
        # for ii in 1:nr
        #     integ = @time solve(problem, TRBDF2(), saveat=tps; p=pinfs[ii])
        #     push!(integu, mapreduce(permutedims, vcat, integ.u))
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
                pldu = plot(DUU_μ[i,:], ribbon=ribbon_duu, fillalpha=0.3, lc=:black, fc=:black,
                title = species[i], label="kernel est", xlabel = "digestion time [hrs]", ylabel = "du", dpi=600, margin=5mm)
                plot!(YPREDU_μ[i,:], ribbon=ribbon_ypredu, fillalpha=0.3, lc=:red, fc=:red, label="predicted")

                # combine
                push!(pp, plot(plu,pldu, layout = (1,2)))
            end

            pplot = plot(pp...; size = default(:size) .* (2,14), layout=(7,14), dpi = 600, margin=10mm)
            savefig(pplot, folderN*"tmp.pdf")
            append_pdf!(folderN*"simulated.pdf", folderN*"tmp.pdf", create=true, cleanup=true)
            counter += chunkSize
        end
    end
    
    print("\n")

end
