include("_odes.jl")

# simulated data
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