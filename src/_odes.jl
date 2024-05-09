### kinetic-network-inference ###
# description:  functions for mass action ODE and Jacobian
# author:       HPR


# ---- mass action ODE -----
function massaction!(du, u, p, t)
    m = exp.(A*log.(u))
    du[1:s] = N*Diagonal(p)*m
    nothing
end

function massaction_stable(du, u, p, t)
    m = prod((transpose(u) .^ A), dims=2)
    du[1:s] = N*Diagonal(p)*m
    nothing
end


function massaction_fast(du, u, p, t, m=m, A=A, N=N)
    u[u .< 0] .= 1.0
    mul!(m, A, log.(u))
    pm = p.*exp.(m)
    mul!(du, N, pm)
    nothing
end




# ----- analytical Jacobian -----
function jacobian!(J, u, p, t)
    M = Diagonal(vec(exp.(A*log.(u))))
    J[1:s,1:s] = N*Diagonal(p)*M*A*inv(Diagonal(u))
    nothing
end

function jacobian_stable(J, u, p, t)
    M = Diagonal(vec(prod((transpose(u) .^ A), dims=2)))
    J[1:s,1:s] = N*Diagonal(p)*M*A*inv(Diagonal(u))
    nothing
end


function jacobian_fast(J, u, p, t, m=m, A=A, N=N, Np=Np, Npm=Npm, Au=Au)
    u[u .< 0] .= 1.0
    mul!(m, A, log.(u))
    mul!(Np, N, Diagonal(p))
    mul!(Npm, Np, Diagonal(exp.(m)))
    mul!(Au, A, inv(Diagonal(u)))
    mul!(J,Npm,Au)
    nothing
end
