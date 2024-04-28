### kinetic-network-inference ###
# description:  functions for mass action ODE and Jacobian
# author:       HPR

# ---- mass action ODE -----
function massaction!(du, u, p, t)
    m = exp.(A*log.(u))
    du[1:s] = N*Diagonal(p)*m
    nothing
end

function massaction_init!(du, u, p, t)
    m = prod((transpose(u) .^ A), dims=2)
    du[1:s] = N*Diagonal(p)*m
    nothing
end


# ----- analytical Jacobian -----
function jacobian!(J, u, p, t)
    M = Diagonal(exp.(A*log.(u)))
    J[1:s,1:s] = N*Diagonal(p)*M*A*inv(Diagonal(u))
    nothing
end
