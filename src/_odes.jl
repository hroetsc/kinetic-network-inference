### kinetic-network-inference ###
# description:  functions for mass action ODE and Jacobian
# author:       HPR

# TODO: try using mul! instead of * for matrix multiplication
# TODO: faster way of computing x^A

# ---- mass action ODE -----
# function massaction!(du, u, p, t)
#     m = exp.(A*log.(u))
#     du[1:s] = N*Diagonal(p)*m
#     nothing
# end

function massaction_stable!(du, u, p, t)
    m = prod((transpose(u) .^ A), dims=2)
    du[1:s] = N*Diagonal(p)*m
    nothing
end

# function massaction_compromise!(du, u, p, t)
#     if all(u .> 0)
#         m = exp.(A*log.(u))
#     else
#         m = prod((transpose(u) .^ A), dims=2)
#     end
#     du[1:s] = N*Diagonal(p)*m
#     nothing
# end


# ----- analytical Jacobian -----
# function jacobian!(J, u, p, t)
#     M = Diagonal(exp.(A*log.(u)))
#     J[1:s,1:s] = N*Diagonal(p)*M*A*inv(Diagonal(u))
#     nothing
# end

function jacobian_stable!(J, u, p, t)
    M = Diagonal(prod((transpose(u) .^ A), dims=2))
    J[1:s,1:s] = N*Diagonal(p)*M*A*inv(Diagonal(u))
    nothing
end

# function jacobian_compromise!(J, u, p, t)
#     if all(u .> 0)
#         M = Diagonal(exp.(A*log.(u)))
#     else
#         M = Diagonal(prod((transpose(u) .^ A), dims=2))
#     end
#     J[1:s,1:s] = N*Diagonal(p)*M*A*inv(Diagonal(u))
#     nothing
# end
