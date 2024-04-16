### kinetic-network-inference ###
# description:  likelihood function
# author:       HPR


# ----- simulate kinetics -----
# NOTE: rates are simulated on signal level
massAction <- function(t,X,p) {
  
  # rates as diagonal matrix
  # K = diag(p,r,r)
  # vector matrix exponentiation of x by A
  M = apply(t(X^t(A)), 1, prod)
  # calculate dX
  dX = t(B-A)%*%(p*M)
  
  list(dX)
}



# ----- calculate likelihood -----
likelihoodFun <- function(param){
  
  sigma = param[numParam]  # sd of prior
  prm = param[-numParam]
  
  # --- ODE part
  L = mclapply(1:length(replicates), function(i){
    
    # start_time = Sys.time()
    out <- ode(func = massAction,
               y = x0[,i],
               parms = prm,
               times = tpoints,
               method = "euler")
    # end_time = Sys.time()
    # difftime(end_time,start_time)
    
    out = out[,species]
    j = which(!is.na(S[,,i]), arr.ind = T)
    likelihood = dnorm(x=out[j], mean=S[j[,1],j[,2],i], sd=sigma, log=T)
    
    # --- account for negative infinity and combine
    likelihood[!is.finite(likelihood)] = -10**11
    
    L = sum(likelihood)
  }, mc.cores = numCPU)
  
  LIKELIHOOD = unlist(L) %>% sum()
  
  return(LIKELIHOOD)
}


