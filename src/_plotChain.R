### pigeons-aQUIRE network ###
# description:  fuctions for plotting chains
# author:       HPR


plotChain <- function(chain){
  
  NN = dim(chain)[1]
  burnIn = round(0.1*NN)
  
  # plot histograms and chains
  pdf(paste(folderN,"/chain.pdf",sep=""), width=9, height=12)
  par(mfrow = c(5,2))
  
  for(i in 1:dim(chain)[2]){
    hist(chain[-(1:burnIn),i], main=paramNames[i], xlab="",breaks=70,col="grey" )
    abline(v = mean(chain[-(1:burnIn),i]), col="red")
    plot(chain[,i],xlab="iteration",ylab=paste0(paramNames[i]),main="",axes=FALSE,type="l")
    axis(1)
    axis(2)
  }
  dev.off()
  
  # get boxplots
  # compute and plot estimated RT values on posterior sample
  # sample from posterior
  post = chain[-(1:burnIn),]
  NNN = min(c(dim(post)[1],100))
  sampledIndex = sample(c(1:dim(post)[1]),size=NNN,replace=FALSE)
  param = post[sampledIndex,-numParam]
  
  
  # ----- get simulated concentrations -----
  OUT = lapply(1:nrow(param), function(i){
    ode(func = massAction,
        y = x0[,1],
        parms = param[i,],
        times = tpoints,
        method = "euler")
  })
  
  
  # plot diagnostics
  pdf(paste(folderN,"/residuals.pdf",sep=""), width=15, height=8)

  # ----- posterior distributions -----
  PP = chain[-(1:burnIn),]
  colnames(PP) = paramNames
  boxplot(PP,
          main = "posterior parameter distribution",
          ylab = "rate",
          xlab = "parameter")
  
  par(mfrow = c(2,3))
  # ----- simulated kinetics -----
  out = abind::abind(OUT, along = 3)
  
  csim_means = apply(out, c(1,2), mean)[,species]
  csim_lower = apply(out, c(1,2), quantile, 0.05)[,species]
  csim_upper = apply(out, c(1,2), quantile, 0.95)[,species]
  
  for (i in 1:length(species)) {
    
    mm = min(csim_means[,i], csim_lower[,i], csim_upper[,i])
    mm = if (mm<0 & is.finite(mm)) mm else 0
    l = c(mm, max(csim_means[,i], csim_lower[,i], csim_upper[,i]) %>% ceiling())
    l[!is.finite(l)] = 0
    
    plot(x = c(0,tpoints), y = c(0,csim_means[,i]),
         type = "l", ylim = l,
         main = species[i], xlab = "time (hrs)", ylab = "simulated concentration")
    
    polygon(x = c(c(0,tpoints), rev(c(0,tpoints))),
            y = c(c(0, csim_lower[,i]), rev(c(0, csim_upper[,i]))),
            col = adjustcolor("lightgray", alpha.f = 0.5), lty = 0)
    
    for (ii in 1:length(replicates)) {
      lines(x = tpoints, y = S[,species[i],ii],
            type = "b", col = "red", pch = 16, cex = 1.2)
    }
    
  }
  
  dev.off()
  
  
}

