### kinetic-network-inference ###
# description:  run inference
# author:       HPR

# renv::init()
library(BayesianTools)
library(deSolve)
library(RColorBrewer)
library(dplyr)
library(stringr)
library(parallel)
source("src/_plotChain.R")
source("src/_likelihoodFun.R")

numCPU = 14
protein_name = "IDH1_WT"
OUTNAME = "test_5"

folderN = paste0("results/inference/",protein_name,"/",OUTNAME)
dir.create(folderN, recursive = T, showWarnings = F)


# ----- INPUT -----
load(paste0("results/graphs/",protein_name,".RData"))

S = DATA$S
A = DATA$A
B = DATA$B

tpoints = dimnames(S)[[1]] %>% as.numeric()
T. = length(tpoints)

x0 = S[1,,]


# ----- preprocessing -----
replicates = dimnames(S)[[3]]
species = dimnames(S)[[2]]
reactions = rownames(A)

A = A[,species]
B = B[,species]

r = length(reactions)
s = length(species)
paramNames = c(rownames(A), "sigma")


# ----- settings -----
# parameters
numParam = length(paramNames)
Niter = 2e03

mini = c(rep(0, r), 0)
maxi = c(rep(10, r), 10)

# prior <- createUniformPrior(lower = mini, upper = maxi)
prior <- createTruncatedNormalPrior(lower = mini, mean = 5, sd = 1, upper = maxi)

sam = prior$sampler(n = 1e04)
plot(density(sam), main = "prior distribution")



# ----- initiate MCMC -----
bayesianSetup <- createBayesianSetup(likelihood = likelihoodFun, prior = prior)

# initialize and run sampler
settings <- list(iterations = Niter,
                  parallel = F,
                  consoleUpdates = 100,
                  nrChains = 8,  # number of chains
                  Z = NULL,  # starting population
                  startValue = 3,
                  pSnooker = 1e-06,  # probability of Snooker update
                  burnin = 0,  # number of iterations as burn in (not recorded)
                  thin = 1,  # thinning parameter
                  f = 2.38,  # scaling factor gamma
                  eps = 0,  # small number to avoid singularity
                  pGamma1 = 0.1,  # probability determining the frequency with which the scaling is set to 1 
                  eps.mult = 2,  # random term (multiplicative error)
                  eps.add = 0,  # random term
                  zUpdateFrequency = 1,
                  currentChain = 1,
                  message = TRUE)

print("START")
out <- runMCMC(bayesianSetup = bayesianSetup, sampler = "DEzs", settings = settings)
print("END")

posterior = getSample(out,end = NULL, thin = 10, parametersOnly=TRUE, whichParameters = 1:numParam)
plotChain(chain = posterior)
save(posterior,file=paste(folderN,"/posterior.RData",sep=""))

for(ii in 1:100) {
  
  print(ii)
  
  print("RESTART")
  out <- runMCMC(out, sampler = "DEzs", settings = settings)
  print("END")
  
  posterior = getSample(out,end = NULL, thin = 10,parametersOnly=TRUE, whichParameters = 1:numParam)
  plotChain(chain = posterior)
  save(posterior,file=paste(folderN,"/posterior.RData",sep=""))
  
}

