### kinetic-network-inference ###
# description:  create simulated network
# author:       HPR

library(data.table)
library(dplyr)
library(stringr)
library(parallel)
library(deSolve)
library(ggplot2)
theme_set(theme_classic())
numCPU = 4

n = 20
OUTNAME = "Ex4_8s"
tpoints = seq(0,4,0.05)
tpoints_coarse = seq(0,4,1)

chosenPeps = c(
    "INFER", "ENCEEVERYDAYANDNIGHT",
    "IN", "FERENCEEVERYDAYANDNIGHT",
    "INFERENCE", "EVERYDAYANDNIGHT",
    "INFERENCEEVERY", "DAYANDNIGHT"
    # "INFERENCEEVERYDAY", "ANDNIGHT",
    # "INFERENCEEVERYDAYAND", "NIGHT",
    # "INF", "ERENCEEVERYDAYANDNIGHT"
)


# ----- INPUT -----
substrateSeq = "INFERENCEEVERYDAYANDNIGHT"
substrateSeq_split = strsplit(substrateSeq, "") %>% unlist()
L = nchar(substrateSeq)
# substrateNode = paste0("1_",L)

# ----- generate peptide sequences -----
allPCP = mclapply(seq(1, L), function(N){
   
    allPCP = sapply(seq(1,L-N+1), function(i){
        j = N+i-1
        positions = paste(i,j, sep = "_")
        pepSeq=str_sub(substrateSeq,i,j)
        return(c(pepSeq,positions,i,j))
    })

    PCP = unlist(allPCP)
    PCP_Pep = data.table(pepSeq = PCP[1,],
                            positions = PCP[2,],
                            P1 = PCP[4,] %>% as.numeric(), # !!!!!
                            P1_ = PCP[3,] %>% as.numeric())

  }, 
mc.preschedule = T,
mc.cleanup = T,
mc.cores = numCPU)

allPCP <- data.table::rbindlist(allPCP) %>%
    dplyr::mutate(N = nchar(pepSeq))

# set.seed(893)
# uniquePeps = allPCP$pepSeq %>% unique()
# paste0("number of unique peptides: ", length(uniquePeps)) %>% print()
# k = sample(1:length(uniquePeps), n)
# chosenPCP = allPCP %>%
#     dplyr::filter(pepSeq %in% c(uniquePeps[k], substrateSeq)) # add the substrate

chosenPCP = allPCP %>%
    dplyr::filter(pepSeq %in% c(chosenPeps, substrateSeq)) # add the substrate

# ----- build graph -----
# NOTE: nodes should be PEPTIDES! edges can account for all positions!
# otherwise simulation will be crap

pcp = allPCP %>%
    dplyr::group_by(pepSeq, N) %>%
    dplyr::reframe(positions = paste(positions, collapse = ";"),
                    P1 = paste(P1, collapse = ";"),
                    P1_ = paste(P1_, collapse = ";")) %>%
    dplyr::arrange(N) %>%
    dplyr::filter(N > 1)


COORD_pcp = mclapply(1:nrow(pcp), function(i){
    
    cnt = pcp[i,] %>%
        tidyr::separate_rows(positions, P1, P1_, sep = ";")

    en = cnt$P1 %>% as.numeric()
    st = cnt$P1_ %>% as.numeric()
    N = pcp$N[i]
    
    reactant = cnt$positions
    reactantSeq = pcp$pepSeq[i]
    reactant_detected = any(reactant %in% chosenPCP$positions)

    # get all possible cleavage sites
    cleave = seq(1,N-1)
    
    # get products resulting from cleavage of current PCP
    # iterate all coordinates of current reactant sequence
    PCP = lapply(1:length(st), function(ii) {

        PCP = lapply(cleave, function(c) {

            abs_site = st[ii]+c-1
            k1 = which(allPCP$P1_ == st[ii] & allPCP$P1 == abs_site)
            k2 = which(allPCP$P1_ == abs_site+1 & allPCP$P1 == en[ii])
            
            product_detected = sapply(c(k1,k2), function(ki){
                allPCP$pepSeq[ki] %in% chosenPCP$pepSeq
            })

            # reactant --> product
            PCP = data.table(reactant_1 = reactant[ii], reactant_2 = NA,
                            product_1 = allPCP$positions[k1], product_2 = allPCP$positions[k2],
                            reactant_1_seq = reactantSeq, reactant_2_seq = NA,
                            product_1_seq = allPCP$pepSeq[k1], product_2_seq = allPCP$pepSeq[k2],
                            reactant_1_detected = reactant_detected, reactant_2_detected = NA,
                            product_1_detected = product_detected[1], product_2_detected = product_detected[2],
                            cleavage_site_abs = abs_site, cleavage_site = c)
            
            return(PCP)
        }) %>%
        rbindlist()

    }) %>%
        rbindlist()
    
    return(PCP)
  }, 
  mc.preschedule = T,
  mc.cleanup = T,
  mc.cores = numCPU)

allPCP = COORD_pcp %>%
    data.table::rbindlist() %>%
    as_tibble()
paste0("number of reactions (all possible): ", nrow(allPCP)) %>% print()

# filter out reactions where neither reactant nor products are observed
reactions = allPCP %>%
    # !!!!!!!!!!!!!!!!!!!!
    dplyr::filter(reactant_1_seq %in% chosenPCP$pepSeq & (product_1_seq %in% chosenPCP$pepSeq | product_2_seq %in% chosenPCP$pepSeq)) %>% 
    # !!!!!!!!!!!!!!!!!!!!
    dplyr::filter(reactant_1_detected | product_1_detected | product_2_detected) %>%
    dplyr::mutate(reaction_ID = 1:n())
paste0("number of reactions (filtered): ", nrow(reactions)) %>% print()

# TODO: number of species actually in network
# TODO (perhaps): set 1 aa products to 0

# ----- build stochiometry matrix from adjacency list -----
species = c(reactions$reactant_1_seq, reactions$reactant_2_seq, reactions$product_1_seq, reactions$product_2_seq) %>%
    na.omit() %>%
    unique() %>%
    sort()
paste0("number of species in actual network: ", length(species)) %>% print()
# species = species[-which(species %in% c(NA,""))]

r = nrow(reactions)
s = length(species)

# reactant and product matrices (to get stochiometry)
A = matrix(0, nrow = r, ncol = s)
B = matrix(0, nrow = r, ncol = s)

for (r in 1:nrow(reactions)) {
  A[r,which(species %in% c(reactions$reactant_1_seq[r],reactions$reactant_2_seq[r]))] <- 1
}
for (r in 1:nrow(reactions)) {
  B[r,which(species %in% c(reactions$product_1_seq[r],reactions$product_2_seq[r]))] <- 1
}

colnames(A) = species
rownames(A) = reactions$reaction_ID

colnames(B) = species
rownames(B) = reactions$reaction_ID

# ----- get distance to substrate node -----
# TODO


# ----- simulate -----
# ----- ODEs
massAction <- function(t,X,p) {
  # vector matrix exponentiation of x by A
  M = apply(t(X^t(A)), 1, prod)
  # calculate dX
  dX = t(B-A)%*%(p*M)
  list(dX)
}

# michaelisMenten <- function(t,X,p) {
#   # vector matrix exponentiation of x by A
#   m = apply(t(X^t(A)), 1, prod)

#   Vmax = p[1:r]
#   Km = p[(r+1):(2*r)]
#   # calculate dX
#   dX = t(B-A)%*%((Vmax*m) / (Km+m))

#   list(dX)
# }

# ----- parameters, initial states
# initial state
x0 = rep(0,length(species))
x0[species == substrateSeq] = 100

# rates
set.seed(345)
rate_ma = runif(nrow(reactions), 0.1, 0.7)
# rate_ma = seq(0.2,0.3,length.out=r)
# rate_vmax = runif(nrow(reactions), 0, 20)
# rate_km = runif(nrow(reactions), 0, 0.5)

# names
names(x0) = species
names(rate_ma) = paste0("k_",reactions$reaction_ID)
# names(rate_vmax) = paste0("vmax_",reactions$reaction_ID)
# names(rate_km) = paste0("km_",reactions$reaction_ID)

# ----- actual simulation
out_MA <- ode(func = massAction,
              y = x0,
              parms = rate_ma,
              times = tpoints)

# out_MM <- ode(func = michaelisMenten,
#               y = x0,
#               parms = c(rate_vmax,rate_km),
#               times = tpoints)

# ----- plot
cols = rainbow(n = s)

plot_ma = out_MA %>%
  as_tibble() %>%
  tidyr::gather(variable,value,-time) %>%
  ggplot(aes(x=time,y=value,color=variable))+
  geom_line(linewidth=.5)+
  scale_color_manual(values = cols, "component") +
  xlim(c(0,4)) +
  labs(x='time (h)',y='concentration') + 
  ggtitle("simulated kinetics - mass action") +
  theme(legend.position = "none")
plot_ma

ggsave(paste0("data/simulation/",OUTNAME,"_MA.png"),
       plot = plot_ma, height = 6, width = 12, dpi = "retina")


# plot_mm = out_MM %>%
#   as_tibble() %>%
#   tidyr::gather(variable,value,-time) %>%
#   ggplot(aes(x=time,y=value,color=variable))+
#   geom_line(linewidth=.5)+
#   scale_color_manual(values = cols, "component") +
#   xlim(c(0,4)) +
#   labs(x='time (h)',y='concentration') + 
#   ggtitle("simulated kinetics - michaelis menten") +
#   theme(legend.position = "none")
# plot_mm

# ggsave(paste0("data/simulation/",OUTNAME,"_MM.png"),
#        plot = plot_ma, height = 6, width = 8, dpi = "retina")


# ----- OUTPUT -----
X_MA = out_MA %>% as.matrix()
X_MA = X_MA[X_MA[,1] %in% tpoints_coarse, species]
reactions = reactions %>%
    dplyr::mutate(rate_ma = rate_ma,
                    rate_name_ma = names(rate_ma))
DATA_MA = list(X = X_MA[,species],
            x0 = X_MA[1,species],
            A = A[reactions$reaction_ID,species],
            B = B[reactions$reaction_ID,species],
            tp = tpoints_coarse,
            species = species,
            reactions = reactions$reaction_ID,
            info = reactions)
save(DATA_MA, file = paste0("data/simulation/",OUTNAME,"_MA_DATA.RData"))


# X_MM = out_MM %>% as.matrix()
# X_MM = X_MM[X_MM[,1] %in% tpoints_coarse, species]
# DATA_MM = list(X = X_MM[,species],
#             x0 = X_MM[1,species],
#             A = A[reactions$reaction_ID,species],
#             B = B[reactions$reaction_ID,species],
#             tp = tpoints_coarse,
#             species = species,
#             reactions = reactions$rate_name,
#             info = reactions)
# save(DATA_MM, file = paste0("data/simulation/",OUTNAME,"_MM_DATA.RData"))

