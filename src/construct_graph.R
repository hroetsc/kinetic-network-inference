### kinetic-network-inference ###
# description:  get graph of non-spliced and spliced peptides for rate inference
# author:       HPR

library(data.table)
library(dplyr)
library(stringr)
library(parallel)
source("src/_utils.R")
source("src/_graph_functions.R")

# FIXME:few spliced peptides are not in the A/B matrices/reaction list!!!

Nmin = 5
Nmax = 40
numCPU = 2

protein_name = "IDH1_WT"
dir.create("results/graphs/", showWarnings = F, recursive = T)

# ----- INPUT -----
finalK = fread("../../../_tools/aSPIRE+invitroSPI/results/IDH1_WT/IDH1_WT_finalKinetics.csv")


# ----- preprocessing -----
finalK = finalK %>%
  dplyr::filter(!is.na(substrateID)) %>%
  disentangleMultimappersType()

pepTbl = finalK %>%
  dplyr::distinct(substrateID, substrateSeq, productType, pepSeq, positions) %>%
  tidyr::separate_rows(positions, sep = ";")


# ----- graph and corresponding rates -----
GRAPH = constructGraphNetwork(pepTbl, numCPU, Nmin, Nmax) %>%
  dplyr::mutate(reaction_ID = 1:n(),
                rate_name = paste0("k_",reaction_ID))

# no off rates

REACTIONS = GRAPH %>%
  dplyr::arrange(reaction_ID)


# ----- matrices -----
species = c(REACTIONS$reactant_1, REACTIONS$reactant_2,
            REACTIONS$product_1, REACTIONS$product_2) %>% na.omit() %>% unique()
reactions = REACTIONS$rate_name
s = length(species)
r = length(reactions)

# reactant and product matrices (to get stochiometry)
A = matrix(0, nrow = r, ncol = s)
B = matrix(0, nrow = r, ncol = s)

for (r in 1:nrow(REACTIONS)) {
  A[r,which(species %in% c(REACTIONS$reactant_1[r],REACTIONS$reactant_2[r]))] <- 1
}
for (r in 1:nrow(REACTIONS)) {
  B[r,which(species %in% c(REACTIONS$product_1[r],REACTIONS$product_2[r]))] <- 1
}

colnames(A) = species
rownames(A) = reactions

colnames(B) = species
rownames(B) = reactions


# ----- OUTPUT -----
DATA = list(A = A,
            B = B,
            species = species,
            reactions = reactions,
            REACTIONS = REACTIONS,
            finalK = finalK,
            pepTbl = pepTbl)

save(DATA, file = paste0("results/graphs/",protein_name,"_v2.RData"))

