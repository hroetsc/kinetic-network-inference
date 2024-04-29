### kinetic-network-inference ###
# description:  get graph of non-spliced and spliced peptides for rate inference
# author:       HPR

library(data.table)
library(dplyr)
library(stringr)
library(parallel)
library(ggraph)
library(tidygraph)
source("src/_utils.R")
source("src/_graph_functions.R")
theme_set(theme_classic())

Nmin = 5
Nmax = 40
numCPU = 64
pcp_colour = "#EC9A56"
psp_colour = "#7B80C7"
substrate_colour = "black"

protein_name = "IDH1_WT"
dir.create("results/graphs/", showWarnings = F, recursive = T)

# ----- INPUT -----
finalK = fread("data/IDH1_WT/IDH1_WT_finalKinetics.csv")

# ----- preprocessing -----
finalK = finalK %>%
  dplyr::filter(!is.na(substrateID)) %>%
  disentangleMultimappersType()

pepTbl = finalK %>%
  dplyr::distinct(substrateID, substrateSeq, productType, pepSeq, positions) %>%
  tidyr::separate_rows(positions, sep = ";")

L = nchar(pepTbl$substrateSeq[1])
pepTbl$positions %>% unique() %>% length()
pepTbl$pepSeq %>% unique() %>% length()


# ----- graph and corresponding rates -----
GRAPH = constructGraphNetwork(pepTbl, numCPU, Nmin, Nmax) %>%
  dplyr::mutate(reaction_ID = 1:n(),
                rate_name = paste0("k_",reaction_ID))

# NOTE: no off rates
REACTIONS = GRAPH %>%
  dplyr::arrange(reaction_ID)

# plot
coord_graph = REACTIONS %>%
    dplyr::select(reactant_1, reactant_2, product_1, product_2) %>%
    tidyr::gather(reactant_cat, reactant, -product_1, -product_2) %>% 
    tidyr::gather(product_cat, product, -reactant_cat, -reactant) %>%
    dplyr::select(-reactant_cat, -product_cat) %>%
    na.omit() %>%
    as_tbl_graph() %>%
    activate(nodes) %>%
    mutate(type = fifelse(str_detect(name,"^[:digit:]+_[:digit:]+$"), "PCP", "PSP"),
           type = fifelse(name == paste0("1_",L), "substrate", type),
           detected = name %in% c(pepTbl$positions, paste0("1_",L)))
  
grph = ggraph(coord_graph, layout = "kk") +
  geom_edge_diagonal(alpha = .2) +
  geom_node_point(aes(colour = type, shape = detected), alpha = .8, size = 2) +
  scale_colour_manual(values = c(pcp_colour, psp_colour, substrate_colour)) +
  scale_shape_manual(values = c(1,16)) +
  ggtitle(protein_name)

ggsave(paste0("results/graphs/",protein_name,"_graph.pdf"), plot = grph,
       height = 8, width = 8, dpi = "retina")


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

# sanity check
all(pepTbl$positions %in% species)

# ----- OUTPUT -----
DATA = list(A = A,
            B = B,
            species = species,
            reactions = reactions,
            REACTIONS = REACTIONS,
            finalK = finalK,
            pepTbl = pepTbl)

save(DATA, file = paste0("results/graphs/",protein_name,"_v2-nofilter.RData"))

