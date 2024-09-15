### kinetic-network-inference ###
# description:  get graph of non-spliced and spliced peptides for rate inference
# author:       HPR

# renv::init()
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
substrate_colour = "#6FA570"

protein_name = "IDH1_WT"
dir.create("results/graphs/", showWarnings = F, recursive = T)


# ----- INPUT -----
finalK = fread("data/IDH1_WT/IDH1_WT_finalKinetics.csv")
substrateSeq = finalK$substrateSeq[1]


# ----- preprocessing -----
finalK = finalK %>%
  dplyr::filter(!is.na(substrateID)) %>%
  disentangleMultimappersType()
# get I/L variants of PCPs!
finalK = finalK %>%
  dplyr::mutate(pepSeq = fifelse(productType == "PCP", substr(substrateSeq, as.numeric((str_extract(positions,"^[:digit:]+(?=_)"))), as.numeric((str_extract(positions,"(?<=_)[:digit:]+$")))), pepSeq))

pepTbl = finalK %>%
  dplyr::distinct(substrateID, substrateSeq, productType, pepSeq, positions) %>%
  tidyr::separate_rows(positions, sep = ";")

L = nchar(pepTbl$substrateSeq[1])
pepTbl$positions %>% unique() %>% length()
pepTbl$pepSeq %>% unique() %>% length()

# ----- graph and corresponding rates -----
GRAPH = constructGraphNetwork(pepTbl, numCPU, Nmin, Nmax) %>%
  dplyr::mutate(reaction_ID = 1:n(),
                rate_name = paste0("on_", reaction_ID))
                # rate_1_name = paste0("Vmax_",reaction_ID),
                # rate_2_name = paste0("Km_",reaction_ID))

# --- with off rates
# get off rates
OFF = GRAPH
nm = str_replace_all(names(GRAPH),"reactant_","x_")
nm = str_replace_all(nm ,"product_","reactant_")
nm = str_replace_all(nm ,"x_","product_")
names(OFF) = nm

OFF = OFF %>%
  dplyr::mutate(rate_name = paste0("off_",reaction_ID))

# combine
REACTIONS = list(GRAPH, OFF) %>%
  rbindlist(use.names = T, fill = T) %>%
  dplyr::arrange(reaction_ID)

# # --- without off rates
# # NOTE: no off rates
# REACTIONS = GRAPH %>%
#   dplyr::arrange(reaction_ID)

psps = unique(pepTbl$pepSeq[pepTbl$productType == "PSP"])

# plot
coord_graph = REACTIONS %>%
    dplyr::select(reactant_1_seq, reactant_2_seq, product_1_seq, product_2_seq) %>%
    tidyr::gather(reactant_cat, reactant, -product_1_seq, -product_2_seq) %>% 
    tidyr::gather(product_cat, product, -reactant_cat, -reactant) %>%
    dplyr::select(-reactant_cat, -product_cat) %>%
    na.omit() %>%
    as_tbl_graph() %>%
    activate(nodes) %>%
    mutate(type = fifelse(name %in% psps, "PSP", "PCP"),
           type = fifelse(name == substrateSeq, "substrate", type),
           detected = name %in% c(pepTbl$pepSeq, substrateSeq))
  
grph = ggraph(coord_graph, layout = "kk") +
  geom_edge_diagonal(alpha = .2) +
  geom_node_point(aes(colour = type, shape = detected), alpha = .8, size = 2) +
  scale_colour_manual(values = c(pcp_colour, psp_colour, substrate_colour)) +
  scale_shape_manual(values = c(1,16)) +
  ggtitle(protein_name)

ggsave(paste0("results/graphs/",protein_name,"_graph.pdf"), plot = grph,
       height = 12, width = 12, dpi = "retina")



# ----- matrices -----
species = c(REACTIONS$reactant_1_seq, REACTIONS$reactant_2_seq,
            REACTIONS$product_1_seq, REACTIONS$product_2_seq) %>% na.omit() %>% unique()
reactions = paste0("rct_",REACTIONS$reaction_ID)
s = length(species)
r = length(reactions)

# reactant and product matrices (to get stochiometry)
A = matrix(0, nrow = r, ncol = s)
B = matrix(0, nrow = r, ncol = s)

for (r in 1:nrow(REACTIONS)) {
  A[r,which(species %in% c(REACTIONS$reactant_1_seq[r],REACTIONS$reactant_2_seq[r]))] <- 1
}
for (r in 1:nrow(REACTIONS)) {
  B[r,which(species %in% c(REACTIONS$product_1_seq[r],REACTIONS$product_2_seq[r]))] <- 1
}

colnames(A) = species
rownames(A) = reactions

colnames(B) = species
rownames(B) = reactions

# sanity check
all(pepTbl$pepSeq %in% species)


# ----- OUTPUT -----
DATA = list(A = A,
            B = B,
            species = species,
            reactions = reactions,
            REACTIONS = REACTIONS,
            finalK = finalK,
            pepTbl = pepTbl)

save(DATA, file = paste0("results/graphs/",protein_name,"_v8-MA.RData"))

