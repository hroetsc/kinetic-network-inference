### kinetic-network-inference ###
# description:  analyse parameters inferred from collocation-based training
# author:       HPR

library(data.table)
library(dplyr)
library(stringr)
library(tidyr)
library(ggplot2)
library(tidygraph)
library(ggraph)
theme_set(theme_classic())

protein_name = "IDH1_WT"
OUTNAME = "nn_v2"
folderN = paste0("results/collocation/",protein_name,"/",OUTNAME,"/")

# ----- INPUT -----
parameters = fread(paste0(folderN,"parameters.csv"))


# ----- filter reliable reactions -----
parameters = parameters %>%
    dplyr::rowwise() %>%
    dplyr::mutate(mean_rate = mean(c(param_1, param_2, param_3)),
                  cv_rate = abs(sd(c(param_1, param_2, param_3))/mean_rate)) %>%
    dplyr::mutate(reliable_rate = cv_rate <= 0.1)

table(parameters$productType, parameters$reliable_rate)
# more PSP parameters (relatively) are deemed reliable

ggplot(parameters, aes(x = productType, y = mean_rate, fill = reliable_rate)) +
    geom_boxplot()
# reliable PCP rates are higher than unreliable ones

# --- for PCP
kpcp = which(parameters$productType == "PCP")
table(parameters$reactant_1_valid[kpcp], parameters$reliable_rate[kpcp])
table(parameters$reactant_1_detected[kpcp], parameters$reliable_rate[kpcp])
# majority of not detected PCP precursors have reliable reaction rates
# majority of detected PCP precursors have unreliable rates

table(parameters$product_1_valid[kpcp], parameters$reliable_rate[kpcp])
table(parameters$product_2_valid[kpcp], parameters$reliable_rate[kpcp])
# majority of not detected products have unreliable rates
# but also majority of detected prpducts have unreliable rates (there seems to be another factor)

# --- for PSP
kpsp = which(parameters$productType == "PSP")
table(parameters$reactant_1_valid[kpsp], parameters$reliable_rate[kpsp])
table(parameters$reactant_2_valid[kpsp], parameters$reliable_rate[kpsp])

table(parameters$reactant_1_detected[kpsp], parameters$reliable_rate[kpsp])
table(parameters$reactant_2_detected[kpsp], parameters$reliable_rate[kpsp])


# ----- plot rates -----
parameters

