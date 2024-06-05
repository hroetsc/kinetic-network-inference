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
library(parallel)
theme_set(theme_classic())
numCPU = 4

protein_name = "IDH1_WT"
# FIXME
OUTNAME = "nn_v2+bias"
folderN = paste0("results/collocation_MM/",protein_name,"/",OUTNAME,"/")
dir.create("results/_analysis/goodness-of-fits/", showWarnings = FALSE, recursive = TRUE)

# ----- INPUT -----
# FIXME
load("results/graphs/IDH1_WT_v5-MM.RData")
REACTIONS = DATA$REACTIONS
# REACTIONS = DATA$REACTIONS %>%
#     dplyr::filter(!grepl("off_", rate_name))

parameters = fread(paste0(folderN,"parameters.csv"))
bias = fread(paste0(folderN,"bias.csv"))

fs = list.files(path = folderN, pattern = "pred_", full.names=TRUE, recursive=TRUE)
predictions = lapply(fs, function(f){
    fread(f)
})

species = DATA$species
L = nchar(unique(DATA$finalK$substrateSeq))
substrateNode = paste0("1_",L)


# ----- calculate prediction error -----
getPredError = function(pred) {
    
    cvals = lapply(1:nrow(pred), function(i){
        cc = cor.test(c(pred$du0[i],pred$du1[i],pred$du2[i],pred$du3[i],pred$du4[i]),
                    c(pred$ypred0[i],pred$ypred1[i],pred$ypred2[i],pred$ypred3[i],pred$ypred4[i]))
        err = sqrt(mean((c(pred$du0[i],pred$du1[i],pred$du2[i],pred$du3[i],pred$du4[i])-
                        c(pred$ypred0[i],pred$ypred1[i],pred$ypred2[i],pred$ypred3[i],pred$ypred4[i]))^2))
        return(c(cc$estimate, cc$p.value, err))
    }) %>%
        plyr::ldply() %>%
        dplyr::rename(cor = cor, pval = V1, rmse = V2) %>%
        dplyr::mutate(species = species)

   return(cvals) 
}

cvals = lapply(1:length(predictions), function(i){
    getPredError(predictions[[i]])
})

predError = data.frame(
    species = cvals[[1]]$species,
    cor = apply(cbind(cvals[[1]]$cor, cvals[[2]]$cor, cvals[[3]]$cor), 1, mean, na.rm = TRUE),
    pval = apply(cbind(cvals[[1]]$pval, cvals[[2]]$pval, cvals[[3]]$pval), 1, mean, na.rm = TRUE),
    rmse = apply(cbind(cvals[[1]]$rmse, cvals[[2]]$rmse, cvals[[3]]$rmse), 1, mean, na.rm = TRUE)
) %>%
    as_tibble()

# ----- get maximum intensity and distance to substrate -----
maxI = apply(DATA$S, c(2,3), mean, na.rm = TRUE) %>%
    as.data.frame() %>%
    tibble::rownames_to_column("species") %>%
    dplyr::rowwise() %>%
    dplyr::mutate(mean_intensity = mean(c(replicate_1,replicate_2,replicate_3), na.rm = TRUE),
                    sd_intensity = sd(c(replicate_1,replicate_2,replicate_3), na.rm = TRUE))

# distance to substrate reaction node
species_mod = species[-which(species == "1_27")]
substrateDegree = mclapply(1:length(species_mod), function(jj){
    sp = species_mod[jj]

    k = which(REACTIONS$product_1 == sp | REACTIONS$product_2 == sp)
    Hop = REACTIONS[k,]
    reactants = unique(c(Hop$reactant_1, Hop$reactant_2))
    deg = 1
    while(!substrateNode %in% reactants) {
        kk = which(REACTIONS$product_1 %in% reactants | REACTIONS$product_2 %in% reactants)
        k = unique(c(k,kk))
        Hop = REACTIONS[k,]
        reactants = unique(c(Hop$reactant_1, Hop$reactant_2))
        deg = deg + 1
    }

    return(c(sp, deg))
}, mc.cores = numCPU, mc.cleanup = TRUE, mc.preschedule = TRUE)


substrateDegree = substrateDegree %>%
    plyr::ldply() %>%
    dplyr::rename(species = V1, degree = V2) %>%
    dplyr::mutate(degree = as.numeric(degree))


# ----- combine and plot -----
MASTER = full_join(full_join(maxI, substrateDegree), predError)
MASTER = MASTER %>%
    dplyr::mutate(productType = ifelse(str_detect(species, "^[:digit:]+_[:digit:]+$"), "PCP", "PSP"),
                    col = ifelse(productType == "PCP", "#EC9A56", "#7B80C7"))
names(MASTER)

png("results/_analysis/goodness-of-fits/intensity-vs-error.png", height = 6, width = 10, units = "in", res = 300)
par(mfrow = c(1,2))
plot(MASTER$mean_intensity, MASTER$rmse, col = MASTER$col,
    xlab = "max intensity of product", ylab = "prediction error on du (RMSE)",
    main = "intensity vs. error")
plot(MASTER$mean_intensity, MASTER$rmse, ylim = c(0,20), col = MASTER$col,
    xlab = "max intensity of product", ylab = "prediction error on du (RMSE)",
    main = "intensity vs. error - zoomed in")
dev.off()
file.copy("results/_analysis/goodness-of-fits/intensity-vs-error.png", folderN, overwrite = TRUE)

ggplot(MASTER %>% dplyr::filter(!is.na(degree)), aes(x = factor(degree), y = rmse, col = productType)) +
    geom_violin(draw_quantiles = 0.5) +
    scale_y_log10() +
    scale_color_manual(values = c("#EC9A56", "#7B80C7")) +
    xlab("distance to substrate (degree)") +
    ylab("prediction error on du (RMSE)") +
    ggtitle("degree vs. error")
ggsave("results/_analysis/goodness-of-fits/degree-vs-error.png", plot = last_plot(),
    height = 6, width = 5, dpi = "retina")
file.copy("results/_analysis/goodness-of-fits/degree-vs-error.png", folderN, overwrite = TRUE)


# ----- introduce bias term -----
bias = bias %>%
    dplyr::rowwise() %>%
    dplyr::mutate(mean_bias = mean(c(b_1,b_2,b_3), na.rm = TRUE))

MASTER = full_join(MASTER, bias)

png("results/_analysis/goodness-of-fits/intensity-vs-bias.png", height = 6, width = 5, units = "in", res = 300)
plot(MASTER$mean_intensity, MASTER$mean_bias, col = MASTER$col,
    xlab = "max intensity of product", ylab = "bias term value",
    main = "intensity vs. bias")
dev.off()
file.copy("results/_analysis/goodness-of-fits/intensity-vs-bias.png", folderN, overwrite = TRUE)

ggplot(MASTER %>% dplyr::filter(!is.na(degree)), aes(x = factor(degree), y = mean_bias, col = productType)) +
    geom_violin(draw_quantiles = 0.5) +
    scale_color_manual(values = c("#EC9A56", "#7B80C7")) +
    xlab("distance to substrate (degree)") +
    ylab("bias term value") +
    ggtitle("degree vs. bias")
ggsave("results/_analysis/goodness-of-fits/degree-vs-bias.png", plot = last_plot(),
    height = 6, width = 5, dpi = "retina")
file.copy("results/_analysis/goodness-of-fits/degree-vs-bias.png", folderN, overwrite = TRUE)


