### kinetic-network-inference ###
# description:  get abundance of species across replicates and time points
# author:       HPR

library(data.table)
library(dplyr)
library(stringr)
library(parallel)

protein_name = "IDH1_WT"


# ----- INPUT -----
load(paste0("results/graphs/",protein_name,".RData"))
finalK = DATA$finalK
substrateSeq = finalK$substrateSeq[1]
L = nchar(substrateSeq)

# TODO: better method to determine substrate degradation
load("../../../_tools/aSPIRE+invitroSPI/results/IDH1_WT/QUANTITIES_raw.RData")
# load("../../../aQUIRE/realData/data/substrateDegradation/DeltaC0_IDH1_221007.RData")

# ----- get abundances -----
intTbl = finalK %>%
  dplyr::distinct(substrateID, productType, positions, pepSeq, biological_replicate, digestTimes, intensities) %>%
  tidyr::separate_rows(positions, sep = ";") %>%
  tidyr::separate_rows(digestTimes, intensities, sep = ";") %>%
  dplyr::rename(digestTime = digestTimes,
                intensity = intensities) %>%
  dplyr::mutate(digestTime = as.numeric(digestTime),
                intensity = as.numeric(intensity)) %>%
  dplyr::mutate(intensity = ifelse(digestTime > 0 & intensity == 0, NA, intensity),
                intensity = log10(intensity),
                intensity = ifelse(digestTime == 0, 0, intensity),
                intensity = ifelse(!is.finite(intensity), NA, intensity)) %>%
  dplyr::mutate(biological_replicate = paste0("replicate_",biological_replicate))

SIGNAL = intTbl %>%
  dplyr::select(positions, biological_replicate, digestTime, intensity) %>%
  dplyr::rename(species = positions)


# ----- substrate degradation -----
substrate = QUANTITIES %>%
  dplyr::ungroup() %>%
  dplyr::filter(pepSeq == substrateSeq) %>%
  dplyr::distinct(substrateID, pepSeq, biological_replicate, digestTime, mean_bioRep) %>%
  dplyr::mutate(species = paste0("1_",L),
                productType = "PCP") %>%
  dplyr::rename(intensity = mean_bioRep) %>%
  dplyr::mutate(digestTime = as.numeric(digestTime),
                intensity = as.numeric(intensity)) %>%
  dplyr::mutate(intensity = log10(intensity),
                intensity = ifelse(!is.finite(intensity), NA, intensity)) %>%
  dplyr::mutate(biological_replicate = paste0("replicate_",biological_replicate)) %>%
  dplyr::select(species, biological_replicate, digestTime, intensity)

SIGNAL = list(SIGNAL, substrate) %>%
  rbindlist()


# ----- fill missing species with 0 -----
reps = SIGNAL$biological_replicate %>% unique()
times = SIGNAL$digestTime %>% unique()

missing = DATA$species[!DATA$species %in% unique(SIGNAL$species)]
missingTbl = tidyr::crossing(missing,
                             tidyr::crossing(reps, times) %>%
                               dplyr::rename(biological_replicate = reps,
                                             digestTime = times)) %>%
  dplyr::rename(species = missing) %>%
  dplyr::mutate(intensity = 0)

SIGNAL = list(SIGNAL, missingTbl) %>%
  rbindlist() %>%
  unique()


# ----- format -----
# array: (time, species, replicates)
SIGNALMATRIX = SIGNAL %>%
  tidyr::spread(digestTime, intensity, fill = NA) %>%
  dplyr::filter(species %in% DATA$species)  # FIXME: temporary!!!

S = array(NA, dim = c(length(times), length(unique(SIGNALMATRIX$species)), length(reps)),
          dimnames = list(times, unique(SIGNALMATRIX$species), reps))
for (r in reps) {
  S[,,r] = SIGNALMATRIX %>%
    dplyr::filter(biological_replicate == r) %>%
    dplyr::select(all_of(as.character(times))) %>%
    t() %>%
    as.matrix()
}


# ----- OUTPUT -----
DATA$S = S
DATA$SIGNALMATRIX = SIGNALMATRIX
DATA$timepoints = dimnames(S)[[1]] %>% as.numeric()
DATA$replicates = dimnames(S)[[3]]
DATA$species = dimnames(S)[[2]]
DATA$reactions = rownames(A)

save(DATA, file = paste0("results/graphs/",protein_name,".RData"))


# FIXME:few spliced peptides are not in the A/B matrices/reaction list!!!
# sanity check
A = DATA$A
species = dimnames(S)[[2]]
eulerr::euler(list(
  A = colnames(A),
  S = species
)) %>%
  plot(quantities = T)


