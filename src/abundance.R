### kinetic-network-inference ###
# description:  get abundance of species across replicates and time points
# author:       HPR

# incorporates results from  construct_graph.R and substrate_degradation.R

library(data.table)
library(dplyr)
library(stringr)
library(parallel)

Nmin = 5
Nmax = 40
protein_name = "IDH1_WT"

# ----- INPUT -----
load(paste0("results/graphs/",protein_name,"_v2-1hop.RData"))
finalK = DATA$finalK
substrateSeq = finalK$substrateSeq[1]
L = nchar(substrateSeq)

load(paste0("results/substrate_degradation/",protein_name,"_degradation.RData"))

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

# sanity check
intTbl$digestTime %>% unique()

# ----- substrate degradation -----
substrate = substrateTbl %>%
  dplyr::mutate(species = paste0("1_",L),
                productType = "PCP") %>%
  dplyr::rename(intensity = mean_int) %>%
  dplyr::mutate(digestTime = as.numeric(digestTime),
                intensity = as.numeric(intensity)) %>%
  dplyr::mutate(intensity = log10(intensity),
                intensity = ifelse(!is.finite(intensity), NA, intensity)) %>%
  dplyr::mutate(biological_replicate = paste0("replicate_",biological_replicate)) %>%
  dplyr::select(species, biological_replicate, digestTime, intensity)

SIGNAL = list(SIGNAL, substrate) %>%
  rbindlist()


# ----- fill missing species with N/A or 0 -----
reps = SIGNAL$biological_replicate %>% unique()
times = SIGNAL$digestTime %>% unique()

missing = DATA$species[!DATA$species %in% unique(SIGNAL$species)]
Ns = apply(str_split_fixed(missing, "_", Inf), 2, as.numeric) 
Ns = Ns[,2]-Ns[,1]+1
validLength = Ns >= Nmin & Ns <= Nmax
table(validLength)


missingTbl_NA = tidyr::crossing(missing[!validLength],
                                tidyr::crossing(reps, times) %>%
                                  dplyr::rename(biological_replicate = reps,
                                                digestTime = times)) %>%
  dplyr::rename(species = `missing[!validLength]`) %>%
  dplyr::mutate(intensity = NA)

missingTbl_0 = tidyr::crossing(missing[validLength],
                                tidyr::crossing(reps, times) %>%
                                  dplyr::rename(biological_replicate = reps,
                                                digestTime = times)) %>%
  dplyr::rename(species = `missing[validLength]`) %>%
  dplyr::mutate(intensity = 0)


SIGNAL = list(SIGNAL, missingTbl_NA, missingTbl_0) %>%
  rbindlist() %>%
  unique()


# ----- format -----
# array: (time, species, replicates)
SIGNALMATRIX = SIGNAL %>%
  tidyr::spread(digestTime, intensity, fill = NA)
  
S = array(NA, dim = c(length(times), length(unique(SIGNALMATRIX$species)), length(reps)),
          dimnames = list(times, unique(SIGNALMATRIX$species), reps))
for (r in reps) {
  S[,,r] = SIGNALMATRIX %>%
    dplyr::filter(biological_replicate == r) %>%
    dplyr::select(all_of(as.character(times))) %>%
    t() %>%
    as.matrix()
}

# NOTE: IMPORTANT!!!!!!!!!
species = dimnames(S)[[2]]
A = DATA$A[,species]
B = DATA$B[,species]

# ----- OUTPUT -----
DATA$S = S
DATA$SIGNALMATRIX = SIGNALMATRIX
DATA$timepoints = dimnames(S)[[1]] %>% as.numeric()
DATA$replicates = dimnames(S)[[3]]
DATA$species = species
DATA$reactions = rownames(A)
DATA$A = A
DATA$B = B

save(DATA, file = paste0("results/graphs/",protein_name,"_v3.RData"))
