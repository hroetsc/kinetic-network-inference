### kinetic-network-inference ###
# description:  reliable substrate degradation via MS1 quantification
# author:       HPR

# renv::init()
library(data.table)
library(dplyr)
library(stringr)
library(parallel)
library(Biostrings)
numCPU = 16

protein_name0 = "IDH1_WT"
mgf_dir = "/data/Hanna/POLYPEPTIDES/MGF/IDH1_WT"
dir.create("results/substrate_degradation/", showWarnings = F, recursive = T)


# ----- INPUT -----
load(paste0("results/graphs/",protein_name0,"_v2.RData"))
finalK = DATA$finalK
substrateSeq = finalK$substrateSeq[1]

sample_list = fread("data/sample_list_aSPIRE_glioma.csv")

# ----- preprocessing -----
mgf_files = list.files(mgf_dir, pattern = ".mgf", full.names = T, recursive = T)

sample_list = sample_list %>%
  dplyr::filter(protein_name == protein_name0) %>%
  dplyr::mutate(raw_file = str_replace(raw_file, "\\.raw", "\\.mgf")) %>%
  dplyr::right_join(data.frame(raw_file = basename(mgf_files),
                               raw_file_path = mgf_files))


# ----- calculate MW of substrate -----
monoisotopic_masses <- data.frame(
  AA = c("A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"),
  mass = c(71.037114,103.009185,115.026943,129.042593,147.068414,57.021464,137.058912,113.084064,128.094963,113.084064,131.040485,114.042927,97.052764,128.058578,156.101111,87.032028,101.047679,99.068414,186.079313,163.06332)
)
computeMW <- function(seq, AAs = monoisotopic_masses$AA, masses = monoisotopic_masses$mass){
  seq = AAStringSet(seq)
  
  if(length(seq) < 1){
    MW = NA
  }
  if(length(seq) > 0){
    aa3 = letterFrequency(seq, letters = AAs) %*% diag(masses)
    MW=rowSums(aa3)+18.01056
    MW[MW==18.01056] = NA
  }
  return(MW)
}

computeMZ = function(MW, charges = seq(1,6)) {
  mz = sapply(charges, function(z){
    (MW+z*1.007825)/z
  })
  return(mz)
}

computeMWfromMZ = function(mz, z) {
  return(z*mz-z*1.007825)
}

mass_combos = tidyr::crossing(monoisotopic_masses$AA, monoisotopic_masses$AA)



# ----- parse mgf file -----
# tmp!
file = mgf_files[1]


parseMGF = function(file, mz) {
  
  print(paste0("parsing mgfs of file: ", basename(file)))
  
  # read in mgf
  mgf = readLines(file)

  # get mzs
  MZ = data.table(k = grep("PEPMASS=",mgf)) %>%
        dplyr::mutate(MZ = str_extract_all(mgf[k],"(?<=PEPMASS=)[[:digit:]|\\.]+") %>% unlist() %>% as.numeric()) %>%
        dplyr::filter(round(MZ,2) %in% round(mz0,2))

  # get corresponding MS1 intensities and RTs
  MZ = MZ %>%
    dplyr::mutate(z = str_extract(mgf[k+1],"(?<=CHARGE=)[:digit:]") %>% as.numeric(),
                  RT = str_extract(mgf[k-1],"(?<=RTINSECONDS=)[:graph:]+$") %>% as.numeric(),
                  I = str_extract(mgf[k],"(?<=[:digit:][:space:])[:graph:]+$") %>% as.numeric())

  # filter out scans outside the RT window
  zs = MZ$z %>% unique() %>% sort()
  
  MZ_outliers = mclapply(zs, function(zi){
    CNT = MZ %>%
      dplyr::filter(z == zi) %>%
      dplyr::mutate(zscore_rt = (RT-mean(RT))/sd(RT),
                    zscore_int = (I-mean(I))/sd(I))
    return(CNT)
  }, mc.cores = numCPU, mc.preschedule = T, mc.cleanup = T) %>%
    rbindlist()
  
  
  
  # png("tmp.png")
  # plot(x = MZ$RT, y = MZ$I)
  # dev.off()
}

# ----- apply to all mgf files -----
mw0 = computeMW(substrateSeq)
mz0 = computeMZ(mw0)

