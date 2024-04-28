### kinetic-network-inference ###
# description:  reliable substrate degradation via MS1 quantification
# author:       HPR

# renv::init()
library(data.table)
library(dplyr)
library(stringr)
library(parallel)
library(Biostrings)
library(MASS)
library(wesanderson)

numCPU = 64
maxRT = 88*60
cc = wes_palette("GrandBudapest1",n=4,type = "discrete")

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


# ----- calculate MW and expected MZ of substrate -----
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
parseMGF = function(file, mz0) {
  
  print(paste0("parsing mgfs of file: ", basename(file)))
  
  # read in mgf
  mgf = readLines(file)

  # get mzs
  MZ = data.table(k = grep("PEPMASS=",mgf)) %>%
        dplyr::mutate(MZ = str_extract_all(mgf[k],"(?<=PEPMASS=)[[:digit:]|\\.]+") %>% unlist() %>% as.numeric()) %>%
        dplyr::filter(round(MZ,1) %in% round(mz0,1))

  # get corresponding MS1 intensities and RTs
  MZ = MZ %>%
    dplyr::mutate(z = str_extract(mgf[k+1],"(?<=CHARGE=)[:digit:]") %>% as.numeric(),
                  RT = str_extract(mgf[k-1],"(?<=RTINSECONDS=)[:graph:]+$") %>% as.numeric(),
                  I = str_extract(mgf[k],"(?<=[:digit:][:space:])[:graph:]+$") %>% as.numeric()) %>%
    na.omit()
  
  print(paste0(basename(file), " - scans found: ", nrow(MZ)))
  return(MZ)
}


# ----- apply to all mgf files -----
mw0 = computeMW(substrateSeq)
mz0 = computeMZ(mw0)

all_MZ = mclapply(mgf_files, function(file){
  MZ = parseMGF(file, mz0) %>%
    dplyr::mutate(raw_file_path = file)
  return(MZ)
}, mc.cores = numCPU, mc.cleanup = T, mc.preschedule = T) %>%
  rbindlist()


# ----- plot substrate XICs and substrate degradation -----
# --- substrate XICs
# filter out scans outside the RT window

pdf(paste0("results/substrate_degradation/",protein_name0,"_XICs.pdf"), height = 16, width = 16)
par(mfrow = c(4,4))

MZ_filtered = lapply(mgf_files, function(file){
  CNT = all_MZ %>%
    dplyr::filter(raw_file_path == file)
  
  smpl = sample(CNT$RT, 100000, prob = CNT$I, replace = T)
  fit = MASS::fitdistr(smpl, "normal")
  para = fit$estimate
  
  plot(x = CNT$RT, y = CNT$I, type = "b", main = basename(file),
      xlab = "RT (s)", ylab = "MS1 intensity", xlim = c(0, maxRT))
  abline(v = para[1]+para[2], lty = "dashed", col = "red")
  abline(v = para[1]-para[2], lty = "dashed", col = "red")
  
  CNT = CNT %>%
    dplyr::filter(RT >= para[1]-para[2] & RT <= para[1]+para[2])

  return(CNT)
}) %>%
  rbindlist()

dev.off()


# --- substrate degradation over time
all_I = MZ_filtered %>%
  dplyr::group_by(raw_file_path) %>%
  dplyr::summarise(Ia = sum(I)) %>%
  dplyr::left_join(sample_list)
replicates = all_I$biological_replicate %>% unique() %>% sort()
lim = max(all_I$Ia)


png(paste0("results/substrate_degradation/",protein_name0,"_degradation_filtered.png"), height = 5, width = 5, units = "in", res = 600)
ICNT = all_I %>%
  dplyr::filter(biological_replicate == replicates[1]) %>%
  dplyr::group_by(digestTime) %>%
  dplyr::summarise(mean_int = mean(Ia),
                   sd_int = sd(Ia))

plot(x = ICNT$digestTime, y = ICNT$mean_int, type = "b", lwd = 1.5, col = cc[1], #pch = 16,
    xlab = "digestion time [hrs]", ylab = "MS1 intensity",
    main = protein_name0, sub = "substrate degradation", ylim = c(0,lim))
arrows(ICNT$digestTime, ICNT$mean_int-ICNT$sd_int,
      ICNT$digestTime, ICNT$mean_int+ICNT$sd_int,
      length=0.03, angle=90, code=3, lty = "solid", lwd = 1,  col = cc[1]) %>%
      suppressWarnings()

for (ii in 2:length(replicates)) {
  ICNT = all_I %>%
    dplyr::filter(biological_replicate == replicates[ii]) %>%
    dplyr::group_by(digestTime) %>%
    dplyr::summarise(mean_int = mean(Ia),
                     sd_int = sd(Ia))

  lines(x = ICNT$digestTime, y = ICNT$mean_int, type = "b", lwd = 1.5, col = cc[ii]) #pch = 16)
  arrows(ICNT$digestTime, ICNT$mean_int-ICNT$sd_int,
        ICNT$digestTime, ICNT$mean_int+ICNT$sd_int,
        length=0.03, angle=90, code=3, lty = "solid", lwd = 1,  col = cc[ii]) %>%
        suppressWarnings()
}

legend("topright",
      legend = replicates, lty = rep("solid",length(replicates)), col = cc[1:length(replicates)],bty="n")

dev.off()


# ----- OUTPUT -----
substrateTbl = MZ_filtered %>%
  dplyr::group_by(raw_file_path) %>%
  dplyr::summarise(Ia = sum(I)) %>%
  dplyr::left_join(sample_list) %>%
  dplyr::ungroup() %>%
  dplyr::group_by(digestTime, biological_replicate) %>%
  dplyr::summarise(mean_int = mean(Ia),
                   sd_int = sd(Ia)) %>%
  dplyr::ungroup()

save(substrateTbl, file = paste0("results/substrate_degradation/",protein_name0,"_degradation.RData"))

