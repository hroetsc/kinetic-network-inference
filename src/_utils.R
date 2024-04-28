options(dplyr.summarise.inform = FALSE)
library(gtools)
library(dtplyr)

# -------------------------------------------
# ----------------- MAPPING -----------------
# -------------------------------------------

# ----- PCP location -----
locate_PCP_substrate <- function(pep, subSeq){
  
  pos = str_locate_all(subSeq,pep)
  pcp <- data.table(pepSeq = pep, pos1 = pos[[1]][,1], pos2 = pos[[1]][,2])
  
  return(pcp)
}


# ----- PSP location -----
generate_SR <- function(pepSeq){
  NMER = nchar(pepSeq)
  
  # for each peptide
  # get all possible combinations of start and end positions of SRs
  # and filter those that fit peptide's length
  s_matrix = gtools::permutations(n=NMER,r=2,v=1:NMER,repeats.allowed=T) %>%
    as_tibble() %>%
    dplyr::filter(V1+V2 == NMER)
  
  out = data.table(pepSeq=pepSeq,
                   SR1 = str_sub(pepSeq,end=s_matrix$V2),
                   SR2 = rev(str_sub(pepSeq,start=-s_matrix$V2)))
  
  return(out)
}

locate_PSP_substrate <- function(pep, subSeq) {
  
  PSPpos = generate_SR(pep) %>%
    as_tibble()
  
  # map SRs as PCP
  sr1_loc = lapply(PSPpos$SR1, function(sr1){
    locate_PCP_substrate(pep = sr1, subSeq)
  }) %>%
    data.table::rbindlist() %>%
    as_tibble() %>%
    dplyr::rename(SR1 = pepSeq) %>%
    suppressWarnings()
  
  sr2_loc = lapply(PSPpos$SR2, function(sr2){
    locate_PCP_substrate(pep = sr2, subSeq)
  }) %>%
    data.table::rbindlist() %>%
    as_tibble() %>%
    dplyr::rename(SR2 = pepSeq,
                  pos3 = pos1, pos4 = pos2) %>%
    suppressWarnings()
  
  # combine and get splice types
  POS = PSPpos %>%
    dplyr::left_join(sr1_loc) %>%
    dplyr::left_join(sr2_loc) %>%
    suppressWarnings() %>%
    suppressMessages() %>%
    na.omit()
  
  # get splice types
  POS = POS %>%
    lazy_dt() %>%
    dplyr::mutate(intervSeq = pos3-pos2,
                  type = fifelse(intervSeq>1 & pos3>pos2, "cis", "trans"),
                  type = fifelse(intervSeq<=0 & pos4<pos1, "revCis", type),
                  intervSeq = abs(intervSeq)-1,
                  SR1length = pos2-pos1+1,
                  SR2length = pos4-pos3+1) %>%
    tidyr::unite(positions, pos1, pos2, pos3, pos4) %>%
    dplyr::group_by(pepSeq) %>%
    dplyr::summarise(spliceType = paste(type, collapse = ";"),
                     positions = paste(positions, collapse = ";")) %>%
    dplyr::ungroup() %>%
    as_tibble() %>%
    suppressWarnings()
  
  return(POS)
}

locate_PSP_substrate_vec = Vectorize(locate_PSP_substrate, vectorize.args = "pep", SIMPLIFY = F, USE.NAMES = T)

# ----- actual mapping -----
locate_peps_substrate = function(peps, subSeq, numCPU) {
  
  # --- PCP mapping
  pcpMAP = mcmapply(FUN = locate_PCP_substrate,
                    pep = peps,
                    subSeq = subSeq,
                    SIMPLIFY = F,
                    mc.cores = numCPU,
                    mc.preschedule = T,
                    mc.cleanup = T)
  k = which(!sapply(pcpMAP, function(x){nrow(x) == 0}))
  
  pcpMAP = rbindlist(pcpMAP[k]) %>%
    na.omit() %>%
    as_tibble() %>%
    dplyr::mutate(spliceType = "PCP",
                  productType = "PCP") %>%
    tidyr::unite(positions, pos1, pos2)
  
  kk = which(!peps %chin% pcpMAP$pepSeq)
  
  # --- PSP mapping
  if (length(kk) > 0) {
    pspMAP = mcmapply(FUN = locate_PSP_substrate_vec,
                      pep = peps[kk],
                      subSeq = subSeq,
                      mc.cores = numCPU,
                      mc.preschedule = T,
                      mc.cleanup = T)
    k = which(!sapply(pspMAP, function(x){nrow(x) == 0}))
    
    pspMAP = rbindlist(pspMAP[k]) %>%
      na.omit() %>%
      as_tibble() %>%
      dplyr::mutate(productType = "PSP")
    
    # --- combine both
    MAP = rbindlist(list(pcpMAP, pspMAP), use.names = TRUE) %>%
      as_tibble() %>%
      dplyr::filter(!is.na(pepSeq))
    
    
  } else {
    # --- combine both
    MAP = pcpMAP %>%
      dplyr::filter(!is.na(pepSeq))
  }
  
  return(MAP)
}



# --------------------------------------------------
# ----------------- POSTPROCESSING -----------------
# --------------------------------------------------

# ----- disentangle rows containing multi-mappers -----
disentangleMultimappersType = function(DB, silent=T) {
  
  pepTable = DB %>%
    dplyr::filter(productType == "PSP") %>%
    dplyr::select(pepSeq, spliceType) %>%
    unique() %>%
    dplyr::mutate(spliceType = sapply(spliceType, function(x){
      
      cnt_types = str_split(x, coll(";"), simplify = T) %>%
        paste() %>%
        unique()
      if (length(cnt_types) == 1) {
        return(cnt_types)
      } else if (any(cnt_types == "trans")) {
        cnt_types = cnt_types[-which(cnt_types == "trans")]
        
        if (length(cnt_types) == 1) {
          return(cnt_types)
        } else {
          return("cis_multi-mapper")
        }
        
      }  else {
        return("cis_multi-mapper")
      }
      
      
    })) %>%
    as.data.frame()
  
  
  DB = dplyr::left_join(DB %>% select(-spliceType), pepTable) %>%
    dplyr::mutate(spliceType = ifelse(productType == "PCP", "PCP", spliceType))
  
  return(DB)
}


