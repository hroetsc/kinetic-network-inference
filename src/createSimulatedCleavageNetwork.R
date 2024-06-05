### kinetic-network-inference ###
# description:  create simulated network
# author:       HPR

library(data.table)
library(dplyr)
library(stringr)
library(parallel)
numCPU = 4

n = 100

# ----- INPUT -----
substrateSeq = "INFERENCEEVERYDAYANDNIGHT"
substrateSeq_split = strsplit(substrateSeq, "") %>% unlist()
L = nchar(substrateSeq)

# ----- generate peptide sequences -----
allPCP = mclapply(seq(1, L-1), function(N){
   
    allPCP = sapply(seq(1,L-N+1), function(i){
        j = N+i-1
        positions = paste(i,j, sep = "_")
        pepSeq=str_sub(substrateSeq,i,j)
        return(c(pepSeq,positions,i,j))
    })

    PCP = unlist(allPCP)
    PCP_Pep = data.table(pepSeq = PCP[1,],
                            positions = PCP[2,],
                            P1 = PCP[3,] %>% as.numeric(),
                            P1_ = PCP[4,] %>% as.numeric())

  }, 
mc.preschedule = T,
mc.cleanup = T,
mc.cores = numCPU)

allPCP <- data.table::rbindlist(allPCP) %>%
    dplyr::mutate(N = nchar(pepSeq))

set.seed(893)
uniquePeps = allPCP$pepSeq %>% unique()
paste0("number of unique peptides: ", length(uniquePeps)) %>% print()
k = sample(1:length(uniquePeps), n)
chosenPCP = allPCP %>%
    dplyr::filter(pepSeq %in% uniquePeps[k])


# ----- build graph -----
# FIXME: nodes should be PEPTIDES! edges can account for all positions!
# otherwise simulation will be crap

pcp = allPCP %>%
    dplyr::group_by(pepSeq, N) %>%
    dplyr::reframe(positions = paste(positions, collapse = ";"),
                    P1 = paste(P1, collapse = ";"),
                    P1_ = paste(P1_, collapse = ";")) %>%
    dplyr::arrange(N) %>%
    dplyr::filter(N > 1)

# FIXME: peptide level!
COORD_pcp = mclapply(1:nrow(pcp), function(i){
    
    cnt = pcp[i,] %>%
        tidyr::separate_rows(positions, P1, P1_, sep = ";")

    en = cnt$P1 %>% as.numeric()
    st = cnt$P1_ %>% as.numeric()
    N = pcp$N[i]
    
    reactant = cnt$positions
    reactantSeq = pcp$pepSeq[i]
    reactant_detected = any(reactant %in% chosenPCP$positions)

    cleave = seq(1,N-1)
    
    # get products resulting from cleavage of current PCP
    PCP = lapply(1:length(st), function(kk){
        PCP = lapply(cleave, function(c) {
      
            abs_site = st[kk]+c-1
            k1 = which(allPCP$P1_ == st[kk] & allPCP$P1 == abs_site)
            k2 = which(allPCP$P1_ == abs_site+1 & allPCP$P1 == en[kk])
            
            product_detected = sapply(c(k1,k2), function(ki){
                allPCP$pepSeq[ki] %in% chosenPCP$pepSeq
            })

            # reactant --> product
            PCP = data.table(cleavage_site_abs = abs_site, cleavage_site = c,
                            reactant_1 = reactant, reactant_2 = NA,
                            product_1 = pcp$positions[k1], product_2 = pcp$positions[k2],
                            reactant_1_detected = reactant_detected, reactant_2_detected = NA,
                            product_1_detected = product_detected[1], product_2_detected = product_detected[2])
            
            return(PCP)
        }) %>%
        rbindlist()

    }) %>%
        rbindlist()
    
    return(PCP)
  }, 
  mc.preschedule = T,
  mc.cleanup = T,
  mc.cores = numCPU)

  

