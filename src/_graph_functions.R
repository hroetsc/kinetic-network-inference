### kinetic-network-inference ###
# description:  utility functions for graph construction
# author:       HPR


# ----- get all possible non-spliced peptides -----
getAllPCPs = function(L, Nmin, Nmax, S) {
  
  P1 = c(1:L)
  P1_ = c(1:L)
  
  # all possible P1-P1' combinations
  P1.P1_ = tidyr::crossing(P1,P1_) %>%
    as_tibble() %>%
    dplyr::mutate(P1.P1_ = paste(P1_,P1, sep = "_"))
  
  # non-spliced peptides: valid combinations of P1 (C-term) and P1' (N-term)
  pcp = P1.P1_ %>%
    dplyr::filter((P1-P1_+1) %in% seq(1, Nmax)) # allow PCPs from as short as 1 aa since they could be SRs
  
  # add the substrate as node
  sub = data.frame(P1 = L, P1_ = 1) %>%
    dplyr::mutate(P1.P1_ = paste(P1_,P1, sep = "_"))  # add +1 to coordinates to account for padded sequence)
  
  pcpall = rbindlist(list(pcp, sub), use.names = T) %>%
    as_tibble() %>%
    unique() %>%
    dplyr::mutate(N = P1-P1_+1)
  
  return(pcpall)
}


# ----- get positions of detected products -----
getPositions = function(DB) {
  
  substrateSeq = DB$substrateSeq[1]
  substrateSeq_pad = paste0("XX",substrateSeq,"XX")
  
  # --- PCPs
  pcp = DB %>%
    dplyr::filter(productType == "PCP") %>%
    dplyr::mutate(P1 = str_split_fixed(positions, "_", Inf)[,2] %>% as.numeric(),
                  P1_ = str_split_fixed(positions, "_", Inf)[,1] %>% as.numeric(),
                  P1.P1_ = paste(P1_,P1, sep = "_"),
                  N = P1-P1_+1) %>%
    unique()
  # add substrate
  pcp = list(pcp,
             data.table(substrateID = DB$substrateID[1],
                        substrateSeq = substrateSeq,
                        productType = "PCP",
                        pepSeq = substrateSeq,
                        positions = paste0("1_",nchar(substrateSeq)),
                        P1 = nchar(substrateSeq),
                        P1_ = 1,
                        P1.P1_ = paste0("1_",nchar(substrateSeq)),
                        N = nchar(substrateSeq))) %>%
    rbindlist()
  
  # --- PSPs
  psp = DB %>%
    dplyr::filter(productType == "PSP") %>%
    dplyr::mutate(Nterm = str_split_fixed(positions, "_", Inf)[,1] %>% as.numeric(),
                  P1 = str_split_fixed(positions, "_", Inf)[,2] %>% as.numeric(),
                  P1_ = str_split_fixed(positions, "_", Inf)[,3] %>% as.numeric(),
                  Cterm = str_split_fixed(positions, "_", Inf)[,4] %>% as.numeric()) %>%
    dplyr::mutate(P1.P1_ = paste(P1,P1_, sep = "_"),
                  N_sr1 = P1-Nterm+1,
                  N_sr2 = Cterm-P1_+1) %>%
    unique()
  
  
  return(list(pcp = pcp, psp = psp))
}


# ----- get PCP clevage templates -----
cleavageTemplate = function(Nmin, Nmax, L) {
  
  # 'normal'
  cleave = lapply((Nmin+1):L, function(N) {
    seq(1,N-1)
  })
  names(cleave) = (Nmin+1):L
  
  return(cleave)
}


# ----- verify that graph is fully connected -----
adjacency_check = function(ALL, subnode) {

  detectedReactants = c(ALL$reactant_1[which(ALL$reactant_1_detected)],  ALL$reactant_2[which(ALL$reactant_2_detected)]) %>% unique()
  detectedProducts = c(ALL$product_1[which(ALL$product_1_detected)],  ALL$product_2[which(ALL$product_2_detected)]) %>% unique()
  species = c(ALL$reactant_1, ALL$reactant_2, ALL$product_1, ALL$product_2) %>% na.omit() %>% unique()

  paste0("found all species in adjacency list: ", all(pepTbl$positions %in% species)) %>% print()

  # adjacency list
  adjList = ALL %>%
    dplyr::select(reactant_1, reactant_2, product_1, product_2, reaction_ID) %>%
    tidyr::gather(reactant_cat, reactant, -product_1, -product_2, -reaction_ID) %>% 
    tidyr::gather(product_cat, product, -reactant_cat, -reactant, -reaction_ID) %>%
    dplyr::select(-reactant_cat, -product_cat) %>%
    na.omit() %>%
    dplyr::mutate(reactantType = fifelse(str_detect(reactant,"^[:digit:]+_[:digit:]+$"), "PCP", "PSP"),
                  reactantType = fifelse(reactant == paste0("1_",L), "substrate", reactantType),
                  productType = fifelse(str_detect(product,"^[:digit:]+_[:digit:]+$"), "PCP", "PSP"),
                  productType = fifelse(product == paste0("1_",L), "substrate", productType),
                  reactantDetected = reactant %in% detectedReactants,
                  productDetected = product %in% detectedProducts)

  # adjacency matrix
  ADJ = matrix(0, nrow = length(species),ncol = length(species))
  colnames(ADJ) = species
  rownames(ADJ) = species
  
  x = match(adjList$reactant, rownames(ADJ))
  y = match(adjList$product, colnames(ADJ))
  ADJ[cbind(x,y)] = 1

  k = which(rownames(ADJ) == subnode)
  ADJ_c = ADJ
  hits = c(species[which(ADJ_c[k,] > 0)],subnode)
  counter = 1

  while(counter <= length(species)) {
    # print(counter)

    ADJ_c = ADJ_c%*%ADJ
    hits = c(hits, species[which(ADJ_c[k,] > 0)]) %>% unique()

    if (all(species %in% hits)) {
      break
    }
    counter = counter + 1
  }

  if (!all(species %in% hits)) {
    print("failed to connect all nodes !!!")
    print(paste0("missing: ", species[!species %in% hits]))
    
  } else {
    print(paste0("took ", counter, " hops to connect all nodes with the substrate node"))
  }

  return(adjList)
}


# ----- CONSTRUCT GRAPH -----
constructGraphNetwork <- function(DB, numCPU, Nmin, Nmax) {
  
  print("CONSTRUCTING NETWORK GRAPH")
  
  S = DB$substrateSeq[1]
  L = nchar(S)
  subID = DB$substrateID[1]
  subnode = paste0("1_",L)

  # get positions
  allpcp = getAllPCPs(L,Nmin,Nmax,S)
  pos = getPositions(DB)
  
  # ----- PCP graph -----
  pcp = allpcp %>%
    as_tibble() %>%
    dplyr::mutate(validLength = N >= Nmin & N <= Nmax,
                  detected = P1.P1_ %in% pos$pcp$P1.P1_)
  
  # ----- cleave the substrate
  cleavage_template = cleavageTemplate(Nmin,Nmax,L)
  st = 1
  en = L
  reactant = subnode
  reactant_detected = TRUE
  HOP1 = lapply(cleavage_template[[as.character(L)]], function(c) {
      
      abs_site = st+c-1
      k1 = which(pcp$P1_ == st & pcp$P1 == abs_site)
      k2 = which(pcp$P1_ == abs_site+1 & pcp$P1 == en)
      
      product_detected = sapply(c(k1,k2), function(j) {
        pcp$detected[j]
      }) %>% as.vector()
      
      # reactant --> product
      PCP = data.table(cleavage_site_abs = abs_site, cleavage_site = c,
                       reactant_1 = reactant, reactant_2 = NA,
                       reactant_1_valid = TRUE, reactant_2_valid = NA,
                       product_1_valid = pcp$validLength[k1], product_2_valid = pcp$validLength[k2],
                       reactant_1_detected = reactant_detected, reactant_2_detected = NA,
                       product_1 = pcp$P1.P1_[k1], product_2 = pcp$P1.P1_[k2],
                       product_1_detected = product_detected[1], product_2_detected = product_detected[2])
      return(PCP)
    }) %>%
      rbindlist()

  # ----- only direkt cleavage products of the substrate are allowed as precursors
  longpcp = pcp %>%
    dplyr::filter(P1.P1_ %in% c(subnode, HOP1$product_1, HOP1$product_2)) %>%
    dplyr::filter(N >= (Nmin+1) & (N <= (2*Nmax) | N == L)) %>%
    dplyr::arrange(N)  # important!!!

  # search for products that could result from cleavage of parental (long) PCPs
  COORD_pcp = mclapply(1:nrow(longpcp), function(k){
    
    en = longpcp$P1[k]
    st = longpcp$P1_[k]
    N = longpcp$N[k]
    
    reactant = longpcp$P1.P1_[k]
    reactant_detected = longpcp$detected[k]
    
    cleave = cleavage_template[[as.character(N)]]
    
    # get products resulting from cleavage of current PCP
    PCP = lapply(cleave, function(c) {
      
      abs_site = st+c-1
      k1 = which(pcp$P1_ == st & pcp$P1 == abs_site)
      k2 = which(pcp$P1_ == abs_site+1 & pcp$P1 == en)
      
      product_detected = sapply(c(k1,k2), function(j) {
        pcp$detected[j]
      }) %>% as.vector()
      
      # reactant --> product
      PCP = data.table(cleavage_site_abs = abs_site, cleavage_site = c,
                       reactant_1 = reactant, reactant_2 = NA,
                       reactant_1_valid = TRUE, reactant_2_valid = NA,
                       product_1_valid = pcp$validLength[k1], product_2_valid = pcp$validLength[k2],
                       reactant_1_detected = reactant_detected, reactant_2_detected = NA,
                       product_1 = pcp$P1.P1_[k1], product_2 = pcp$P1.P1_[k2],
                       product_1_detected = product_detected[1], product_2_detected = product_detected[2])
      
      return(PCP)
    }) %>%
      rbindlist()
    
    return(PCP)
  }, 
  mc.preschedule = T,
  mc.cleanup = T,
  mc.cores = numCPU)
  
  allPCP = COORD_pcp %>%
    data.table::rbindlist() %>%
    as_tibble()
  
  
  # ----- PSP graph -----
  psp = pos$psp %>%
    dplyr::mutate(SR1 = paste0(Nterm,"_",P1),
                  SR2 = paste0(P1_,"_",Cterm)) %>%
    as_tibble() %>%
    unique()
  
  COORD_psp = mclapply(1:nrow(psp), function(k){
    
    product = psp$positions[k]
    
    # PSP generated from PCP/substrate through splice
    ki = which(pcp$P1.P1_ == psp$SR1[k])
    kj = which(pcp$P1.P1_ == psp$SR2[k])
    
    # add to results
    PSP = data.table(reactant_1 = pcp$P1.P1_[ki], reactant_2 = pcp$P1.P1_[kj],
                     reactant_1_valid = pcp$validLength[ki], reactant_2_valid = pcp$validLength[kj],
                     product_1_valid = TRUE, product_2_valid = NA,
                     reactant_1_detected = pcp$detected[ki], reactant_2_detected = pcp$detected[kj],
                     product_1 = product, product_2 = NA,
                     product_1_detected = TRUE, product_2_detected = NA)
    
    return(PSP)
  }, 
  mc.preschedule = T,
  mc.cleanup = T,
  mc.cores = numCPU)
  
  allPSP = COORD_psp %>%
    data.table::rbindlist() %>%
    as_tibble()
  
  
  # ----- build adjacency list -----
  ALL = rbindlist(
    list(
      allPCP %>% dplyr::mutate(productType = "PCP"),
      allPSP %>% dplyr::mutate(productType = "PSP")
    ),
    use.names = T,
    fill = T
  ) %>%
    dplyr::mutate(substrateID = subID, .before = 1,
                  reaction_ID = 1:n()) %>%
    as.data.table()

  # ensure that all products are connected to substrate node
  # adjacency list
  adjList = adjacency_check(ALL, subnode)

  # # ----- filter PCPs -----
  # hop1_species = c(HOP1$reactant_1, HOP1$product_1, HOP1$product_2) %>% unique()
  
  # # filter PCPs
  # ALL1 = ALL %>%
  #   dplyr::filter(!(productType == "PCP" & !reactant_1_detected & !reactant_1 %in% hop1_species & !product_1 %in% hop1_species & !product_2 %in% hop1_species)) %>% # precursor not detected
  #   dplyr::filter(!(productType == "PCP" & !product_1_detected & !product_2_detected & !reactant_1 %in% hop1_species & !product_1 %in% hop1_species & !product_2 %in% hop1_species)) # precursor detected, but none of the cleavage products
  # # sanity check
  # adjList1 = adjacency_check(ALL1)

  # # remove redundant PCP reactions
  # toRemove = c()
  # for (pcpid in pcp$P1.P1_) {
    
  #   # check in which reactions the current PCP occurs as a product
  #   prdct = ALL1 %>%
  #     dplyr::filter(product_1 == pcpid | product_2 == pcpid)
  #   # # check in which reactions the current PCP occurs as a reactant
  #   # rctnt = ALL1 %>%
  #   #   dplyr::filter(reactant_1 == pcpid | reactant_2 == pcpid)
    
  #   if (nrow(prdct) > 0) {
      
  #     # are both products detected in any of the reactions?
  #     # any of the PCPs used for splicing ?

  #     kk = sapply(1:nrow(prdct), function(i){

  #       # both products observed?
  #       k1 = prdct$product_1_detected[i] & prdct$product_2_detected[i]
  #       # none of the reactants/products are in 1-hop distance
  #       k2 = !prdct$hop1[i]
  #       # used as reactant for splicing
  #       k3 = ALL1 %>%
  #         dplyr::filter(productType == "PSP" & 
  #                         (reactant_1 %in% c(prdct$product_1[i], prdct$product_2[i]) | reactant_2 %in% c(prdct$product_1[i], prdct$product_2[i]))) %>%
  #         nrow() == 0
  #       # used as reactant for cleavage
  #       k4 = ALL1 %>%
  #         dplyr::filter(productType == "PCP" & 
  #                         (reactant_1 %in% c(prdct$product_1[i],prdct$product_2[i]))) %>%
  #         nrow() == 0

  #       (k1|k3|k4) & k2
  #     })
      
  #     k = which(kk)
  #     if (length(k) > 0) {
  #       toRemove = c(toRemove, prdct$reaction_ID[-k])
  #       print("beepop")
  #     }

  #   }
  # }

  # ALL2 = ALL1 %>%
  #   dplyr::filter(!reaction_ID %in% toRemove)
  # # sanity check for connectivity
  # adjList2 = adjacency_check(ALL2)

  return(ALL)
}


