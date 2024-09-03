### kinetic-network-inference ###
# description:  utility functions for graph construction
# author:       HPR

substr_vec = Vectorize(substr, vectorize.args = c("start","stop"))
substr_vec_all = Vectorize(substr, vectorize.args = c("x","start","stop"))

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
    dplyr::mutate(N = P1-P1_+1,
                  pepSeq = substr_vec(S, P1_, P1))
  
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
  allpcp = getAllPCPs(L,Nmin,Nmax,S) %>%
    dplyr::mutate(validLength = N %in% seq(Nmin, Nmax))
  pos = getPositions(DB)
  
  # ----- PCP graph -----
  pcp = allpcp %>%
    as_tibble() %>%
    dplyr::mutate(validLength = N >= Nmin & N <= Nmax,
                  detected = pepSeq %in% pos$pcp$pepSeq) %>%
    dplyr::group_by(pepSeq, N, validLength, detected) %>%
    dplyr::reframe(positions = paste(P1.P1_, collapse = ";"),
                    P1 = paste(P1, collapse = ";"),
                    P1_ = paste(P1_, collapse = ";")) %>%
    dplyr::arrange(N) %>%
    dplyr::filter(N > 1)
  
  table(pcp$validLength, pcp$detected)
  
  # search for products that could result from cleavage of parental (long) PCPs
  COORD_pcp = mclapply(1:nrow(pcp), function(i){
    
    cnt = pcp[i,] %>%
        tidyr::separate_rows(positions, P1, P1_, sep = ";")

    en = cnt$P1 %>% as.numeric()
    st = cnt$P1_ %>% as.numeric()
    N = pcp$N[i]
    
    reactant = cnt$positions
    reactantSeq = pcp$pepSeq[i]
    reactant_detected = pcp$detected[i]
    reactant_valid = pcp$validLength[i]
    
    # get all possible cleavage sites
    cleave = seq(1,N-1)

    # get products resulting from cleavage of current PCP
    # iterate all coordinates of current reactant sequence
    PCP = lapply(1:length(st), function(ii) {

        PCP = lapply(cleave, function(c) {

            abs_site = st[ii]+c-1
            k1 = which(allpcp$P1_ == st[ii] & allpcp$P1 == abs_site)
            k2 = which(allpcp$P1_ == abs_site+1 & allpcp$P1 == en[ii])
            
            product_detected = sapply(c(k1,k2), function(ki){
                allpcp$pepSeq[ki] %in% pos$pcp$pepSeq
            })
            product_valid = allpcp$validLength[c(k1,k2)]

            # reactant --> product
            PCP = data.table(reactant_1 = reactant[ii], reactant_2 = NA,
                            product_1 = allpcp$P1.P1_[k1], product_2 = allpcp$P1.P1_[k2],
                            reactant_1_seq = reactantSeq, reactant_2_seq = NA,
                            product_1_seq = allpcp$pepSeq[k1], product_2_seq = allpcp$pepSeq[k2],
                            reactant_1_detected = reactant_detected, reactant_2_detected = NA,
                            product_1_detected = product_detected[1], product_2_detected = product_detected[2],
                            reactant_1_valid = reactant_valid, reactant_2_valid = NA,
                            product_1_valid = product_valid[1], product_2_valid = product_valid[2],
                            cleavage_site_abs = abs_site, cleavage_site = c)
            
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

  allPCP = COORD_pcp %>%
    data.table::rbindlist() %>%
    as_tibble()


  # ----- PSP graph -----
  allpsp = pos$psp %>%
    dplyr::mutate(N = nchar(pepSeq))
  psp = allpsp %>%
    dplyr::mutate(SR1 = paste0(Nterm,"_",P1),
                  SR2 = paste0(P1_,"_",Cterm)) %>%
    as_tibble() %>%
    dplyr::group_by(pepSeq, N) %>%
    dplyr::reframe(positions = paste(positions, collapse = ";"),
                    P1 = paste(P1, collapse = ";"),
                    P1_ = paste(P1_, collapse = ";"),
                    SR1 = paste(SR1, collapse = ";"),
                    SR2 = paste(SR2, collapse = ";")) %>%
    unique()
  

  COORD_psp = mclapply(1:nrow(psp), function(i){
    
    productSeq = psp$pepSeq[i]
    cnt = psp[i,] %>%
        tidyr::separate_rows(positions, P1, P1_, SR1, SR2, sep = ";")

    # PSP generated from PCP/substrate through splice
    # iterate all positions of current psp
    PSP = lapply(1:nrow(cnt), function(ii) {

      ki = which(allpcp$P1.P1_ == cnt$SR1[ii])
      kj = which(allpcp$P1.P1_ == cnt$SR2[ii])
      
      reactant_detected = sapply(c(ki,kj), function(kii){
                allpcp$pepSeq[kii] %in% pos$pcp$pepSeq
      })
      reactant_valid = allpcp$validLength[c(ki,kj)]

      # add to results
      PSP = data.table(reactant_1 = allpcp$P1.P1_[ki], reactant_2 = allpcp$P1.P1_[kj],
                            product_1 = cnt$positions[ii], product_2 = NA,
                            reactant_1_seq = allpcp$pepSeq[ki], reactant_2_seq = allpcp$pepSeq[kj],
                            product_1_seq = productSeq, product_2_seq = NA,
                            reactant_1_detected = reactant_detected[1], reactant_2_detected = reactant_detected[2],
                            product_1_detected = TRUE, product_2_detected = NA,
                            reactant_1_valid = reactant_valid[1], reactant_2_valid = reactant_valid[2],
                            product_1_valid = TRUE, product_2_valid = NA)

      return(PSP)
    }) %>%
      rbindlist()
    
    return(PSP)
  }, 
  mc.preschedule = T,
  mc.cleanup = T,
  mc.cores = numCPU)
  
  allPSP = COORD_psp %>%
    data.table::rbindlist() %>%
    as_tibble()
  
  # filter out not detected PCPs that are not used for splicing
  paste0("all hydrolysis reactions: ", nrow(allPCP)) %>% print()
  allPCP = allPCP %>%
    dplyr::mutate(product_1_sr = product_1 %in% allPSP$reactant_1,
                  product_2_sr = product_2 %in% allPSP$reactant_2) %>%
    dplyr::filter((product_1_detected | product_2_detected) | (product_1_sr | product_2_sr)) # NOTE: this also includes 1-hop distance nodes!
  paste0("filtered hydrolysis reactions: ", nrow(allPCP)) %>% print()
  

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

  print(paste0("total number of reactions: ",nrow(ALL)))
  
  # ensure that all products are connected to substrate node
  # adjacency list
  adjList = adjacency_check(ALL, subnode)

  return(ALL)
}


