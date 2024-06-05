### kinetic-network-inference ###
# description:  simulate in silico dataset
# author:       HPR

library(dplyr)
library(stringr)
library(deSolve)
library(ggplot2)
theme_set(theme_classic())

dir.create("data/simulation/", showWarnings = F, recursive = T)

# ----- INPUT -----
reactions = read.csv("data/simulation/Ex3.csv", stringsAsFactors = F)
reactions = reactions %>%
    dplyr::mutate(rate_name_ma = paste0("on_",reaction_ID),
                  rate_name_vmax = paste0("Vmax_",reaction_ID),
                  rate_name_km = paste0("Km_",reaction_ID))

# time points
tpoints = seq(0,4,0.01)
tpoints_coarse = seq(0,4,1)

set.seed(42)
# rate_ma = reactions$rate_ma/5
rate_ma = 0.25/reactions$rate_ma
# rate_ma = runif(nrow(reactions), 0, 1)
rate_vmax = runif(nrow(reactions), 0, 20)
rate_km = runif(nrow(reactions), 0, 0.5)

reactions = reactions %>%
  dplyr::mutate(rate_ma = rate_ma,
                rate_vmax = rate_vmax,
                rate_km = rate_km)

# ----- build stochiometry matrix from adjacency list -----
species = c(reactions$reactant1, reactions$reactant1, reactions$product1, reactions$product2) %>%
  unique() %>%
  sort()
species = species[-which(species %in% c(NA,""))]

# initial state
x0 = c(rep(0,length(species)-1), 100)

r = nrow(reactions)
s = length(species)

# reactant and product matrices (to get stochiometry)
A = matrix(0, nrow = r, ncol = s)
B = matrix(0, nrow = r, ncol = s)

for (r in 1:nrow(reactions)) {
  A[r,which(species %in% c(reactions$reactant1[r],reactions$reactant2[r]))] <- 1
}
for (r in 1:nrow(reactions)) {
  B[r,which(species %in% c(reactions$product1[r],reactions$product2[r]))] <- 1
}

colnames(A) = species
rownames(A) = reactions$reaction_ID

colnames(B) = species
rownames(B) = reactions$reaction_ID


# ----- simulate ODE -----
massAction <- function(t,X,p) {
  
  # vector matrix exponentiation of x by A
  M = apply(t(X^t(A)), 1, prod)
  # calculate dX
  dX = t(B-A)%*%(p*M)
  
  list(dX)
}

michaelisMenten <- function(t,X,p) {

  # vector matrix exponentiation of x by A
  m = apply(t(X^t(A)), 1, prod)

  Vmax = p[1:r]
  Km = p[(r+1):(2*r)]
  # calculate dX
  dX = t(B-A)%*%((Vmax*m) / (Km+m))
  
  list(dX)
}


names(x0) = species

names(rate_ma) = reactions$rate_name_ma
names(rate_vmax) = reactions$rate_name_vmax
names(rate_km) = reactions$rate_name_km


out_MA <- ode(func = massAction,
              y = x0,
              parms = rate_ma,
              times = tpoints)

out_MM <- ode(func = michaelisMenten,
              y = x0,
              parms = c(rate_vmax,rate_km),
              times = tpoints)

# ----- plot -----
cols = rainbow(n = s)

plot_ma = out_MA %>%
  as_tibble() %>%
  tidyr::gather(variable,value,-time) %>%
  ggplot(aes(x=time,y=value,color=variable))+
  geom_line(linewidth=1.5)+
  scale_color_manual(values = cols, "component") +
  xlim(c(0,4)) +
  labs(x='time (h)',y='concentration') + 
  ggtitle("simulated kinetics - mass action") +
  theme(legend.position = "none")
plot_ma

ggsave("data/simulation/Ex3_MA.png",
       plot = plot_ma, height = 4, width = 5, dpi = "retina")


plot_mm = out_MM %>%
  as_tibble() %>%
  tidyr::gather(variable,value,-time) %>%
  ggplot(aes(x=time,y=value,color=variable))+
  geom_line(linewidth=1.5)+
  scale_color_manual(values = cols, "component") +
  xlim(c(0,4)) +
  labs(x='time (h)',y='concentration') + 
  ggtitle("simulated kinetics - michaelis menten") +
  theme(legend.position = "none")
plot_mm

ggsave("data/simulation/Ex3_MM.png",
       plot = plot_ma, height = 4, width = 5, dpi = "retina")


# ----- OUTPUT -----
X_MA = out_MA %>% as.matrix()
X_MA = X_MA[X_MA[,1] %in% tpoints_coarse, species]
DATA_MA = list(X = X_MA[,species],
            x0 = X_MA[1,species],
            A = A[reactions$reaction_ID,species],
            B = B[reactions$reaction_ID,species],
            tp = tpoints_coarse,
            species = species,
            reactions = reactions$rate_name,
            info = reactions)
save(DATA_MA, file = "data/simulation/Ex3_MA_DATA.RData")


X_MM = out_MM %>% as.matrix()
X_MM = X_MM[X_MM[,1] %in% tpoints_coarse, species]
DATA_MM = list(X = X_MM[,species],
            x0 = X_MM[1,species],
            A = A[reactions$reaction_ID,species],
            B = B[reactions$reaction_ID,species],
            tp = tpoints_coarse,
            species = species,
            reactions = reactions$rate_name,
            info = reactions)
save(DATA_MM, file = "data/simulation/Ex3_MM_DATA.RData")
