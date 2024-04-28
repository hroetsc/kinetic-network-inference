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
reactions = read.csv("data/simulation/Ex2.csv", stringsAsFactors = F)

k = reactions$rate

# time points
tpoints = seq(0,4,0.01)
tpoints_coarse = seq(0,4,1)


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
rownames(A) = reactions$rate_name

colnames(B) = species
rownames(B) = reactions$rate_name


# ----- simulate ODE -----
massAction <- function(t,X,p) {
  
  # rates as diagonal matrix
  K = diag(k,r,r)
  # vector matrix exponentiation of x by A
  M = apply(t(X^t(A)), 1, prod)
  # calculate dX
  dX = t(B-A)%*%(k*M)
  
  list(dX)
}

names(x0) = species
names(k) = reactions$rate_name

out <- ode(func = massAction,
            y = x0,
            parms = k,
            times = tpoints)



# ----- plot -----
cols = rainbow(n = s)

plot_out = out %>%
  as_tibble() %>%
  tidyr::gather(variable,value,-time) %>%
  ggplot(aes(x=time,y=value,color=variable))+
  geom_line(linewidth=1.5)+
  scale_color_manual(values = cols, "component") +
  xlim(c(0,4)) +
  labs(x='time (h)',y='concentration') + 
  ggtitle("simulated kinetics")
plot_out

ggsave("data/simulation/Ex2.png",
       plot = plot_out, height = 4, width = 5, dpi = "retina")


# ----- OUTPUT -----
X = out %>% as.matrix()
X = X[X[,1] %in% tpoints_coarse, species]

DATA = list(X = X[,species],
            x0 = X[1,species],
            A = A[reactions$rate_name,species],
            B = B[reactions$rate_name,species],
            tp = tpoints_coarse,
            species = species,
            reactions = reactions$rate_name,
            info = reactions)

save(DATA, file = "data/simulation/Ex2_DATA.RData")


