
# delta_k
delta <- c(1/3,1/3,1/3)
delta <- c(1/3,1/12,1/6,1/4,1/6)
# p_k
pi <- delta[1]
for(k in 2:length(delta)) {
  pi <- c(pi, delta[k]/prod(1 - pi))
}
# n_k
X <- matrix(ncol = 5)
N <- 1000
for(i in 1:10000) {
  n <- N
  x <- c()
  for(k in 1:length(delta)) {
    D <- rbinom(1,n,pi[k])
    x <- c(x,D)
    n <- n - D
  }
  X <- rbind(X,x)
}
X <- X[-1,]

colMeans(X) / N
delta

# plot densities
library(ggplot2)
library(dplyr)
data.frame(x = 1:5, delta.k = delta, pi.k = pi) %>%
  ggplot(aes(x = x, y = delta.k, col = "delta_k")) +
  geom_point() +
  geom_line() +
  geom_point(aes(y = pi, col = "pi_k")) +
  geom_line(aes(y = pi, col = "pi_k")) +
  ggtitle("Transitional probabilities", "Example results") +
  labs(y = "Probability", col = "Probability")
