
priors <- list(cz = c(.0025, 1.5),
               it = c(.0035, 1.1),
               se = c(.003, 1.3),
               pl = c(.001, 3))

get.prior <- function(country) {
  params <- priors[[country]]
  tau <- params[1]
  alpha <- params[2]
  return(c(alpha, alpha/tau))
}

#'
#'
plot.prior <- function(country) {#prob_pos) {
  # parameters
  #alpha <- (tau+1)/(sigma2*tau^2)+tau/(tau+1)
  prior <- get.prior(country)
  alpha <- prior[1]
  beta <- prior[2]
  #alpha <- prob_pos
  #beta <- alpha / test_ratio
  cat("alpha:", alpha, "\n")
  cat("beta:", beta, "\n")
  # compute beta
  xgrid <- seq(0,.01,by=.00001)
  fx <- dbeta(xgrid, alpha, beta)
  # plot
  plot(xgrid, fx, type="l")
}

plot.prior('cz')
plot.prior('it')
plot.prior('se')
plot.prior('pl')

get.data <- function(samples = 1000, sample_prob = .1) {
  sample(c(0,1), size = samples, replace = T, prob = c(1 - sample_prob, sample_prob))
  #c(0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0)
}

plot.posterior <- function(country, posterior_draws = 1000, samples = 100, sample_prob = .1) { 
  # get prior parameters
  prior <- get.prior(country)
  # get data
  data <- get.data(samples)
  # random draw from posterior
  N <- length(data)
  xbar <- mean(data)
  draw <- rbeta(posterior_draws, prior[1] + N * xbar, prior[2] + N - N * xbar)
  # plot
  hist(draw, breaks=70, probability = T)
  lines(density(draw))
}

# cz = c(.0025, 1.5),
# it = c(.0035, 2),
# se = c(.003, 1.9),
# pl = c(.001, 1.2))
plot.posterior('it', posterior_draws = 100000, samples = 142419, sample_prob = .1)

