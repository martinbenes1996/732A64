
#priors <- list(cz = c(.0025,.70),
#               it = c(.0035,.80),
#               se = c(.0030,.70),
#               pl = c(.0010,.45))

priors <- list(cz = c(.0025,.70),
               it = c(.0035,.80),
               se = c(.0030,.70),
               pl = c(.0010,.45))


#
# 0.0 - 0.1, 0.2 - 0.3

xgrid <- seq(0,0.5,length.out = 1000)
#fx <- dbeta(xgrid,0.0025 + 1801 * 1.29103e-8, 0.05 + 1801*(1 - 1.29103e-8))
fx <- dbeta(xgrid, 0.25155568403722195, 0.30983778140500917)
hist(covid_stats$I / covid_tests$T, probability = T, breaks = 30)
lines(xgrid, fx)

get.prior <- function(country) {
  params <- priors[[country]]
  return(c(params[1]*1e4, params[2]))
}

#'
#'
plot.prior <- function(country) {#prob_pos) {
  # parameters
  prior <- get.prior(country)
  alpha <- prior[1]
  beta <- prior[2]
  cat("alpha:", alpha, "\n")
  cat("beta:", beta, "\n")
  # compute beta
  xgrid <- seq(0,1,by=.00001)
  fx <- dbeta(xgrid, alpha, beta)
  # plot
  plot(xgrid, fx, type="l")
}

#plot.prior('cz')
#plot.prior('it')
#plot.prior('se')
#plot.prior('pl')

get.data <- function(samples = 1000, sample_prob = .1) {
  sample(c(0,1), size = samples, replace = T, prob = c(1 - sample_prob, sample_prob))
  #c(0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0)
}

plot.posterior <- function(country, posterior_draws = 1000, xbar = 5e-8,
                           samples = 100, sample_prob = .1) { 
  # get prior parameters
  prior <- get.prior(country)
  # get data
  data <- get.data(samples = samples, sample_prob = sample_prob)
  # random draw from posterior
  K <- length(data)
  #xbar <- sum(data) / N
  alpha <- prior[1] + K * xbar
  beta <- prior[2] + K * (1 - xbar)
  cat("alpha:", alpha, "\n")
  cat("beta:", beta, "\n")
  draw <- rbeta(posterior_draws, alpha, beta)
  # plot
  hist(draw, breaks=70, probability = T)
  lines(density(draw))
}
plot.posterior('pl', posterior_draws = 10000, xbar = 5e-7,
               samples = 500, sample_prob = .15)

