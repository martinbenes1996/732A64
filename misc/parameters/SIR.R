
library(dplyr)
library(COVID19)
library(rstan)
library(bayesplot)

# parameters
country <- 'CZ'
date.min <- '2020-03-10'
date.max <- '2021-01-31'

# fetch data
covid_data <- covid19(country, level = 1)
covid_stats <- covid_data %>%
  dplyr::ungroup() %>%
  dplyr::transmute(country = iso_alpha_2, dates = date,
                   I = c(0,diff(confirmed)), R = c(0,diff(confirmed)), D = c(0,diff(deaths))) %>%
  dplyr::mutate(I = ifelse(is.na(I), 0, I),
                R = ifelse(is.na(R), 0, R),
                D = ifelse(is.na(D), 0, D)) %>%
  dplyr::mutate(I = as.integer(I),
                R = as.integer(R),
                D = as.integer(D)) %>%
  dplyr::mutate(I = ifelse(I < 0, 0, I),
                R = ifelse(R < 0, 0, R),
                D = ifelse(D < 0, 0, D)) %>% # remove corrections
  dplyr::filter(dates <= as.Date(date.max), dates > as.Date(date.min)) %>%
  dplyr::filter(country == 'CZ')

covid_tests <- read.csv('data/tests.csv', header=T) %>%
  dplyr::transmute(country, dates = date, T = tests) %>%
  dplyr::filter(dates <= as.Date(date.max), dates > as.Date(date.min)) %>%
  dplyr::filter(country == 'CZE')

# distributions
incub_d <- read.csv('data/distr/incubation.csv', header = F, col.names = c('Day','Prob','Trans'))
symp_d <- read.csv('data/distr/symptoms.csv', header = F, col.names = c('Day','Prob','Trans'))

DAYS <- nrow(covid_stats)
#downC <- 100
POP <- as.integer(10629928)
to_time_axis <- function(v) {
  matrix(v, nrow = DAYS, ncol = length(v), byrow = T)
}
stan_data <- list(
  DAYS = DAYS, POP = POP,
  TS = 1:DAYS,
  prior_a = c(c=1.725482,sigma=.373588),
  prior_c = c(a=2.736960,b=28.970829),
  prior_b = c(a=3.622493,b=14.421170),
  prior_d = c(a=3.622493,b=14.421170),
  prior_test = to_time_axis(c(.02,.8)),
  tests = covid_tests$T,
  confirmed = covid_stats$I / POP,
  recovered = covid_stats$R / POP,
  deaths = covid_stats$D / POP,
  init_solution = c(S = (POP - 1) / POP, E = 0, I = 1 / POP, R = 0, D = 0)
)

MAXITER <- 1000
model <- stan_model("model/quotient.stan", verbose = TRUE)
fit_sir <- sampling(model,
                    data = stan_data,
                    iter = MAXITER*2,
                    chains = 2,
                    seed = 0,
                    init = 1)

# SIR parameters
stan_dens(fit_sir, pars = c("a_sir","c_sir","b_sir","d_sir"), separate_chains = TRUE)
# derived parameters
stan_dens(fit_sir, pars = c("R0","recovery_time"), separate_chains = TRUE)

# get R0
#r0 <- extract(fit_sir, pars = c('R0'))$R0 #/ POP
#plot(fit_sir, pars = c("R0"))

# get latent variables
y <- extract(fit_sir, pars = c('y'))

susceptible <- matrix(nrow=DAYS, ncol=MAXITER)
exposed <- matrix(nrow=DAYS, ncol=MAXITER)
infected <- matrix(nrow=DAYS, ncol=MAXITER)
deaths <- matrix(nrow=DAYS, ncol=MAXITER)
recovered <- matrix(nrow=DAYS, ncol=MAXITER)
for(i in 1:MAXITER) {
  for(d in 1:DAYS) {
    susceptible[d,i] <- y$y[i,d,1] * POP
    exposed[d,i] <- y$y[i,d,2] * POP
    infected[d,i] <- y$y[i,d,3] * POP
    deaths[d,i] <- y$y[i,d,4] * POP
    recovered[d,i] <- y$y[i,d,5] * POP
  }
}

plot.sim <- function(samples, sims = T, lab = 'Density') {
  if(sims) {
    plot(1:DAYS, samples[,1], col = 'red', type = 'l',
         xlab = 'Day of pandemic', ylab = lab)
    for(i in 2:MAXITER)
      lines(1:DAYS, samples[,i], col = 'red')
    lines(1:DAYS, apply(samples, 1, mean))
  } else {
    plot(1:DAYS, apply(samples, 1, mean), type = 'l',
         xlab = 'Day of pandemic', ylab = lab)
  }
}
 
# plot infected
plot.sim(susceptible, sims = F, lab = 'Susceptible')
plot.sim(exposed, sims = F, lab = 'Exposed')
plot.sim(infected, sims = F, lab = 'Infected')
plot.sim(deaths, sims = F, lab = 'Deaths')
plot.sim(recovered, sims = F, lab = 'Recovered')


# plot infected against confirmed
plot(1:DAYS, covid_stats$I, type="l", col="blue")
lines(1:DAYS, apply(infected, 1, mean), col="gray")


plot(fit_sir_negbin, pars = c("R0", 'recovery_time'))

#print(fit_sir_negbin, pars = c("R0", "recovery_time"))
stan_dens(fit_sir_negbin, pars = c("R0", "recovery_time"), separate_chains = TRUE)
#fit_rstan <- stan(
#  file = "../../model/sir_model.stan",
#  data = stan_data
#)
fit_sir %>% mcmc_trace()
