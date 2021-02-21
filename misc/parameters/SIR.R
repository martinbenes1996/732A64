
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
incub_d <- read.csv('data/incubation.csv', header = F, col.names = c('Day','Prob','Trans'))
symp_d <- read.csv('data/symptoms.csv', header = F, col.names = c('Day','Prob','Trans'))

DAYS <- nrow(covid_stats)
POP <- 10629928
stan_data <- list(
  DAYS = DAYS, POP = POP,
  TS = 1:DAYS,
  #INCUB = nrow(incub_d), SYMP = nrow(symp_d),
  prior_a = c(a=0.9,b=1),
  prior_c = c(a=2.927733,b=14737.430269,loc=0.0454624,scale=768.263174),
  #prior_b = c(a=1.632731,b=1836.076787,loc=-0.017817,scale=82.54667),
  #prior_d = c(a=1e-7,b=1e-1),
  #prior_d = c(a = 1.5, b = 1.5),
  ifr = 0.064, 
  prior_test = c(.02,.8),
  tests = covid_tests$T / POP,
  confirmed = covid_stats$I / POP,
  #recovered = covid_stats$R / POP,
  #deaths = covid_stats$D / POP,
  init_solution = c(S = (POP - 1) / POP, E = 0, I = 1 / POP) #, R = 0)#, D = 0)
)

MAXITER <- 500
model <- stan_model("model/quotient.stan", verbose = TRUE)
fit_sir_negbin <- sampling(model,
                           data = stan_data,
                           iter = MAXITER*2,
                           chains = 2,
                           seed = 0)

plot(fit_sir_negbin, pars = c("R0"))


y <- extract(fit_sir_negbin, pars = c('y'))

susceptible <- matrix(nrow=DAYS, ncol=MAXITER)
exposed <- matrix(nrow=DAYS, ncol=MAXITER)
infected <- matrix(nrow=DAYS, ncol=MAXITER)
for(i in 1:MAXITER) {
  for(d in 1:DAYS) {
    susceptible[d,i] <- y$y[i,d,1]
    exposed[d,i] <- y$y[i,d,2]
    infected[d,i] <- y$y[i,d,3]
  }
}

# plot infected
plot(1:DAYS, covid_stats$I, type="l", col="blue")
for(i in 1:MAXITER) {
  lines(1:DAYS, POP * infected[,i], col="red", type="l")
}
lines(1:DAYS, apply(infected * POP, 1, mean))


plot(fit_sir_negbin, pars = c("R0", 'recovery_time'))
 plot(fit_sir_negbin, pars = c("y"))
print(fit_sir_negbin, pars = c('beta', 'gamma', "R0", "recovery_time"))
stan_dens(fit_sir_negbin, pars = c('beta', 'gamma', "R0", "recovery_time"), separate_chains = TRUE)
#fit_rstan <- stan(
#  file = "../../model/sir_model.stan",
#  data = stan_data
#)
fit_rstan %>% mcmc_trace()