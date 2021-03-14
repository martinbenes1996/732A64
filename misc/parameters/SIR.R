
library(dplyr)
library(COVID19)
library(rstan)
library(bayesplot)

# parameters
country <- 'CZ'
date.min <- '2020-04-01' #'2020-03-20' #'2020-09-01'#
date.max <- '2020-12-31' #'2020-12-31'#

# fetch data
covid_data <- covid19(country, level = 1)
covid_stats <- covid_data %>%
  dplyr::ungroup() %>%
  dplyr::transmute(country = iso_alpha_2, dates = date,
                   I = c(confirmed[1],diff(confirmed)),
                   R = c(recovered[1],diff(recovered)),
                   D = c(deaths[1],diff(deaths))) %>%
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
saturate <- function(v, max = 1, min = 0, eps=1e-5) {
  if(v >= max) return(max-eps)
  else if(v <= min) return(min+eps)
  else return(v)
}
stan_data <- list(
  DAYS = DAYS, WINDOW = 7, TS = 1:DAYS,
  POP = POP, INDIV = 1/POP,
  prior_a = c(c=1.725482,sigma=.373588),#c(1,2),
  prior_c = c(a=2.736960,b=28.970829),#c(1,2),
  prior_b = c(a=3.622493,b=14.421170),#c(1,2),
  prior_d = c(a=3.622493,b=14.421170),#c(1,2),
  prior_test = c(0.25155568403722195, 0.30983778140500917), #.7,
  prior_test_rec = c(0.25155568403722195, 0.30983778140500917),
  prior_deaths = c(0.25155568403722195, 0.30983778140500917),#c(10,1),
  tests = covid_tests$T,
  confirmed = covid_stats$I / covid_tests$T,
  recovered = sapply((covid_stats$R+1) / covid_tests$T, saturate),
  deaths = (covid_stats$D+1) / POP,
  init_solution = c(S = (POP-covid_stats$I[1]-covid_stats$R[1]-covid_stats$D[1])
                      / POP,
                    E = saturate(covid_stats$I[1]/3/POP*10),
                    I = saturate(covid_stats$I[1]*2/3/POP*10),
                    R = saturate(covid_stats$R[1]/POP*10),
                    D = (covid_stats$D[1]+1) / POP)
)

MAXITER <- 400
model <- stan_model("model/quotient.stan", verbose = TRUE)
fit_sir <- sampling(model,
                    data = stan_data,
                    iter = MAXITER*2,
                    chains = 1,
                    cores = 2,
                    seed = 0,
                    init = 1)

# parameters
params <- extract(fit_sir, pars = c("a_sir","c_sir","b_sir","d_sir"))
der.params <- extract(fit_sir, pars = c("R0","recovery_time"))
y <- extract(fit_sir, pars = c('y'))
write.csv(covid_tests$T, 'results/tests.csv')
write.csv(params$a_sir, 'results/a_sir.csv')
write.csv(params$c_sir, 'results/c_sir.csv')
write.csv(params$b_sir, 'results/b_sir.csv')
write.csv(params$d_sir, 'results/d_sir.csv')
write.csv(der.params$R0, 'results/R0.csv')
write.csv(der.params$recovery_time, 'results/recovery_time.csv')
write.csv(y$y, 'results/y.csv')


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
    susceptible[d,i] <- round(y$y[i,d,1] * POP)
    exposed[d,i] <- y$y[i,d,2]
    infected[d,i] <- y$y[i,d,3] * POP
    deaths[d,i] <- round(abs(y$y[i,d,4]) * POP)
    recovered[d,i] <- round(y$y[i,d,5] * POP)
  }
}

plot.sim <- function(samples, ref = NA, sims = T, lab = 'Density') {
  if(sims) {
    ylim = c(min(samples, ref, 0, na.rm = T),
             max(samples, ref, na.rm = T))
    plot(1:DAYS, samples[,1], col = 'red', type = 'l',
         xlab = 'Day of pandemic', ylab = lab, ylim = ylim)
    for(i in 2:MAXITER)
      lines(1:DAYS, samples[,i], col = 'red')
    lines(1:DAYS, apply(samples, 1, mean))
  } else {
    ylim = c(min(apply(samples, 1, mean), ref, 0, na.rm = T),
             max(apply(samples, 1, mean), ref, na.rm = T))
    plot(1:DAYS, apply(samples, 1, mean), type = 'l',
         xlab = 'Day of pandemic', ylab = lab, ylim = ylim)
  }
  if(!is.na(ref)) {
    lines(1:DAYS, ref, col = 'blue')
  }
}
 
# plot infected
plot.sim(susceptible, sims = F, lab = 'Susceptible')
plot.sim(exposed, sims = F, lab = 'Exposed')
plot.sim(infected, sims = F, lab = 'Infected')
plot.sim(deaths, ref = covid_stats$D, sims = F, lab = 'Deaths')
plot.sim(recovered, ref = covid_stats$R, sims = F, lab = 'Recovered')

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


#stan_data <- list(
#  n_days = DAYS, t0 = 0,
#  y0 = c(S = POP - 1, E = 0, I = 1, R = 0),
#  N = POP,
#  ts = 1:DAYS,
#  cases = covid_stats$I
#)

#MAXITER <- 200
#model <- stan_model("model/school.stan", verbose = TRUE)
#fit_school <- sampling(model,
#                    data = stan_data,
#                    iter = MAXITER*2,
#                    chains = 1,
#                    seed = 0,
#                    init = 1)



