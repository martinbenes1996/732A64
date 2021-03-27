

library(dplyr)
library(COVID19)
library(rstan)
library(bayesplot)

# parameters
country <- 'CZ'
date.min <- '2020-03-20'
date.max <- '2021-01-31'

# fetch data
covid_data <- covid19(country, level = 1)
covid_stats <- covid_data %>%
  dplyr::ungroup() %>%
  dplyr::transmute(country = iso_alpha_2, dates = date, I = c(0,diff(confirmed))) %>%
  dplyr::mutate(I = ifelse(is.na(I), 0, I)) %>%
  dplyr::mutate(I = as.integer(I)) %>%
  dplyr::mutate(I = ifelse(I < 0, 0, I)) %>% # remove corrections
  dplyr::filter(dates <= as.Date(date.max))
covid_tests <- read.csv('data/tests.csv', header=T) %>%
  dplyr::transmute(country, dates = as.Date(date), T = tests) %>%
  dplyr::filter(dates >= date.min, dates <= date.max) %>%
  dplyr::filter(country == 'CZE')
covid_tests$country = country

covid <- covid_stats %>%
  dplyr::inner_join(covid_tests, by=c('dates','country'))

sim <- c()
alpha <- 1e3
beta <- 4
for(r in 1:nrow(covid)) {
  spl <- rbeta(1, alpha + covid$I[r], beta + covid$T[r] - covid$I[r])
  sim <- c(sim, spl)
}
plot(covid$dates, sim, type="l", ylim=c(0,1))
lines(covid$dates, covid$I / covid$T, type = "l", col = "red")



POP <- 1e7
fx <- rbeta(1000,alpha,beta)
hist(fx, probability=T, breaks=50)
lines(density(fx))



