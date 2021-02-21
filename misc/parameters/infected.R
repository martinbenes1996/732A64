

library(dplyr)
library(COVID19)
library(rstan)
library(bayesplot)

# parameters
country <- 'CZ'
date.min <- '2020-03-20'
date.max <- '2021-01-31'

# fetch data
covid_data <- covid19(country, level = 2)
covid_stats <- covid_data %>%
  dplyr::ungroup() %>%
  dplyr::transmute(country = iso_alpha_2, dates = date, I = c(0,diff(confirmed))) %>%
  dplyr::mutate(I = ifelse(is.na(I), 0, I)) %>%
  dplyr::mutate(I = as.integer(I)) %>%
  dplyr::mutate(I = ifelse(I < 0, 0, I)) %>% # remove corrections
  dplyr::filter(dates <= as.Date(date.max))