
library(dplyr)
library(COVID19)
library(rstan)
library(bayesplot)

saturate <- function(v, max = 1, min = 0, eps=1e-5) {
  if(v >= max) return(max-eps)
  else if(v <= min) return(min+eps)
  else return(v)
}

runSim <- function(country, date.min, date.max, window = 5, niter = 500) {
  # dates
  date.min <- as.Date(date.min)
  date.max <- as.Date(date.max)
  date.len <- as.integer(date.max - date.min) + 1
  date.win.min <- date.min - window
  date.win.max <- date.max + window
  
  # fetch data
  covid_data <- covid19(country, level = 1)
  covid_stats <- covid_data %>%
    dplyr::ungroup() %>%
    dplyr::transmute(country = iso_alpha_2, dates = as.Date(date),
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
    dplyr::filter(dates >= date.win.min, dates <= date.win.max) %>%
    dplyr::filter(country == 'CZ')
  covid_tests <- read.csv('data/tests.csv', header=T) %>%
    dplyr::transmute(country, dates = as.Date(date), T = tests) %>%
    dplyr::filter(dates >= date.win.min, dates <= date.win.max) %>%
    dplyr::filter(country == 'CZE')
  covid_tests$country <- country
  # crop to have same range
  date.act.min <- max(c(min(covid_stats$dates), min(covid_tests$dates)))
  date.act.max <- min(c(max(covid_stats$dates), max(covid_tests$dates)))
  covid_stats <- covid_stats %>%
    dplyr::filter(dates >= date.act.min, dates <= date.act.max)
  covid_tests <- covid_tests %>%
    dplyr::filter(dates >= date.act.min, dates <= date.act.max)
  # statistics
  DAYS <- nrow(covid_stats)
  POP <- as.integer(10629928)
  
  # compile model
  model <- stan_model("model/quotient.stan", verbose = TRUE)
  
  # input data
  stan_data <- list(
    # time units
    DAYS = DAYS, # time axis length
    TS = 1:DAYS, # time axis
    WINDOW = window, # parameter window size
    # population
    POP = POP, # population size
    INDIV = 1/POP, # individual ratio
    # priors
    prior_a = c(2.0073,1/1.5453),#c(1.67,1/1.675572)#c(c=1.836352,sigma=.365743), # prior [S -> E]
    prior_c = c(a=2.773984,b=15.192478), # prior [E -> I]
    prior_b = c(a=3.402405,b=37.713360), # prior [I -> R]
    prior_d = c(a=3.151443,b=5438.488333), # prior [I -> D]
    prior_test = c(0.25155568403722195, 0.30983778140500917), # prior [I gets tested]
    prior_test_rec = c(0.25155568403722195, 0.30983778140500917), # prior [R gets tested]
    prior_deaths = c(20,1), # prior [D is connected with Covid-19]
    # statistics
    tests = covid_tests$T, # number of tests
    confirmed = covid_stats$I / covid_tests$T, # positive test ratio
    recovered = sapply((covid_stats$R+1) / covid_tests$T, saturate), # recovered test ratio
    deaths = (covid_stats$D+1) / POP, # deaths per capita
    # initial solution
    init_solution = c(
      S = (POP-(covid_stats$I[1]-covid_stats$R[1])*10-covid_stats$D[1])/ POP, # S
      E = saturate(covid_stats$I[1]*10/3/POP), # E
      I = saturate(covid_stats$I[1]*20/3/POP), # I
      R = saturate(covid_stats$R[1]/POP), # R
      D = (covid_stats$D[1]+1) / POP # D
    )
  )
  
  # simulate
  fit_sir <- sampling(model,
                      data = stan_data,
                      iter = niter*2,
                      chains = 1,
                      seed = 0,
                      init = 1)
  
  # extract parameters
  get_param_df <- function(param_values, value_column) {
    param_sir <- data.frame(draw=1:(dim(param_values)[1]), param_values)
    colnames(param_sir) <- c('draw', 1:(ncol(param_sir)-1))
    param_sir %>%
      tidyr::pivot_longer(!draw, names_to = 'time', values_to = value_column) %>%
      dplyr::mutate(time = as.integer(time))
  }
  # params
  params <- extract(fit_sir, pars = c("a_sir","c_sir","b_sir","d_sir"))
  a_sir <- get_param_df(params$a_sir, 'a')
  c_sir <- get_param_df(params$c_sir, 'c')
  b_sir <- get_param_df(params$b_sir, 'b')
  d_sir <- get_param_df(params$d_sir, 'd')
  der.params <- extract(fit_sir, pars = c("R0","recovery_time"))
  r0 <- get_param_df(t(der.params$R0), 'r0')
  recovery_time <- get_param_df(der.params$recovery_time, 'recovery_time')
  params <- a_sir %>%
    dplyr::full_join(c_sir, c('draw','time')) %>%
    dplyr::full_join(b_sir, c('draw','time')) %>%
    dplyr::full_join(d_sir, c('draw','time')) %>%
    dplyr::full_join(r0, c('draw','time')) %>%
    dplyr::full_join(recovery_time, c('draw','time')) %>%
    dplyr::mutate(date = date.win.min + window * (time-1)) %>%
    #dplyr::filter(date >= date.min, date <= date.max) %>%
    dplyr::transmute(draw,date,a,c,b,d,r0,recovery_time)
  
  # latent variables
  y <- extract(fit_sir, pars = c('y'))$y * POP
  y_s <- get_param_df(y[,,1], 's')
  y_e <- get_param_df(y[,,2], 'e')
  y_i <- get_param_df(y[,,3], 'i')
  y_r <- get_param_df(y[,,4], 'r')
  y_d <- get_param_df(y[,,5], 'd')
  y <- y_s %>%
    dplyr::full_join(y_e, c('draw','time')) %>%
    dplyr::full_join(y_i, c('draw','time')) %>%
    dplyr::full_join(y_r, c('draw','time')) %>%
    dplyr::full_join(y_d, c('draw','time')) %>%
    dplyr::mutate(date = date.win.min + (time-1)) %>%
    #dplyr::filter(date >= date.min, date <= date.max) %>%
    dplyr::transmute(draw,date,s,e,i,r,d)
  
  # join covid stats
  covid <- covid_stats %>%
    dplyr::full_join(covid_tests, c('dates','country'))
  
  # directory
  today = format(Sys.Date(), format = '%d_%m_%Y')
  start = format(date.min, format = '%d_%m_%Y')
  end = format(date.max, format = '%d_%m_%Y')
  save_dir = sprintf("results/%s-%s-%s-w%d-prior2", today, start, end, window)
  # save parameters
  dir.create(save_dir, showWarnings = FALSE)
  write.csv(covid, sprintf('%s/covid.csv',save_dir))
  write.csv(params, sprintf('%s/params.csv',save_dir))
  write.csv(y, sprintf('%s/y.csv',save_dir))
}

#runSim('CZ', '2020-04-01', '2020-04-15')
#runSim('CZ', '2020-04-16', '2020-04-30')
#runSim('CZ', '2020-05-01', '2020-05-15', niter = 700)
runSim('CZ', '2020-04-01', '2020-04-30', window = 10)
#runSim('CZ', '2020-04-01', '2020-04-30', niter = 1000)


