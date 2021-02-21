
# import
library(tidyr)
library(ggplot2)
library(plotly)
library(patchwork)
library(EpiEstim)
library(COVID19)


# parameters
#date.min <- as.Date("2020-03-08") # as.Date("2020-03-10")
#country <- 'CZ'

plot_gamma <- function(lim.max = 20) {
  # continuous distribution
  xgrid <- seq(0.01,lim.max,length.out = 1000)
  gamma.continuous <- data.frame(
    x = xgrid,
    y = dgamma(xgrid,11.39,2.504)
  )
  # discrete distribution
  y.discrete <- pgamma(1:(lim.max+1),11.39,2.504) - pgamma(0:lim.max,11.39,2.504)
  gamma.discrete <- data.frame(
    x = c(0, sapply(1:lim.max, rep, 2), lim.max+1),
    y = c(sapply(y.discrete, rep, 2))
  )
  
  # plot
  gamma.discrete %>%
    ggplot(mapping = aes(x = x, y = y)) +
    geom_polygon(alpha = .8) +
    geom_line(data = gamma.continuous) +
    labs(x = 'Serial interval (days)', y = 'Density')
}

get_restrictions <- function(country) {
  restrictions <- read.csv('data/restrictions.csv')
  restrictions <- restrictions %>%
    dplyr::filter(Country == country) %>%
    dplyr::mutate(Date = as.Date(Date), y = 1)
  return(restrictions)
}

get_tests <- function(covid_data, country, date.min) {
  if(country %in% c('PL','POL','Poland','SE','SWE','Sweden')) {
    # Poland
    if(country %in% c('PL','POL','Poland')) {
      # parse tests
      url <- 'https://raw.githubusercontent.com/martinbenes1996/covid19poland/master/data/tests.csv'
      tests <- read.csv(url) %>%
        dplyr::mutate(Date = as.Date(date), Tests = tests)
      tests.total <- tests %>%
        dplyr::filter(region == '') %>%
        dplyr::arrange(Date)
      tests.group <- tests %>%
        dplyr::filter(region != '') %>%
        dplyr::group_by(Date) %>%
        dplyr::summarise(tests = sum(tests), .groups = 'drop')
      tests <- tests.total %>%
        dplyr::full_join(tests.group, by = 'Date') %>%
        dplyr::transmute(Date, tests = ifelse(is.na(tests.x), tests.y, tests.x)) %>%
        dplyr::arrange(Date) %>%
        dplyr::transmute(Date, Tests = c(0,diff(tests))) %>%
        dplyr::mutate(Tests = ifelse(Tests > 0, Tests, 0))
      max.date <- max(tests$Date)
      tests.weekly <- tests %>%
        dplyr::filter(Date >= as.Date('2020-09-07')) %>%
        dplyr::filter(Tests > 0) %>%
        dplyr::mutate(days = diff(c(Date, max.date))) %>%
        dplyr::mutate(days = as.integer(days)) %>%
        dplyr::filter(Date > date.min, Tests > 0)
    # Sweden
    } else {
      # parse tests
      tests <- read.csv('data/se_tests.csv')
      colnames(tests) <- c('Year','Week','Tests','Performed','URL')
      tests <- tests %>%
        dplyr::mutate(Week = ifelse(Year == 2021, Week + 1, Week)) %>%
        dplyr::mutate(Date = sprintf('%d-%02d-1', Year,Week-1)) %>%
        dplyr::transmute(Date = as.Date(Date, '%Y-%W-%w'), Tests)
      max.date <- max(tests$Date) + 7
      tests.weekly <- tests %>%
        dplyr::mutate(days = diff(c(Date, max.date))) %>%
        dplyr::mutate(days = as.integer(days)) %>%
        dplyr::filter(Date > date.min, tests > 0)
    }
    # fill empty days
    for(i in 1:nrow(tests.weekly)) {
      row <- tests.weekly[i,]
      end.day <- row$Date + row$days - 1
      week <- rmultinom(1, row$Tests, rep(1/row$days, row$days))
      for(j in 0:(row$days - 1)) {
        dt <- row$Date + j
        if(nrow(tests[which(tests$Date == dt),]) == 0) {
          tests <- rbind(tests, data.frame(Date = dt, Tests = NA))
        }
      }
      tests <- tests %>% dplyr::arrange(Date)
      tests[which(tests$Date >= row$Date & tests$Date <= end.day),]$Tests <- week
    }
  # Covid 19 Data-Hub
  } else {
    tests <- covid_data %>%
      dplyr::ungroup() %>%
      dplyr::filter(date >= date.min, !is.na(tests)) %>%
      dplyr::mutate(Tests = c(0,diff(tests)), Date = date) %>%
      dplyr::filter(Date > date.min, Tests > 0)
  }
  # to view structure
  tests <- tests %>%
    dplyr::transmute(Date, Tests)
  # fill missing dates
  missing_days <- c()
  for(day_offset in 0:as.numeric(max(tests$Date)-min(tests$Date))) {
    day <- min(tests$Date) + day_offset
    if(!(day %in% tests$Date)) {
      missing_days <- c(missing_days, strftime(day))
    }
  }
  tests <- tests %>%
    dplyr::mutate(Date = strftime(Date))
  if(length(missing_days) > 0) {
    tests <- rbind(tests, data.frame(Date = missing_days, Tests = 0))
  }
  # to view structure
  tests <- tests %>%
    dplyr::arrange(Date) %>%
    dplyr::transmute(Date = as.Date(Date), Tests, Variable = 'Tests')
  return(tests)
}

estimate_reproduction <- function(incid, date.min) {
  lim.max <- 20
  y.discrete <- pgamma(1:(lim.max+1),11.39,2.504) - pgamma(0:lim.max,11.39,2.504)
  y.discrete[1] <- 0
  y.discrete <- y.discrete / sum(y.discrete)
  res <- estimate_R(incid = incid, method = "non_parametric_si",
                    config = make_config(list(si_distr = y.discrete)))
  R <- data.frame(
    Date = res$dates[res$R$`t_start`],
    R0 = res$R$`Mean(R)`,
    lower95 = res$R$`Quantile.0.025(R)`,
    upper95 = res$R$`Quantile.0.975(R)`) %>%
    dplyr::filter(Date >= date.min) %>%
    dplyr::mutate(Variable = 'R0')
}

plot_R0_series <- function(country, date.min = '2020-03-08') {

  # restrictions
  restrictions <- get_restrictions(country)

  # daily incidence
  covid_data <- covid19(country, level = 1)
  covid_stats <- covid_data %>%
    dplyr::ungroup() %>%
    dplyr::transmute(dates = date, I = c(0,diff(confirmed))) %>%
    dplyr::mutate(I = ifelse(is.na(I), 0, I)) %>%
    dplyr::mutate(I = as.integer(I)) %>%
    dplyr::mutate(I = ifelse(I < 0, 0, I)) # remove corrections
  
  # tests
  tests <- get_tests(covid_data, country, date.min)

  # estimate R0
  R <- estimate_reproduction(covid_stats, date.min)
  dt.min <- min(R$Date)
  dt.max <- max(R$Date)
  
  # plot
  plot_ly() %>%
    add_lines(x = ~Date, y = ~R0, color = ~Variable, data = R,
              opacity = 1) %>%
    add_lines(x = ~Date, y = ~Tests, color = ~Variable, data = tests,
              yaxis = "y2", opacity = .25) %>%
    add_segments(x = ~Date, xend = ~Date, y = ~lower95, yend = ~upper95,
                 data = R, opacity = .4, name = '95% CI') %>%
    add_segments(x = dt.min, xend = dt.max, y = 1, yend = 1,
                 opacity = .4, name = 'R0 = 1') %>%
    add_markers(x = ~Date, y = ~y, color = ~Restriction, text = ~Title,
                data = restrictions) %>%
    layout(yaxis2 = list(overlaying = "y", side = "right"))

}

plot_tests_R0_series <- function(country, date.min = '2020-03-08') {
  
  # restrictions
  restrictions <- get_restrictions(country)
  
  # daily incidence
  covid_data <- covid19(country, level = 1)
  covid_stats <- covid_data %>%
    dplyr::ungroup() %>%
    dplyr::transmute(dates = date, I = c(0,diff(confirmed))) %>%
    dplyr::mutate(I = ifelse(is.na(I), 0, I)) %>%
    dplyr::mutate(I = as.integer(I)) %>%
    dplyr::mutate(I = ifelse(I < 0, 0, I)) # remove corrections
  
  # estimate R0
  R <- estimate_reproduction(covid_stats, date.min)
  
  # tests
  tests <- get_tests(covid_data, country, date.min)
  tests <- tests %>% dplyr::transmute(dates = Date, I = Tests)
  # estimate R0
  R_tests <- estimate_reproduction(tests, date.min)
  tests <- tests %>%
    dplyr::mutate(Variable = 'R0 on Tests')
  
  dt.min <- min(R$Date)
  dt.max <- max(R$Date)
  
  # plot
  plot_ly() %>%
    add_lines(x = ~Date, y = ~R0, color = '#0000ff', data = R,
              opacity = 1, name = 'R0') %>%
    add_lines(x = ~Date, y = ~R0, color = '#ff0000', data = R_tests,
              opacity = 1, name = 'R0 on Tests') %>%
    #add_lines(x = ~Date, y = ~Tests, color = ~Variable, data = tests,
    #          yaxis = "y2", opacity = .15) %>%
    add_segments(x = dt.min, xend = dt.max, y = 1, yend = 1,
                 opacity = .4, name = 'R0 = 1', color = 'black') %>%
    add_markers(x = ~Date, y = ~y, color = ~Restriction, text = ~Title,
                data = restrictions) %>%
    layout(yaxis2 = list(overlaying = "y", side = "right"))
  
}

plot_R0_series <- function(country, date.min = '2020-03-08') {
  
  # restrictions
  restrictions <- get_restrictions(country)
  
  # daily incidence
  covid_data <- covid19(country, level = 1)
  covid_stats <- covid_data %>%
    dplyr::ungroup() %>%
    dplyr::transmute(dates = date, I = c(0,diff(confirmed))) %>%
    dplyr::mutate(I = ifelse(is.na(I), 0, I)) %>%
    dplyr::mutate(I = as.integer(I)) %>%
    dplyr::mutate(I = ifelse(I < 0, 0, I)) # remove corrections
  
  # tests
  tests <- get_tests(covid_data, country, date.min)
  
  # estimate R0
  R <- estimate_reproduction(covid_stats, date.min)
  dt.min <- min(R$Date)
  dt.max <- max(R$Date)
  
  # plot
  plot_ly() %>%
    add_lines(x = ~Date, y = ~R0, color = ~Variable, data = R,
              opacity = 1) %>%
    add_lines(x = ~Date, y = ~Tests, color = ~Variable, data = tests,
              yaxis = "y2", opacity = .15) %>%
    add_segments(x = dt.min, xend = dt.max, y = 1, yend = 1,
                 opacity = .4, name = 'R0 = 1') %>%
    add_markers(x = ~Date, y = ~y, color = ~Restriction, text = ~Title,
                data = restrictions) %>%
    layout(yaxis2 = list(overlaying = "y", side = "right"))
  
}

plot_R0_box <- function(country = NA, date.min = '2020-03-08', date.max = '2021-01-01') {
  
  # daily incidence
  covid_data <- covid19(country, level = 1)
  covid_stats <- covid_data %>%
    dplyr::ungroup() %>%
    dplyr::transmute(country = iso_alpha_2, dates = date, I = c(0,diff(confirmed))) %>%
    dplyr::mutate(I = ifelse(is.na(I), 0, I)) %>%
    dplyr::mutate(I = as.integer(I)) %>%
    dplyr::mutate(I = ifelse(I < 0, 0, I)) %>% # remove corrections
    dplyr::filter(dates <= as.Date(date.max))
  # tests
  #tests <- get_tests(covid_data, country, date.min)
  
  Rs <- NA
  Rs.empty <- T
  for(country in unique(covid_stats$country)) {
    country_stats <- covid_stats[covid_stats$country == country,] %>%
      dplyr::transmute(dates,I)
    R <- estimate_reproduction(country_stats, date.min)
    R <- cbind(R, data.frame(country = country))
    if(Rs.empty) {
      Rs <- R
      Rs.empty <- F
    }
    else Rs <- rbind(Rs,R)
  }
  
  Rs %>%
    ggplot(mapping = aes(x = country, y = R0)) +
    geom_boxplot(outlier.colour="red")
}

plot_trace_R0_box <- function(country = NA, date.min = '2020-03-08', date.max = '2021-01-31') {
  
  # daily incidence
  covid_data <- covid19(country, level = 1)
  covid_stats <- covid_data %>%
    dplyr::ungroup() %>%
    dplyr::transmute(Country = iso_alpha_2, dates = date, I = c(0,diff(confirmed))) %>%
    dplyr::mutate(I = ifelse(is.na(I), 0, I)) %>%
    dplyr::mutate(I = as.integer(I)) %>%
    dplyr::mutate(I = ifelse(I < 0, 0, I)) %>% # remove corrections
    dplyr::filter(dates <= as.Date(date.max))
  # tests
  #tests <- get_tests(covid_data, country, date.min)
  
  Rs <- NA
  Rs.empty <- T
  for(country in unique(covid_stats$Country)) {
    country_stats <- covid_stats[covid_stats$Country == country,] %>%
      dplyr::transmute(dates,I)
    R <- estimate_reproduction(country_stats, date.min)
    R <- cbind(R, data.frame(Country = country))
    if(Rs.empty) {
      Rs <- R
      Rs.empty <- F
    }
    else Rs <- rbind(Rs,R)
  }
  
  Rs <- Rs %>%
    dplyr::mutate(Month = format(Date,'%Y-%m'))
  
  Rs %>%
    ggplot(mapping = aes(x = Month, y = R0)) +
    geom_boxplot(outlier.colour="red") +
    facet_wrap(~Country)
}

plot_gamma()
plot_R0_series('CZ','2020-03-10')
plot_R0_series('PL','2020-03-10')
plot_R0_series('SE','2020-03-01')
plot_R0_series('IT','2020-03-01')

plot_tests_R0_series('CZ','2020-03-10')
plot_tests_R0_series('PL','2020-03-10')
plot_tests_R0_series('SE','2020-03-01')
plot_tests_R0_series('IT','2020-03-01')

# R0 box plots
plot_R0_box(c('CZ','IT','PL','SE'), '2020-03-25','2021-02-01')
plot_R0_box(c('CZ','IT','PL','SE'), '2020-03-01','2020-03-31') # March - initial period
plot_R0_box(c('CZ','IT','PL','SE'), '2020-04-01','2020-04-30') # April - first wave
plot_R0_box(c('CZ','IT','PL','SE'), '2020-05-01','2020-05-31') # May - easing
plot_R0_box(c('CZ','IT','PL','SE'), '2020-06-01','2020-06-30') # June
plot_R0_box(c('CZ','IT','PL','SE'), '2020-07-01','2020-07-31') # July
plot_R0_box(c('CZ','IT','PL','SE'), '2020-08-01','2020-08-31') # August
plot_R0_box(c('CZ','IT','PL','SE'), '2020-09-01','2020-09-30') # September - second wave
plot_R0_box(c('CZ','IT','PL','SE'), '2020-10-01','2020-10-31') # October
plot_R0_box(c('CZ','IT','PL','SE'), '2020-11-01','2020-11-30') # November - easing
plot_R0_box(c('CZ','IT','PL','SE'), '2020-12-01','2020-12-31') # December - third wave
plot_R0_box(c('CZ','IT','PL','SE'), '2021-01-01','2021-01-31') # January

plot_trace_R0_box(c('CZ','IT','PL','SE'), '2020-03-20')


R0.stan <- function(country, date.min, date.max) {
  library(rstan)
  library(bayesplot)
  covid_data <- covid19(country, level = 1)
  covid_stats <- covid_data %>%
    dplyr::ungroup() %>%
    dplyr::transmute(country = iso_alpha_2, dates = date, I = c(0,diff(confirmed))) %>%
    dplyr::mutate(I = ifelse(is.na(I), 0, I)) %>%
    dplyr::mutate(I = as.integer(I)) %>%
    dplyr::mutate(I = ifelse(I < 0, 0, I)) %>% # remove corrections
    dplyr::filter(dates <= as.Date(date.max))
  
  #stan_data <- list(
  #  a = 1.5, b = 1.5,
  #  #p = 0.25,
  #  N = nrow(covid_stats),
  #  V = 1000000,
  #  incidence = covid_stats$I
  #)
  
  N <- nrow(covid_stats)
  stan_data <- list(
    n_days = N,
    y0 = c(S = N - 1, I = 1, R = 0),
    t0 = 0,
    ts = 1:N,
    N = 10000000,
    cases = covid_stats$I
  )
  
  library(outbreaks)
  N <- nrow(influenza_england_1978_school)
  stan_data <- list(
    n_days = N,
    y0 = c(S = N - 1, I = 1, R = 0),
    t0 = 0,
    ts = 1:N,
    N = 763,
    cases = influenza_england_1978_school$in_bed
  )
  
  model <- stan_model("../../model/sir_model.stan")
  fit_sir_negbin <- sampling(model,
                             data = stan_data,
                             iter = 1000,
                             chains = 4, 
                             seed = 0)
  print(fit_sir_negbin, pars = c('beta', 'gamma', "R0", "recovery_time"))
  stan_dens(fit_sir_negbin, pars = c('beta', 'gamma', "R0", "recovery_time"), separate_chains = TRUE)
  #fit_rstan <- stan(
  #  file = "../../model/sir_model.stan",
  #  data = stan_data
  #)
  fit_rstan %>% mcmc_trace()
}

xgrid <- seq(0.01,10,by = 0.01)
y <- dgamma(xgrid, shape = 1.5, rate = 1.5)
plot(xgrid, y)





# serial interval implementation
#y.discrete <- pgamma(1:(lim.max+1),11.39,2.504) - pgamma(0:lim.max,11.39,2.504)
#serial.interval <- function(tj, ti) y.discrete[ti - tj + 1]

# estimate probabilities
#probs <- matrix(nrow = nrow(covid_stats), ncol = lim.max)
#for(i in 20:nrow(covid_stats)) {
#  for(j in 1:lim.max) {
#    Ni <- covid_stats[i,]$incidence
#    w <- serial.interval(i - j, i)
#    probs[i,j] <- Ni * w
#  }
#  for(j in 1:lim.max) {
#    k <- -j
#    probs[i,j] <- probs[i,j] / sum(probs[i,k])
#  }
#}

#Rj <- apply(probs, 1, sum)
#Rt <- Rj

#covid_stats$R0 <- Rt

#covid_stats %>%
#  filter(date > as.date(""))
#  ggplot(aes(y = R0)) +
#  geom_line()

#probs[267,]

 # 267 2020-11-22


MAXITER <- 10000
POP <- 1000
prior.pi_neg <- .02
prior.pi_pos <- .8
ratios <- rbeta(MAXITER, 10,10)

#prior.alpha <- prior.pi_neg
#prior.beta <- 1 - prior.pi_pos
prior.alpha <- 2
prior.beta <- 2
xbar <- mean(ratios)

post.alpha <- prior.alpha + xbar
post.beta <- prior.beta + 1 - xbar

xgrid_01 <- seq(0,1,by=.01)
post.y <- dbeta(xgrid_01, post.alpha, post.beta)
plot(xgrid_01, post.y, type="l")

betas <- c()
for(it in 1:MAXITER) betas <- c(betas, sum(rbeta(POP, post.alpha, post.beta)))
xgrid <- 0:POP
beta.mean <- post.alpha / (post.alpha + post.beta)
beta.sigma2 <- post.alpha*post.beta/(((post.alpha+post.beta)^2)*(post.alpha+post.beta+1))
dens <- dnorm(xgrid, beta.mean * POP, sqrt(beta.sigma2 * POP))
hist(betas, probability = T, breaks=20)
lines(xgrid, dens)

plot.beta <- function(piN = NA,piP = NA) {
  if(is.na(piN)) piN <- 1
  if(is.na(piP)) piP <- 0
  xgrid <- seq(0,1,by=.01)
  fx <- dbeta(xgrid,piN,1-piP)
  plot(xgrid,fx,type="l")
}
plot.beta(.02,.8)

