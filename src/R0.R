
# import
library(tidyr)
library(ggplot2)
library(plotly)
library(patchwork)
library(EpiEstim)
library(COVID19)

#serial.pars <- c(11.39,2.504)
serial.pars <- c(1.901,0.41781)
#'
#'
plot_serial_interval <- function(lim.max = 20, show.hist = TRUE) {
  # continuous distribution
  xgrid <- seq(0.01,lim.max,length.out = 1000)
  gamma.continuous <- data.frame(
    x = xgrid,
    y = dgamma(xgrid,serial.pars[1],serial.pars[2])
  )
  # discrete distribution
  y.discrete <- pgamma(1:(lim.max+1),serial.pars[1],serial.pars[2]) -
                pgamma(0:lim.max,serial.pars[1],serial.pars[2])
  gamma.discrete <- data.frame(
    x = c(0,0,sapply(1:lim.max, rep, 2), lim.max+1),
    y = c(0,sapply(y.discrete, rep, 2))
  )
  # plot
  p <- gamma.discrete %>%
    ggplot(mapping = aes(x = x, y = y))
  if(show.hist)
    p <- p + geom_polygon(alpha = .7)
  p <- p +
    geom_line(data = gamma.continuous, size = 1) +
    labs(x = 'Serial interval (days)', y = 'Density')
  plot(p)
}

#'
#'
get_restrictions <- function(country) {
  restrictions <- read.csv('data/calendar/restrictions.csv')
  restrictions <- restrictions %>%
    dplyr::filter(Country == country) %>%
    dplyr::mutate(Date = as.Date(Date), y = 1)
  return(restrictions)
}

#'
#'
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

#'
#'
estimate_reproduction <- function(incid, date.min) {
  lim.max <- 20
  y.discrete <- pgamma(1:(lim.max+1),serial.pars[1],serial.pars[2]) -
                pgamma(0:lim.max,serial.pars[1],serial.pars[2])
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

#'
#'
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

#'
#'
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

#'
#'
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

#'
#'
plot_trace_R0_box <- function(country = c('CZ','IT','SE','PL'),
                              date.min = '2020-03-15', date.max = '2021-01-31') {
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
    ggplot(mapping = aes(x = Month, y = R0, color = Country)) +
    geom_boxplot(width=0.5) #+
    #facet_wrap(~Country)
}

plot_trace_test_R0_box <- function(countries = c('CZE','ITA','SWE','POL'),
                              date.min = '2020-03-15', date.max = '2021-01-31') {
  # daily incidence
  covid_data <- covid19(countries, level = 1)
  covid_tests <- read.csv('data/tests.csv', header=T) %>%
    dplyr::transmute(country, dates = as.Date(date), I = tests) %>%
    dplyr::mutate(I = ifelse(is.na(I), 0, I)) %>%
    dplyr::mutate(I = as.integer(I)) %>%
    dplyr::mutate(I = ifelse(I < 0, 0, I)) %>% # remove corrections
    dplyr::filter(dates <= as.Date(date.max), dates > as.Date(date.min)) %>%
    dplyr::filter(country %in% countries)
  # tests
  Rs <- NA
  Rs.empty <- T
  for(country in unique(covid_tests$country)) {
    country_stats <- covid_tests[covid_tests$country == country,] %>%
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
    dplyr::mutate(Month = format(Date,'%Y-%m')) %>%
    dplyr::mutate(R0 = ifelse(R0 < 2.5, R0, 2.5)) %>%
    dplyr::mutate(R0 = ifelse(R0 > .7, R0, .7))
  Rs %>%
    dplyr::mutate(Test_R0 = R0) %>%
    ggplot(mapping = aes(x = Month, y = Test_R0, color = Country)) +
    geom_boxplot(width=0.5) #+
  #facet_wrap(~Country)
}


plot_trace_test_R0_box()

