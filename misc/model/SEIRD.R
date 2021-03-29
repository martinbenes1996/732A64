
library(GA)
library(ggplot2)
library(mosaicCalc)

run_SEIRD <- function(Tmax, POP, pars, init) {
  dS_dt <- dS ~ -a*S*I + f*R + g*S
  dE_dt <- dE ~  a*S*I - c*E
  dI_dt <- dI ~  c*E - b*I - e*I
  dR_dt <- dR ~  b*I - f*R
  dD_dt <- dD ~  e*I
  pars$e <- ifelse(is.null(pars$e), 0, pars$e)
  pars$g <- ifelse(is.null(pars$g), 0, pars$g)
  epi = integrateODE(dS_dt, dE_dt, dI_dt, dR_dt, dD_dt,
                     a=pars$a, c=pars$c, b=pars$b, e=pars$d, f=pars$e, g=pars$g,
                     #init_state = init,
                     S=init$S, E=init$E, I=init$I, R=init$R, D=init$D,
                     POP=POP, tdur=Tmax)
  list(
    S = sapply(1:Tmax, epi$S),
    E = sapply(1:Tmax, epi$E),
    I = sapply(1:Tmax, epi$I),
    R = sapply(1:Tmax, epi$R),
    D = sapply(1:Tmax, epi$D)
  )
}

plot_SEIRD <- function(Tmax, POP, pars, init, ylog=T) {
  # parameters
  init$E <- ifelse(is.null(init$E), 0, init$E)/POP
  init$I <- ifelse(is.null(init$I), 0, init$I)/POP
  init$R <- ifelse(is.null(init$R), 0, init$R)/POP
  init$D <- ifelse(is.null(init$D), 0, init$D)/POP
  init$S <- 1 - init$E - init$I - init$R - init$D
  # fit
  res <- run_SEIRD(Tmax=Tmax, POP=POP, pars = pars, init=init)
  # transform
  res <- rbind(
    data.frame(dates=1:Tmax, value=res$S, type='S'),
    data.frame(dates=1:Tmax, value=res$E, type='E'),
    data.frame(dates=1:Tmax, value=res$I, type='I'),
    data.frame(dates=1:Tmax, value=res$R, type='R'),
    data.frame(dates=1:Tmax, value=res$D, type='D')
  )
  # plot
  p <- res %>%
    ggplot() +
    geom_line(aes(x = dates, y = value, color = type), size=1) +
    xlab('Days') + ylab('Population ratio')
  
  if(ylog)
    p <- p + scale_y_log10()
  return(p)
}

#plot_SEIRD(100, 10000, list(a = .7, c = .4, b = .2, d = .05, e=0, g=0), list(I = 1), ylog = F)
#plot_SEIRD(150, 10000, list(a = .5, c = .4, b = .2, d = .05, e=0, g=0), list(I = 1), ylog = F)
#plot_SEIRD(150, 10000, list(a = .7, c = .1, b = .2, d = .05, e=0, g=0), list(I = 1), ylog = F)
#plot_SEIRD(100, 10000, list(a = .7, c = .4, b = .12, d = .05, e=0, g=0), list(I = 1), ylog = F)
#plot_SEIRD(100, 10000, list(a = .7, c = .4, b = .3, d = .05, e=0, g=0), list(I = 1), ylog = F)
#plot_SEIRD(100, 10000, list(a = .7, c = .4, b = .1, d = .15, e=0, g=0), list(I = 1), ylog = F)
#plot_SEIRD(500, 10000, list(a = .7, c = .4, b = .1, d = 1e-4, e=.2, g=.2), list(I = 1), ylog = F)



# load data
library(dplyr)
library(COVID19)
library(rstan)
library(bayesplot)

get.covid.data <- function(country, POP, date.min, date.max) {
  
  saturate <- function(v, max = 1, min = 0, eps=1e-5) {
    if(v >= max) return(max-eps)
    else if(v <= min) return(min+eps)
    else return(v)
  }
  # dates
  date.min <- as.Date(date.min)
  date.max <- as.Date(date.max)
  date.len <- as.integer(date.max - date.min) + 1
  # fetch data
  covid_data <- covid19('CZE', level = 1)
  covid_stats <- covid_data %>%
    dplyr::ungroup() %>%
    dplyr::transmute(country = iso_alpha_2, dates = as.Date(date),
                     I = c(confirmed[1],diff(confirmed)),
                     R = recovered,
                     D = deaths) %>%
    dplyr::mutate(I = ifelse(is.na(I), 0, I),
                  R = ifelse(is.na(R), 0, R),
                  D = ifelse(is.na(D), 0, D)) %>%
    dplyr::mutate(I = as.integer(I),
                  R = as.integer(R),
                  D = as.integer(D)) %>%
    dplyr::mutate(I = ifelse(I < 0, 0, I),
                  R = ifelse(R < 0, 0, R),
                  D = ifelse(D < 0, 0, D)) %>% # remove corrections
    dplyr::filter(country == 'CZ')
  # tests
  covid_tests <- read.csv('data/tests.csv', header=T) %>%
    dplyr::transmute(country, dates = as.Date(date), T = tests) %>%
    dplyr::mutate(Tcum = cumsum(T)) %>%
    dplyr::filter(country == 'CZE')
  covid_tests$country <- 'CZ'
  # crop to have same range
  date.act.min <- max(c(min(covid_stats$dates), min(covid_tests$dates)))
  date.act.max <- min(c(max(covid_stats$dates), max(covid_tests$dates)))
  covid_stats <- covid_stats %>%
    dplyr::filter(dates >= date.act.min, dates <= date.act.max)
  covid_tests <- covid_tests %>%
    dplyr::filter(dates >= date.act.min, dates <= date.act.max)
  covid_data <- covid_stats %>%
    dplyr::full_join(covid_tests, by = c('country','dates'))
  covid_stats_before <- covid_data %>%
    dplyr::filter(dates <= date.min)
  covid_data <- covid_data  %>%
    dplyr::filter(dates >= date.min, dates <= date.max)

  # prepare inputs
  covid_data <- covid_stats %>%
    dplyr::full_join(covid_tests, by = c('country','dates')) %>%
    dplyr::filter(dates >= date.min, dates <= date.max) %>%
    dplyr::mutate(I = I / T, D = D / Tcum, R = R / Tcum)
  init_I <- covid_data$I[1] * covid_data$T[1] * 1/2
  init_E <- covid_data$I[1] * covid_data$T[1] * 1/2
  init_state <- list(
    S = (POP - (init_I+init_E)
             - covid_data$R[1]* covid_data$Tcum[1]
             - covid_data$D[1]* covid_data$Tcum[1]) / POP,
    E = init_E / covid_data$T[1],
    I = init_I / covid_data$T[1],
    R = covid_data$R[1] * covid_data$Tcum[1] / POP,
    D = covid_data$D[1] * covid_data$Tcum[1] / POP
  )
  POP <- as.integer(POP)
  DAYS <- nrow(covid_data)
  
  return(list(
    data = covid_data,
    init = init_state,
    POP  = POP
  ))
}


model_all <- function(data, priors, POP, init) {
  
  objF <- function(pars, data, priors, POP, init_state) {
    # parameters
    Tmax <- nrow(data)
    a <- pars[1]; c <- pars[2]; b <- pars[3]; d <- pars[4]
    # priors
    objV <- 0
    #objV <- objV + dweibull(a, priors$a[1], priors$a[2], log = TRUE)#dweibull(a, priors$a[1], priors$a[2], log = TRUE)
    #objV <- objV + log(dbeta(c/priors$c[3], priors$c[1], priors$c[2])/priors$c[3])
    #objV <- objV + log(dbeta(b/priors$b[3], priors$b[1], priors$b[2])/priors$b[3])
    #objV <- objV + log(dbeta(d/priors$d[3], priors$d[1], priors$d[2])/priors$d[3])
    # optimize latent
    res <- run_SEIRD(
      Tmax=Tmax,
      pars=list(a=a,c=c,b=b,d=d),
      init=init_state,
      POP = POP)
    # emission
    objV <- objV + sum(sapply(1:Tmax, function(t) {
      Tt <- data$T[t]
      fx <- dbeta(data$I[t],
            priors$confirmed[1] + Tt * (res$E[t] + res$I[t]),
            priors$confirmed[2] + Tt * (1 - res$E[t] - res$I[t]), log=TRUE)
      #if(is.infinite(fx)) return(-1e5)
      fx
    }))
    #print(objV)
    objV <- objV + sum(sapply(1:Tmax, function(t) {
      Tt <- data$T[t]
      fx <- dbeta(data$R[t] + 1e-8,
            priors$recovered[1] + Tt * res$R[t],
            priors$recovered[2] + Tt * (1 - res$R[t]), log=TRUE)
      #if(is.infinite(fx)) return(-1e5)
      fx
    }))
    #print(objV)
    objV <- objV + sum(sapply(1:Tmax, function(t) {
      Tt <- data$T[t]
      fx <- dbeta(data$D[t] + 1e-8,
            priors$deaths[1] + Tt * res$D[t],
            priors$deaths[2] + Tt * (1 - res$D[t]), log=TRUE)
      #if(is.infinite(fx)) return(-1e5)
    }))
    #cat("obj (", objV, ") - ", pars, "\n")
    #objV <- ifelse(!is.nan(objV), objV, 0)
    #objV <- ifelse(!is.infinite(objV), objV, 1e5)
    objV / Tmax
    
  }
  initV <- c(
    #a = rweibull(1, priors$a[1], priors$a[2]),
    c = rbeta(1, priors$c[1], priors$c[2]),
    b = rbeta(1, priors$b[1], priors$b[2]),
    d = rbeta(1, priors$d[1], priors$d[2])
  )
  
  ga(type = "real-valued", 
           fitness =  function(x) objF(x,data,priors,POP,init),
           lower = c(0,.1,0,0), upper = c(.1,1,.1,.01),
           #suggestions = matrix(c(.015,.25,.002,.005),ncol=4,byrow=T),
           pmutation = .65, popSize = 70, maxiter = 40, run = 30)
  #optim(c(.34,.7,.039,.012),
  #      function(x) -objF(x,data,priors,POP,init_state),
  #      method='L-BFGS-B', lower = 1e-8, upper = 1-1e-8)
}

spline_SEIRD <- function(country, POP, priors, init_values = NULL,
                         date.min = '2020-03-20', date.max = '2020-04-30', window = 7) {
  # initialize data
  date.min <- as.Date(date.min)
  date.max <- as.Date(date.max)
  if(is.null(init_values))
    init_values <- get.covid.data(country, POP, date.min, date.min)$init
  # iterate
  date.i <- date.min
  dt <- c()
  params <- list(a = c(), c = c(), b = c(), d = c())
  latent <- list(S = c(), E = c(), I = c(), R = c(), D = c())
  vars <- list(T = c(), Tcum = c())
  while(date.i < date.max) {
    date.j <- date.i + window
    if(date.j > date.max) date.j <- date.max
    
    # todo
    # ...
    print(date.i)
    print(date.j)
    # fetch data
    data <- get.covid.data(country, POP, date.i, date.j)$data
    Tmax <- nrow(data)
    print(data)
    
    # fit model
    print(init_values)
    fit <- model_all(data, priors, POP, init_values)
    pars <- as.vector(fit@solution[1,])
    print(pars)
    
    # predict
    res <- run_SEIRD(Tmax=Tmax, init=init_values, POP=POP,
                     pars=list(a=pars[1], c=pars[2], b=pars[3], d=pars[4]))
    
    # Exposed
    par(mfrow=c(2,2))
    lim <- c(min(c(res$E*data$T,#data$I*POP,
                   data$I*data$T)),
             max(c(res$E*data$T,#data$I*POP,
                   data$I*data$T)))
    plot(data$dates,res$E*data$T, col="orange", type="l",ylim=lim,xlab='Date',ylab='Exposed')
    #points(data$dates,data$I*POP)
    points(data$dates,data$I*data$T)
    # Infections
    lim <- c(min(c(res$I*data$T,#data$I*POP,
                   data$I*data$T)),
             max(c(res$I*data$T,#data$I*POP,
                   data$I*data$T)))
    plot(data$dates,res$I*data$T, col="red", type="l",ylim=lim,xlab='Date',ylab='Infected')
    #points(data$dates,data$I * POP)
    points(data$dates,data$I * data$T)
    # Recovered
    lim <- c(min(c(res$R*data$Tcum,#data$R*POP,
                   data$R*data$Tcum)),
             max(c(res$R*data$Tcum,#data$R*POP,
                   data$R*data$Tcum)))
    plot(data$dates,res$R*data$Tcum, col="green", type="l", ylim = lim,
         xlab = 'Date', ylab = 'Recovered')
    #points(data$dates,data$R*POP)
    points(data$dates,data$R*data$Tcum)
    # Deaths
    lim <- c(min(c(res$D*data$Tcum,#data$D*POP,
                   data$D*data$Tcum)),
             max(c(res$D*data$Tcum,#data$D*POP,
                   data$D*data$Tcum)))
    plot(data$dates,res$D*data$Tcum, col="black",type="l",ylim=lim,xlab='Date',ylab='Deaths')
    #points(data$dates,data$D*POP)
    points(data$dates,data$D*data$Tcum)                
    
    dt <- c(dt, sapply(data$dates, format))
    params$a <- c(params$a, rep(pars[1],Tmax))
    params$c <- c(params$c, rep(pars[2],Tmax))
    params$b <- c(params$b, rep(pars[3],Tmax))
    params$d <- c(params$d, rep(pars[4],Tmax))
    latent$S <- c(latent$S, res$S)
    latent$E <- c(latent$E, res$E)
    latent$I <- c(latent$I, res$I)
    latent$R <- c(latent$R, res$R)
    latent$D <- c(latent$D, res$D)
    vars$T <- c(vars$T, data$T)
    vars$Tcum <- c(vars$Tcum, data$Tcum)
    
    # increment
    date.i <- date.i + window
    # change init values
    init_values <- list(S = 1 - (res$E[Tmax]+res$I[Tmax]+res$R[Tmax]+res$D[Tmax]),
                        E = res$E[Tmax], I = res$I[Tmax], R = res$R[Tmax], D = res$D[Tmax])
  }
  
  return(list(
    data = data.frame(
      dates = dt,
      a = params$a,
      c = params$c,
      b = params$b,
      d = params$d,
      S = latent$S,
      E = latent$E,
      I = latent$I,
      R = latent$R,
      D = latent$D,
      T = vars$T,
      Tcum = vars$Tcum
    ),
    POP = POP
  ))
}





priors <- list(
  # parameters
  a = c(1,1), # [S -> E]
  c = c(1,1), # [E -> I]
  b = c(1,1), # [I -> R]
  d = c(1,1), # [I -> D]
  # emission
  confirmed = c(1,1e2), # [I gets tested]
  recovered = c(1,1e2), # [R gets tested]
  deaths = c(1,2)      # [D gets tested]
)


POP <- 1e7
ga <- spline_SEIRD('CZ', POP, priors, list(S=750/1000,E=50/1000,I=200/1000,R=0,D=0),
                   '2020-03-15', '2020-05-31', window = 10)

data.ga <- ga$data %>%
  dplyr::mutate(dates = as.Date(dates)) %>%
  dplyr::transmute(
    dates = as.Date(dates),
    a, b, c, d,
    S = S,
    E = E * T,
    I = I * T,
    R = R * Tcum,
    D = D * Tcum
  )

data.covid <- get.covid.data('CZ', 1e7,'2020-03-15', '2020-05-31')$data %>%
  dplyr::transmute(
    dates,
    confirmed = I * T,
    recovered = R * Tcum,
    deaths = D * Tcum) %>%
  dplyr::full_join(data.ga, by = 'dates')

data.covid %>%
  #dplyr::mutate(confirmed = cumsum(confirmed), I = cumsum(I)) %>%
  ggplot() +
  geom_line(aes(x = dates, y = confirmed), color='blue', size=1.5) +
  geom_line(aes(x = dates, y = I), color='red', size=1.5)
data.covid %>%
  ggplot() +
  geom_line(aes(x = dates, y = recovered), color='blue', size=1.5) +
  geom_line(aes(x = dates, y = R), color='red', size=1.5)
data.covid %>%
  ggplot() +
  geom_line(aes(x = dates, y = deaths), color='blue', size=1.5) +
  geom_line(aes(x = dates, y = D), color='red', size=1.5)


# fit model
#fit <- model_all(covid_data, priors, POP, init_state)
#pars <- fit$par
# predict
#data_1503_2003 <- get.covid.data('CZ', 1e7,'2020-03-15', '2020-04-30')
#covid_data <- data_1503_2003$data
#POP <- 1e7

#par1 <- list(a=.0137, c=.1156, b=.0285, d=.0348)
#par2 <- list(a=.9323,c=.7857,b=.0271,d=.0202)
#par3 <- list(a=.95382,c=.97893,b=.00347,d=.00178)
#par4 <- list(a=.01723,c=.00803,b=.00737,d=.00505)
#par5 <- list(a=.01281,c=.22021,b=.00202,d=.04218)
#par5 <- list(a=.01281,c=.22021,b=.00202,d=.004218)
#par6 <- list(a=.09524,c=.84633,b=.001,#647,
#             d=.0005)#21)
#res <- run_SEIRD(Tmax=nrow(data_1503_2003$data),
#                 pars=par6,#pars[1], c = pars[2], b = pars[3], d = pars[4]),
#                 init=data_1503_2003$init,
#                 POP = data_1503_2003$POP)


# Susceptible
#plot(covid_data$dates,res$S * POP, col="gray", type="l")
# Exposed
#par(mfrow=c(2,2))
#lim <- c(min(c(res$E*POP,covid_data$I*POP,
#               covid_data$I*covid_data$T)),
#         max(c(res$E*POP,covid_data$I*POP,
#               covid_data$I*covid_data$T)))
#plot(covid_data$dates,res$E*POP, col="orange", type="l", ylim = lim,
#     xlab = 'Date', ylab = 'Exposed')
#points(covid_data$dates, covid_data$I * POP)
#points(covid_data$dates, covid_data$I * covid_data$T)
# Infections
#lim <- c(min(c(res$I*POP,covid_data$I*POP,
#               covid_data$I*covid_data$T)),
#         max(c(res$I*POP,covid_data$I*POP,
#               covid_data$I*covid_data$T)))
#plot(covid_data$dates,res$I*POP, col="red", type="l", ylim = lim,
#     xlab = 'Date', ylab = 'Infected')
#points(covid_data$dates, covid_data$I * POP)
#points(covid_data$dates, covid_data$I * covid_data$T)
# Recovered
#lim <- c(min(c(res$R*POP,#covid_data$R*POP,
#               covid_data$R*covid_data$Tcum)),
#         max(c(res$R*POP,#covid_data$R*POP,
#               covid_data$R*covid_data$Tcum)))
#plot(covid_data$dates,res$R*POP, col="green", type="l", ylim = lim,
#     xlab = 'Date', ylab = 'Recovered')
#points(covid_data$dates, covid_data$R * POP)
#points(covid_data$dates, covid_data$R * covid_data$Tcum)
# Deaths
#lim <- c(min(c(res$D*POP,covid_data$D*POP,
#               covid_data$D*covid_data$Tcum)),
#         max(c(res$D*POP,covid_data$D*POP,
#               covid_data$D*covid_data$Tcum)))
#plot(covid_data$dates,res$D*POP, col="black", type="l", ylim = lim,
#     xlab = 'Date', ylab = 'Deaths')
#points(covid_data$dates, covid_data$D * POP)
#points(covid_data$dates, covid_data$D * covid_data$Tcum)

