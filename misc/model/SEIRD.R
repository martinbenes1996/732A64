
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
    cat("obj -", pars, "\n")
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
      dbeta(data$I[t],
            priors$confirmed[1] + Tt * (res$E[t] + res$I[t]),
            priors$confirmed[2] + Tt * (1 - res$E[t] - res$I[t]))
    }))
    objV <- objV + sum(sapply(1:Tmax, function(t) {
      Tt <- data$T[t]
      dbeta(data$R[t],
            priors$recovered[1] + Tt * res$R[t],
            priors$recovered[2] + Tt * (1 - res$R[t]))
    }))
    objV <- objV + sum(sapply(1:Tmax, function(t) {
      Tt <- data$T[t]
      dbeta(data$D[t],
            priors$deaths[1] + Tt * res$D[t],
            priors$deaths[2] + Tt * (1 - res$D[t]))
    }))
    print(objV)
    -objV
  }
  initV <- c(
    a = rweibull(1, priors$a[1], priors$a[2]),
    c = rbeta(1, priors$c[1], priors$c[2]),
    b = rbeta(1, priors$b[1], priors$b[2]),
    d = rbeta(1, priors$d[1], priors$d[2])
  )
  optim(initV,#c(.25,.1,.2,.01),#c(.25,.1,.1,.005),
        objF, method='L-BFGS-B',
        data = data, priors = priors, POP = POP, init = init,
        lower = 1e-12, upper = 1-1e-12, control=list(maxit = 200))
}

model_bayes <- function(data, priors, POP, init) {
  
  as <- c()
  cs <- c()
  bs <- c()
  ds <- c()
  Is <- matrix(ncol = nrow(data))
  Rs <- matrix(ncol = nrow(data))
  Ds <- matrix(ncol = nrow(data))
  for(i in 1:100) {
    print(i)
    #cat("obj -", pars, "\n")
    # parameters
    Tmax <- nrow(data)
    # priors
    par_a <- rweibull(1, priors$a[1], priors$a[2])
    par_c <- rbeta(1, priors$c[1], priors$c[2])
    par_b <- rbeta(1, priors$b[1], priors$b[2])
    par_d <- rbeta(1, priors$d[1], priors$d[2])
    # optimize latent
    res <- run_SEIRD(
      Tmax=Tmax,
      pars=list(a=par_a,c=par_c,b=par_b,d=par_d),
      init=init,
      POP = POP)
    # emission
    It <- sum(sapply(1:Tmax, function(t) {
      Tt <- data$T[t]
      rbeta(1,
            priors$confirmed[1] + Tt * (res$I[t]),
            priors$confirmed[2] + Tt * (1 - res$I[t]))
    }))
    Rt <- sum(sapply(1:Tmax, function(t) {
      Tt <- data$T[t]
      rbeta(1,
            priors$recovered[1] + Tt * res$R[t],
            priors$recovered[2] + Tt * (1 - res$R[t]))
    }))
    Dt <- sum(sapply(1:Tmax, function(t) {
      Tt <- data$T[t]
      rbeta(1,
            priors$deaths[1] + Tt * res$D[t],
            priors$deaths[2] + Tt * (1 - res$D[t]))
    }))
    
    as <- c(par_a, as)
    cs <- c(par_c, cs)
    bs <- c(par_b, bs)
    ds <- c(par_d, ds)
    
    Is <- rbind(Is, It)
    Rs <- rbind(Rs, Rt)
    Ds <- rbind(Ds, Dt)
  }
  Is <- Is[-1,]
  Rs <- Rs[-1,]
  Ds <- Ds[-1,]
  
  list(a=as,c=cs,b=bs,d=ds,
       I=Is,R=Rs,D=Ds)
}

spline_SEIRD <- function(country, POP, priors,
                         date.min = '2020-03-15', date.max = '2020-04-30', window = 7) {
  
  date.min <- as.Date(date.min)
  date.max <- as.Date(date.max)
  date.i <- date.min
  while(date.i < date.max) {
    date.j <- date.i + window
    if(date.j > date.max) date.j <- date.max
    
    # todo
    # ...
    print(date.i)
    print(date.j)
    # fetch data
    data <- get.covid.data(country, POP, date.i, date.j)
    print(data)
    
    # fit model
    fit <- model_all(data$data, priors, POP, data$init)
    print(fit)
    
    date.i <- date.i + window
    break
  }
}






# fit model
#fit <- model_bayes(covid_data, priors, POP, init_state)
#I <- colMeans(fit$I)
#R <- colMeans(fit$R)
#D <- colMeans(fit$D)

# Infections
#par(mfrow=c(2,2))
#lim <- c(min(c(I*POP,covid_data$I*covid_data$T#,covid_data$I*POP
#               )),
#         max(c(I*POP,covid_data$I*covid_data$T#,covid_data$I*POP
#               )))
#plot(covid_data$dates,I * POP, col="red", type="l", ylim = lim)
#points(covid_data$dates, covid_data$I * POP)
#points(covid_data$dates, covid_data$I * covid_data$T)
# Recovered
#lim <- c(min(c(R*POP,covid_data$R*covid_data$Tcum#,covid_data$R*POP
#               )),
#         max(c(R*POP,covid_data$R*covid_data$Tcum#,covid_data$R*POP
#              )))
#plot(covid_data$dates,R * POP, col="green", type="l", ylim = lim)
#points(covid_data$dates, covid_data$R * POP)
#points(covid_data$dates, covid_data$R * covid_data$Tcum)
# Deaths
#lim <- c(min(c(D*POP,covid_data$D*covid_data$Tcum#,covid_data$D*POP
#               )),
#         max(c(D*POP,covid_data$D*covid_data$Tcum#,covid_data$D*POP
#               )))
#plot(covid_data$dates,D * POP, col="black", type="l", ylim = lim)
#points(covid_data$dates, covid_data$D * POP)
#points(covid_data$dates, covid_data$D * covid_data$Tcum)







# fit model
#fit <- model_all(covid_data, priors, POP, init_state)
#pars <- fit$par
# predict
#res <- run_SEIRD(Tmax=nrow(covid_data),
#                 pars=list(a = pars[1], c = pars[2], b = pars[3], d = pars[4]),
#                 init=init_state,
#                 POP = POP)


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
#lim <- c(min(c(res$D*POP,#covid_data$D*POP,
#               covid_data$D*covid_data$Tcum)),
#         max(c(res$D*POP,#covid_data$D*POP,
#               covid_data$D*covid_data$Tcum)))
#plot(covid_data$dates,res$D*POP, col="black", type="l", ylim = lim,
#     xlab = 'Date', ylab = 'Deaths')
#points(covid_data$dates, covid_data$D * POP)
#points(covid_data$dates, covid_data$D * covid_data$Tcum)





priors <- list(
  # parameters
  a = c(1,1),#c(1.63157,4.1984e-4),#c(1,1),#c(3,2),#3,20),#1.836352,36.5743),     # [S -> E]
  c = c(1,1),#c(3.47768,30),#51.05856,2.54478),#c(1,1),#c(2,20),#c(2.773984,12),#1.5192478),   # [E -> I]
  b = c(1,1),#c(2.718446,2186.1217,50),#540947),#c(3.402405,377.13360),   # [I -> R]
  d = c(1,1),#c(2.29282,9468.9540676,200),#20821096.605),#c(3.151443,5000),#5438.488333), # [I -> D]
  # emission
  confirmed = c(1,POP),#c(1,50000),#POP/1e2),#POP), # [I gets tested]
  recovered = c(1,POP*10),#1000),#POP/1e2),#POP), # [R gets tested]
  deaths = c(1,POP*10)#POP/3),#POP)      # [D gets tested]
)

spline_SEIRD('CZ', 1e7, priors, '2020-03-15', '2020-04-04', window = 5)


