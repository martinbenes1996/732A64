library(ggplot2)
library(mosaicCalc)

run_SEIRD <- function(Tmax, POP, pars, init) {
  dS_dt <- dS ~ -a*S*I/POP + f*R - v*POP
  dE_dt <- dE ~  a*S*I/POP - c*E
  dI_dt <- dI ~  c*E - b*I
  dR_dt <- dR ~  b*(1-e)*I - f*R
  dD_dt <- dD ~  b*e*I
  pars$e <- ifelse(is.null(pars$e), 0, pars$e)
  pars$v <- ifelse(is.null(pars$v), 0, pars$v)
  epi = integrateODE(dS_dt, dE_dt, dI_dt, dR_dt, dD_dt,
                     a=pars$a, c=pars$c, b=pars$b, e=pars$d, f=pars$e, v=pars$v,
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
  init$E <- ifelse(is.null(init$E), 0, init$E)#/POP
  init$I <- ifelse(is.null(init$I), 0, init$I)#/POP
  init$R <- ifelse(is.null(init$R), 0, init$R)#/POP
  init$D <- ifelse(is.null(init$D), 0, init$D)#/POP
  init$S <- POP - init$E - init$I - init$R - init$D
  # fit
  res <- run_SEIRD(Tmax=Tmax, POP=POP, pars = pars, init=init)
  res$Total = res$S + res$E + res$I + res$R + res$D
  print(res$D)
  # transform
  res <- rbind(
    data.frame(dates=1:Tmax, value=res$S, type='S'),
    data.frame(dates=1:Tmax, value=res$E, type='E'),
    data.frame(dates=1:Tmax, value=res$I, type='I'),
    data.frame(dates=1:Tmax, value=res$R, type='R'),
    data.frame(dates=1:Tmax, value=res$D, type='D'),
    data.frame(dates=1:Tmax, value=res$Total, type='Population')
  )
  # plot
  p <- res %>%
    ggplot() +
    geom_line(aes(x = dates, y = value, color = type), size=1) +
    xlab('Days') + ylab('Population ratio') +
    theme(panel.background = element_rect(fill = 'white', colour = 'grey'))
  
  if(ylog)
    p <- p + scale_y_log10()
  return(p)
}

run_LotkaVolterra <- function(Tmax, POP, pars, init) {
  dS_dt <- dS ~ -a*S*I/POP + b*S
  dI_dt <- dI ~  a*S*I/POP - b*I
  epi = integrateODE(dS_dt, dI_dt,
                     a=pars$a, b=pars$b, S=init$S, I=init$I, POP=POP, tdur=Tmax)
  list(
    S = sapply(1:Tmax, epi$S),
    I = sapply(1:Tmax, epi$I)
  )
}

plot_LotkaVolterra <- function(Tmax, POP, pars, init, ylog=T) {
  # parameters
  init$I <- ifelse(is.null(init$I), 0, init$I)
  init$S <- POP - init$I
  print(init)
  # fit
  res <- run_LotkaVolterra(Tmax=Tmax, POP=POP, pars = pars, init=init)
  res$POP = res$S + res$I
  # transform
  res <- rbind(
    data.frame(dates=1:Tmax, value=res$S, type='Prey (S)'),
    data.frame(dates=1:Tmax, value=res$I, type='Predator (I)'),
    data.frame(dates=1:Tmax, value=res$POP, type='Total (S+I)')
  )
  # plot
  p <- res %>%
    ggplot() +
    geom_line(aes(x = dates, y = value, color = type), size=1) +
    xlab('Days') + ylab('Population') +
    theme(panel.background = element_rect(fill = 'white', colour = 'grey'))
  
  if(ylog)
    p <- p + scale_y_log10()
  return(p)
}

# epidemic with herd immunity
plot_SEIRD(100, 10000, list(a = .8, c = .3, b = .3, d = .05), list(I = 1), ylog = F)
# epidemic without herd immunity
plot_SEIRD(100, 10000, list(a = .9, c = .3, b = .2, d = .05), list(I = 1), ylog = F)
# no-epidemic
plot_SEIRD(100, 10000, list(a = .5, c = .3, b = .5, d = .05), list(I = 1), ylog = T)

# Lotka-Volterra oscillation
plot_LotkaVolterra(150, 50, list(a = .1, b = .1), list(I = 20), ylog = F)
# epidemic with herd immunity and non-permanent immunity
plot_SEIRD(300, 10000, list(a = .8, c = .3, b = .3, d = .05, e = 1/30), list(I = 1), ylog = F)
# epidemic with vaccination
plot_SEIRD(100, 10000, list(a = .8, c = .3, b = .3, d = .05, v = 25/10000), list(I = 1), ylog = F)
