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

run_SEIARD <- function(Tmax, POP, pars, init) {
  # formulas
  dS_dt <- dS ~ -a1*S*I/POP - a2*S*A/POP
  dE_dt <- dE ~  a1*S*I/POP + a2*S*A/POP - c1*E - c2*E
  dI_dt <- dI ~  c1*E - b1*I - d1*I
  dA_dt <- dA ~  c2*E - b2*A - d2*A
  dR_dt <- dR ~  b1*I + b2*A
  dD_dt <- dD ~  d1*I + d2*A
  # parameters
  pars$a1 <- ifelse(is.null(pars$a1), pars$a, pars$a1)
  pars$a2 <- ifelse(is.null(pars$a2), pars$a, pars$a2)
  pars$c1 <- ifelse(is.null(pars$c1), pars$c, pars$c1)
  pars$c2 <- ifelse(is.null(pars$c2), pars$c, pars$c2)
  pars$b1 <- ifelse(is.null(pars$b1), pars$b, pars$b1)
  pars$b2 <- ifelse(is.null(pars$b2), pars$b, pars$b2)
  pars$d1 <- ifelse(is.null(pars$d1), pars$d, pars$d1)
  pars$d2 <- ifelse(is.null(pars$d2), pars$d, pars$d2)
  epi = integrateODE(dS_dt, dE_dt, dI_dt, dA_dt, dR_dt, dD_dt,
                     a1=pars$a1, a2=pars$a2, c1=pars$c1, c2=pars$c2,
                     b1=pars$b1, b2=pars$b2, d1=pars$d1, d2=pars$d2,
                     S=init$S, E=init$E, I=init$I, A=init$A, R=init$R, D=init$D,
                     POP=POP, tdur=Tmax)
  list(
    S = sapply(1:Tmax, epi$S),
    E = sapply(1:Tmax, epi$E),
    I = sapply(1:Tmax, epi$I),
    A = sapply(1:Tmax, epi$A),
    R = sapply(1:Tmax, epi$R),
    D = sapply(1:Tmax, epi$D)
  )
}

plot_SEIARD <- function(Tmax, POP, pars, init, ylog=T) {
  # parameters
  init$E <- ifelse(is.null(init$E), 0, init$E)#/POP
  init$I <- ifelse(is.null(init$I), 0, init$I)#/POP
  init$A <- ifelse(is.null(init$A), 0, init$A)#/POP
  init$R <- ifelse(is.null(init$R), 0, init$R)#/POP
  init$D <- ifelse(is.null(init$D), 0, init$D)#/POP
  init$S <- POP - init$E - init$I - init$A - init$R - init$D
  # fit
  res <- run_SEIARD(Tmax=Tmax, POP=POP, pars = pars, init=init)
  res$Total = res$S + res$E + res$I + res$A + res$R + res$D
  # transform
  res <- rbind(
    data.frame(dates=1:Tmax, value=res$S, type='S'),
    data.frame(dates=1:Tmax, value=res$E, type='E'),
    data.frame(dates=1:Tmax, value=res$I, type='I'),
    data.frame(dates=1:Tmax, value=res$A, type='A'),
    data.frame(dates=1:Tmax, value=res$R, type='R'),
    data.frame(dates=1:Tmax, value=res$D, type='D')#,
    #data.frame(dates=1:Tmax, value=res$Total, type='Population')
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

# SEIRD with asymptotic cases
plot_SEIARD(100, 10000, list(a1 = .5, a2 = .8, c = .3, b1 = .5, b2 = .2, d = .05), list(I = 1), ylog = T)
