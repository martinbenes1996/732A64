# -*- coding: utf-8 -*-
"""Model emission component.

Module containing operations of emission component in HMM.

Example:
    Emission model is executed with
    
        emission.emission(
            xbar = np.array([.3,.4,.4,.3,.3]),
            T = np.array([20,30,35,30,35]),
            a = 2,
            b = 3
        )
    
    Get emission model negative log likelihood with
    
        emission.emission_objective(
            xbar = np.array([.3,.4,.4,.3,.3]),
            T = np.array([20,30,35,30,35]),
            a = 2,
            b = 3
        )
        
    Construct plot of emission model with moving average transition with
    
        emission.plot_MA()
        
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta
from statsmodels.tsa.arima_process import ArmaProcess

def emission(xbar, T, a, b):
    """Simulation of emission model.
    
    Args:
        xbar (np.array): Confirmed tests ratio.
        T (np.array): Number of performed tests
        a,b (float): Prior parameters.
    """
    # parameters
    alpha_ = (a + T * xbar)
    beta_ = (b + T * (1 - xbar))
    # simulate
    D = T.shape[0]
    draw = np.zeros((D,))
    for i in range(D):
        try:
            draw[i] = beta.rvs(alpha_[i], beta_[i], size = 1)
        except:
            print(i, a, b, T[i], xbar[i], alpha_[i], beta_[i])
            raise
    # result
    return draw

def emission_objective(infected, xbar, T, a, b):
    """Objective value of emission model for `infected`.
    
    Args:
        infected (np.array): Infected to be tested against the emission.
        xbar (np.array): Confirmed tests ratio.
        T (np.array): Number of performed tests
        a,b (float): Prior parameters.
    """
    # parameters
    alpha_ = (a + T * xbar)
    beta_ = (b + T * (1 - xbar))
    # compute loglik
    D = T.shape[0]
    logL = 0
    for i in range(D):
        logL += beta.logpdf(infected[i] + 1e-11, # log stability
                            alpha_[i], beta_[i])
    # result
    return -logL

def plot_MA(maxit = 100, N = 365, T = 1000, save=False, name='img/results/emission.png'):
    """Generate emission model output with transition replaced with MA.
    
    Transition model is defined as
    
        z_t = z_{t-1} + 3*sin(2*pi*t/T)
    
    Args:
        maxit (int): Number of samples
        N (int): Size of time range.
        T (int): Constant test size.
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    # create model    
    MA = ArmaProcess(ma = [.2,-.4,.2,-.7])
    # iterate
    z = np.zeros((maxit, N))
    x = np.zeros((maxit, N))
    for i in range(maxit):
        # create z[t] sample
        z[i,:] = MA.generate_sample(nsample=N) + 3*np.sin(np.array(range(N))/N*2*np.pi)
        z[i,:] = (z[i,:] - z[i,:].min()) / (z[i,:].max() - z[i,:].min())
        # create x[t] sample
        x[i,:] = emission(z[i,:], np.array([T for i in range(N)]), 1, 50)
    def get_mu_ci(ts):
        mu = ts.mean(axis = 0)
        ci = np.quantile(ts, [.025,.975],axis=0)
        return mu,ci
    z_mu,z_ci = get_mu_ci(z)
    x_mu,x_ci = get_mu_ci(x)
    # plot
    fig1, ax1 = plt.subplots()
    ax1.plot(range(N), z_mu, color='red', label='z[t]')
    ax1.fill_between(range(N), z_ci[0,:], z_ci[1,:], color = 'red', alpha = .1)
    ax1.plot(range(N), x_mu, color='blue', label='x[t]')
    ax1.fill_between(range(N), x_ci[0,:], x_ci[1,:], color = 'blue', alpha = .1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    if save: fig1.savefig(name)
