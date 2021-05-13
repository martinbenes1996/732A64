# -*- coding: utf-8 -*-
"""Covid-19 incubation internal module.

Module containing operations with incubation period.
Incubation is modelled with Gamma distribution Gamma(5.807,).

Example:
    List distributions with its parameters by

        distr = incubation.continuous()
        
    Compute the quantile MSE of distributions to compare them

        MSEs = incubation.mse()
        
    Module constructs the plot of IFR distributions with
    
        incubation.plot.continuous()
        
    The discretized distribution of IFR is plotted by
    
        incubation.plot.discretized()
        
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import lognorm,gamma,erlang

class weibull:
    """Implementation of Weibull in style of `scipy.stats` module."""
    def pdf(x, n, a):
        """Weibull probability density function.
        
        Args:
            x (float): Point to estimate pdf at.
            n,a (float): Parameters of Weibull.
        Returns:
            (float): pdf of Weibull(n,a) at x.
        """
        return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)
    def ppf(q, n, a):
        """Weibull percent point function.
        
        Args:
            q (float): Quantile to estimate ppf at.
            n,a (float): Parameters of Weibull.
        Returns:
            (float): ppf of Weibull(n,a) at q.
        """
        return n * (-np.log(1 - np.array(q)))**(1/a)

def continuous():
    """Distribution of incubation period (https://doi.org/10.7326/M20-0504)."""
    return {
        'gamma': (5.807, 0, 1/.948),
        'lognorm': (.418,0,np.exp(1.621)),
        'weibull': (6.258,2.453),
        'erlang': (6,0,.88)
    }

def mse():
    """Compute MSE of quantiles from the distributions.
    
    Returns:
        (dict): MSEs of `lognorm`, `gamma`, `weibull` and `erlang` in a dict.
    """
    # quantiles
    quantile_points = [.05,.25,.5,.75,.95]
    def _qMSE(q):
        return np.mean((q - np.array([2.2,3.8,5.1,6.7,11.5]))**2)
    # distributions
    distr = continuous()
    lognorm_quantiles = lognorm.ppf(quantile_points, *distr['lognorm'])
    gamma_quantiles = gamma.ppf(quantile_points, *distr['gamma'])
    weibull_quantiles = weibull.ppf(quantile_points, *distr['weibull'])
    erlang_quantiles = erlang.ppf(quantile_points, *distr['erlang'])
    # MSE
    return {
        'lognorm': _qMSE(lognorm_quantiles),
        'gamma': _qMSE(gamma_quantiles),
        'weibull': _qMSE(weibull_quantiles),
        'erlang': _qMSE(erlang_quantiles)
    }

def discretized(N = 21):
    """Discretizes the continuous Gamma distribution. Slots of size 1.
    
    Args:
        N (int): Max number. By default 21.
    Returns:
        (pandas.DataFrame): Dataframe of x and Px over time.
    """
    probs = []
    for i in range(N):
        P = gamma.cdf(i+1, *continuous()['gamma']) - gamma.cdf(i, *continuous()['gamma'])
        probs.append(P)
    distribution = pd.DataFrame({'x': range(N), 'Px': probs})
    return distribution

class plot:
    """Plotting of the module."""
    @staticmethod
    def continuous(save = False, name = 'img/parameters/incubation.png'):
        """Plot incubation period distributions.
        
        After call, use `plt.show()` to show the figure.
    
        Args:
            save (bool, optional): Whether to save the figure, defaultly not.
            name (str, optional): Path to save the plot to.
        """
        def _pars(a,b): return '%.3f,%.3f' % (a,b)
        # grid
        xgrid = np.linspace(0, 14, 1000)
        # distributions
        distr = continuous()
        lognorm_pdf = lognorm.pdf(xgrid, *distr['lognorm'])
        gamma_pdf = gamma.pdf(xgrid, *distr['gamma'])
        weibull_pdf = weibull.pdf(xgrid, *distr['weibull'])
        erlang_pdf = erlang.pdf(xgrid, *distr['erlang'])
        # plot
        fig1, ax1 = plt.subplots()
        ax1.plot(xgrid, lognorm_pdf,
                 label=f"LN({_pars(np.log(distr['lognorm'][2]),distr['lognorm'][0])}^2)")#'LN(1.621,0.418)')
        ax1.plot(xgrid, gamma_pdf,
                 label=f"Gamma({_pars(distr['gamma'][0],1/distr['gamma'][2])})")
        ax1.plot(xgrid, weibull_pdf,
                 label=f"W({_pars(*distr['weibull'])})")
        ax1.plot(xgrid, erlang_pdf,
                 label=f"E({_pars(distr['erlang'][0],distr['erlang'][2])})")
        ax1.set_xlabel('Incubation period')
        ax1.set_ylabel('Density')
        ax1.legend()
        if save: fig1.savefig(name)

    @staticmethod
    def discrete(N = 21, save = False, name = 'img/parameters/incubation_discrete.png'):
        """Plot incubation period discretized Gamma distribution fit.
        
        After call, use `plt.show()` to show the figure.
    
        Args:
            N (int, optional): 
            save (bool, optional): Whether to save the figure, defaultly not.
            name (str, optional): Path to save the plot to.
        """
        def _pars(a,b): return '%5.3f,%5.3f' % (a,b)
        # get distribution
        distr = continuous()
        distribution = discretized(N = N)
        # grid
        xgrid = np.linspace(0, distribution.shape[0] - 1, 1000)
        def find_X(i):
            idx = np.argmax(i < distribution.x) - 1
            if idx < 0: idx = distribution.shape[0]-1
            prob = distribution.loc[idx].Px
            return prob
        grid_probs = pd.Series(xgrid).apply(find_X)
        # plot
        fig1, ax1 = plt.subplots()
        ax1.plot(xgrid, grid_probs,
                 label=f"Discretized Gamma({_pars(distr['gamma'][0],1/distr['gamma'][2])})")
        ax1.set_xlabel('Incubation period')
        ax1.set_ylabel('Density')
        ax1.legend()
        if save: fig1.savefig(name)
    