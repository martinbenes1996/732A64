# -*- coding: utf-8 -*-
"""Incubation internal module.

Module containing operation for incubation period.
Incubation is modelled with Gamma distribution Gamma(5.807,).

Example:
    Module constructs the plot of IFR with
    
        ifr.plot()
        
    It is also possible to get simulations of IFR
    
        sim100 = ifr.rvs(100)
        

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import lognorm,gamma,erlang
import scipy
#plt.rcParams["figure.figsize"] = (12,10)
#plt.rcParams.update({'font.size': 18})

# Weibull
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
    @staticmethod
    def continuous(save = False, name = 'img/parameters/incubation.png'):
        """Plot incubation period fitted distributions.
        
        After call, use `plt.show()` to show the figure.
    
        Args:
            save (bool, optional): Whether to save the figure, defaultly not.
            name (str, optional): Path to save the plot to.
        """
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
        ax1.plot(xgrid, lognorm_pdf, label='LN(1.621,0.418)')
        ax1.plot(xgrid, gamma_pdf, label='Gamma(5.807,0.948)')
        ax1.plot(xgrid, weibull_pdf, label='W(2.453,6.258)')
        ax1.plot(xgrid, erlang_pdf, label='E(6,0.88)')
        ax1.set_xlabel('Incubation period')
        ax1.set_ylabel('Density')
        ax1.legend()
        if save: fig1.savefig(name)

    @staticmethod
    def discretized(N = 21, save = False, name = 'img/parameters/incubation_discrete.png'):
        """Plot incubation period discretized Gamma distribution fit.
        
        After call, use `plt.show()` to show the figure.
    
        Args:
            N (int, optional): 
            save (bool, optional): Whether to save the figure, defaultly not.
            name (str, optional): Path to save the plot to.
        """
        # get distribution
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
        ax1.plot(xgrid, grid_probs, label='Discretized Gamma(5.807,0.948)')
        ax1.set_xlabel('Incubation period')
        ax1.set_ylabel('Density')
        ax1.legend()
        if save: fig1.savefig(name)
    