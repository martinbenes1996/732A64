# -*- coding: utf-8 -*-
"""Covid-19 symptoms' duration internal module.

Module containing operations with symptoms' duration.
Symptoms' duration data comes from https://doi.org/10.1038/s41467-020-20568-4.
Module fits various distributions to the data and chooses the best.

Example:
    List distributions with its parameters by

        distr = incubation.continuous()
    
    Get data summary with
    
        data_summary = incubation.data_summary()
        
    Get AIC over various distribution fits by
    
        aic = incubation.aic()
        
    Module constructs the plot of IFR with
    
        symptoms.plot()
        
    It is also possible to get simulations of IFR
    
        sim100 = symptoms.rvs(100)
        
"""
import numpy as np
import pandas as pd
from scipy.stats import lognorm,norm,gamma,beta
import matplotlib.pyplot as plt

def _symptoms_data():
    """Get symptoms' duration data (https://doi.org/10.1038/s41467-020-20568-4)."""
    # load
    path = 'data/41467_2020_20568_MOESM4_ESM.xlsx'
    df = pd.read_excel(path, engine='openpyxl')
    # parse
    x = df['duration of symptoms in days']\
        .apply(int)\
        .to_numpy()
    x[x == 0] = 1
    return x

def continuous():
    """Fit distributions to symptoms' duration data."""
    # fetch data
    x = _symptoms_data()
    # fit distributions
    return {
        'x': x,
        'norm': norm.fit(x),
        'lognorm': lognorm.fit(x, floc=0),
        'gamma': gamma.fit(x, floc=0)
    }

def data_summary():
    """Compute data summary."""
    # fetch data
    x = _symptoms_data()
    return {
        'mean': x.mean(),
        'ci95': np.quantile(x,[.025,.975]),
        'ci50': np.quantile(x,[.25,.75])
    }
    
def aic():
    """Get AIC of distribution fitted to the data. Lower is better."""
    # get distribution
    fit = continuous()
    # aic
    def _AIC(x, logpdf, *args, **kw):
        return 6 - 2*np.sum(logpdf(x, *args, **kw))
    return {
        'norm': _AIC(fit['x'], norm.logpdf, *fit['norm']),
        'lognorm': _AIC(fit['x'], lognorm.logpdf, *fit['lognorm']),
        'gamma': _AIC(fit['x'], gamma.logpdf, *fit['gamma'])
    }

def discrete(N = 40):
    """Discretizes symptoms' duration.
    
    Args:
        N (int): Number of time slots.
    """
    # fit distributions
    fit = continuous()
    params = fit['gamma']
    # discretize
    probs = []
    for i in range(N):
        P = gamma.cdf(i+1, *params) - gamma.cdf(i, *params)
        probs.append(P)
    distribution = pd.DataFrame({'x': range(N), 'Px': probs})
    return distribution

class plot:
    """Plotting of the module."""
    @staticmethod
    def continuous(save = False, name = 'img/parameters/symptoms.png'):
        """Plot symptoms' duration fitted distributions.
        
        After call, use `plt.show()` to show the figure.
    
        Args:
            save (bool, optional): Whether to save the figure, defaultly not.
            name (str, optional): Path to save the plot to.
        """
        def _pars(a,b): return '%.3f,%.3f' % (a,b)
        # get distribution
        fit = continuous()
        # generate pdf
        xgrid = np.linspace(1,40,100)
        y_norm = norm.pdf(xgrid, *fit['norm'])
        y_lognorm = lognorm.pdf(xgrid, *fit['lognorm'])
        y_gamma = gamma.pdf(xgrid, *fit['gamma'])
        # plot
        fig1, ax1 = plt.subplots()
        ax1.hist(fit['x'], bins = 40, alpha = .6, density=True)
        ax1.plot(xgrid, y_norm,
                 label=f"N({_pars(*fit['norm'][:2])})")
        ax1.plot(xgrid, y_lognorm,
                 label=f"LN({_pars(fit['lognorm'][0],fit['lognorm'][2])})")
                 #label = 'Lognorm(%.3f,%.3f)' % (fit['lognorm'][0],fit['lognorm'][2]))
        ax1.plot(xgrid, y_gamma,
                 label=f"Gamma({_pars(fit['gamma'][0],fit['gamma'][2])})")
                 #label = 'Gamma(%.3f,%.3f)' % (fit['gamma'][0],fit['gamma'][2]))
        ax1.xlabel('Days from symptom onset')
        ax1.ylabel('Density')
        ax1.legend()
        if save: fig1.savefig(name)
        
    @staticmethod
    def discrete(N = 40, save = False, name = 'img/parameters/symptoms_discrete.png'):
        """Plot symptoms' duration discretized Gamma distribution fit.
        
        After call, use `plt.show()` to show the figure.
    
        Args:
            N (int, optional): 
            save (bool, optional): Whether to save the figure, defaultly not.
            name (str, optional): Path to save the plot to.
        """
        def _pars(a,b): return '%.3f,%.3f' % (a,b)
        # get disribution
        fit = continuous()
        distribution = discrete(N = N)
        xgrid = np.linspace(0, N - 2, 1000)
        def find_X(i):
            idx = np.argmax(i < distribution.x) - 1
            if idx < 0: idx = distribution.shape[0]-1
            prob = distribution.loc[idx].Px
            return prob
        grid_probs = pd.Series(xgrid).apply(find_X)
        # plot
        fig1, ax1 = plt.subplots()
        ax1.plot(xgrid, grid_probs,
                 label=f"Discretized Gamma({_pars(fit['gamma'][0],fit['gamma'][2])}")
        ax1.xlabel('Days from symptom onset')
        ax1.ylabel('Density')
        ax1.legend()
        if save: fig1.savefig(name)
