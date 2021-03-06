# -*- coding: utf-8 -*-
"""Module with main functionality for model.

Module containing main model functionality.

Example:
    Compute the negative log likelihood of data with given parameters with

        posterior.posterior_objective()
        
    Simulate from the HMM model with
    
        posterior.simulate_posterior()

    Simulates the posterior with the given parameters, saves and plots it with
    
        posterior.plot_posterior()
        
    Compute simulation mean from simulated samples with
    
        posterior.compute_sim_mean()
    
    Compute simulation CI from simulated samples with
    
        posterior.compute_sim_ci()
        
    Run example pandemic with
    
        posterior.run_covid_characteristics()

"""
from datetime import datetime,timedelta
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys
pd.options.mode.chained_assignment = None
sys.path.append('src')
from covid19 import src
from demographic import population
from emission import emission,emission_objective
from transition import transition

def _parse_params(params, fixparams):
    """Joins optimized parameters with the parameters set to be fixed.
    
    Args:
        params (list): Optimized parameters.
        fixparams (list): Fixed parameters.
    """
    # no fixparams set
    if fixparams is None:
        return params
    ptr = 0
    # a
    a = fixparams[0]
    if a is None:
        a = params[ptr]
        ptr += 1
    # c
    c = fixparams[1]
    if c is None:
        c = params[ptr]
        ptr += 1 
    # b
    b = fixparams[2]
    if b is None:
        b = params[ptr]
        ptr += 1
    # d
    d = fixparams[3]
    if d is None:
        d = params[ptr]
    # result
    return a,c,b,d

_data = {}
def _posterior_data(region, dates, weekly=False):
    """Load data for the model.
    
    Args:
        region (str): Region for the data.
        dates (tuple (2)): Date range of the data.
        weekly (bool, optional): Weekly time step if True, otherwise daily.
    """
    global _data
    # get data
    if region not in _data:
        x = src.get_data(region, weekly=weekly)
        # parse tests
        x['cumtests'] = x.tests.cumsum()
        x['tests'] = x.tests.replace({0:1})
        x['cumtests'] = x.cumtests.replace({0:1})
        _data[region] = x
    x = _data[region]
    # filter by dates
    x = x[(x.date >= dates[0]) & (x.date <= dates[1])]\
        .fillna(0)
    return x

def posterior_objective(params, region, dates, initial, fixparams = None, weekly=False,
                        attributes = 'IRD', parI = (1,1), parR = (1,1), parD = (1,1)):
    """Objective function of the HMM model for optimization.
    
    Args:
        params (list): Optimized parameters.
        region (str): Region for the data.
        dates (tuple (2)): Date range of the data.
        initial (dict): Initial values in dict with keys S,E,I,R,D.
        fixparams (list): Fixed parameters.
        weekly (bool, optional): Weekly time step if True, otherwise daily.
        attributes (str, optional): Attributes used for optimization, 'I', 'R' or 'D'.
        parI (tuple (2)): Prior parameters for emission model I.
        parR (tuple (2)): Prior parameters for emission model R.
        parD (tuple (2)): Prior parameters for emission model D.
    """
    x = _posterior_data(region, dates, weekly=weekly)
    POP = population.get_population(region)
    # construct params dataframe
    a,c,b,d = _parse_params(params, fixparams)
    params = pd.DataFrame({'start': [dates[0]], 'end': [dates[1]],
                            'a': [a], 'b': [b], 'c': [c], 'd': [d]})
    # compute score
    D = (dates[1] - dates[0]).days + 1
    score = 0
    latent = transition(POP, initial, params)
    latent.loc[latent.I < 0,'I'] = 0
    x = x.merge(latent, how='left', on=['date'])
    if 'I' in attributes:
        score += emission_objective(x.confirmed.to_numpy() / x.tests.to_numpy(),
                                    np.abs(x.I.to_numpy()), x.tests.to_numpy(), *parI)
    if 'D' in attributes:
        score += emission_objective(x.deaths.cumsum().to_numpy() / x.cumtests.to_numpy(),
                                    x.D.to_numpy(), x.cumtests.to_numpy(), *parD)
    if 'R' in attributes:
        score += emission_objective(x.recovered.cumsum().to_numpy() / x.cumtests.to_numpy(),
                                    x.R.to_numpy(), x.cumtests.to_numpy(), *parR)
    return score / D

def simulate_posterior(region, params, dates, initial, N = 1000, weekly = False,
                       parI = (1,1), parR = (1,1),parD = (1,1), random_params = False):
    """Simulate from the HMM model.
    
    Args:
        region (str): Region for the data.
        params (list): Optimized parameters.
        dates (tuple (2)): Date range of the data.
        initial (dict): Initial values in dict with keys S,E,I,R,D.
        N (int): Number of samples.
        weekly (bool, optional): Weekly time step if True, otherwise daily.
        parI (tuple (2)): Prior parameters for emission model I. By default (1,1).
        parR (tuple (2)): Prior parameters for emission model R. By default (1,1).
        parD (tuple (2)): Prior parameters for emission model D. By default (1,1).
        random_params (bool, optional): Bayesian parameters if True, otherwise single point.
    """
    x = _posterior_data(region, dates, weekly=weekly)\
        .reset_index(drop = True)
    POP = population.get_population(region)
    # filter param
    params = params[params.start <= dates[1]]
    if (params.end > dates[1]).any():
        params.loc[params.end > dates[1], 'end'] = dates[1]
    latent = transition(POP, initial, params, random_params=random_params)
    xx = x.merge(latent, how='left', on=['date'])
    Dw = xx.shape[0]
    D = (dates[1] - dates[0]).days + 1
    sim_lat = np.zeros((5,N,Dw))
    sim_obs = np.zeros((5,N,Dw))
    for i in range(N):
        if i == 0 or (i+1) % 100 == 0:
            print('%4d / %d' % (i+1,N))
        # transition
        latent = transition(POP, initial, params, random_params=random_params)
        latent[latent.I < 0]['I'] = 0
        xx = x.merge(latent, how='left', on=['date'])
        xx.tests = xx['tests'].apply(lambda t: t if t >= 0 else 1)
        sim_lat[:,i,:] = xx[['S','E','I','R','D']].to_numpy().T
        # emission
        try:
            sim_obs[2,i,:] = emission(np.abs(xx.I.to_numpy()), xx.tests.to_numpy(), *parI)
        except:
            print(xx.I)
            print(xx.tests)
            raise
        sim_obs[3,i,:] = emission(xx.R.to_numpy(), xx.cumtests.to_numpy(), *parR)
        sim_obs[4,i,:] = emission(xx.D.to_numpy(), xx.cumtests.to_numpy(), *parD)
    # spare last
    last_values = sim_lat[:,:,-1].mean(axis = 1)
    # denormalize probability
    sim_lat[1:3,:,:] = sim_lat[1:3,:,:] * x.tests.to_numpy()
    sim_lat[3:5,:,:] = sim_lat[3:5,:,:] * x.cumtests.to_numpy()
    sim_obs[1:3,:,:] = sim_obs[1:3,:,:] * x.tests.to_numpy()
    sim_obs[3:5,:,:] = sim_obs[3:5,:,:] * x.cumtests.to_numpy()
    return (sim_lat, sim_obs), last_values

def plot_posterior(region, params, dates, initial, N = 1000,
                   parI = (1,1), parR = (1,1), parD = (1,1), random_params=False):
    """Simulates the posterior with the given parameters, saves and plots it.
    
    Args:
        region (str): Region for the data.
        params (list): Optimized parameters.
        dates (tuple (2)): Date range of the data.
        initial (dict): Initial values in dict with keys S,E,I,R,D.
        N (int): Number of samples.
        parI (tuple (2)): Prior parameters for emission model I. By default (1,1).
        parR (tuple (2)): Prior parameters for emission model R. By default (1,1).
        parD (tuple (2)): Prior parameters for emission model D. By default (1,1).
        random_params (bool, optional): Bayesian parameters if True, otherwise single point.
    """
    # run simulation
    (sim_lat,sim_obs),_ = simulate_posterior(
        region=region, params=params, dates=dates, N=N, initial=initial,
        parI=parI, parR=parR, parD=parD, random_params=random_params
    )
    _plot_posterior(sim=(sim_lat,sim_obs), region=region, dates=dates)

def _plot_confirmed(mean, ci, x):
    """Plots the predicted (latent, observed) confirmed vs. data.

    Args:
        mean (tuple (2) of np.array): Mean of simulations.
        ci (tuple (2) of np.array): CI of simulation mean.
        x (pandas.DataFrame): Data
    """
    # data
    sim_mean,sim_obs_mean = mean
    if ci is None: ci = (None,None)
    sim_ci,sim_obs_ci = ci if ci is not None else (None,None)
    # plot
    fig1, ax1 = plt.subplots()
    ax1.plot(x.date, sim_mean[2,:], color='orange', label='Infected z[t]')
    if sim_ci is not None:
        ax1.fill_between(x.date, sim_ci[0,2,:], sim_ci[1,2,:], color = 'orange', alpha = .25)
    ax1.plot(x.date, sim_obs_mean[2,:], color='red', label='Infected x[t]')
    if sim_obs_ci is not None:
        ax1.fill_between(x.date, sim_obs_ci[0,2,:], sim_obs_ci[1,2,:], color = 'red', alpha = .1)
    if 'confirmed' in x.columns:
        ax1.plot(x.date, x.confirmed, color = 'blue', label='Data')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Infected')
    ax1.set_yscale('log')
    ax1.legend()

def _plot_recovered(mean, ci, x):
    """Plots the predicted (latent, observed) recovered vs. data.

    Args:
        mean (tuple (2) of np.array): Mean of simulations.
        ci (tuple (2) of np.array): CI of simulation mean.
        x (pandas.DataFrame): Data
    """
    # data
    sim_mean,sim_obs_mean = mean
    if ci is None: ci = (None,None)
    sim_ci,sim_obs_ci = ci if ci is not None else (None,None)
    # plot
    fig1, ax1 = plt.subplots()
    ax1.plot(x.date, sim_mean[3,:], color='orange', label='Recovered z[t]')
    if sim_ci is not None:
        ax1.fill_between(x.date, sim_ci[0,3,:], sim_ci[1,3,:], color = 'orange', alpha = .25)
    ax1.plot(x.date, sim_obs_mean[3,:], color='red', label='Recovered x[t]')
    if sim_obs_ci is not None:
        ax1.fill_between(x.date, sim_obs_ci[0,3,:], sim_obs_ci[1,3,:], color = 'red', alpha = .1)
    if 'recovered' in x.columns:
        ax1.plot(x.date, x.recovered.cumsum(), color = 'blue', label='Recovered (data)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Recovered')
    plt.yscale('log')
    plt.legend()

def _plot_deaths(mean, ci, x):
    """Plots the predicted (latent, observed) deaths vs. data.

    Args:
        mean (tuple (2) of np.array): Mean of simulations.
        ci (tuple (2) of np.array): CI of simulation mean.
        x (pandas.DataFrame): Data
    """
    # data
    sim_mean,sim_obs_mean = mean
    if ci is None: ci = (None,None)
    sim_ci,sim_obs_ci = ci if ci is not None else (None,None)
    # plot
    fig1, ax1 = plt.subplots()
    ax1.plot(x.date, sim_mean[4,:], color='red', label='Deaths z[t]')
    if sim_ci is not None:
        ax1.fill_between(x.date, sim_ci[0,4,:], sim_ci[1,4,:], color = 'red', alpha = .25)
    ax1.plot(x.date, sim_obs_mean[4,:], color='orange', label='Deaths x[t]')
    if sim_obs_ci is not None:
        ax1.fill_between(x.date, sim_obs_ci[0,4,:], sim_obs_ci[1,4,:], color = 'orange', alpha = .1)
    if 'deaths' in x.columns:
        ax1.plot(x.date, x.deaths.cumsum(), color = 'blue', label='Deaths (data)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Deaths')
    plt.yscale('log')
    plt.legend()

def compute_sim_mean(sim):
    """Compute mean for latent and observed.
    
    Args:
        sim (tuple (2) of np.array): Matrix with samples (latent, observed).
    """
    sim_lat,sim_obs = sim
    sim_mean = sim_lat.mean(axis=1)
    sim_obs_mean = sim_obs.mean(axis=1)
    return sim_mean,sim_obs_mean

def compute_sim_ci(sim):
    """Compute 95% CI for latent and observed.
    
    Args:
        sim (tuple (2) of np.array): Matrix with samples (latent, observed).
    """
    sim_lat,sim_obs = sim
    sim_ci = np.quantile(sim_lat, [.025,.975], axis = 1)
    sim_obs_ci = np.quantile(sim_obs, [.025,.975], axis = 1)
    return sim_ci,sim_obs_ci 

def _plot_posterior(sim=None, region=None, dates=None, sim_mean=None):
    """Plots the posterior simulations.
    
    If samples `sim` is given, plots it with CI. If not, only `sim_mean` mean
    is plotted without the CI.
    
    Args:
        sim (tuple (2) of np.array): Matrix with samples (latent, observed).v
        region (str): Region for the data.
        dates (tuple (2)): Date range of the data.
        sim_mean (tuple (2) of np.array): Mean of simulations, if sim not given.
    """
    assert(region is not None)
    assert(dates is not None)
    # fetch data
    x = _posterior_data(region, dates)
    # aggregate results
    if sim is not None:
        sim_mean,sim_obs_mean = compute_sim_mean(sim)        
        sim_ci,sim_obs_ci = compute_sim_ci(sim)
    else:
        sim_mean,sim_obs_mean = sim_mean
        sim_ci,sim_obs_ci = None,None
    # plot
    _plot_confirmed((sim_mean,sim_obs_mean),(sim_ci,sim_obs_ci),x)
    plt.show()
    _plot_recovered((sim_mean,sim_obs_mean),(sim_ci,sim_obs_ci),x)
    plt.show()
    _plot_deaths((sim_mean,sim_obs_mean),(sim_ci,sim_obs_ci),x)
    plt.show()

def run_covid_characteristics():
    """Run example Covid-19 pandemic."""
    params = pd.DataFrame({
        'start': [datetime(2020,3,1)],
        'end': [datetime(2020,9,30)],
        'a': [(2.20007645778212,27.37944426070125,0,6.619279818359113e-05)],
        'c': [(3.4776857980163545,51.0585661816971,.03465259257309947,2.5447897386852354)],
        'b': [(2.58506397436604,1581376.9982822197,0.01431241008785283,41895.140446301935)],
        'd': [(.004,.01-.004)]
    })
    plot_posterior(
        'PL', N=1000, params=params, dates=(datetime(2020,3,1),datetime(2020,9,30)), 
        initial=(700/1000,150/1000,150/1000,0,0),
        parI=(1,1e3), parR=(1,1e3), parD=(1,1), random_params=True
    )
