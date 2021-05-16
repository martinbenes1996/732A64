# -*- coding: utf-8 -*-
"""Saving and plotting of results.

Module containing operations with results, such as saving and plotting.

Example:
    Load result of simulation with
    
        sim,dates,region,params = results.load(
            dates=(datetime.datetime(2020,3,10),datetime.datetime(2020,5,31)),
            region='CZ',
            now=datetime.datetime(2021,4,12)
        )
    
    Save result of simulation with
    
        results.save(
            sim=(sim_lat,sim_obs),
            dates=(datetime.datetime(2020,3,10),datetime.datetime(2020,5,31)),
            region='CZ',
            params=datetime.datetime(2021,4,12)
        )
    
    Plot parameters of Covid-19 with
    
        results.plot_params(
            dates=(datetime.datetime(2020,3,10),datetime.datetime(2020,5,31)),
            region='CZ',
            now=datetime.datetime(2021,4,12)
        )
    
    Plot estimates of Covid-19 characteristics with
    
        results.plot_characteristics(
            dates=(datetime.datetime(2020,3,10),datetime.datetime(2020,5,31)),
            region='CZ',
            now=datetime.datetime(2021,4,12),
            par='r0' # 'ifr','symptom duration','incubation period'
        )
    
    Plot weekly results with
    
        results.plot_weekly(
            dates=(datetime.datetime(2020,3,3),datetime.datetime(2021,4,16)),
            region='PL',
            now=datetime.datetime(2021,4,18),
            weekly=False
        )
    
    Plot weekly susceptibles for PL with
    
        results.plotSusceptible_PL_Weekly()
        
    Plot weekly susceptibles for SE224 with
    
        results.plotSusceptible_SE224_Weekly()
        
"""
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import sys
sys.path.append('src')
import posterior

def _get_path(dates, region, now = None, create_if_not_exists = True):
    """Get path of simulation results for given parameters.
    
    Args:
        dates (tuple (2) of datetime): Time range for simulation (start, end).
        region (str): Region to load result for.
        now (datetime, optional): Time of production of the simulation. By default today.
        create_if_not_exists (bool): Creates if path does not exist, if True.
    """
    # parse dates
    date_start = dates[0].strftime('%Y-%m-%d')
    date_end = dates[1].strftime('%Y-%m-%d')
    if now is None: now = datetime.now()
    now = now.strftime('%Y-%m-%d')
    # paths
    def _create_path(path):
        try: os.makedirs(path) 
        except OSError as error: pass
    projdir = f'results/{now}/{region}_{date_start}_{date_end}'
    if create_if_not_exists:
        _create_path(projdir)
    return projdir

def save(sim, dates, region, params):
    """Save results to file.
    
    Args:
        sim (tuple (2) of matrices): Simulation results (latent [5xD], observed [5xD]).
        dates (tuple (2) of datetime): Time range for simulation (start, end).
        region (str): Region to load result for.
        now (datetime, optional): Time of production of the simulation. By default today.
    """
    # path
    path = _get_path(dates, region)
    # parse simulations
    sim_lat,sim_obs = sim
    lat = sim_lat.mean(axis = 1)
    obs = sim_obs.mean(axis = 1)
    x = posterior._posterior_data(region, dates)
    # save
    df = pd.DataFrame({
        'date': x.date,
        'region': region,
        'param_a': None, 'param_c': None, 'param_b': None, 'param_d': None,
        'latent_S': lat[0,:],
        'latent_E': lat[1,:],
        'latent_I': lat[2,:],
        'latent_R': lat[3,:],
        'latent_D': lat[4,:],
        'observed_I': obs[2,:], 'observed_R': obs[3,:], 'observed_D': obs[4,:],
        'data_I': x.confirmed,
        'data_R': x.recovered if 'recovered' in x.columns else np.nan,
        'data_D': x.deaths})\
        .reset_index(drop=True)
    # set parameters a,c,b,d
    for df_i in df.index:
        df_date = df.loc[df_i,'date']
        params_i = (params.start <= df_date) & (params.end >= df_date)
        params_i = params[params_i].start.idxmax()
        for col in ['a','c','b','d']:
            df.loc[df_i,'param_'+col] = float(params.loc[params_i,col])
    # save
    df.to_csv(f'{path}/data.csv', index = False)
    # aggregate
    sim_mean,sim_obs_mean = posterior.compute_sim_mean(sim)        
    sim_ci,sim_obs_ci = posterior.compute_sim_ci(sim)
    # plot
    posterior._plot_confirmed((sim_mean,sim_obs_mean),(sim_ci,sim_obs_ci),x)
    plt.savefig(f'{path}/confirmed.png')
    posterior._plot_recovered((sim_mean,sim_obs_mean),(sim_ci,sim_obs_ci),x)
    plt.savefig(f'{path}/recovered.png')
    posterior._plot_deaths((sim_mean,sim_obs_mean),(sim_ci,sim_obs_ci),x)
    plt.savefig(f'{path}/deaths.png')
    plot_characteristics(dates, region, datetime.now(), par='R0')
    plt.savefig(f'{path}/R0.png')
    plot_characteristics(dates, region, datetime.now(), par='Incubation period')
    plt.savefig(f'{path}/incubation.png')
    plot_characteristics(dates, region, datetime.now(), par='IFR')
    plt.savefig(f'{path}/ifr.png')
    plot_characteristics(dates, region, datetime.now(), par='Symptom duration')
    plt.savefig(f'{path}/symptom.png')    

def load(dates, region, now=None):
    """Loads result of the simulation.
    
    Args:
        dates (tuple (2) of datetime): Time range for simulation (start, end).
        region (str): Region to load result for.
        now (datetime, optional): Time of production of the simulation. By default today.
    """
    # load
    path = _get_path(dates, region, now=now, create_if_not_exists=False)
    x = pd.read_csv(f'{path}/data.csv')
    x['date'] = x.date.apply(lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
    # parse
    lat = x[['latent_S','latent_E','latent_I','latent_R','latent_D']].to_numpy().T
    obs = np.zeros((5,x.shape[0]))
    obs[2:5,:] = x[['observed_I','observed_R','observed_D']].to_numpy().T
    region = x.loc[0,'region']
    dates = (x.date.min(), x.date.max())
    params = x[['param_a','param_c','param_b','param_d']].to_numpy().T
    return (lat,obs),x.date,region,params

def plot_params(dates, region, now=None, save=False, name='img/results/%s_params.png'):
    """Plot parameters of the simulation.
    
    Args:
        dates (tuple (2) of datetime): Time range for simulation (start, end).
        region (str): Region to load result for.
        now (datetime, optional): Time of production of the simulation. By default today.
    """
    # load
    name = name % region
    sim,dt,region,params = load(dates,region,now)
    # plot
    fig1, ax1 = plt.subplots()
    ax1.plot(dt, params[0,:], color='red', label='a')
    ax1.plot(dt, params[1,:], color='orange', label='c')
    ax1.plot(dt, params[2,:], color='green', label='b')
    ax1.plot(dt, params[3,:], color='black', label='d')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.set_yscale('log')
    ax1.legend()
    if save: fig1.savefig(name)

def plot_characteristics(dates, region, now=None, par='r0', crop_last=0):
    """Plot Covid-19 characteristics of the simulation.
    
    Args:
        dates (tuple (2) of datetime): Time range for simulation (start, end).
        region (str): Region to load result for.
        now (datetime, optional): Time of production of the simulation. By default today.
        par (str, optional): Characteristic to plot, one of 'r0','incubation period','ifr','symptom duration'.
        crop_last (int, optional):
    """
    # load
    sim,dt,region,params = load(dates,region,now)
    if crop_last > 0:
        dt,params = dt[:-crop_last],params[:,:-crop_last]
    # compute
    r0 = params[0,:] / params[2,:]
    incubation = 1 / params[1,:]
    ifr = params[3,:]
    symptoms = 1 / params[2,:]
    # create plot
    fig1, ax1 = plt.subplots()
    if par.lower() == 'r0':
        ax1.plot(dt, r0, label=par)
        ax1.axhline(y=1, color='grey', alpha=.4)
    if par.lower() == 'incubation period':
        ax1.plot(dt, incubation, label=par)
    if par.lower() == 'ifr':
        ax1.plot(dt, ifr, label=par)
    if par.lower() == 'symptom duration':
        ax1.plot(dt, symptoms, label=par)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    plt.legend()

def _savefig(path):
    """Save current figure into `path`.
    
    Args:
        path (str): Path to save figure to.
    """
    # remove the file
    try: os.remove(path)
    except: pass
    # save plot
    plt.savefig(path)
    
def plot_weekly(dates, region, now=None, weekly=True, save=False):
    """Generates the plots of the simulation.
    
    Args:
        dates (tuple (2) of datetime): Time range for simulation (start, end).
        region (str): Region to load result for.
        now (datetime, optional): Time of production of the simulation. By default today.
        weekly (bool, optional): Weekly data time slots if True, otherwise daily.
        save (bool, optional): Whether to save the figure, defaultly not.
    """
    # path
    path = _get_path(dates, region, now=now, create_if_not_exists=False)
    # load
    x = posterior._posterior_data(region, dates, weekly=weekly)
    print(x)
    (sim_mean,sim_obs_mean),dt,region,params = load(dates, region, now)
    print(x.shape)
    print(sim_mean.shape)
    # plot
    posterior._plot_confirmed((sim_mean,sim_obs_mean),None,x)
    if save: _savefig(f'{path}/confirmed.png')
    posterior._plot_recovered((sim_mean,sim_obs_mean),None,x)
    if save: _savefig(f'{path}/recovered.png')
    posterior._plot_deaths((sim_mean,sim_obs_mean),None,x)
    if save: _savefig(f'{path}/deaths.png')
    plot_characteristics(dates, region, now, par='R0')
    if save: _savefig(f'{path}/R0.png')
    plot_characteristics(dates, region, now, par='Incubation period')
    if save: _savefig(f'{path}/incubation.png')
    plot_characteristics(dates, region, now, par='IFR')
    if save: _savefig(f'{path}/ifr.png')
    plot_characteristics(dates, region, now, par='Symptom duration')
    if save: _savefig(f'{path}/symptom.png') 

def plotSusceptible_SE224_Weekly(save=False, name='TODO'):
    """Generates plot of weekly susceptible for region SE224.
    
    Args:
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    # load result
    (sim_mean,sim_obs_mean),dates,region,params = load(
        (datetime(2020,7,1),datetime(2021,3,15)), 'SE224', datetime(2021,4,14))
    x = posterior._posterior_data('SE224', (datetime(2020,7,1),datetime(2021,3,15)), weekly=True)
    # plot
    fig,ax = plt.subplots()
    ax.plot(dates, sim_mean[0,:], label='S')
    ax.legend()
    if save: fig.savefig(name)

def plotSusceptible_PL_Weekly(save=False, name='TODO'):
    """Generates plot of weekly susceptible for Poland.
    
    Args:
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    # load result
    (sim_mean,sim_obs_mean),dates,region,params = load(
        (datetime(2020,3,3),datetime(2021,4,16)), 'PL', datetime(2021,4,18))
    x = posterior._posterior_data('PL', (datetime(2020,3,3),datetime(2021,4,16)), weekly=False)
    # plot
    fig,ax = plt.subplots()
    ax.plot(dates, sim_mean[0,:], label='S')
    ax.legend()
    if save: fig.savefig(name)
