
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import sys
sys.path.append('src')
import posterior

def get_path(dates, region, now = None, create_if_not_exists = True):
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
    #projdir = f'results/{region}w/{region}_{date_start}_{date_end}'
    
    if create_if_not_exists:
        _create_path(projdir)
    return projdir

def save_result(sim, dates, region, params):
    # path
    path = get_path(dates, region)
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

def load_result(dates, region, now=None):
    path = get_path(dates, region, now=now, create_if_not_exists=False)
    # load
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

def plot_params(dates, region, now=None):
    # load
    sim,dt,region,params = load_result(dates,region,now)
    #dt = pd.date_range(dates[0],dates[1])
    # create plot
    fig1, ax1 = plt.subplots()
    ax1.plot(dt, params[0,:], color='red', label='a')
    ax1.plot(dt, params[1,:], color='orange', label='c')
    ax1.plot(dt, params[2,:], color='green', label='b')
    ax1.plot(dt, params[3,:], color='black', label='d')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    plt.yscale('log')
    plt.legend()

def plot_characteristics(dates, region, now=None, par='r0', crop_last=0):
    # load
    sim,dt,region,params = load_result(dates,region,now)
    #print(params.shape)
    #dt = pd.date_range(dates[0],dates[1])
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
    #plt.yscale('log')
    plt.legend()

def _savefig(path):
    try:
        os.remove(path)
    except:
        pass
    plt.savefig(path)
def plot_weekly_results(dates, region, now=None):
    # path
    path = get_path(dates, region, now=now, create_if_not_exists=False)
    # load
    x = posterior._posterior_data(region, dates, weekly=True)
    (sim_mean,sim_obs_mean),dt,region,params = load_result(dates, region, now)
    # plot
    posterior._plot_confirmed((sim_mean,sim_obs_mean),None,x)
    _savefig(f'{path}/confirmed.png')
    posterior._plot_recovered((sim_mean,sim_obs_mean),None,x)
    _savefig(f'{path}/recovered.png')
    posterior._plot_deaths((sim_mean,sim_obs_mean),None,x)
    _savefig(f'{path}/deaths.png')
    plot_characteristics(dates, region, datetime.now(), par='R0')
    _savefig(f'{path}/R0.png')
    plot_characteristics(dates, region, datetime.now(), par='Incubation period')
    _savefig(f'{path}/incubation.png')
    plot_characteristics(dates, region, datetime.now(), par='IFR')
    _savefig(f'{path}/ifr.png')
    plot_characteristics(dates, region, datetime.now(), par='Symptom duration')
    _savefig(f'{path}/symptom.png') 

def plotSusceptible_SE224_Weekly():
    (sim_mean,sim_obs_mean),dates,region,params = load_result(
        (datetime(2020,7,1),datetime(2021,3,15)), 'SE224', datetime(2021,4,14))
    x = posterior._posterior_data('SE224', (datetime(2020,7,1),datetime(2021,3,15)), weekly=True)
    #print(x)
    #sim_mean,sim_obs_mean = posterior.compute_sim_mean(sim)        
    #sim_ci,sim_obs_ci = posterior.compute_sim_ci(sim)
    plt.plot(dates, sim_mean[0,:], label='S')
    plt.legend()
    plt.show()
    #posterior._plot_confirmed((sim_mean,sim_obs_mean),None,x)
    #plt.show()
    #posterior._plot_recovered((sim_mean,sim_obs_mean),None,x)
    #plt.show()
    #posterior._plot_deaths((sim_mean,sim_obs_mean),None,x)
    #plt.show()

def plotSusceptible_PL_Weekly():
    (sim_mean,sim_obs_mean),dates,region,params = load_result(
        (datetime(2020,3,3),datetime(2021,4,16)), 'PL', datetime(2021,4,18))
    x = posterior._posterior_data('PL', (datetime(2020,3,3),datetime(2021,4,16)), weekly=False)
    #print(x)
    #sim_mean,sim_obs_mean = posterior.compute_sim_mean(sim)        
    #sim_ci,sim_obs_ci = posterior.compute_sim_ci(sim)
    plt.plot(dates, sim_mean[0,:], label='S')
    plt.legend()
    plt.show()
    #posterior._plot_confirmed((sim_mean,sim_obs_mean),None,x)
    #plt.show()
    #posterior._plot_recovered((sim_mean,sim_obs_mean),None,x)
    #plt.show()
    #posterior._plot_deaths((sim_mean,sim_obs_mean),None,x)
    #plt.show()
#plotSusceptible_PL_Weekly()

if __name__ == '__main__':
    plot_weekly_results((datetime(2020,8,1),datetime(2021,3,13)),
                        'PL', datetime(2021,4,21))
