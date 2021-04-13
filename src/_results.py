
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import sys
sys.path.append('src')
import posterior

def get_path(dates, region):
    # parse dates
    date_start = dates[0].strftime('%Y-%m-%d')
    date_end = dates[1].strftime('%Y-%m-%d')
    now = datetime.now().strftime('%Y-%m-%d')
    # paths
    def _create_path(path):
        try: os.makedirs(path) 
        except OSError as error: pass
    projdir = f'results/{now}/{region}_{date_start}_{date_end}'
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
    
def load_result(filename):
    # load
    x = pd.read_csv(filename)
    x['date'] = x.date.apply(lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
    # parse
    lat = x[['latent_S','latent_E','latent_I','latent_R','latent_D']].to_numpy().T
    obs = np.zeros((5,x.shape[0]))
    obs[2:5,:] = x[['observed_I','observed_R','observed_D']].to_numpy().T
    region = x.loc[0,'region']
    dates = (x.date.min(), x.date.max())
    
    return (lat,obs),dates,region

if __name__ == '__main__':
    #sim,dates,region = load_result('res.csv')
    #posterior._plot_posterior(sim_mean = sim, region = region, dates = dates)
    get_path((datetime(2020,3,1),datetime(2020,3,31)),'CZ')
    get_path((datetime(2020,3,1),datetime(2020,3,31)),'CZ010')
#load_result('res.csv')