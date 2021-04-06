
from datetime import datetime
import numpy as np
import pandas as pd
import sys
sys.path.append('src')
import posterior

def save_result(sim, dates, country, filename):
    # parse simulations
    sim_lat,sim_obs = sim
    lat = sim_lat.mean(axis = 1)
    obs = sim_obs.mean(axis = 1)
    x = posterior._posterior_data(country, dates)
    # save
    pd.DataFrame({'date': pd.date_range(*dates),
                  'country': country,
                  'latent_S': lat[0,:],
                  'latent_E': lat[1,:],
                  'latent_I': lat[2,:],
                  'latent_R': lat[3,:],
                  'latent_D': lat[4,:],
                  'observed_I': obs[2,:],
                  'observed_R': obs[3,:],
                  'observed_D': obs[4,:],
                  'data_I': x.confirmed,
                  'data_R': x.recovered,
                  'data_D': x.deaths})\
        .to_csv(filename, index = False)

def load_result(filename):
    # load
    x = pd.read_csv(filename)
    x['date'] = x.date.apply(lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
    # parse
    lat = x[['latent_S','latent_E','latent_I','latent_R','latent_D']].to_numpy().T
    obs = np.zeros((5,x.shape[0]))
    obs[2:5,:] = x[['observed_I','observed_R','observed_D']].to_numpy().T
    country = x.loc[0,'country']
    dates = (x.date.min(), x.date.max())
    
    return (lat,obs),dates,country

if __name__ == '__main__':
    sim,dates,country = load_result('res.csv')
    posterior._plot_posterior(sim_mean = sim, country = country, dates = dates)
    
#load_result('res.csv')