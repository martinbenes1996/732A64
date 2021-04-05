
from datetime import datetime,timedelta
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import sys
sys.path.append('src')

import _src
from emission import emission,emission_objective
from transition import transition

def _parse_params(params, fixparams):
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
def _posterior_data(country, dates):
    global _data
    # get data
    if country not in _data:
        x = _src.get_data()
        # filter by country
        x = x[x.iso_alpha_3 == country]
        # parse tests
        x['cumtests'] = x.tests.cumsum()
        x['tests'] = x.tests.replace({0:1})
        x['cumtests'] = x.cumtests.replace({0:1})
        _data[country] = x
    x = _data[country]
    # filter by dates
    x = x[(x.date >= dates[0]) & (x.date <= dates[1])]
    if x.date.min() > dates[0]:
        x_init = pd.DataFrame({
            'dates': pd.date_range(dates[0],x.date.min()),
            'iso_alpha_3': country,
            'tests': 0,'confirmed': 0,'recovered': 0,'deaths': 0})
        x = x.append(x_init)\
            .sort_values('dates')
    return x

def posterior_objective(params, country, POP, dates, initial_values,
                        fixparams = None, parI = (1,1), parR = (1,1), parD = (1,1)):
    """"""
    assert(country in {'CZE','SWE','ITA','POL'})
    x = _posterior_data(country, dates)
    # construct params dataframe
    a,c,b,d = _parse_params(params, fixparams)
    params = pd.DataFrame({'start': [dates[0]], 'end': [dates[1]],
                            'a': [a], 'b': [b], 'c': [c], 'd': [d]})
    # compute score
    D = (dates[1] - dates[0]).days + 1
    score = 0
    latent = transition(POP, initial_values, params)
    latent.loc[latent.I < 0,'I'] = 0
    latent.loc[latent.dR < 0,'dR'] = 0
    latent.loc[latent.dD < 0,'dD'] = 0
    score += emission_objective(x.confirmed.to_numpy() / x.tests.to_numpy(),
                                latent.I.to_numpy(), x.tests.to_numpy(), *parI)
    score += emission_objective(x.recovered.cumsum().to_numpy() / x.tests.to_numpy(),
                                latent.R.to_numpy(), x.tests.to_numpy(), *parR)
    score += emission_objective(x.deaths.cumsum().to_numpy() / x.tests.to_numpy(),
                                latent.D.to_numpy(), x.tests.to_numpy(), *parD)
    return score / D

def simulate_posterior(country, params, dates, initial_values,
                       POP = 1e7, N = 1000, parI = (1,1), parR = (1,1),parD = (1,1)):
    """"""
    assert(country in {'CZE','SWE','ITA','POL'})
    x = _posterior_data(country, dates)\
        .reset_index(drop = True)
    # filter param
    params = params[params.start <= dates[1]]
    if (params.end > dates[1]).any():
        params.loc[params.end > dates[1], 'end'] = dates[1]
    # simulate
    D = (dates[1] - dates[0]).days + 1
    sim_lat = np.zeros((5,N,D))
    sim_obs = np.zeros((5,N,D))
    for i in range(N):
        if i == 0 or (i+1) % 100 == 0:
            print('%4d / %d' % (i+1,N))
        # transition
        latent = transition(POP, initial_values, params)
        latent[latent.I < 0]['I'] = 0
        latent[latent.dR < 0]['dR'] = 0
        latent[latent.dD < 0]['dD'] = 0
        sim_lat[:,i,:] = latent[['S','E','I','R','D']].to_numpy().T
        # emission
        sim_obs[2,i,:] = emission(latent.I.to_numpy(), x.tests.to_numpy(), *parI)
        sim_obs[3,i,:] = emission(latent.R.to_numpy(), x.tests.to_numpy(), *parR)
        sim_obs[4,i,:] = emission(latent.D.to_numpy(), x.tests.to_numpy(), *parD)
        #sim_obs[2,i,:][sim_obs[2,i,:] < 0] = 0
        #sim_obs[3,i,:][sim_obs[3,i,:] < 0] = 0
        #sim_obs[4,i,:][sim_obs[4,i,:] < 0] = 0
    # spare last
    last_values = sim_lat[:,:,-1].mean(axis = 1)
    # denormalize probability
    #sim_lat[3:5,:,:] = np.diff(sim_lat[3:5,:,:], axis=2, prepend=sim_lat[3:5,:,2:3])
    sim_lat[1:3,:,:] = sim_lat[1:3,:,:] * x.tests.to_numpy()
    sim_lat[3:5,:,:] = sim_lat[3:5,:,:] * x.tests.to_numpy()#x.cumtests.to_numpy()
    #sim_lat[3:5,:,:] = sim_lat[3:5,:,:]
    sim_obs[3:5,:,:] = np.diff(sim_obs[3:5,:,:], axis=2, prepend=sim_obs[3:5,:,2:3])
    sim_obs[1:3,:,:] = sim_obs[1:3,:,:] * x.tests.to_numpy()
    sim_obs[3:5,:,:] = sim_obs[3:5,:,:] * x.tests.to_numpy()#x.cumtests.to_numpy()
    #sim_obs[3:5,:,:] = sim_obs[3:5,:,:]
    return (sim_lat, sim_obs), last_values

def plot_posterior(country, params, dates, initial_values,
                   POP = 1e7, N = 1000, parI = (1,1),parR = (1,1),parD = (1,1)):
    """"""
    # run simulation
    (sim_lat,sim_obs),_ = simulate_posterior(
        country = country, params = params, dates = dates, POP = POP, N = N,
        initial_values = initial_values, parI = parI, parR = parR, parD = parD)
    _plot_posterior(sim = (sim_lat,sim_obs), country = country, dates = dates)
    
def _plot_posterior(sim, country, dates):
    # fetch data
    x = _posterior_data(country, dates)
    sim_lat,sim_obs = sim
    #sim_lat[3:5,:,:] = sim_lat[3:5,:,:].cumsum(axis = 2)
    #sim_obs[3:5,:,:] = sim_obs[3:5,:,:].cumsum(axis = 2)
    # aggregate results
    sim_mean = sim_lat.mean(axis = 1)
    sim_ci = np.quantile(sim_lat, [.025,.975], axis = 1)
    sim_obs_mean = sim_obs.mean(axis = 1)
    sim_obs_ci = np.quantile(sim_obs, [.025,.975], axis = 1)
    # plot
    fig1, ax1 = plt.subplots()
    ax1.plot(x.date, sim_mean[2,:], color='orange', label='Infected (latent)')
    ax1.fill_between(x.date, sim_ci[0,2,:], sim_ci[1,2,:], color = 'orange', alpha = .25)
    ax1.plot(x.date, sim_obs_mean[2,:], color='red', label='Infected (observed)')
    ax1.fill_between(x.date, sim_obs_ci[0,2,:], sim_obs_ci[1,2,:], color = 'red', alpha = .1)
    ax1.plot(x.date, x.confirmed, color = 'blue', label='Confirmed')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Infected')
    plt.legend()
    plt.show()
    
    fig1, ax1 = plt.subplots()
    ax1.plot(x.date, sim_mean[3,:], color='orange', label='Recovered (latent)')
    ax1.fill_between(x.date, sim_ci[0,3,:], sim_ci[1,3,:], color = 'orange', alpha = .25)
    ax1.plot(x.date, sim_obs_mean[3,:], color='red', label='Recovered (observed)')
    ax1.fill_between(x.date, sim_obs_ci[0,3,:], sim_obs_ci[1,3,:], color = 'red', alpha = .1)
    ax1.plot(x.date, x.recovered, color = 'blue', label='Recovered')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Recovered')
    plt.legend()
    plt.show()
    
    fig1, ax1 = plt.subplots()
    ax1.plot(x.date, sim_mean[4,:], color='orange', label='Deaths (latent)')
    ax1.fill_between(x.date, sim_ci[0,4,:], sim_ci[1,4,:], color = 'orange', alpha = .25)
    ax1.plot(x.date, sim_obs_mean[4,:], color='red', label='Deaths (observed)')
    ax1.fill_between(x.date, sim_obs_ci[0,4,:], sim_obs_ci[1,4,:], color = 'red', alpha = .1)
    ax1.plot(x.date, x.deaths, color = 'blue', label='Deaths')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Deaths')
    plt.legend()
    plt.show()


#params = pd.DataFrame({
#    'start': [datetime(2020,3,15), datetime(2020,3,25), datetime(2020,4,4), datetime(2020,4,14)],
#    'end': [datetime(2020,3,25), datetime(2020,4,4), datetime(2020,4,14), datetime(2020,4,15)],
#    'a': [.00744432423122573, .00256929147327853, .00222884179092944, .000310343377761511],
#    'c': [.202934525296726, .428593541196657, .529646578160122, .358540222347213],
#    'b': [.00355981379148359, .000883336900733411, .0103621398903399, .0723453141705167],
#    'd': [.000155681618489325, .00224380836567308, .0010705482843332, .00982152286218479]
#})

#params = pd.DataFrame({
#    'start': [datetime(2020,3,15), datetime(2020,4,1), datetime(2020,5,1), datetime(2020,6,1), datetime(2020,6,15), datetime(2020,7,1)],
#    'end': [datetime(2020,4,1), datetime(2020,5,1), datetime(2020,6,1), datetime(2020,6,15), datetime(2020,7,1), datetime(2020,7,30)],
#    'a': [(2,900), (2,1200), (2,1200), (2,50), (2,50), (2,50)],
#    'c': [(4,8), (4,8), (4,8), (4,8), (4,8), (4,8)],
#    'b': [(3,700), (3,500), (3,300), (3,400), (3,400), (3,400)],
#    'd': [(2,1e4), (5,1e4), (5,1e4), (2,1e4), (2,1e4), (2,1e4)]
#})
#params = pd.DataFrame({
#    'start': [datetime(2020,2,28), datetime(2020,3,10), datetime(2020,4,1), datetime(2020,5,1), datetime(2020,6,1), datetime(2020,6,15), datetime(2020,7,1)],
#    'end': [datetime(2020,3,10), datetime(2020,4,1), datetime(2020,5,1), datetime(2020,6,1), datetime(2020,6,15), datetime(2020,7,1), datetime(2020,7,30)],
#    'a': [(2,6), (2,20), (2,200), (2,1e3), (2,1e3), (2,400), (2,400)],
#    'c': map(lambda _:(3,51),range(7)),
#    'b': map(lambda _:(1.1,10),range(7)),
#    'd': map(lambda _:(1.1,100),range(7))
#})
#params = pd.DataFrame({
#    'start': [datetime(2020,3,1), datetime(2020,3,15), datetime(2020,3,29)],
#    'end': [datetime(2020,3,15), datetime(2020,3,29), datetime(2020,4,12)],
#    'a': [.23455118,.0048916962,.009328],
#    'c': [.2,.2,.2],
#    'b': [.00038252,.007314,.0326747],
#    'd': [.000714116,.009550056,.00985778]
#})


def run_0304():
    params = pd.DataFrame({
        'start': [datetime(2020,3,1), datetime(2020,3,15), datetime(2020,3,29), datetime(2020,4,12), datetime(2020,4,26)],
        'end': [datetime(2020,3,15), datetime(2020,3,29), datetime(2020,4,12), datetime(2020,4,26), datetime(2020,4,30)],
        'a': [.0119831,.0038502,.00067523,.0039271,.0047303],
        'c': [.2,.2,.2,.2,.2],
        'b': [.00092836,.00761937,.0425419445,.0934527,.0194829],
        'd': [.00187313,.001527467,.0096889,.001792207,.0002380]
    })
    plot_posterior(
        'CZE', POP = 1e7, N = 300,
        params = params, dates = (datetime(2020,3,1),datetime(2020,4,30)), 
        initial_values = (700/1000,300/1000,0/1000,0,0),
        parI=(1,1), parR=(1,1), parD=(1,1))



#get_obj()

#POP = 1e7
#run_country('CZE', N = 300, params = params, dates = (datetime(2020,3,15),datetime(2020,6,30)), POP = 1e7,
#            initial_values = (820/1000,80/1000,100/1000,0,0), alpha = 2, beta = 10000)
#run_country('CZE', N = 300, params = params, dates = (datetime(2020,2,28),datetime(2020,7,30)), POP = 1e7,
#            initial_values = (700/1000,300/1000,0/1000,0,0), parI=(2,5e4), parR=(2,5e4), parD=(2,1e8))
#run_country('CZE', N = 300, params = params, dates = (datetime(2020,3,1),datetime(2020,4,12)), POP = 1e7,
#            initial_values = (700/1000,300/1000,0/1000,0,0), parI=(2,1e5), parR=(2,1e4), parD=(2,1e4))

#run_country('CZE', dates = (datetime(2020,9,1),datetime(2020,11,30)), alpha = 1000, beta = 300)




# plot
#fig1, ax1 = plt.subplots()
#ax1.plot(x.date, x.E, color='orange', label='Expected')
#ax1.plot(x.date, x.I, color='red', label='Infected')
#ax1.plot(x.date, x.R, color='blue', label='Recovered')
#ax1.plot(x.date, x.D, color='black', label='Deaths')
#ax1.set_xlabel('Date')
#ax1.set_ylabel('Infected')
#plt.legend()
#plt.show()
