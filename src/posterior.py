
from datetime import datetime,timedelta
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta,betaprime
from scipy.integrate import odeint
import sys
sys.path.append('src')

import _src

def _seird(y, t, POP, a, c, b, d):
    S, E, I, R, D = y
    dSdt = - a*S*I
    dEdt = a*S*I - c*E
    dIdt = c*E - b*I - d*I
    dRdt = b*I
    dDdt = d*I
    return dSdt, dEdt, dIdt, dRdt, dDdt
def _transition(POP, initial_values, parameters):
    """Transition sampler.
    
    Args:
        POP (int): Population size.
        initial_values (tuple): Initial values (S0, E0, I0, R0, D0).
        parameters (tuple): Dataframe of parameters: start, end, a, c, b, d.
    """
    assert(len(initial_values) == 5)
    # iterate over time slots
    result = {'date': [], 'S': [], 'E': [], 'I': [], 'R': [], 'D': []}
    for row in parameters.itertuples():
        D = (row.end - row.start).days
        t = np.linspace(0, D, D+1)
        # integrate
        a = beta.rvs(*row.a, size = 1)[0]
        c = beta.rvs(*row.c, size = 1)[0]
        b = beta.rvs(*row.b, size = 1)[0]
        d = beta.rvs(*row.d, size = 1)[0]
        r = odeint(_seird, initial_values, t, args=(POP, a, c, b, d))
        for dt in pd.date_range(row.start, row.end-timedelta(days=1)):
            result['date'].append(dt)
        #result['date'] = [*result['date'], )
        result['S'] = [*result['S'], *r.T[0,:D]]
        result['E'] = [*result['E'], *r.T[1,:D]]
        result['I'] = [*result['I'], *r.T[2,:D]]
        result['R'] = [*result['R'], *r.T[3,:D]]
        result['D'] = [*result['D'], *r.T[4,:D]]
        initial_values = r[D,:]
    # add last
    result['date'].append(row.end)
    result['S'].append(r.T[0,D])
    result['E'].append(r.T[1,D])
    result['I'].append(r.T[2,D])
    result['R'].append(r.T[3,D])
    result['D'].append(r.T[4,D])
    # return
    result = pd.DataFrame(result)
    return result

def _emission(xbar, T, a, b):
    # parameters
    alpha_ = (a + T * xbar)
    beta_ = (b + T * (1 - xbar))
    # simulate
    D = T.shape[0]
    draw = np.zeros((D,))
    for i in range(D):
        draw[i] = betaprime.rvs(alpha_[i], beta_[i], size = 1)
    # result
    return draw

def run_country(country, params, dates = (datetime(2020,3,15),datetime(2021,2,28)),
                initial_values = (800/1000,50/1000,150/1000,0,0), POP = 1e7, N = 1000,
                parI = (1,1),parR = (1,1),parD = (1,1)):
    """"""
    assert(country in {'CZE','SWE','ITA','POL'})
    # get data and filter by country
    x = _src.get_data()
    x = x[x.iso_alpha_3 == country]
    # filter param
    params = params[params.start < dates[1]]
    if (params.end > dates[1]).any():
        params.loc[params.end > dates[1], 'end'] = dates[1]

    # daily to cumsum and normalize by tests
    x['cumtests'] = x.tests.cumsum()
    x['confirmed'] = x.confirmed / x.tests
    x[x.confirmed < 0]['confirmed'] = 0
    x['recovered'] = (x.recovered / x.tests)
    x[x.recovered < 0]['recovered'] = 0
    x['deaths'] = (x.deaths / x.tests)
    x[x.deaths < 0]['deaths'] = 0
    # filter by dates
    x = x[(x.date >= dates[0]) & (x.date <= dates[1])]
    if x.date.min() > dates[0]:
        x_init = pd.DataFrame({
            'dates': pd.date_range(dates[0],x.date.min()),
            'tests': 100,
            'confirmed': 0,
            'recovered': 0,
            'deaths': 0
        })
        x = x.append(x_init)\
            .sort_values('dates')
    
    # simulate
    D = (dates[1] - dates[0]).days + 1
    sim_lat = np.zeros((5,N,D))
    sim_obs = np.zeros((5,N,D))
    for i in range(N):
        if i == 0 or (i+1) % 100 == 0:
            print('%4d / %d' % (i+1,N))
        latent = _transition(POP, initial_values, params)
        sim_lat[:,i,:] = latent[['S','E','I','R','D']].to_numpy().T
        sim_obs[2,i,:] = _emission(latent.I.to_numpy(), x.tests.to_numpy(), *parI)
        sim_obs[3,i,:] = _emission(latent.R.to_numpy(), x.tests.to_numpy(), *parR)
        sim_obs[4,i,:] = _emission(latent.D.to_numpy(), x.tests.to_numpy(), *parD)
    # denormalize probability
    sim_lat[3:5,:,:] = np.diff(sim_lat[3:5,:,:], axis=2, prepend=sim_lat[3:5,:,2:3])
    sim_lat = sim_lat * x.tests.to_numpy()
    sim_lat[sim_lat > 1e5] = 0
    sim_lat[sim_lat < 0] = 0
    sim_obs = sim_obs * x.tests.to_numpy()
    
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
    ax1.plot(x.date, x.confirmed * x.tests, color = 'blue', label='Confirmed')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Infected')
    plt.legend()
    plt.show()
    
    fig1, ax1 = plt.subplots()
    ax1.plot(x.date, sim_mean[3,:], color='orange', label='Recovered (latent)')
    ax1.fill_between(x.date, sim_ci[0,3,:], sim_ci[1,3,:], color = 'orange', alpha = .25)
    ax1.plot(x.date, sim_obs_mean[3,:], color='red', label='Recovered (observed)')
    ax1.fill_between(x.date, sim_obs_ci[0,3,:], sim_obs_ci[1,3,:], color = 'red', alpha = .1)
    ax1.plot(x.date, x.recovered * x.cumtests, color = 'blue', label='Recovered')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Recovered')
    plt.legend()
    plt.show()
    
    fig1, ax1 = plt.subplots()
    ax1.plot(x.date, sim_mean[4,:], color='orange', label='Deaths (latent)')
    ax1.fill_between(x.date, sim_ci[0,4,:], sim_ci[1,4,:], color = 'orange', alpha = .25)
    ax1.plot(x.date, sim_obs_mean[4,:], color='red', label='Deaths (observed)')
    ax1.fill_between(x.date, sim_obs_ci[0,4,:], sim_obs_ci[1,4,:], color = 'red', alpha = .1)
    ax1.plot(x.date, x.deaths * x.cumtests, color = 'blue', label='Deaths')
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
params = pd.DataFrame({
    'start': [datetime(2020,2,28), datetime(2020,3,10), datetime(2020,4,1), datetime(2020,5,1), datetime(2020,6,1), datetime(2020,6,15), datetime(2020,7,1)],
    'end': [datetime(2020,3,10), datetime(2020,4,1), datetime(2020,5,1), datetime(2020,6,1), datetime(2020,6,15), datetime(2020,7,1), datetime(2020,7,30)],
    'a': [(2,6), (2,20), (2,200), (2,1e3), (2,1e3), (2,400), (2,400)],
    'c': map(lambda _:(3,51),range(7)),
    'b': map(lambda _:(1.1,10),range(7)),
    'd': map(lambda _:(1.1,100),range(7))
})



#POP = 1e7
#run_country('CZE', N = 300, params = params, dates = (datetime(2020,3,15),datetime(2020,6,30)), POP = 1e7,
#            initial_values = (820/1000,80/1000,100/1000,0,0), alpha = 2, beta = 10000)
run_country('CZE', N = 300, params = params, dates = (datetime(2020,2,28),datetime(2020,7,30)), POP = 1e7,
            initial_values = (700/1000,300/1000,0/1000,0,0), parI=(2,5e4), parR=(2,5e4), parD=(2,1e8))
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
