
from datetime import datetime,timedelta
from geneticalgorithm import geneticalgorithm as ga
import numpy as np
import pandas as pd
import sys
sys.path.append('src')

import posterior

def optimize_segment(country, dates, initial_values):
    fixparam = [None,.2,None,None]
    def _obj(pars):
        return posterior.posterior_objective(
            pars, country = country, fixparams = fixparam,
            POP = 1e7, dates = dates,
            initial_values = initial_values,
            parI=(1,1), parR=(1,1), parD=(1,1))
    algorithm_param = {
        'max_num_iteration': 500,
        'population_size': 70,
        'mutation_probability': .65,
        'elit_ratio': .05,
        'crossover_probability': .8,
        'parents_portion': .5,
        'crossover_type':'uniform',
        'max_iteration_without_improv': 20
    }
    varbound = np.array([[0,.25],[0,.1],[0,.01]])
    model = ga(function=_obj,
               dimension=3,
               variable_type='real',
               variable_boundaries=varbound,
               convergence_curve = False,
               progress_bar = True,
               algorithm_parameters=algorithm_param)
    model.run()
    # best params
    params = posterior._parse_params(model.output_dict['variable'], fixparam)
    return params

def optimize_spline(country, dates, initial_values, emission_parameters = [(1,1),(1,1),(1,1)], window = 7):
    fixparam = [None,.2,None,None]
    # iterate windows
    parameters = {'start': [], 'end': [], 'a': [], 'b': [], 'c': [], 'd': []}
    for start in pd.date_range(dates[0], dates[1], freq=f'{window}D'):
        end = min(dates[1], start + timedelta(days=window))
        if start == end: continue
        print("Segment", start, "to", end)
        # optimize
        p = optimize_segment(country, (start,end), initial_values)
        parameters['start'].append(start)
        parameters['end'].append(end)
        parameters['a'].append(p[0])
        parameters['c'].append(p[1])
        parameters['b'].append(p[2])
        parameters['d'].append(p[3])
        # run simulation
        segment_pars = pd.DataFrame({
            'start': [start], 'end': [end],
            'a': [p[0]], 'c': [p[1]], 'b': [p[2]], 'd': [p[3]]
        })
        (sim_lat,sim_obs),last_values = posterior.simulate_posterior(
            country = country, params = segment_pars, dates = (start,end), POP = 1e7, N = 1,
            initial_values = initial_values,
            parI = emission_parameters[0], parR = emission_parameters[1], parD = emission_parameters[2])
        # plot
        #posterior._plot_posterior(sim = (sim_lat,sim_obs), country = country, dates = (start,end))
        # change initial values
        initial_values = last_values
    return pd.DataFrame(parameters)

if __name__ == '__main__':
    # parameters
    country = 'SWE'
    dates = (datetime(2020,3,1),datetime(2020,6,30))
    initial_values = (900/1000,100/1000,0/1000,0/1000,0/1000)
    # optimize
    params = optimize_spline(country, dates, initial_values,
                             emission_parameters = [(1,1e3),(1,1e3),(1,1)], window = 14)
    print(params)
    (sim_lat,sim_obs),last_values = posterior.simulate_posterior(
        country = country, params = params, dates = dates, POP = 1e7, N = 1000,
        initial_values = initial_values, parI = (1,1e3), parR = (1,1e3), parD = (1,1))
    
    lat = sim_lat.mean(axis = 1)
    obs = sim_obs.mean(axis = 1)
    print(lat.shape, obs.shape)
    pd.DataFrame({'date': pd.date_range(*dates),
                  'latent_S': lat[0,:],
                  'latent_E': lat[1,:],
                  'latent_I': lat[2,:],
                  'latent_R': lat[3,:],
                  'latent_D': lat[4,:],
                  'observed_I': obs[0,:],
                  'observed_R': obs[1,:],
                  'observed_D': obs[2,:]})\
        .to_csv('res.csv', index = False)
    
    posterior.plot_posterior(country, params, dates, initial_values, POP = 1e7)
    exit()
    
    
    # optimize
    pars = optimize_segment('CZE', dates, initial_values)
    # plot
    params = pd.DataFrame({'start': [dates[0]], 'end': [dates[1]],
                            'a': [pars[0]], 'c': [pars[1]], 'b': [pars[2]], 'd': [pars[3]]})
    run_country('CZE', params,
                dates = dates,
                initial_values = initial_values,
                POP = 1e7, N = 500,
                parI = (1,1),parR = (1,1),parD = (1,1))