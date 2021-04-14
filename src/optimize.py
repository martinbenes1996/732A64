
from datetime import datetime,timedelta
from geneticalgorithm import geneticalgorithm as ga
import json
import numpy as np
import pandas as pd
import sys
sys.path.append('src')

import population
import posterior
import _results

def optimize_segment(region, dates, initial, attributes):
    fixparams = [None,.2,None,None]#.0064]
    def _obj(pars):
        return posterior.posterior_objective(
            pars, region=region, fixparams=fixparams,
            dates=dates, initial=initial, attributes=attributes,
            parI=(1,1), parR=(1,1), parD=(1,1))
    algorithm_param = {
        'max_num_iteration': 500,
        'population_size': 70,
        'mutation_probability': .65,
        'elit_ratio': .05,
        'crossover_probability': .8,
        'parents_portion': .5,
        'crossover_type':'uniform',
        'max_iteration_without_improv': 40
    }
    varbound = np.array([[0,.25],[0,.1],[0,.1]])
    model = ga(function=_obj,
               dimension=sum([i is None for i in fixparams]),
               variable_type='real',
               variable_boundaries=varbound,
               convergence_curve = False,
               progress_bar = True,
               algorithm_parameters=algorithm_param)
    model.run()
    # best params
    params = posterior._parse_params(model.output_dict['variable'], fixparams)
    return params

def optimize_spline(region, dates, initial, emission = [(1,1),(1,1),(1,1)], attributes = 'IRD', window = 7):
    #fixparam = [None,.2,None,.0064]
    # iterate windows
    parameters = {'start': [], 'end': [], 'a': [], 'b': [], 'c': [], 'd': []}
    for start in pd.date_range(dates[0], dates[1], freq=f'{window}D'):
        end = min(dates[1], start + timedelta(days=window))
        if start == end: continue
        print("Segment", start, "to", end)
        # optimize
        p = optimize_segment(region, (start,end), initial, attributes)
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
            region=region, params=segment_pars, dates=(start,end), N=1,
            initial=initial, parI=emission[0], parR=emission[1], parD=emission[2])
        # plot
        #posterior._plot_posterior(sim = (sim_lat,sim_obs), country = country, dates = (start,end))
        # change initial values
        initial_values = last_values
    return pd.DataFrame(parameters)

def run(region, N = 1000):
    region = region.upper().strip()
    print(region)
    # load config
    with open("model/regions.json") as fp:
        _config = json.load(fp)
    config = _config[region]
    config = {
        'dates': ('2020-03-15','2020-12-31'),
        'window': 7, 'attributes': 'IRD',
        'initial': {'E':.1,'I':.1,'R':0,'D':0},
        'emission': {'I':(1,1),'R':(1,1),'D':(1,1)},
        **config}
    POP = population.get_population(region)
    # parse
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in config['dates']]
    window = config['window']
    attributes = config['attributes'].upper()
    initial = [1-sum(config['initial'].values()), # S
               config['initial'].get('E',0), # E
               config['initial'].get('I',0), # I
               config['initial'].get('R',0), # R     
               config['initial'].get('D',0)] # D
    emission = [config['emission'].get('I',(1,1)),
                config['emission'].get('R',(1,1)),
                config['emission'].get('D',(1,1))]
    # optimize
    params = optimize_spline(
        region, dates, initial=initial, attributes=attributes,
        emission=emission, window=window)
    print(params)
    # simulate result
    (sim_lat,sim_obs),last_values = posterior.simulate_posterior(
        region=region, params=params, dates=dates, N=N, initial=initial,
        parI=emission[0], parR=emission[1], parD=emission[2])
    # save result
    _results.save_result((sim_lat,sim_obs), dates, region, params)

if __name__ == '__main__':
    # countries
    #run('CZ')
    #run('SE')
    run('IT')
    # CZ regions
    #run('CZ010')
    #run('CZ020')
    #run('CZ031')
    #run('CZ032')
    #run('CZ041')
    #run('CZ042')
    #run('CZ051')
    #run('CZ052')
    #run('CZ053')
    #run('CZ063')
    #run('CZ064')
    #run('CZ071')
    #run('CZ072')
    #run('CZ080')
    # SE regions
    #run('SE110')
    #run('SE121')
    #run('SE122')
    #run('SE123')
    #run('SE124')
    #run('SE125')
    #run('SE211')
    #run('SE212')
    #run('SE213')
    #run('SE214')
    #run('SE221')
    #run('SE224')
    #run('SE231')
    #run('SE232')
    #run('SE311')
    #run('SE312')
    #run('SE313')
    #run('SE321')
    #run('SE331')
    #run('SE332')
    # IT regions
    run('ITC1')
    run('ITC2')
    run('ITC3')
    run('ITC4')
    run('ITF1')
    run('ITF2')
    run('ITF3')
    run('ITF4')
    run('ITF5')
    run('ITF6')
    run('ITG1')
    run('ITG2')
    run('ITH10')
    run('ITH20')
    run('ITH3')
    run('ITH4')
    run('ITH5')
    run('ITI1')
    run('ITI2')
    run('ITI3')
    run('ITI4')