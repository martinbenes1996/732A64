
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import beta
import sys

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 10})

def _param_mean(param):
    return param\
        .groupby('date')\
        .mean()\
        .reset_index()    
def _param_ci(param):
    param_groups = param\
        .groupby('date')
    ci_low = param_groups\
        .quantile(q = .025)\
        .reset_index()
    ci_high = param_groups\
        .quantile(q = .975)\
        .reset_index()
    return ci_low\
        .merge(ci_high, on = 'date', suffixes = ('_low','_high'))
    
def plot_params(path, save = False):
    params = pd.read_csv(f'{path}/params.csv', header=0)
    params['date'] = params.date.apply(lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
    params_mean = _param_mean(params[['date','a','c','b','d']])
    params_ci = _param_ci(params[['date','a','c','b','d']])
    fig,ax = plt.subplots()
    for col in ['a','c','b','d']:
        ax.plot(params_mean.date, params_mean[col], label = col)
        ax.fill_between(params_ci.date, params_ci[col + '_low'], params_ci[col + '_high'], alpha=.1)
    ax.legend()
    if save: fig.savefig(f'{path}/params.png')

def plot_eird(path, save = False):
    params = pd.read_csv(f'{path}/y.csv', header=0)
    params['date'] = params.date.apply(lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
    params_mean = _param_mean(params[['date','e','i','r','d']])
    params_ci = _param_ci(params[['date','e','i','r','d']])
    fig,ax = plt.subplots()
    for col in ['e','i','r','d']:
        ax.plot(params_mean.date, params_mean[col], label = col)
        ax.fill_between(params_ci.date, params_ci[col + '_low'], params_ci[col + '_high'], alpha=.1)
    ax.legend()
    if save: fig.savefig(f'{path}/eird.png')

def plot_s(path, save = False):
    params = pd.read_csv(f'{path}/y.csv', header=0)
    params['date'] = params.date.apply(lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
    params_mean = _param_mean(params[['date','s']])
    params_ci = _param_ci(params[['date','s']])
    fig,ax = plt.subplots()
    for col in ['s']:
        ax.plot(params_mean.date, params_mean[col], label = col)
        ax.fill_between(params_ci.date, params_ci[col + '_low'], params_ci[col + '_high'], alpha=.1)
    ax.legend()
    if save: fig.savefig(f'{path}/s.png')

def plot_param_distribution2(path, save = False):
    # load
    x = pd.read_csv(f'{path}/data.csv', header=0)
    x['date'] = x.date.apply(lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
    # plot distribution
    fig,ax = plt.subplots()
    ax.hist(x.param_a, bins=50, label='Optimized', alpha=.5, density=True)
    # prior density
    with open('data/distr/prior.json') as fp:
        prior = json.load(fp)
    xgrid = np.linspace(0,.25,1000)
    fx = [beta.pdf(i, *prior['SI']['params'][:2]) for i in xgrid]
    ax.plot(xgrid,fx,label='Prior')
    ax.legend()

def plot_param_distribution(path, save = False):
    # load
    x = pd.read_csv(f'{path}/data.csv', header=0)
    x['date'] = x.date.apply(lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
    # plot distribution
    beta_fit = beta.fit(x.param_d.unique(), floc=0, fscale=1)
    fig,ax = plt.subplots()
    ax.hist(x.param_d.unique(), bins=150, label='Optimized', alpha=.5, density=True)
    # prior density
    xgrid = np.linspace(0,.1,1000)
    fx = [beta.pdf(i, *beta_fit) for i in xgrid]
    ax.plot(xgrid,fx,label='Prior')
    ax.legend()

def plot(path):
    plot_param_distribution(path, save = True)
    plt.show()

if __name__ == '__main__':
    try:
        params = sys.argv[1]
    except:
        print("Usage: python src/result.py result/<result-dir>", file = sys.stderr)
        exit(1)
    plot(params)
    


