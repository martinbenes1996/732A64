
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
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

def plot(path):
    plot_params(path, save = True)
    plot_eird(path, save = True)
    plot_s(path, save = True)

if __name__ == '__main__':
    try:
        params = sys.argv[1]
    except:
        print("Usage: python src/result.py result/<result-dir>", file = sys.stderr)
        exit(1)
    plot(params)


