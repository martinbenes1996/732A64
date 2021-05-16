"""TODO

Example:
    TODO

        TODO
"""
from datetime import datetime
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.path.append('src')
from demographic import population
import posterior

_cache = None
def _load_data(dates):
    """
    
    Args:
        dates ():
    """
    global _cache
    if _cache is not None:
        return _cache
    # regions
    with open('model/regions.json') as fp:
        regions = [k for k in json.load(fp) if len(k) > 2]
    # fetch data
    data_c,data_d,data_r = None,None,None
    dateaxis = None
    regions_r = []
    for reg in regions:
        x = posterior._posterior_data(reg,dates,weekly=True)
        if dateaxis is None:
            dateaxis = x.date
        POP = population.get_population(reg)
        # normalize by popylation
        x['I1K'] = x.confirmed / POP * 1e3
        x['D1K'] = x.deaths / POP * 1e3
        # confirmed
        c = x.I1K.to_numpy().reshape((1,-1))
        data_c = np.concatenate([data_c,c],axis=0) if data_c is not None else c
        # deaths
        d = x.D1K.to_numpy().reshape((1,-1))
        data_d = np.concatenate([data_d,d],axis=0) if data_d is not None else d
        # recovered
        if 'recovered' in x:
            x['R1K'] = x.recovered / POP * 1e3
            r = x.recovered.to_numpy().reshape((1,-1))
            data_r = np.concatenate([data_r,r],axis=0) if data_r is not None else r
            regions_r.append(reg)
    _cache = (data_c,data_d,data_r),dateaxis,(regions,regions,regions_r)
    return _cache

def _plot_clusters(data, dates, regions):
    """
    
    Args:
        data ():
        dates ():
        regions ():
    """
    sns.clustermap(
        data,
        metric="cosine",
        col_cluster=False,
        cbar_pos=None,
        yticklabels=regions,
        xticklabels=dates.apply(lambda dt: datetime.strftime(dt,'%Y-w%W')),
        figsize=(12,10))
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

def plot_confirmed():
    """"""
    data,dates,cols = _load_data((datetime(2020,8,1),datetime(2021,3,15)))
    _plot_clusters(data[0], dates, cols[0])
    
def plot_deaths():
    """"""
    data,dates,cols = _load_data((datetime(2020,8,1),datetime(2021,3,15)))
    _plot_clusters(data[1], dates, cols[1])
    
def plot_recovered():
    """"""
    data,dates,cols = _load_data((datetime(2020,8,1),datetime(2021,3,15)))
    _plot_clusters(data[2], dates, cols[2])
