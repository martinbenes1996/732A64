# -*- coding: utf-8 -*-
"""Component performing comparison of simulation results.

Module containing operations to compare simulation results.

Example:
    Plot cluster plot of confirmed with
    
        regional.plot_confirmed()
    
    Plot cluster plot of deaths with
    
        regional.plot_deaths()
    
    Plot cluster plot of recovered with
    
        regional.plot_recovered()
        
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
    """Load Covid-19 statistics in the appropriate format.
    
    Args:
        dates (tuple (2)): Date range of the data.
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

def _plot_clusters(data, dates, regions, save=False, name=None):
    """Plot clusters with data.
    
    Args:
        data (np.array): Data matrix to plot.
        dates (pd.Series): Date axis.
        regions (list): List of region, used for labels in the plot.
    """
    # plot
    sns.clustermap(
        data,
        metric="cosine",
        col_cluster=False,
        cbar_pos=None,
        yticklabels=regions,
        xticklabels=dates.apply(lambda dt: datetime.strftime(dt,'%Y-w%W'))
    )
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    if save and name is not None: plt.savefig(name)

def plot_confirmed(save=False, name='img/results/clust_I.png'):
    """Plot clusters of regions using confirmed cases.
    
    Args:
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    data,dates,cols = _load_data((datetime(2020,8,1),datetime(2021,3,15)))
    _plot_clusters(data[0], dates, cols[0])
    
def plot_deaths(save=False, name='img/results/clust_D.png'):
    """Plot clusters of regions using deaths.
    
    Args:
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    data,dates,cols = _load_data((datetime(2020,8,1),datetime(2021,3,15)))
    _plot_clusters(data[1], dates, cols[1])
    
def plot_recovered():
    """Plot clusters of regions using recovered cases.
    
    Args:
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    data,dates,cols = _load_data((datetime(2020,8,1),datetime(2021,3,15)))
    _plot_clusters(data[2], dates, cols[2])
