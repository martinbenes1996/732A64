
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
sys.path.append('src')
import population
import posterior

def linear_spline():
    xgrid = np.linspace(-6,6,1000)
    y = []
    for x in xgrid:
        if x <= -3: y.append(-3-x)
        elif x <= 0: y.append(x+3)
        elif x <= 3: y.append(3-2*x)
        else: y.append(.5*x - 4.5)
    plt.plot(xgrid,y)
    for x in [-6,-3,0,3,6]:
        plt.axvline(x = x, color = 'b', alpha=.1)
    plt.show()

def cubic_spline_deg2():
    """Comes from https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation."""
    xgrid = np.linspace(0,3,100)
    y = []
    for x in xgrid:
        if x <= 1: y.append(.5*x**3-.15*x**2+.15*x)
        elif x <= 2: y.append(-1.2*(x-1)**3+1.35*(x-1)**2+1.35*(x-1)+.5)
        else: y.append(1.3*(x-2)**3-2.25*(x-2)**2+.45*(x-2)+2)
    plt.plot(xgrid,y)
    for x in [0,1,2,3]:
        plt.axvline(x = x, color = 'b', alpha=.1)
    plt.show()

def cubic_spline_deg0():
    xgrid = np.linspace(-3,3,100)
    y = []
    for x in xgrid:
        if x <= -1: y.append((x+1)**2)
        elif x <= 1: y.append(.25*(x+1)**2)
        else: y.append(-(2/3*(x-2.5)**2) + 2.5)
    plt.plot(xgrid,y)
    for x in [-3,-1,1,3]:
        plt.axvline(x = x, color = 'b', alpha=.1)
    #plt.scatter(3,42/18)
    plt.show()

def covid_confirmed():
    # get data
    xx = pd.concat([
        posterior._posterior_data(country, (datetime(2020,3,1),datetime(2021,5,1)))
        for country in ['CZ','PL','IT','SE']
    ])
    # population
    POP = {country: population.get_population(country) for country in ['CZ','PL','IT','SE']}
    xx['POP'] = xx.region.apply(POP.get)
    # normalize
    xx['confirmed100K'] = xx.confirmed / xx.POP * 1e5
    # to weekly
    xx['year'] = xx.date.apply(lambda d: int(datetime.strftime(d,'%Y')))
    xx['week'] = xx.date.apply(lambda d: int(datetime.strftime(d,'%W')))
    xx = xx\
        .groupby(['year','week','region'])\
        .aggregate({'confirmed100K': 'sum'})\
        .reset_index(drop=False)
    xx['date'] = xx.apply(lambda r: datetime.strptime('%04d-%02d-1' % (r.year,r.week), '%Y-%W-%w'), axis=1)
    # plot
    fig, ax = plt.subplots(figsize=(8,6))
    for label,df in xx.groupby('region'):
        ax.plot(df.date, df.confirmed100K, label=label)
    ax.set_xlabel('Date')
    ax.set_ylabel('Confirmed cases per 100K')
    plt.legend()
    plt.show()

def covid_deaths():
    # get data
    xx = pd.concat([
        posterior._posterior_data(country, (datetime(2020,3,1),datetime(2021,5,1)))
        for country in ['CZ','PL','IT','SE']
    ])
    # population
    POP = {country: population.get_population(country) for country in ['CZ','PL','IT','SE']}
    xx['POP'] = xx.region.apply(POP.get)
    # normalize
    xx['deaths100K'] = xx.deaths / xx.POP * 1e5
    # to weekly
    xx['year'] = xx.date.apply(lambda d: int(datetime.strftime(d,'%Y')))
    xx['week'] = xx.date.apply(lambda d: int(datetime.strftime(d,'%W')))
    def q025(x): return x.quantile(0.)
    def q975(x): return x.quantile(1.)
    xx = xx\
        .groupby(['year','week','region'])\
        .aggregate({'deaths100K': 'sum'})\
        .reset_index(drop=False)
    xx['date'] = xx.apply(lambda r: datetime.strptime('%04d-%02d-1' % (r.year,r.week), '%Y-%W-%w'), axis=1)
    # plot
    fig, ax = plt.subplots(figsize=(8,6))
    for label,df in xx.groupby('region'):    
        ax.plot(df.date, df.deaths100K, label=label)
    ax.set_xlabel('Date')
    ax.set_ylabel('Deaths per 100K')
    plt.legend()
    plt.show()

def covid_recovered():
    # get data
    countries = ['CZ','PL','IT']#,'SE'] ,
    xx = pd.concat([
        posterior._posterior_data(country, (datetime(2020,3,1),datetime(2021,5,1)))
        for country in countries
    ])
    # population
    POP = {country: population.get_population(country) for country in countries}
    xx['POP'] = xx.region.apply(POP.get)
    # normalize
    xx['deaths100K'] = xx.deaths / xx.POP * 1e5
    # to weekly
    xx['year'] = xx.date.apply(lambda d: int(datetime.strftime(d,'%Y')))
    xx['week'] = xx.date.apply(lambda d: int(datetime.strftime(d,'%W')))
    def q025(x): return x.quantile(0.)
    def q975(x): return x.quantile(1.)
    #xx = xx\
    #    .groupby(['year','week','region'])\
    #    .aggregate({'recovered100K': 'sum'})\
    #    .reset_index(drop=False)
    #xx['date'] = xx.apply(lambda r: datetime.strptime('%04d-%02d-1' % (r.year,r.week), '%Y-%W-%w'), axis=1)
    # plot
    fig, ax = plt.subplots(figsize=(8,6))
    for label,df in xx.groupby('region'):
        alpha = 1 if label != 'PL' else .5
        ax.plot(df.date, df.deaths100K, label=label, alpha=alpha)
    ax.set_xlabel('Date')
    ax.set_ylabel('Deaths per 100K')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    #cubic_spline_deg0()
    #covid_deaths()
    #covid_confirmed()
    covid_recovered()