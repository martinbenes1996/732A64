# -*- coding: utf-8 -*-
"""Covid-19 tests internal module.

Module containing operations with tests.

Example:
    List distributions with its parameters by

        data = tests.get()
        
    Plot positive test ratio with
    
        tests.plot_positive_test_ratio()
    
"""
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import resample
from . import src

def get(country = None, date_min = None, date_max = None):
    """Fetch the Covid-19 tests data.
    
    Args:
        country (str|list): Country or list of countries {'CZ','IT','SE','PL'}.
        date_min,date_max (str): Date range of the data in format `YY-mm-dd`.
    Returns:
        (pandas.DataFrame): Dataframe with columns `country`,`date`,`tests`,`confirmed`,`ratio`.
    """
    # fetch data
    if country is None:
        country = ['CZ','IT','SE','PL']
    else:
        try: country = [*country]
        except: country = [country]
    df_countries = []
    for c in country:
        df_country = src.get_data(c)
        df_country['iso_alpha_3'] = c
        df_countries.append(df_country)
    x = pd.concat(df_countries)
    # filter segment
    if date_min is not None:
        date_min = datetime.strptime(date_min,'%Y-%m-%d')-timedelta(days=1)
        x = x[(x.date >= date_min)]
    if date_max is not None:
        date_max = datetime.strptime(date_max,'%Y-%m-%d')
        x = x[(x.date < date_max)]
    # parse
    res = pd.DataFrame({
        'country': x.iso_alpha_3,
        'date': x.date,
        'tests': x.tests,
        'confirmed': x.confirmed
    })
    # compute test ratio
    res['ratio'] = res.confirmed / res.tests
    res['ratio'] = res.ratio.fillna(0.0)
    res['ratio'] = res.ratio.apply(lambda i: i if abs(i) != float('inf') else 0)
    return res

cmap = {'CZ': 'r','IT': 'g', 'SE': 'b'}
def plot_positive_test_ratio(*args, save=False, name='img/parameters/positive_tests_ratio.png', **kw):
    """Constructs plot of positive tests' ratio.
    
    Args:
        *args (): Delegated to tests.get().
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
        **kw (): Delegated to tests.get().    
    """
    global cmap
    # get ratio
    x = get(*args, **kw)
    x['month'] = x.date.apply(lambda d: d.strftime("%Y-%m"))
    # plot
    fig1, ax1 = plt.subplots()
    # estimate CI
    for country,country_data in x.groupby('country'):
        df_tests = {'date': [], 'mu': [], 'ci_low': [], 'ci_high': []}
        for month,monthly in country_data.groupby('month'):
            # data
            monthly_ratio = monthly.ratio.to_numpy()
            monthly_ratio = monthly_ratio[~np.isnan(monthly_ratio)]
            # nonparametric bootstrap
            gen_bs_sample = lambda: resample(monthly_ratio, replace=True, n_samples=3)
            bs_sample = np.array([np.mean(gen_bs_sample()) for _ in range(1000)])
            # statistical description
            ci_mean = np.mean(monthly_ratio)
            ci_low = ci_mean - 1.96 * np.std(bs_sample)
            ci_high = ci_mean + 1.96 * np.std(bs_sample)
            # append
            df_tests['date'].append(monthly.date.min())
            df_tests['mu'].append(ci_mean)
            df_tests['ci_low'].append(ci_low)
            df_tests['ci_high'].append(ci_high)
        # plot
        df_tests = pd.DataFrame(df_tests)
        color = cmap.get(country, 'k')
        ax1.plot(df_tests.date, df_tests.mu, color = color, label = country)
        ax1.fill_between(df_tests.date, df_tests.ci_low,  df_tests.ci_high, color = color, alpha = .1)
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('P ( Infected | Tested )')
    if save: fig1.savefig(name)
