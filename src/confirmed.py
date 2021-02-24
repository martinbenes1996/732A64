
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import resample
import sys
sys.path.append('src')

import _testing

def plot_test_ratio(*args, cmap = {}, **kw):
    # get ratio
    x = _testing._get_test_ratios(*args, **kw)
    x['month'] = x.date.apply(lambda d: d.strftime("%Y-%m"))
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
        plt.plot(df_tests.date, df_tests.mu, color = color, label = country)
        plt.fill_between(df_tests.date, df_tests.ci_low,  df_tests.ci_high, color = color, alpha = .1)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('P ( Infected | Tested )')
    plt.show()

def test_ratios(*args, **kw):
    # get ratio
    x = _testing._get_test_ratios(*args, **kw)
    # compute average per country
    return x\
        .groupby('country')\
        .aggregate({'ratio': 'mean'})\
        .reset_index()
        
def export_tests():
    r = _testing.tests()
    r.to_csv('data/tests.csv', index=False)
    
def plot_test_ratio_all():
    plot_test_ratio(cmap = {'CZE': 'r','ITA': 'g', 'SWE': 'b'})
