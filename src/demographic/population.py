# -*- coding: utf-8 -*-
"""Population mortality internal module.

Module containing operations with population.
Population comes from Eurostat database.

Example:
    Fetch mortality data with
    
        data = mortality.data()
        
    
"""
import math
import matplotlib.pyplot as plt
import pandas as pd
import re
from scipy.stats import multinomial, t, f
import seaborn as sns
import eurostat_deaths

def _get_populations():
    """Fetches population from Eurostat."""
    return eurostat_deaths.populations()

def _filter_country(regional = False):
    """Returns function to filter the regions (`regional` is True) or countries."""
    def _filter_f(s):
        """Filters region `s`."""
        # regions
        if regional:
            res = False
            res = res or (s[:2] == 'CZ' and len(s) == 5)
            res = res or (s[:2] == 'SE' and len(s) == 5)
            res = res or (s[:2] == 'PL' and len(s) == 4 and s[:3] != 'PL9')
            res = res or (s[:2] == 'PL' and len(s) == 3)
            res = res or (s[:2] == 'IT' and s in {'ITH10','ITH20'})
            res = res or (s[:2] == 'IT' and len(s) == 4 and s not in {'ITH1','ITH2'})
        # countries
        else:
            res = s in ['CZ','IT','PL','SE']
        # result
        return res
    return _filter_f

def _populations_data():
    """Fetches and parses the populations' data."""
    # get data
    x = _get_populations()
    # filter everything but countries 2020
    x = x[x['geo\\time'].apply(_filter_country(regional = False)) &
          x.sex.isin(['M','F']) & ~x.age.isin(['TOTAL','UNK'])]
    x = x[['sex','age','geo\\time','2020']]\
        .reset_index(drop = True)\
        .rename({'geo\\time': 'region', '2020': 'population'}, axis = 1)
    # parse age group start
    x['age'] = x.age.apply(str)
    def get_age_start(i):
        """Parses age start from the age group `i`."""
        x = re.match(r'(90)|(85)|(\d+)_\d+', i)
        for i in range(1,4):
            if x[i] is not None:
                return int(x[i])
        else:
            raise RuntimeError(f'invalid format of age: {i}')
    x['age_start'] = x.age.apply(get_age_start)
    # parse age group end
    def get_age_end(i):
        """Parses age end from the age group `i`."""
        x = re.match(r'(90)|(85)|\d+_(\d+)', i)
        for i in range(1,4):
            if x[i] is not None:
                if i == 1: return 99
                elif i == 2: return 89
                else: return int(x[i])
        else:
            raise RuntimeError(f'invalid format of age: {i}')
    x['age_end'] = x.age.apply(get_age_end)
    # return
    return x

def _countries():
    """Fetches country populations data."""
    # get data
    x = _get_populations()
    # filter everything but countries 2020
    x = x[(x['geo\\time'].apply(_filter_country(regional = False))) &
          (x.sex == 'T') & (x.age == 'TOTAL')]
    x = x[['geo\\time','2020']]\
        .reset_index(drop = True)\
        .rename({'geo\\time': 'region', '2020': 'population'}, axis = 1)
    return x

def _regions():
    """Fetches regions' populations data."""
    # get data
    x = _get_populations()
    # filter everything but regions 2020
    x = x[(x['geo\\time'].apply(_filter_country(regional = True))) &
          (x.sex == 'T') & (x.age == 'TOTAL')]
    x = x[['geo\\time','2020']]\
        .reset_index(drop = True)\
        .rename({'geo\\time': 'region', '2020': 'population'}, axis = 1)
    return x

def population(save = False, name = 'tmp/cache/population.csv'):
    """Get populations, both regional and country.
    
    Args:
        save (bool, optional): Whether to cache or not.
        name (str, optional): Path of caching.
    """
    # load data
    try: return pd.read_csv(name)
    except: pass
    # get data
    x_c = _countries()
    x_r = _regions()
    # concat
    x = pd.concat([x_c, x_r])
    # save
    if save: x.to_csv(name, index = False)
    return x

def get_population(region):
    """Get population of a region.
    
    Args:
        region (str): Region to get population of.
    """
    x = population()
    return int(x.loc[x.region == region,'population'])

def plot_violin(save = False, name = 'img/demographic/population.png'):
    """Constructs a violin plot of population (so called demographic curve).
    
    Args:
        save (bool, optional): Whether to cache or not.
        name (str, optional): Path of caching.
    """
    # fetch data
    df = _populations_data()
    # upsample
    cases = {'sex': [], 'age': [], 'country': []}
    for row in df.itertuples():
        age_cat = row.age_end - row.age_start + 1
        random_pops = multinomial.rvs(int(row.population / 100), [1/age_cat]*age_cat)
        ages = list(range(row.age_start, row.age_end + 1))
        for age,deaths in zip(ages, random_pops):
            for _ in range(deaths):
                cases['country'].append(row.region)
                cases['sex'].append(row.sex)
                cases['age'].append(age)
    cases = pd.DataFrame(cases)\
        .sort_values(by = 'sex', ascending = False)
    cases['date'] = None
    # plot
    fig1, ax1 = plt.subplots()
    sns.violinplot(x="country", y="age", hue="sex", data = cases, ax=ax1)
    if save: fig1.savefig(name)

def test_countries_equal(c1, c2):
    """Test countries' populations are equal.
    
    Args:
        c1,c1 (str): Countries to compare.
    """
    # fetch data
    df = _populations_data()\
        .groupby(['age','region','age_start','age_end'])\
        .aggregate({'population': 'sum'})\
        .reset_index()
    df['population'] = df.population / 1e4
    # filter country data
    country1 = df[df.region == c1]
    country2 = df[df.region == c2]
    # country statistics
    n1 = country1.population.sum()
    n2 = country2.population.sum()
    x1 = (country1.age_end + country1.age_start) / 2
    x2 = (country2.age_end + country2.age_start) / 2
    mu1 = (country1.population @ x1) / n1
    mu2 = (country2.population @ x2) / n2
    var1 = (country1.population @ (x1 - mu1)**2) / n1
    var2 = (country2.population @ (x2 - mu2)**2) / n2
    var_pooled = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
    # f test
    if var1 > var2:
        f_df1,f_df2 = n1-1,n2-1
        fstat = var1 / var2
    else:
        f_df1,f_df2 = n2-1,n1-1
        fstat = var2 / var1
    fpi = 1 - f.cdf(fstat, f_df1, f_df2) # f test pi value
    # t test
    if fpi > .05:
        tstat = abs(mu1 - mu2) / math.sqrt(var_pooled * (n1+n2)/n1/n2)
        t_df = n1 + n2 - 2
    else:
        tstat = abs(mu1 - mu2) / math.sqrt(var1 / n1 + var2 / n2)
        t_df = (var1**2/n1 + var2**2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    tpi = 1 - t.cdf(tstat, t_df) # t test pi value
    return {
        'country1': c1,
        'country2': c2,
        'f_pi': fpi, 'f_accept': 'Y' if fpi > .05 else 'N',
        't_pi': tpi, 't_accept': 'Y' if tpi > .05 else 'N'
    }
