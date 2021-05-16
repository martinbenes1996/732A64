# -*- coding: utf-8 -*-
"""Population mortality internal module.

Module containing operations with population.
Population comes from Eurostat database.

Example:
    Fetch mortality data with
    
        data = mortality.data()
        
    Produce mortality violinplot with
    
        mortality.plot_violin()
        
    Produce plot of Polish mortality over years with
    
        mortality.plot_poland_years(range(2010,2021))
        
    Produce plot of Polish mortality in age group 0-4y over years with

        mortality.plot_poland_0_4()
    
    Test that countries are equal in mortality with
    
        mortality.test_countries_equal('CZ','PL')
    
    Test that mortality of genders per country is equal with
    
        mortality.test_country_gender_equal('CZ')
        
    Test that Poland is greater in age group 0-4 years and the rest are equal with
    
        mortality.test_0_4_greater()
    
"""
from datetime import datetime
import eurostat_deaths as eurostat
import math
import matplotlib.pyplot as plt
import pandas as pd
import re
from scipy.stats import multinomial, ttest_ind, f, t
import seaborn as sns
from . import population

def data():
    """Fetch mortality data. By default uses cached version."""
    # fetch
    try: df = pd.read_csv('tmp/cache/deaths.csv')
    except: df = eurostat.deaths()
    # filter
    df = df[df.region.isin(['CZ','PL','SE','IT']) &
            df.sex.isin(['M','F']) &
            ~df.age.isin(['TOTAL','UNK'])]\
        .reset_index(drop = True)
    # parse
    df['deaths'] = df.deaths.apply(lambda i: int(i) if not math.isnan(float(i)) else 0)
    df['age'] = df.age.apply(str)
    def get_age_start(i):
        x = re.match(r'(90)|(\d+)_\d+', i)
        return int(x[1]) if x[1] is not None else int(x[2])
    df['age_start'] = df.age.apply(get_age_start)
    def get_age_end(i):
        x = re.match(r'(90)|\d+_(\d+)', i)
        return 99 if x[1] is not None else int(x[2])
    df['age_end'] = df.age.apply(get_age_end)
    # writeback to cache and return
    df.to_csv('tmp/cache/deaths.csv', index = False)
    return df

def _upsample_mortality(years = None, regions = None):
    """Returns deaggregated (per-case) mortality for density plots.
    
    Args:
        years (list, optional): List with years to contain. All by default.
        regions (list, optional): List with years to contain. All by default.
    Returns:
        (pandas.DataFrame): Upsampled per-case mortality data.
    """
    # get data
    x = data()
    # filter
    if regions is not None:
        x = x[x.region.isin(regions)]
    if years is not None:
        x = x[x.year.isin(years)]
    # upsample
    cases = {'sex': [], 'age': [], 'country': [], 'year': []}
    for row in x.itertuples():
        age_cat = row.age_end - row.age_start + 1
        random_deaths = multinomial.rvs(int(row.deaths / 10), [1/age_cat]*age_cat)#, random_state = 12345)
        ages = list(range(row.age_start, row.age_end + 1))
        for age,deaths in zip(ages, random_deaths):
            for _ in range(deaths):
                cases['country'].append(row.region)
                cases['year'].append(row.year)
                cases['sex'].append(row.sex)
                cases['age'].append(age)
    cases = pd.DataFrame(cases)\
        .sort_values(by = 'sex', ascending = False)
    cases['date'] = None
    # return
    return cases

def plot_violin(save = False, name = 'img/demographic/mortality.png'):
    """Constructs violin plot of mortality in 2020.
    
    Args:
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    # fetch
    cases = _upsample_mortality(years = [2020])
    # plot
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 10})
    sns.violinplot("country", "age", hue="sex", data = cases, ax=ax)
    ax.legend()
    if save: fig.savefig(name)

def plot_poland_years(years = None, save = False, name = 'img/demographic/mortality_pl.png'):
    """Constructs violin plot of mortality in Poland over years.
    
    Args:
        years (list,optional): Years to construct the plot for.
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    # fetch
    cases = _upsample_mortality(years = years, regions = ['PL'])
    # plot
    plt.rcParams.update({'font.size': 10})
    g = sns.FacetGrid(cases, row="sex", hue="sex")
    g.map(sns.violinplot, "year", "age")
    if save: plt.savefig(name)

def plot_poland_0_4(save = False, name = 'img/demographic/mortality_pl_05.png'):
    """Constructs trace plot of mortality in Poland in age group 0-4y.
    
    Args:
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    # get data
    x = data()
    # filter poland
    x = x[(x.region == 'PL') & (x.age.isin(['0_4'])) &
          (x.year < 2021) & (x.week < 54)]\
        .reset_index(drop = True)
    # aggregate
    x = x\
        .groupby(['week','year'])\
        .aggregate({'deaths': 'sum'})\
        .reset_index(drop = False)
    # plot
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 10})
    sns.lineplot(x = 'week', y = 'deaths', hue = 'year', data = x, ax = ax)
    if save: fig.savefig(name)

# cache pops
pops = None
def test_countries_equal(c1, c2, years = [2020]):
    """Hypothesis test of similarity of countries' mortality.
    
    Args:
        c1,c2 (str): Countries' iso2 codes to compare, codes are {'CZ','IT','PL','SE'}.
        years (list): List of years.
    """
    global pops
    if pops is None:
        pops = population._populations_data()
        pops.population = pops.population / 1000
    # fetch data, filter by year
    df = data()
    df = df[df.year.isin(years)]
    # join population
    df = df\
        .merge(pops, on=['region','sex','age_start','age_end','age'], suffixes=('','_2'))
    df.deaths = df.deaths / df.population
    # filter country data
    country1 = df[df.region == c1]
    country2 = df[df.region == c2]
    # country statistics
    n1 = country1.deaths.sum()
    n2 = country2.deaths.sum()
    x1 = (country1.age_end + country1.age_start) / 2
    x2 = (country2.age_end + country2.age_start) / 2
    mu1 = (country1.deaths @ x1) / n1
    mu2 = (country2.deaths @ x2) / n2
    var1 = (country1.deaths @ (x1 - mu1)**2) / n1
    var2 = (country2.deaths @ (x2 - mu2)**2) / n2
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
        'f_pi': fpi,
        'f_accept': 'Y' if fpi > .05 else 'N',
        't_pi': tpi,
        't_accept': 'Y' if tpi > .05 else 'N'
    }

def test_country_gender_equal(c1, years = [2020]):
    """Hypothesis test of similarity of gender countries' mortality.
    
    Args:
        c1 (str): Country to test gender equality, from {'CZ','IT','PL','SE'}.
        years (list): List of years.
    """
    # fetch data
    df = data()
    # filter year
    df = df[df.year.isin(years)]
    # join population
    df = df\
        .merge(pops, on=['region','sex','age_start','age_end','age'], suffixes=('','_2'))
    df.deaths = df.deaths / df.population
    # filter country data
    df1 = df[(df.region == c1) & (df.sex == 'F')]
    df2 = df[(df.region == c1) & (df.sex == 'M')]
    # country statistics
    n1 = df1.deaths.sum()
    n2 = df2.deaths.sum()
    x1 = (df1.age_end + df1.age_start) / 2
    x2 = (df2.age_end + df2.age_start) / 2
    mu1 = (df1.deaths @ x1) / n1
    mu2 = (df2.deaths @ x2) / n2
    var1 = (df1.deaths @ (x1 - mu1)**2) / n1
    var2 = (df2.deaths @ (x2 - mu2)**2) / n2
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
        tstat = (mu1 - mu2) / math.sqrt(var_pooled * (n1+n2)/n1/n2)
        t_df = n1 + n2 - 2
    else:
        tstat = (mu1 - mu2) / math.sqrt(var1 / n1 + var2 / n2)
        t_df = (var1**2/n1 + var2**2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    tpi = 1 - t.cdf(tstat, t_df) # t test pi value
    return {
        'country': c1,
        'f_pi': fpi,
        'f_accept': 'Y' if fpi > .05 else 'N',
        't_pi': tpi,
        't_accept': 'Y' if tpi > .05 else 'N'
    }

def plot_CZ(save = False, name = 'img/demographic/mortality_cz.png'):
    """Plot Czech mortality.
    
    Args:
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    # load data
    x = data()
    x = x[x.region == 'CZ']\
        .groupby(['year','week'])\
        .aggregate({'deaths':'sum'})\
        .reset_index()
    x = x[(x.week <= 53) & (x.year >= 2005)]
    x['date'] = x.apply(lambda r: datetime.strptime(f'{r.year}-{r.week}-1','%Y-%W-%w'), axis=1)
    # plot
    fig, ax = plt.subplots()
    x.plot(x = 'date', y = 'deaths', ax=ax)
    if save: fig.savefig(name)

def plot_children(country, save = False, name = 'img/discussion/age_%s.png'):
    """Constructs trace plot of mortality for age groups 0-4, 5-9, 10-14, 15-19.
    
    Args:
        country (str): Country to construct the plot for, from {'CZ','IT','PL','SE'}.
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    # get data
    x = data()
    name = name % country
    # filter poland
    age_groups = ['0_4','5_9','10_14','15_19']
    x = x[x.region.apply(lambda r: r[:2] == country) &
          (x.age.isin(age_groups)) &
          (x.year >= 2014) & (x.year < 2021) &
          (x.week < 54)]\
        .reset_index(drop = True)
    # aggregate
    x = x\
        .groupby(['year','week','region','age'])\
        .aggregate({'deaths': 'sum'})\
        .reset_index(drop = False)
    x['year'] = x.year.apply(int)
    # get population
    POP = population._get_populations()
    POP = POP[(POP.age.isin(age_groups)) & (POP.sex == 'T') &
              POP['geo\\time'].apply(lambda r: r[:2] == country)]\
        .rename({'geo\\time': 'region'}, axis=1)
    POP = pd.melt(POP.drop(['sex'], axis=1),
                  id_vars=['region','age'], var_name='year', value_name='population')
    POP['year'] = POP.year.apply(int)
    x = x.merge(POP, how='left', on=['year','region','age'])
    # prepare
    x['date'] = x.apply(lambda r: datetime.strptime('%04d-%02d-1'%(r.year,r.week),
                                                    '%Y-%W-%w'), axis=1)
    x['Deaths per 100K'] = x.deaths / x.population * 1e5
    # plot
    g = sns.FacetGrid(x, col="age", col_order=['0_4','5_9','10_14','15_19'])
    g.map(sns.lineplot, 'week', 'Deaths per 100K', 'year')
    plt.ylim(0,3)
    plt.legend()
    if save: plt.savefig(name)

def test_0_4_greater():
    """Test Poland greater in 0-4 years against other countries.
    
    The rest of the countries is tested for equality.
    """
    # get data
    x = data()
    # filter poland
    age_groups = ['0_4','5_9','10_14','15_19']
    x = x[x.region.isin(['CZ','PL','SE','IT']) &
          (x.age.isin(age_groups)) &
          (x.year >= 2014) & (x.year < 2021) &
          (x.week < 54)]\
        .reset_index(drop = True)
    # aggregate
    x = x\
        .groupby(['year','week','region','age'])\
        .aggregate({'deaths': 'sum'})\
        .reset_index(drop = False)
    x['year'] = x.year.apply(int)
    # get population
    POP = population._get_populations()
    POP = POP[(POP.age.isin(age_groups)) & (POP.sex == 'T') &
              POP['geo\\time'].isin(['CZ','PL','SE','IT'])]\
        .rename({'geo\\time': 'region'}, axis=1)
    POP = pd.melt(POP.drop(['sex'], axis=1),
                  id_vars=['region','age'], var_name='year', value_name='population')
    POP['year'] = POP.year.apply(int)
    x = x.merge(POP, how='left', on=['year','region','age'])
    x['deaths100K'] = x.deaths / x.population * 1e5
    x = x[x.year == 2020]
    # filter
    def filter_region(x, reg):
        return x[x.region.apply(lambda r: r[:2] == reg)]\
            .reset_index(drop=True)
    x_pl = filter_region(x,'PL')
    x_it = filter_region(x,'IT')
    x_se = filter_region(x,'SE')
    x_cz = filter_region(x,'CZ')
    # test
    t_pl_cz = ttest_ind(x_pl.deaths100K[x_pl.age == '0_4'],
                        x_cz.deaths100K[x_cz.age == '0_4'], alternative='less')
    t_pl_it = ttest_ind(x_pl.deaths100K[x_pl.age == '0_4'],
                        x_it.deaths100K[x_it.age == '0_4'], alternative='less')
    t_pl_se = ttest_ind(x_pl.deaths100K[x_pl.age == '0_4'],
                        x_se.deaths100K[x_se.age == '0_4'], alternative='less')
    t_cz_it = ttest_ind(x_cz.deaths100K[x_cz.age == '0_4'],
                        x_it.deaths100K[x_it.age == '0_4'], alternative='two-sided')
    t_cz_se = ttest_ind(x_cz.deaths100K[x_cz.age == '0_4'],
                        x_se.deaths100K[x_se.age == '0_4'], alternative='two-sided')
    t_it_se = ttest_ind(x_it.deaths100K[x_it.age == '0_4'],
                        x_se.deaths100K[x_se.age == '0_4'], alternative='two-sided')
    def decide(pvalue, thres = .05):
        return 'Y' if pvalue > thres else 'N'
    return [
        {
            'country1': 'PL', 'country2': 'CZ',
            't_pi': t_pl_cz.pvalue,
            't_accept': decide(t_pl_cz.pvalue)
        },
        {
            'country1': 'PL', 'country2': 'IT',
            't_pi': t_pl_it.pvalue,
            't_accept': decide(t_pl_it.pvalue)
        },
        {
            'country1': 'PL', 'country2': 'SE',
            't_pi': t_pl_se.pvalue,
            't_accept': decide(t_pl_se.pvalue)      
        },
        {
            'country1': 'CZ', 'country2': 'IT',
            't_pi': t_cz_it.pvalue,
            't_accept': decide(t_cz_it.pvalue, .025)
        },
        {
            'country1': 'CZ', 'country2': 'SE',
            't_pi': t_cz_se.pvalue,
            't_accept': decide(t_cz_se.pvalue, .025)
        },
        {
            'country1': 'IT', 'country2': 'SE',
            't_pi': t_it_se.pvalue,
            't_accept': decide(t_it_se.pvalue, .025)
        }
    ]
