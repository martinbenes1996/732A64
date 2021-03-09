
import math
import matplotlib.pyplot as plt
import pandas as pd
import re
from scipy.stats import multinomial, ttest_ind
import seaborn as sns

import eurostat_deaths as eurostat

import logging
logging.basicConfig(level = logging.INFO)

def _mortality_data():
    try:
        df = pd.read_csv('data/deaths.csv')
    except:
        # fetch
        df = eurostat.deaths()
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
    # write
    df.to_csv('data/deaths.csv', index = False)
    return df

def _upsample_mortality(years = None, regions = None):
    # get data
    x = _mortality_data()
    # filter poland
    if regions is not None:
        x = x[x.region.isin(regions)]
    #x = x[(x.region == 'PL')]
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

def plot_mortality_violin(save = False, name = 'img/demographic/mortality.png'):
    # fetch
    cases = _upsample_mortality(years = [2020])
    # plot
    plt.rcParams.update({'font.size': 10})
    sns.violinplot("country", "age", hue="sex", data = cases)
    plt.legend()
    if save: plt.savefig(name)

def plot_poland_years(years = None, save = False, name = 'img/demographic/mortality.png'):
    # fetch
    cases = _upsample_mortality(regions = ['PL'])
    # plot
    plt.rcParams.update({'font.size': 10})
    g = sns.FacetGrid(cases, row="sex", hue="sex")
    print(g.axes[0])
    #g.axes[0].axhline(y=70, ls='--', c='black')
    #g.axes[0][1].axhline(y=80, ls='--', c='black')
    g.map(sns.violinplot, "year", "age")
    #g.axes[0][0].text(70,2010,"70 years")
    #g.axes[0][1].text(80,2010,"80 years")
    if save: plt.savefig(name)

def plot_poland_0_5(save = False, name = 'img/demographic/mortality.png'):
    # get data
    x = _mortality_data()
    # filter poland
    x = x[(x.region == 'SE') & (x.age.isin(['5_9'])) &
          (x.year < 2021) & (x.week < 54)]\
        .reset_index(drop = True)
    # aggregate
    x = x\
        .groupby(['week','year'])\
        .aggregate({'deaths': 'sum'})\
        .reset_index(drop = False)
    # normalize
    #for year in x.year.unique():
    #    denorm = x[x.year == year].deaths.sum()
    #    x.loc[x.year == year,'deaths'] = x[x.year == year].deaths / denorm
    # plot
    plt.rcParams.update({'font.size': 10})
    sns.lineplot(x = 'week', y = 'deaths', hue = 'year', data = x)
    #g = sns.FacetGrid(cases, row="sex", hue="sex")
    #g.map(sns.violinplot, "year", "age")
    if save: plt.savefig(name)

def test_country1_lower(c1, c2):
    # fetch data
    df = _mortality_data()
    # filter country data
    country1 = df[df.region == c1]
    country2 = df[df.region == c2]
    # country statistics
    n1 = country1.deaths.sum()
    n2 = country2.deaths.sum()
    x1 = (country1.age_end + country1.age_start) / 2
    x2 = (country2.age_end + country2.age_start) / 2
    mu1 = (country1.deaths * x1).sum() / n1
    mu2 = (country2.deaths * x2).sum() / n2
    var1 = (country1.deaths * (x1 - mu1)**2).sum() / n1
    var2 = (country2.deaths * (x2 - mu2)**2).sum() / n2
    sd_pooled = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)
    tstat = (mu1 - mu2) / math.sqrt(sd_pooled) / (n1+n2-2)
    
    print(mu1, mu2)
    print(var1, var2)
    print(tstat)
    return
    # upsample
    cases = {'sex': [], 'age': [], 'country': []}
    for row in df.itertuples():
        random_deaths = multinomial.rvs(int(row.deaths / 10), [0.1]*10)#, random_state = 12345)
        ages = list(range(row.age_start, row.age_end + 1))
        for age,deaths in zip(ages, random_deaths):
            for _ in range(deaths):
                cases['country'].append(row.region)
                cases['sex'].append(row.sex)
                cases['age'].append(age)
    cases = pd.DataFrame(cases)\
        .sort_values(by = 'sex', ascending = False)
    cases['date'] = None
    
    # test
    test_res = ttest_ind(country_data1, country_data2, equal_var = False, alternative = 'two-sided')
    print(test_res)
    
plot_mortality_violin()
plt.show()

#test_country1_lower('PL','SE')
