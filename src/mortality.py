
from datetime import datetime
import math
import matplotlib.pyplot as plt
import pandas as pd
import re
from scipy.stats import multinomial, ttest_ind, f, t
import seaborn as sns

import eurostat_deaths as eurostat

import sys
sys.path.append('src')
import population

#import logging
#logging.basicConfig(level = logging.INFO)

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

def plot_mortality_population(years = [2020]):
    # fetch data
    df = _mortality_data()
    # filter year
    df = df[df.year.isin(years)]
    # join population
    df = df\
        .merge(pops, on=['region','sex','age_start','age_end','age'], suffixes=('','_2'))
    df.deaths = df.deaths / df.population
    

pops = population._populations_data()
pops.population = pops.population / 1000
def test_countries_equal(c1, c2, years = [2020]):
    # fetch data
    df = _mortality_data()
    # filter year
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
    # f test pi value
    fpi = 1 - f.cdf(fstat, f_df1, f_df2)
    
    # t test
    if fpi > .05:
        tstat = abs(mu1 - mu2) / math.sqrt(var_pooled * (n1+n2)/n1/n2)
        t_df = n1 + n2 - 2
    else:
        tstat = abs(mu1 - mu2) / math.sqrt(var1 / n1 + var2 / n2)
        t_df = (var1**2/n1 + var2**2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    # t test pi value
    tpi = 1 - t.cdf(tstat, t_df)
    print("Countries %s - %s" % (c1, c2))
    print("* F-test %.5f [%s]" % (fpi, 'Y' if fpi > .05 else 'N'))
    print("* T-test %.5f [%s]" % (tpi, 'Y' if tpi > .05 else 'N'))

def test_country_age_equal(c1, years = [2020]):
    # fetch data
    df = _mortality_data()
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
    # f test pi value
    fpi = 1 - f.cdf(fstat, f_df1, f_df2)
    
    # t test
    if fpi > .05:
        tstat = (mu1 - mu2) / math.sqrt(var_pooled * (n1+n2)/n1/n2)
        t_df = n1 + n2 - 2
    else:
        tstat = (mu1 - mu2) / math.sqrt(var1 / n1 + var2 / n2)
        t_df = (var1**2/n1 + var2**2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    # t test pi value
    tpi = 1 - t.cdf(tstat, t_df)
    print("Country %s" % (c1))
    print("* F-test %.5f [%s]" % (fpi, 'Y' if fpi > .05 else 'N'))
    print("* T-test %.5f [%s]" % (tpi, 'Y' if tpi > .05 else 'N'))

def CZ_mortality():
    x = _mortality_data()
    print(x)
    x = x[x.region == 'CZ']\
        .groupby(['year','week'])\
        .aggregate({'deaths':'sum'})\
        .reset_index()
    x = x[x.week <= 53]
    x['date'] = x.apply(lambda r: datetime.strptime(f'{r.year}-{r.week}-1','%Y-%W-%w'), axis=1)
    
    print(x)
    x.plot(x = 'date', y = 'deaths')
    plt.show()

def plot_0_4(country):
    # get data
    print(country)
    x = _mortality_data()
    # filter poland
    age_groups = ['0_4','5_9','10_14','15_19']
    x = x[x.region.apply(lambda r: r[:2] == country) &#x.region.apply(lambda r: len(r) == 2) &
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
    g = sns.FacetGrid(x, col="age")
    g.map(sns.lineplot, 'week', 'Deaths per 100K', 'year')
    plt.legend()
    plt.savefig(f'img/discussion/age_{country}.png')
    plt.show()

def test_PL_0_4_greater():
    # get data
    x = _mortality_data()
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
    print('PL-CZ', t_pl_cz)
    print('PL-IT', t_pl_it)
    print('PL-SE', t_pl_se)
    print('CZ-IT', t_cz_it)
    print('CZ-SE', t_cz_se)
    print('IT-SE', t_it_se)


if __name__ == '__main__':
    test_PL_0_4_greater()
    #plot_0_4('PL')
    #plot_0_4('CZ')
    #plot_0_4('IT')
    #plot_0_4('SE')
    #plot_0_4('FR')
    #plot_0_4('DE')
    
#plot_mortality_violin()
#plt.show()

#test_countries_equal('IT','SE')
#test_countries_equal('IT','PL')
#test_countries_equal('IT','CZ')
#test_countries_equal('SE','PL')
#test_countries_equal('SE','CZ')
#test_countries_equal('PL','CZ')

#CZ_mortality()

#test_country_age_equal('IT')
#test_country_age_equal('SE')
#test_country_age_equal('PL')
#test_country_age_equal('CZ')

#plot_mortality_population