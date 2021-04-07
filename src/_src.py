
import covid19dh
import covid19czechia as CZ
import covid19poland as PL
import covid19sweden as SE
from datetime import datetime
import numpy as np
import pandas as pd

def _se_tests(level = 1):
    """"""
    # read data
    x = pd.read_csv('data/se_tests.csv')
    if level == 1:
        x = x[x.Region.isna()]
        # fill date
        x['Monday'] = x.Monday.apply(lambda m: datetime.strptime(m, '%Y-%m-%d'))
        r = pd.date_range(start=x.Monday.min(), end=x.Monday.max())
        x = x.set_index('Monday').reindex(r)
        x['Tests'] = x.Tests.fillna(method='ffill')
        x['Performed'] = x.Performed.fillna(method='ffill')
        x['Tests'] = x.Tests.fillna(0.0)
        x['Performed'] = x.Performed.fillna(0.0)
        x = x.rename_axis('date').reset_index()
        # parse
        x['Year'] = x.date.apply(lambda i: int(i.strftime("%Y")))
        x['Week'] = x.date.apply(lambda i: int(i.strftime("%W")))
        x['Year'] = x.Year.apply(int)
        x['Week'] = x.Week.apply(int)
        x['Tests'] = x.Tests.apply(int)
        x['Performed'] = x.Performed.apply(int)
        # sort
        x = x.sort_values(by = 'date', axis = 0)
    else:
        x = x[~x.Region.isna()]\
            .sort_values(['Monday','Region'])
    return x

def _dead_gender():
    """"""
    # read data
    x = pd.read_csv('data/dead_gender.csv')
    x.columns = ['date','deaths_m','deaths_f']
    # parse
    x['date'] = x.date.apply(lambda i: datetime.strptime(i, "%Y-%m-%d"))
    x['deaths_m'] = x.deaths_m\
        .fillna(0.0)\
        .apply(int)
    x['deaths_f'] = x.deaths_f\
        .fillna(0.0)\
        .apply(int)
    x['deaths'] = x.deaths_m + x.deaths_f
    # result
    return x[['date','deaths']]

def _ffill0(x):
    x = x\
        .fillna(method='ffill')\
        .fillna(0.)\
        .apply(int)
    return x
def _cum_to_new(x):
    x = x\
        .reset_index(drop=True)
    res = [float(x[0])]
    for i in x.diff()[1:]:
        res.append(float(i))
    x = _ffill0(pd.Series(res))
    return x.to_numpy()
    
def get_data():
    """"""
    # load
    x,sources = covid19dh.covid19(['CZE','ITA','Poland','SWE'], verbose = False)
    # parse
    data = None
    for country,country_data in x.groupby('iso_alpha_3'):
        # Sweden
        if country == 'SWE':
            se_tests = _se_tests()[['date','Tests']]\
                .rename({'Tests': 'tests'}, axis = 1)
            country_data = country_data\
                .drop('tests', axis=1)\
                .merge(se_tests, how = 'outer', on = 'date')
            country_data['tests'] = country_data.tests / 7.
        # fill from top
        country_data['tests'] = _ffill0(country_data.tests)
        country_data['confirmed'] = _ffill0(country_data.confirmed)
        country_data['deaths'] = _ffill0(country_data.deaths)
        country_data['recovered'] = _ffill0(country_data.recovered)
        # cumulative to newly
        if country != 'SWE':
            country_data['tests'] = _cum_to_new(country_data.tests)
        country_data['confirmed'] = _cum_to_new(country_data.confirmed)
        country_data['deaths'] = _cum_to_new(country_data.deaths)
        country_data['recovered'] = _cum_to_new(country_data.recovered)
        # data
        if data is None:
            data = country_data
        else:
            data = pd.concat([data, country_data])
    return data[['date','tests','confirmed','recovered','deaths','iso_alpha_3']]\
        .reset_index(drop = True)

def _CZ_data():
    """"""
    # load czech
    def _per_day(df, key):
        return df\
            .groupby(['date','week','region'])\
            .aggregate({key: 'sum'})\
            .reset_index()
    cz_deaths = CZ.covid_deaths(level = 2, usecache=True)
    cz_confirmed = CZ.covid_confirmed(level = 2, usecache=True)
    cz_tests = CZ.covid_tests(level = 2, usecache=True)
    cz_recovered = CZ.covid_recovered(level = 2, usecache=True)
    # merge czech
    cz = _per_day(cz_tests, 'tests')\
        .merge(_per_day(cz_confirmed,'confirmed'), how='outer', on=['date','week','region'])\
        .merge(_per_day(cz_deaths,'deaths'), how='outer', on=['date','week','region'])\
        .merge(_per_day(cz_recovered,'recovered'), how='outer', on=['date','week','region'])\
        .fillna(0)\
        .sort_values(['date','region'])
    
    return cz
    
def _PL_data():
    # fetch deaths and tests per region
    pl_deaths = PL.covid_deaths(level = 2,from_github = True)\
        .rename({'NUTS2': 'region'}, axis=1)
    pl_tests = PL.covid_tests(level = 2, offline = False, from_github=True)
    print(pl_tests)
    return

def _SE_data(level = 1):
    # fetch data
    se_data = SE.covid_deaths(level = level)
    se_data_change = (se_data.year == 2020) & (se_data.week == 53)
    se_data.loc[se_data_change,'year'] = 2021
    se_data.loc[se_data_change,'week'] = 0
    se_data.loc[se_data.year == 2021,'week'] = se_data.loc[se_data.year == 2021,'week'] + 1
    # fetch tests
    se_tests = _se_tests(level = level)\
            .rename({'Year':'year','Week':'week','Tests':'tests','Region':'region'}, axis=1)
    # country level
    if level == 1:
        # select columns
        se_data = se_data[['year','week','deaths','confirmed']]
        se_tests = se_tests[['year','week','tests']]
        # to weeks
        se_tests = se_tests[~se_tests.duplicated(['year','week'])]
        # merge
        se = se_data.merge(se_tests, how='outer', on=['year','week'])
        
    # region level
    else:
        # select columns
        se_data = se_data[['year','week','region','deaths','confirmed']]
        se_tests = se_tests[['year','week','region','tests']]
        # merge
        se = se_data.merge(se_tests, how='outer', on=['year','week','region'])
        
    # postprocess
    se['date'] = se.apply(lambda r: datetime.strptime(f'{int(r.year)}-{int(r.week)}-1','%Y-%W-%w'), axis=1)
    se = se[~se.deaths.isna() & ~se.confirmed.isna() & ~se.tests.isna()]
    se = se.sort_values('date').reset_index(drop = True)
    return se
    
x = _SE_data(level = 2)
#x.to_csv('res.csv', index=False)
print(x)
#from matplotlib import pyplot as plt
#plt.plot(x.date, x.tests)
#plt.show()