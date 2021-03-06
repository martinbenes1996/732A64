
import covid19dh
import covid19czechia as CZ
import covid19poland as PL
import covid19sweden as SE
from datetime import datetime
import math
import numpy as np
import pandas as pd

# make NUTS mapping
def _name_to_nuts():
    nuts = pd.read_csv('data/demo/NUTS.csv')
    return {r.name: r.nuts for r in nuts.itertuples()}

class cache:
    @staticmethod
    def read(country, level):
        """"""
        name = f'tmp/cache/{country}_{level}.csv'
        try: x = pd.read_csv(name)
        except: return None
        x['date'] = x.date.apply(lambda d: datetime.strptime(d, "%Y-%m-%d"))
        x['week'] = x.date.apply(lambda d: int(d.strftime('%W')))
        #x['week'] = x.week.apply(int)
        for c in set(x.columns) - {'date','week','region'}:
            x[c] = x[c].apply(float)
        #print(x)
        return x
    @staticmethod
    def write(x, country, level):
        """"""
        name = f'tmp/cache/{country}_{level}.csv'
        x.to_csv(name, index = False)

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
    x = pd.read_csv('data/gender/dead_gender.csv')
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
    
def _CZ_data(level = 1):
    """"""
    # read cache
    cz = cache.read(country='CZ', level=level)
    #cz['week'] = cz.date.apply(lambda dt: int(datetime.strftime(dt, '%W')))
    #cz['year'] = cz.date.apply(lambda dt: int(datetime.strftime(dt, '%Y')))
    #cz['date'] = cz.apply(lambda r: datetime.strptime('%04d-%02d-1'%(r.year,r.week), '%Y-%W-%w'), axis=1)
    if cz is not None: return cz
    # aggregation
    def _aggregator(keys):
        def _per_day(df, key):
            df['week'] = df.date.apply(lambda d: int(d.strftime('%W')))
            df = df\
                .groupby(keys)\
                .aggregate({key: 'sum'})\
                .reset_index()
            return df
        return _per_day
    on_cols = ['date','week'] if level == 1 else ['date','week','region']
    aggregate = _aggregator(on_cols)  
    # fetch data
    cz_deaths = CZ.covid_deaths(level = level, usecache=True)
    cz_confirmed = CZ.covid_confirmed(level = level, usecache=True)
    cz_tests = CZ.covid_tests(level = level, usecache=True)
    cz_recovered = CZ.covid_recovered(level = level, usecache=True)
    # parse
    cz_deaths = aggregate(cz_deaths,'deaths')
    cz_confirmed = aggregate(cz_confirmed,'confirmed')
    #cz_tests['date'] = cz_tests.date.apply(lambda d: datetime.strptime(d,'%Y-%m-%d'))
    cz_tests = aggregate(cz_tests,'tests')
    cz_recovered = aggregate(cz_recovered,'recovered')
    # merge czech
    cz = cz_tests\
        .merge(cz_confirmed, how='outer', on=on_cols)\
        .merge(cz_deaths, how='outer', on=on_cols)\
        .merge(cz_recovered, how='outer', on=on_cols)\
        .sort_values(on_cols)
    cz['week'] = cz.date.apply(lambda dt: int(datetime.strftime(dt, '%W')))
    #cz.reset_index(drop=True).to_csv('here.csv')
    if level == 1:
        cz['region'] = 'CZ'
    cache.write(cz, country='CZ', level=level)
    return cz

def _PL_confirmed(level = 1):
    # country
    if level == 1:
        pass
    # region
    else:
        # load
        df_confirmed = pd.read_excel(
            io = 'data/COVID-19 w Polsce.xlsx', engine='openpyxl',
            sheet_name='Wzrost w województwach', skiprows=7, nrows=17, header=None)
        # crop and set column names
        df_confirmed.columns = pd.Series(['region']).append(df_confirmed.loc[0,1:].apply(str))
        df_confirmed = df_confirmed.iloc[1:,:-2]
        # wide to long
        df_confirmed = pd.melt(df_confirmed, id_vars='region', var_name='date', value_name='confirmed')
        df_confirmed['date'] = df_confirmed.date.apply(lambda dt: datetime.strptime(dt, '%Y-%m-%d %H:%M:%S'))
        df_confirmed['region'] = df_confirmed.region.apply(str)\
            .replace({'Świętokrzyskie *': 'Świętokrzyskie'})\
            .apply(_name_to_nuts().get)
        df_confirmed['confirmed'] = df_confirmed.confirmed.apply(float)
        # merge
        df = df_confirmed\
            .sort_values(['date','region'])\
            .reset_index(drop=True)
    df['week'] = df.date.apply(lambda dt: datetime.strftime(dt, '%W'))
    return df

def _PL_tests(level = 1):
    # country
    if level == 1:
        # read
        df_tests = pd.read_excel(
            io = 'data/COVID-19 w Polsce.xlsx', engine='openpyxl',
            sheet_name='Testy', skiprows=2, nrows=411, header=None)
        # columns
        df_tests = df_tests.loc[1:,1:5]
        df_tests.columns = ['date','x1','x2','x3','tests']
        df = df_tests[['date','tests']]
    # region
    else:
        # load
        df_tests = pd.read_excel(
            io = 'data/COVID-19 w Polsce.xlsx', engine='openpyxl',
            sheet_name=' Testy w województwach od 11.05', skiprows=2, nrows=17, header=None)
        # crop and set column names
        df_tests.columns = pd.Series(['region']).append(df_tests.loc[0,1:].apply(str))
        df_tests = df_tests.iloc[1:,:-5]
        # wide to long
        df_tests = pd.melt(df_tests, id_vars='region', var_name='date', value_name='tests')
        df_tests['date'] = df_tests.date.apply(lambda dt: datetime.strptime(dt, '%Y-%m-%d %H:%M:%S'))
        df_tests['region'] = df_tests.region\
            .apply(str)\
            .replace({'Świętokrzyskie *': 'Świętokrzyskie'})\
            .apply(_name_to_nuts().get)
        df_tests['tests'] = df_tests.tests.apply(float)
        # load
        df_tests2 = pd.read_excel(
            io = 'data/COVID-19 w Polsce.xlsx', engine='openpyxl',
            sheet_name='Testy w województwach', skiprows=3, nrows=17, header=None)
        # crop and set column names
        df_tests2 = df_tests2.loc[:,1:]
        df_tests2.columns = pd.Series(['region']).append(df_tests2.loc[0,2:].apply(str))
        df_tests2 = df_tests2.iloc[1:,:-2]
        # wide to long
        df_tests2 = pd.melt(df_tests2, id_vars='region', var_name='date', value_name='tests')
        df_tests2['date'] = df_tests2.date.apply(lambda dt: datetime.strptime(dt, '%Y-%m-%d %H:%M:%S'))
        df_tests2['region'] = df_tests2.region\
            .apply(str)\
            .apply(_name_to_nuts().get)
        df_tests2['tests'] = df_tests2.tests.apply(float)
        # merge
        df = df_tests.append(df_tests2)\
            .sort_values(['date','region'])\
            .reset_index(drop=True)
        df = df[['date','region','tests']]
    df['week'] = df.date.apply(lambda dt: int(datetime.strftime(dt, '%W')))
    return df

def _PL_data(level = 1):
    # read cache
    pl = cache.read(country='PL', level=level)
    if pl is not None: return pl
    # fetch data
    pl_tests = _PL_tests(level=level)
    # country level
    if level == 1:
        # read
        pl = pd.read_excel(
            io = 'data/COVID-19 w Polsce.xlsx', engine='openpyxl',
            sheet_name='Wzrost', skiprows=2, nrows=411, header=None)
        pl = pl.loc[:,:8]
        pl.columns = ['date','confirmed','x1','x2','x3','x4','x5','deaths','recovered']
        pl = pl[['date','confirmed','deaths','recovered']]
        pl['week'] = pl.date.apply(lambda dt: int(datetime.strftime(dt, '%W')))
        # merge
        pl = pl\
            .merge(pl_tests, how='outer', on=['date','week'])\
            .sort_values(['date','week'])\
            .reset_index(drop = True)
        pl = pl[~pl.deaths.isna() & ~pl.tests.isna() & ~pl.confirmed.isna() & ~pl.recovered.isna()]
        pl['region'] = 'PL'
    # region
    else:
        # deaths
        pl_deaths = PL.covid_deaths(level = level, from_github = True)
        pl_deaths = pl_deaths\
            .rename({'NUTS2': 'region'}, axis = 1)\
            .groupby(['date','week','region'])\
            .aggregate({'deaths': 'sum'})\
            .reset_index()
        pl_deaths_9 = pl_deaths[pl_deaths.region.isin({'PL91','PL92'})]\
            .groupby(['date','week'])\
            .aggregate({'deaths': 'sum'})\
            .reset_index()
        pl_deaths_9['region'] = 'PL9'
        pl_deaths = pl_deaths[~pl_deaths.region.isin({'PL91','PL92'})]\
            .append(pl_deaths_9, ignore_index=True)
        pl_deaths['week'] = pl_deaths.date.apply(lambda dt: int(datetime.strftime(dt, '%W')))
        # tests
        pl_tests = pl_tests[['date','week','region','tests']]
        pl_tests['week'] = pl_tests.week.apply(int)
        # confirmed
        pl_confirmed = _PL_confirmed(level=level)
        pl_confirmed['week'] = pl_confirmed.date.apply(lambda dt: int(datetime.strftime(dt, '%W')))
        pl_confirmed = pl_confirmed[['date','week','region','confirmed']]
        # merge
        pl = pl_deaths\
            .merge(pl_tests, how='outer', on=['date','week','region'])\
            .merge(pl_confirmed, how='outer', on=['date','week','region'])\
            .sort_values(['date','week','region'])\
            .reset_index(drop = True)\
            .fillna(0)
    cache.write(pl, country='PL', level=level)
    return pl

def _SE_data(level = 1):
    # read cache
    se = cache.read(country='SE', level=level)
    if se is not None: return se
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
        if level == 1:
            se['region'] = 'SE'
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
    cache.write(se, country='SE', level=level)
    return se

def _IT_data(level = 1):
    # read cache
    it = cache.read(country='IT', level=level)
    if it is not None: return it
    # fetch
    x,_ = covid19dh.covid19('ITA', level = level, verbose = False)
    # country level
    if level == 1:
        x = x[['date','confirmed','tests','deaths','recovered']]\
            .sort_values('date')\
            .reset_index(drop = True)
        # to diff
        x['confirmed'] = x.confirmed.diff().fillna(x.loc[0,'confirmed'])
        x['tests'] = x.tests.diff().fillna(x.loc[0,'tests'])
        x['deaths'] = x.deaths.diff().fillna(x.loc[0,'deaths'])
        x['recovered'] = x.recovered.diff().fillna(x.loc[0,'recovered'])
        x = x[~x.confirmed.isna() & ~x.tests.isna() & ~x.deaths.isna() & ~x.recovered.isna()]
        if level == 1:
            x['region'] = 'IT'
    # region level
    else:
        name_mapper = _name_to_nuts()
        x['region'] = x.administrative_area_level_2.apply(name_mapper.get)
        x = x[['date','region','confirmed','tests','deaths','recovered']]\
            .sort_values('date')\
            .reset_index(drop = True)
        # to diff
        for reg,df in x.groupby('region'):
            x.loc[x.region == reg,'confirmed'] = df.confirmed\
                .diff().fillna(df.reset_index(drop=True).loc[0,'confirmed'])
            x.loc[x.region == reg,'tests'] = df.tests\
                .diff().fillna(df.reset_index(drop=True).loc[0,"tests"])
            x.loc[x.region == reg,'deaths'] = df.deaths\
                .diff().fillna(df.reset_index(drop=True).loc[0,"deaths"])
            x.loc[x.region == reg,'recovered'] = df.recovered\
                .diff().fillna(df.reset_index(drop=True).loc[0,"recovered"])
    cache.write(x, country='IT', level=level)
    return x

def get_data(region, weekly=False):
    """"""
    # parse input
    region = region.upper().strip()
    level = 1 if len(region) == 2 else 2
    # fetch data
    if region[:2] == 'CZ':
        x = _CZ_data(level = level)
    elif region[:2] == 'PL':
        x = _PL_data(level = level)
    elif region[:2] == 'SE':
        x = _SE_data(level = level)
    elif region[:2] == 'IT':
        x = _IT_data(level = level)
    else:
        raise NotImplementedError(f'not implemented region {region}')
    # filter region
    x = x[x.region == region]
    # weekly
    if weekly:
        x['year'] = x.date.apply(lambda d: int(datetime.strftime(d, '%Y')))
        x['date'] = x.apply(lambda r: datetime.strptime(f'{r.year}-{r.week}-1', '%Y-%W-%w'), axis=1)
        x = x\
            .groupby(['date','region'])\
            .sum()\
            .reset_index(drop=False)
        x['year'] = x.date.apply(lambda d: int(datetime.strftime(d, '%Y')))
        x['week'] = x.date.apply(lambda d: int(datetime.strftime(d, '%W')))
        
    return x

#x = get_data(region='CZ010', weekly=True)
#x = x[(x.date >= datetime(2020,12,28)) & (x.date <= datetime(2021,1,7))]
#print(x)
