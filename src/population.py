
import pandas as pd
import eurostat_deaths

def _get_populations(agg = True):
    x = eurostat_deaths.populations()
    return x

def _filter_country(regional = False):
    def _filter_f(s):
        if regional:
            res = False
            res = res or (s[:2] == 'CZ' and len(s) == 5)
            res = res or (s[:2] == 'SE' and len(s) == 5)
            res = res or (s[:2] == 'PL' and len(s) == 4 and s[:3] != 'PL9')
            res = res or (s[:2] == 'PL' and len(s) == 3)
            res = res or (s[:2] == 'IT' and len(s) == 5) # todo
        else:
            res = s in ['CZ','IT','PL','SE']
        return res
    return _filter_f

def countries():
    # get data
    x = _get_populations()
    # filter everything but countries 2020
    x = x[(x['geo\\time'].apply(_filter_country(regional = False))) &
          (x.sex == 'T') & (x.age == 'TOTAL')]
    x = x[['geo\\time','2020']]\
        .reset_index(drop = True)\
        .rename({'geo\\time': 'region', '2020': 'population'}, axis = 1)
    #x.columns = ['region','population']
    return x

def regions():
    # get data
    x = _get_populations()
    # filter everything but regions 2020
    x = x[(x['geo\\time'].apply(_filter_country(regional = True))) &
          (x.sex == 'T') & (x.age == 'TOTAL')]
    x = x[['geo\\time','2020']]\
        .reset_index(drop = True)\
        .rename({'geo\\time': 'region', '2020': 'population'}, axis = 1)
    return x

def population(save = False, name = 'data/population.csv'):
    # load data
    try: return pd.read_csv(name)
    except: pass
    # get data
    x_c = countries()
    x_r = regions()
    # concat
    x = pd.concat([x_c, x_r])
    # save
    if save: x.to_csv(name, index = False)
    return x

if __name__ == '__main__':
    x = population(save = True)