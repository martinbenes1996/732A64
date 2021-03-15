
import matplotlib.pyplot as plt
import pandas as pd
import re
from scipy.stats import multinomial
import seaborn as sns
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

def _populations_data():
    # get data
    x = _get_populations()
    # filter everything but countries 2020
    x = x[x['geo\\time'].apply(_filter_country(regional = False)) &
          x.sex.isin(['M','F']) & ~x.age.isin(['TOTAL','UNK'])]
    x = x[['sex','age','geo\\time','2020']]\
        .reset_index(drop = True)\
        .rename({'geo\\time': 'region', '2020': 'population'}, axis = 1)
    x['age'] = x.age.apply(str)
    def get_age_start(i):
        x = re.match(r'(90)|(85)|(\d+)_\d+', i)
        for i in range(1,4):
            if x[i] is not None:
                return int(x[i])
        else:
            raise RuntimeError(f'invalid format of age: {i}')
    x['age_start'] = x.age.apply(get_age_start)
    def get_age_end(i):
        x = re.match(r'(90)|(85)|\d+_(\d+)', i)
        for i in range(1,4):
            if x[i] is not None:
                if i == 1: return 99
                elif i == 2: return 89
                else: return int(x[i])
        else:
            raise RuntimeError(f'invalid format of age: {i}')
    x['age_end'] = x.age.apply(get_age_end)
    return x


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

def plot_population_violin(df = None, save = False, name = 'img/demographic/population.png'):

    # fetch data
    df = _populations_data() if df is None else df
    print(df)
    #return
    # upsample
    cases = {'sex': [], 'age': [], 'country': []}
    for row in df.itertuples():
        age_cat = row.age_end - row.age_start + 1
        random_pops = multinomial.rvs(int(row.population / 100), [1/age_cat]*age_cat)#, random_state = 12345)
        ages = list(range(row.age_start, row.age_end + 1))
        for age,deaths in zip(ages, random_pops):
            for _ in range(deaths):
                cases['country'].append(row.region)
                cases['sex'].append(row.sex)
                cases['age'].append(age)
    cases = pd.DataFrame(cases)\
        .sort_values(by = 'sex', ascending = False)
    cases['date'] = None
    
    print(cases)
    # plot
    plt.rcParams.update({'font.size': 20})
    sns.violinplot(x="country", y="age", hue="sex", data = cases)
    if save: plt.savefig(name)

if __name__ == '__main__':
    plot_population_violin()
    plt.show()
    #x = population(save = True)
    #print(x)