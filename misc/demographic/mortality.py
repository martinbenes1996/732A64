
import math
import matplotlib.pyplot as plt
import pandas as pd
import re
from scipy.stats import multinomial
import seaborn as sns

import eurostat_deaths as eurostat

import logging
logging.basicConfig(level = logging.INFO)

def mortality_data():
    
    # fetch
    df = eurostat.deaths()
    df = df[df.region.isin(['CZ','PL','SE','IT']) &
            df.sex.isin(['M','F']) &
            ~df.age.isin(['TOTAL','UNK'])]\
        .reset_index(drop = True)

    # parse
    print(df.deaths.unique())
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

    return df

def plot_mortality_violin(df = None):

    # fetch data
    df = mortality_data() if df is None else df
    
    # upsample
    cases = {'sex': [], 'age': [], 'country': []}
    for row in df.itertuples():
        random_deaths = multinomial.rvs(row.deaths, [0.1]*10)#, random_state = 12345)
        ages = list(range(row.age_start, row.age_end + 1))
        for age,deaths in zip(ages, random_deaths):
            for _ in range(deaths):
                cases['country'].append(row.region)
                cases['sex'].append(row.sex)
                cases['age'].append(age)
    cases = pd.DataFrame(cases)\
        .sort_values(by = 'sex', ascending = False)
    cases['date'] = None
    
    # plot
    plt.rcParams.update({'font.size': 20})
    sns.violinplot(x="country", y="age", hue="sex", data = cases)
    plt.show()

df = mortality_data()
df.to_csv('deaths.csv')
plot_mortality_violin(df)