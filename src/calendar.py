
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import sys
sys.path.append('src')
import population
import posterior

def load_calendar():
    x = pd.read_csv('data/calendar/restrictions.csv')
    x['Date'] = x.Date.apply(lambda d: datetime.strptime(d, '%Y-%m-%d'))
    x['Year'] = x.Date.apply(lambda d: int(datetime.strftime(d, '%Y')))
    x['Week'] = x.Date.apply(lambda d: int(datetime.strftime(d, '%W')))
    return x

def plot_confirmed(region):
    # load calendar
    cal = load_calendar()
    cal = cal[cal.Country == region]
    cal['Y'] = -.05
    # load data
    x = posterior._posterior_data(region, (datetime(2020,3,1),datetime(2021,3,15)), weekly=False)
    POP = population.get_population(region)
    x['Confirmed cases per 1000 people'] = x.confirmed / POP * 1e3
    
    # plot
    plt.rcParams["figure.figsize"] = (12,6)
    fix,ax = plt.subplots()
    sns.lineplot(x='date',y='Confirmed cases per 1000 people',data=x,ax=ax)
    ax.axhline(0, alpha=.4, color='grey')
    sns.scatterplot(x='Date',y='Y',hue='Restriction',data=cal,ax=ax)
    plt.show()

if __name__ == '__main__':
    #x = load_calendar()
    #print(x[x.Country == 'IT'])
    plot_confirmed('CZ')
    plot_confirmed('SE')
    plot_confirmed('IT')
    plot_confirmed('PL')