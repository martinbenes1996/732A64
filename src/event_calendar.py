# -*- coding: utf-8 -*-
"""Event calendar module.

Module containing operations with Covid-19 events listed in calendar.

Example:
    Load calendar of events with

        event_calendar.load_calendar()
    
    Construct plot with events with
    
        event_calendar.plot_confirmed('SE')
        
    Construct plot of single event with
    
        event_calendar.plot_segment(
            region='CZ',
            dates=(datetime.datetime(2020,10,10),datetime.datetime(2020,11,30)),
            event=datetime.datetime(2020,10,28)
        )
    
"""
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import sys
sys.path.append('src')
import demographic
import posterior

def load_calendar():
    """Load calendar."""
    # load
    x = pd.read_csv('data/calendar.csv')
    # process
    x['Date'] = x.Date.apply(lambda d: datetime.strptime(d, '%Y-%m-%d'))
    x['Year'] = x.Date.apply(lambda d: int(datetime.strftime(d, '%Y')))
    x['Week'] = x.Date.apply(lambda d: int(datetime.strftime(d, '%W')))
    return x

def plot_confirmed(region, dates=(datetime(2020,3,1),datetime(2021,3,15)), event=None, save=False, name='img/discussion/restrictions.png'):
    """Create plot of confirmed with restrictions.
    
    Args:
        region (str): Region to make the plot of.
        dates (tuple): Date range (date start, date end).
        event (datetime.datetime): Single event to show.
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    # load calendar
    cal = load_calendar()
    cal = cal[cal.Country == region]
    if event is not None:
        cal = cal[cal.Date == event]
        cal['Restriction'] = cal['Title']
    else:
        cal = cal[(cal.Date >= dates[0]) & (cal.Date <= dates[1])]
    cal['Y'] = -.05
    # load data
    x = posterior._posterior_data(region, dates, weekly=False)
    POP = demographic.population.get_population(region)
    x['Confirmed cases per 1000 people'] = x.confirmed / POP * 1e3
    x = x[(x.date >= dates[0]) & (x.date <= dates[1])]
    # plot
    plt.rcParams["figure.figsize"] = (12,6)
    fig,ax = plt.subplots()
    sns.lineplot(x='date',y='Confirmed cases per 1000 people',data=x,ax=ax)
    ax.axhline(0, alpha=.4, color='grey')
    sns.scatterplot(x='Date',y='Y',hue='Restriction',data=cal,ax=ax)
    if save: fig.savefig(name)
    
def plot_segment(region, dates=(datetime(2020,3,1),datetime(2021,3,15)), event=None, save=False, name='img/discussion/restrictions.png'):
    """Plot restrictions together with segment.
    
    Args:
        region (str): Region to make the plot of.
        dates (tuple): Date range (date start, date end).
        event (datetime.datetime): Single event to show.
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    # load calendar
    cal = load_calendar()
    cal = cal[cal.Country == region]
    if event is not None:
        cal = cal[cal.Date == event]
        cal['Restriction'] = cal['Title']
    else:
        cal = cal[(cal.Date >= dates[0]) & (cal.Date <= dates[1])]
    cal['Y'] = -.05
    # load data
    x = posterior._posterior_data(region, dates, weekly=False)
    POP = demographic.population.get_population(region)
    x['Confirmed cases per 1000 people'] = x.confirmed / POP * 1e3
    x = x[(x.date >= dates[0]) & (x.date <= dates[1])]
    # plot
    plt.rcParams["figure.figsize"] = (12,6)
    fig,ax = plt.subplots()
    ax.axvline(cal.Date, alpha=.4, label='Lockdown', color='blue')
    sns.lineplot(x='date',y='Confirmed cases per 1000 people',data=x,ax=ax)
    ax.axhline(0, alpha=.4, color='grey')
    ax.set_ylim(-.05,2)
    if save: fig.savefig(name)

# plot lockdowns
#plot_segment('CZ', (datetime(2020,10,10),datetime(2020,11,30)), event=datetime(2020,10,28))
#plot_segment('CZ', (datetime(2020,12,15),datetime(2021,1,31)), event=datetime(2020,12,27))
#plot_segment('CZ', (datetime(2021,2,15),datetime(2021,3,31)), event=datetime(2021,2,26))
#plot_segment('IT', (datetime(2020,3,1),datetime(2020,4,30)), event=datetime(2020,3,9))
#plot_segment('IT', (datetime(2020,12,15),datetime(2021,1,31)), event=datetime(2020,12,18))
#plot_segment('PL', (datetime(2020,3,15),datetime(2020,4,30)), event=datetime(2020,3,25))
