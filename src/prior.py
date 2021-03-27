
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import lognorm,norm,gamma,beta,norm,uniform,dweibull,argus,triang
from sklearn.utils import resample
import sys

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 10})
sys.path.append('src')

import _ifr
import _incubation
import _src
import _symptoms
import _testing
import population

def EI():
    """"""
    # seed
    np.random.seed(seed=12345)
    # draw from incubation period
    pars = _incubation.continuous()['gamma']
    draws = gamma.rvs(*pars, size = 1000000, random_state = 12345)
    # fit beta to 1/draw
    samples = 1 / draws
    samples = samples[(samples > 0) & (samples < 1)]
    return {'x': samples,
            'beta': beta.fit(samples),
            'gamma': gamma.fit(samples, loc = .2, scale = 10)}
    
def IR():
    """"""
    # seed
    np.random.seed(seed=12345)
    # draw from symptoms period
    pars = _symptoms.continuous()['gamma']
    draws = gamma.rvs(*pars, size = 100000, random_state = 54321)
    draws_ifr = _ifr.rvs(size = 100000)
    # fit beta to 1 / draw
    samples = (1 - draws_ifr) / draws
    samples = samples[(samples > 0) & (samples < 1)]
    return {'x': samples,
            'beta': beta.fit(samples),
            'gamma': gamma.fit(samples)}

def ID():
    """"""
    # seed
    np.random.seed(seed=12345)
    # draw from symptoms period
    pars = _symptoms.continuous()['gamma']
    draws = gamma.rvs(*pars, size = 100000, random_state = 54321)
    draws_ifr = _ifr.rvs(size = 100000)
    # fit beta to 1 / draw
    samples = draws_ifr / draws
    samples = samples[(samples > 0) & (samples < 1)]
    return {'x': samples,
            'beta': beta.fit(samples),
            'gamma': gamma.fit(samples)}

def draw_R0(K):
    return uniform.rvs(2,2, size = K)

def SI():
    """"""
    # seed
    np.random.seed(seed=12345)
    # get ir
    fit_IR = IR()
    fit_ID = ID()
    # sample
    K = 100000
    r0 = draw_R0(K)
    ir = beta.rvs(*fit_IR['beta'][:2], size = K)
    id = beta.rvs(*fit_ID['beta'][:2], size = K)
    # 
    samples = r0 * (ir + id)
    #print(samples)
    #samples = samples[samples < 1]
    return {'x': samples,
            #'gamma': gamma.fit(samples),
            'weib': dweibull.fit(samples, floc = 0)}

def _get_beta_label(params):
    # parameters
    a = params[0]
    b = params[1]
    mu = params[2]
    s = params[3]
    # format
    label = 'Beta((X - %.3f)/%.3f ; a=%.3f, b=%.3f)'
    return label % (mu, s, a, b)
def _get_gamma_label(params):
    # parameters
    a = params[0]
    b = 1 / params[2]
    mu = params[1]
    # format
    label = 'Gamma(X - %.3f ; a=%.3f, b=%.3f)'
    return label % (mu, a, b)

def plot_R0(save = False, name = 'img/parameters/R0.png'):
    # get fit
    r0 = draw_R0(10000)
    # generate curve
    xgrid = np.linspace(1.5,4.5,1000)
    fx = uniform.pdf(xgrid, 2,2)
    # plot
    fig1, ax1 = plt.subplots()
    ax1.hist(r0, density = True, bins = 50)
    ax1.plot(xgrid, fx)
    ax1.set_xlabel('R0')
    ax1.set_ylabel('Density')
    # save plot
    if save: fig1.savefig(name)
    

def plot_SI(save = False, name = 'img/sir/SI.png'):
    # get fit
    fit = SI()
    # generate curve
    xgrid = np.linspace(0,2e-6,100)
    fx = dweibull.pdf(xgrid, *fit['weib']) * 2
    # plot
    fig1, ax1 = plt.subplots()
    ax1.hist(fit['x'], density = True, bins = 150)
    ax1.plot(xgrid, fx)
    ax1.set_xlabel('R0 * (Deaths + Recovered)')
    ax1.set_ylabel('Density')
    ax1.set_xlim(0,2e-6)
    # save plot
    if save: fig1.savefig(name)

def plot_EI(save = False, name = 'img/sir/EI.png'):
    """"""
    # get fit
    fit = EI()
    # generate curve
    xgrid = np.linspace(0,1,100)
    fx_gamma = gamma.pdf(xgrid, *fit['gamma'])
    fx_beta = beta.pdf(xgrid, *fit['beta'])
    # plot
    fig1, ax1 = plt.subplots()
    ax1.hist(fit['x'], density = True, bins = 50)
    gamma_label = _get_gamma_label(fit['gamma'])
    beta_label = _get_beta_label(fit['beta'])
    ax1.plot(xgrid,fx_gamma, label=gamma_label)
    ax1.plot(xgrid,fx_beta, label=beta_label)
    ax1.set_xlabel('1 / Incubation')
    ax1.set_ylabel('Density')
    ax1.legend()
    # save plot
    if save: fig1.savefig(name)
    
def plot_IR(save = False, name = 'img/sir/IR.png'):
    """"""
    # get fit
    fit = IR()
    # generate curve
    xgrid = np.linspace(0,1,1000)
    fx = beta.pdf(xgrid, *fit['beta'])
    # plot
    fig1, ax1 = plt.subplots()
    ax1.hist(fit['x'], density = True, bins = 100)
    ax1.plot(xgrid,fx)
    ax1.set_xlabel('(1-IFR) / Symptoms')
    ax1.set_ylabel('Density')
    ax1.set_xlim(0,1)
    # save plot
    if save: fig1.savefig(name)

def plot_ID(save = False, name = 'img/sir/ID.png'):
    """"""
    # get fit
    fit = ID()
    # generate curve
    xgrid = np.linspace(0,.005,1000)
    fx = beta.pdf(xgrid, *fit['beta'])
    # plot
    fig1, ax1 = plt.subplots()
    ax1.hist(fit['x'], density = True, bins = 200)
    ax1.plot(xgrid,fx)
    ax1.set_xlabel('IFR / Symptoms')
    ax1.set_ylabel('Density')
    ax1.set_xlim(0,.005)
    # save plot
    if save: fig1.savefig(name)

def plot_parameters():
    plot_SI(save = True)
    plot_EI(save = True)
    plot_IR(save = True)
    plot_ID(save = True)

def priors(save = False, name = 'data/distr/prior.json'):
    """"""
    _si = SI()['weib']
    _ei = EI()['beta']
    _ir = IR()['beta']
    _id = ID()['beta']
    prior_params = {
        'SI': {
            'distribution': 'weib',
            'params': _si
        },
        'EI': {
            'distribution': 'beta',
            'params': _ei
        },
        'IR': {
            'distribution': 'beta',
            'param': _ir
        },
        'ID': {
            'distribution': 'beta',
            'param': _id
        }
    }
    if save:
        with open(name,'w') as fp:
            json.dump(prior_params, fp, indent = 2)
    return prior_params

def test_prior(save = False, name = 'data/distr/testratio.csv'):
    """"""
    # get data
    pop = population.countries()
    tests = _testing.tests()
    # group
    iso3_iso2 = {'CZE':'CZ','SWE':'SE','POL':'PL','ITA':'IT'}
    for country3 in tests.country.unique():
        # get country population
        country2 = iso3_iso2[country3]
        country_pop = float(pop.population[pop.region == country2])
        # normalize data by population
        country_tests = tests[tests.country == country3].tests
        tests.loc[tests.country == country3,'ratio'] = country_tests / country_pop    
    tests = tests[['country','date','ratio']]
    # save
    if save: tests.to_csv(name, index = False)
    return tests

def tested(country = None, date_min = None, date_max = None):
    # get data
    df = test_prior()
    # filter by country and dates

    if country is not None:
        df = df[df.country == country]
    if date_min is not None:
        df = df[df.date >= date_min]
    if date_max is not None:
        df = df[df.date <= date_max]
    # perform computation
    return {
        'x': df.ratio,
        'beta': beta.fit(df.ratio)
    }

def confirmed_prior(save = False, name = 'data/distr/confirmedratio.csv'):
    """"""
    # get data
    pop = population.countries()
    df = _src.get_data()
    tests = _testing.tests()
    # group
    iso3_iso2 = {'CZE':'CZ','SWE':'SE','POL':'PL','ITA':'IT'}
    for country3 in df.iso_alpha_3.unique():
        # get country population
        country2 = iso3_iso2[country3]
        country_pop = float(pop.population[pop.region == country2])
        # normalize confirmed by tests
        country_confirmed = df[df.iso_alpha_3 == country3].confirmed.apply(lambda c: c if c > 0 else 1)
        country_tests = tests[tests.country == country3].tests.apply(lambda t: t if t > 0 else 1)
        df.loc[df.iso_alpha_3 == country3,'ratio'] = (country_confirmed / country_tests).fillna(0)
        df['ratio'] = df.ratio.apply(lambda r: r if r < 1 else 1 - 1e-6)
        df['ratio'] = df.ratio.apply(lambda r: r if r > 0 else 1e-6)
        df[df.iso_alpha_3 == country3]['tests'] = country_tests
        df[df.iso_alpha_3 == country3]['confirmed'] = country_confirmed
    df = df[['iso_alpha_3','date','confirmed','tests','ratio']]
    confirmed_fit = beta.fit(df.ratio, floc = 0, fscale = 1)
    # save
    if save: df.to_csv(name, index = False)
    return df

def plot_test_prior(cmap = {}):
    # get ratio
    x = test_prior()
    x['month'] = x.date.apply(lambda d: d.strftime("%Y-%m"))
    # estimate CI
    fig1, ax1 = plt.subplots()
    for country,country_data in x.groupby('country'):
        df_tests = {'date': [], 'mu': [], 'ci_low': [], 'ci_high': []}
        for month,monthly in country_data.groupby('month'):
            # data
            monthly_ratio = monthly.ratio.to_numpy()
            monthly_ratio = monthly_ratio[~np.isnan(monthly_ratio)]
            # nonparametric bootstrap
            gen_bs_sample = lambda: resample(monthly_ratio, replace=True, n_samples=3)
            bs_sample = np.array([np.mean(gen_bs_sample()) for _ in range(1000)])
            # statistical description
            ci_mean = np.mean(monthly_ratio)
            ci_low = ci_mean - 1.96 * np.std(bs_sample)
            ci_high = ci_mean + 1.96 * np.std(bs_sample)
            # append
            df_tests['date'].append(monthly.date.min())
            df_tests['mu'].append(ci_mean)
            df_tests['ci_low'].append(max(ci_low,0))
            df_tests['ci_high'].append(ci_high)
        # plot
        df_tests = pd.DataFrame(df_tests)
        color = cmap.get(country, 'k')
        ax1.plot(df_tests.date, df_tests.mu, color = color, label = country)
        ax1.fill_between(df_tests.date, df_tests.ci_low,  df_tests.ci_high, color = color, alpha = .1)
    ax1.legend()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('P ( Tested )')

def plot_test_ratio_all(save = False, name = 'img/parameters/test_ratio.png'):
    plot_test_prior(cmap = {'CZE': 'r','ITA': 'g', 'SWE': 'b'})
    if save: plt.savefig(name)

#plot_SI()
#plt.show()
#priors(save = True)
#test_prior(save = True)

#confirmed_prior(save = True)

if __name__ == "__main__":
    #plot_IR()
    #plt.show()
    #plot_ID()
    #plt.show()
    #plot_SI()
    #plt.show()
    #plot_parameters()
    priors(save = True)
    #plot_parameters()
    #test_prior(save = True)
    #confirmed_prior(save = True)
    #pr = tested(country = 'CZE')
    #print(pr)
    
    #plot_R0(save = True)
    #plt.show()
    
    