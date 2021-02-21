
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm,norm,gamma,beta,norm
import sys

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 18})
sys.path.append('src')

import _incubation
import _symptoms

def EI():
    """"""
    # draw from incubation period
    pars = _incubation.continuous()['gamma']
    draws = gamma.rvs(*pars, size = 10000, random_state = 12345)
    # fit beta to 1/draw
    samples = 1 / draws
    return {'x': samples,
            'beta': beta.fit(samples)}
    
def IR():
    """"""
    # draw from symptoms period
    pars = _symptoms.continuous()['gamma']
    draws = gamma.rvs(*pars, size = 10000, random_state = 54321)
    # fit beta to 1/draw
    samples = 1 / draws
    return {'x': samples[samples < 1],
            'beta': beta.fit(samples, loc = 0)}
    
def plot_EI(save = False, name = 'img/sir/EI.png'):
    """"""
    # get fit
    fit = EI()
    # generate curve
    xgrid = np.linspace(0,1,100)
    fx = beta.pdf(xgrid, *fit['beta'])
    # plot
    fig1, ax1 = plt.subplots()
    ax1.hist(fit['x'], density = True, bins = 50)
    ax1.plot(xgrid,fx)
    ax1.set_xlabel('1 / Incubation')
    ax1.set_ylabel('Density')
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
    ax1.hist(fit['x'], density = True, bins = 300)
    ax1.plot(xgrid,fx)
    ax1.set_xlabel('1 / Symptoms')
    ax1.set_ylabel('Density')
    ax1.set_xlim(0,1)
    # save plot
    if save: fig1.savefig(name)

def priors(save = False, name = 'data/distr/prior.json'):
    """"""
    _ei = EI()['beta']
    _ir = IR()['beta']
    prior_params = {
        'EI': {
            'distribution': 'beta',
            'params': list(_ei[:2]),
            'mean': _ei[2],
            'scale': _ei[3]
        },
        'IR': {
            'distribution': 'beta',
            'param': list(_ir[:2]),
            'mean': _ir[2],
            'scale': _ir[3]
        }
    }
    if save:
        with open(name,'w') as fp:
            json.dump(prior_params, fp, indent = 2)
    return prior_params
    
    