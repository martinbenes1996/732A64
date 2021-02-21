
from scipy.stats import lognorm,norm,gamma,beta,norm
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 18})

import sys
sys.path.append('src')
import _incubation
import _symptoms

def EI():
    # draw from incubation period
    pars = _incubation.continuous()['gamma']
    draws = gamma.rvs(*pars, size = 10000, random_state = 12345)
    # fit beta to 1/draw
    samples = 1 / draws
    return {'x': samples,
            'beta': beta.fit(samples)}
    
def plot_EI():
    # get fit
    fit = EI()
    # generate curve
    xgrid = np.linspace(0,1,100)
    fx = beta.pdf(xgrid, *fit['beta'])
    # plot
    plt.hist(fit['x'], density = True, bins = 50)
    plt.plot(xgrid,fx)
    plt.xlabel('1 / Incubation')
    plt.ylabel('Density')
    plt.show()

def IR():
    # draw from symptoms period
    pars = _symptoms.continuous()['gamma']
    draws = gamma.rvs(*pars, size = 10000, random_state = 54321)
    # fit beta to 1/draw
    samples = 1 / draws
    return {'x': samples[samples < 1],
            'beta': beta.fit(samples, loc = 0)}
    
def plot_IR():
    # get fit
    fit = IR()
    # generate curve
    xgrid = np.linspace(0,1,1000)
    fx = beta.pdf(xgrid, *fit['beta'])
    # plot
    plt.hist(fit['x'], density = True, bins = 300)
    plt.plot(xgrid,fx)
    plt.xlabel('1 / Symptoms')
    plt.ylabel('Density')
    plt.xlim(0,1)
    plt.show()

def priors():
    return {
        'incubation': {
            'distribution': 'beta',
            'params': EI()['beta']
        },
        'symptoms': {
            'distribution': 'beta',
            'param': IR()['beta']
        }
    }
    
if __name__ == "__main__":
    #plot_EI()
    #plot_IR()
    print(priors())
    