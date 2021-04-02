
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import lognorm,gamma,erlang
import scipy
#plt.rcParams["figure.figsize"] = (12,10)
#plt.rcParams.update({'font.size': 18})

# Weibull
class weibull:
    """"""
    def pdf(x, n, a):
        """"""
        return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)
    def ppf(q, n, a):
        """"""
        return n * (-np.log(1 - np.array(q)))**(1/a)

def continuous():
    return {'gamma': (5.807, 0, 1/0.948)}

def continuous_plot():
    """Plot of incubation distributions."""
    # grid
    xgrid = np.linspace(0, 14, 1000)
    # distributions
    lognormal_pdf = lognorm.pdf(xgrid, 0.418, 0, np.exp(1.621))
    gamma_pdf = gamma.pdf(xgrid, 5.807, 0, 1/0.948)
    weibull_pdf = weibull.pdf(xgrid, 6.258, 2.453)
    erlang_pdf = erlang.pdf(xgrid, 6, 0, 0.88)
    # plot
    plt.plot(xgrid, lognormal_pdf, label='LN(1.621,0.418)')
    plt.plot(xgrid, gamma_pdf, label='Gamma(5.807,0.948)')
    plt.plot(xgrid, weibull_pdf, label='W(2.453,6.258)')
    plt.plot(xgrid, erlang_pdf, label='E(6,0.88)')
    plt.xlabel('Incubation period')
    plt.ylabel('Density')
    plt.legend()

def mse():
    """"""
    # quantiles
    quantile_points = [.05,.25,.5,.75,.95]
    lognormal_quantiles = lognorm.ppf(quantile_points, 0.418, 0, np.exp(1.621))
    gamma_quantiles = gamma.ppf(quantile_points, 5.807, 0, 1/0.948)
    weibull_quantiles = weibull.ppf(quantile_points, 6.258, 2.453)
    erlang_quantiles = erlang.ppf(quantile_points, 6, 0, 0.88)
    ref_quantiles = np.array([2.2,3.8,5.1,6.7,11.5])
    # MSE
    return {
        'lognormal': np.mean((lognormal_quantiles - ref_quantiles)**2),
        'gamma': np.mean((gamma_quantiles - ref_quantiles)**2),
        'weibull': np.mean((weibull_quantiles - ref_quantiles)**2),
        'erlang': np.mean((erlang_quantiles - ref_quantiles)**2)
    }

def discrete(N = 21):
    """"""
    probs = []
    for i in range(N):
        P = gamma.cdf(i+1, 5.807, 0, 1/0.948) - gamma.cdf(i, 5.807, 0, 1/0.948)
        probs.append(P)
    distribution = pd.DataFrame({'x': range(N), 'Px': probs})
    return distribution

def discrete_plot(N = 21):
    """"""
    # get distribution
    distribution = discrete(N = 21)
    # grid
    xgrid = np.linspace(0, distribution.shape[0] - 1, 1000)
    def find_X(i):
        idx = np.argmax(i < distribution.x) - 1
        prob = distribution.loc[idx].Px
        return prob
    grid_probs = pd.Series(xgrid).apply(find_X)
    # plot
    plt.plot(xgrid, grid_probs, label='Discretized Gamma(5.807,0.948)')
    plt.xlabel('Incubation period')
    plt.ylabel('Density')
    plt.legend()
    
