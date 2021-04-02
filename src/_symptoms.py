
import numpy as np
import pandas as pd
from scipy.stats import lognorm,norm,gamma,beta
import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = (12,10)
#plt.rcParams.update({'font.size': 18})


def continuous():
    """"""
    # fetch data (https://doi.org/10.1038/s41467-020-20568-4)
    path = 'data/41467_2020_20568_MOESM4_ESM.xlsx'
    df = pd.read_excel(path, engine='openpyxl')
    x = df['duration of symptoms in days']\
        .apply(int)\
        .to_numpy()
    x[x == 0] = 1
    # fit distributions
    return {
        'x': x,
        'norm': norm.fit(x),
        'lognorm': lognorm.fit(x, floc=0),
        'gamma': gamma.fit(x, floc=0)
    }

def continuous_plot(save = False, name = 'img/parameters/symptoms.png'):
    """"""
    # get distribution
    fit = continuous()
    # generate pdf
    xgrid = np.linspace(1,40,100)
    y_norm = norm.pdf(xgrid, *fit['norm'])
    y_lognorm = lognorm.pdf(xgrid, *fit['lognorm'])
    y_gamma = gamma.pdf(xgrid, *fit['gamma'])
    # plot
    plt.hist(fit['x'], bins = 40, alpha = .6, density=True)
    plt.plot(xgrid, y_norm, label = 'Norm(%.3f,%.3f)' % fit['norm'][:2])
    plt.plot(xgrid, y_lognorm, label = 'Lognorm(%.3f,%.3f)' % (fit['lognorm'][0],fit['lognorm'][2]))
    plt.plot(xgrid, y_gamma, label = 'Gamma(%.3f,%.3f)' % (fit['gamma'][0],fit['gamma'][2]))
    plt.xlabel('Days from symptom onset')
    plt.ylabel('Density')
    plt.legend()
    if save: plt.savefig(name)

def distribution_aic():
    """"""
    # get distribution
    fit = continuous()
    # aic
    aic_norm = 6 - 2*np.sum( norm.logpdf(fit['x'], *fit['norm']) )
    aic_lognorm = 6 - 2*np.sum( lognorm.logpdf(fit['x'], *fit['lognorm']) )
    aic_gamma = 6 - 2*np.sum( gamma.logpdf(fit['x'], *fit['gamma']) )
    return {
        'norm': aic_norm,
        'lognorm': aic_lognorm,
        'gamma': aic_gamma
    }
    
def discrete(N = 40):
    """"""
    # fit distributions
    fit = continuous()
    params = fit['gamma']
    # discretize
    probs = []
    for i in range(N):
        P = gamma.cdf(i+1, *params) - gamma.cdf(i, *params)
        probs.append(P)
    distribution = pd.DataFrame({'x': range(N), 'Px': probs})
    return distribution
def discrete_plot(N = 40, save = False, name = 'img/parameters/symptoms_discrete.png'):
    """"""
    # get disribution
    fit = continuous()
    distribution = discrete(N = N)
    xgrid = np.linspace(0, N - 2, 1000)
    def find_X(i):
        idx = np.argmax(i < distribution.x) - 1
        prob = distribution.loc[idx].Px
        return prob
    grid_probs = pd.Series(xgrid).apply(find_X)
    # plot
    plt.plot(xgrid, grid_probs, label='Discretized Gamma(%.3f,%.3f)' % (fit['gamma'][0],fit['gamma'][2]))
    plt.xlabel('Days from symptom onset')
    plt.ylabel('Density')
    plt.legend()
    if save: plt.savefig(name)

if __name__ == '__main__':
    continuous_plot(save=True)
    plt.show()
#discrete_plot(save = True)
#print(continuous())
#print(distribution_aic())