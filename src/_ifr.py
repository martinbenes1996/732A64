
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform,bernoulli

def rvs(a = .004, b = .01, size = 10000):
    # create draws
    draws = uniform.rvs(a, b - a, size = size, random_state=12345)
    return draws
def pdf(a = .004, b = .01, size = 10000):
    # create draws
    xgrid = np.linspace(a - (b-a)/4, b + (b-a)/4, num=size)
    fx = uniform.pdf(xgrid, a, b - a)
    #draws = uniform.rvs(a, b - a, size = size, random_state=12345)
    #draws = bernoulli.rvs(prior_draw, size = size, random_state=12345)
    return xgrid, fx

def plot_ifr(save = False, name = 'img/sir/ifr.png'):
    # simulate
    draws = rvs(size = 10000)
    # get density
    xgrid, fx = pdf()
    # plot
    fig1, ax1 = plt.subplots()
    ax1.hist(draws, density = True, bins = 50, alpha = .3)
    ax1.plot(xgrid, fx)
    ax1.set_xlabel('IFR')
    ax1.set_ylabel('Density')
    if save: fig1.savefig(name)

if __name__ == "__main__":
    plot_ifr(save = True)
    plt.show()
