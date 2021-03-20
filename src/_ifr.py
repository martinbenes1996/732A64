
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform,bernoulli

def rvs(size = 10000):
    # create draws
    a,b = 0.004,0.01
    draws = uniform.rvs(a, b - a, size = size, random_state=12345)
    #draws = bernoulli.rvs(prior_draw, size = size, random_state=12345)
    return draws

def plot_ifr(save = False, name = 'img/sir/ifr.png'):
    # simulate
    draws = rvs(size = 10000)
    # plot
    fig1, ax1 = plt.subplots()
    ax1.hist(draws, density = True, bins = 50)
    ax1.set_xlabel('IFR')
    ax1.set_ylabel('Density')
    if save: fig1.savefig(name)

if __name__ == "__main__":
    plot_ifr(save = True)
    plt.show()
