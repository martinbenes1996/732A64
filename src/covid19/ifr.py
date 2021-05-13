# -*- coding: utf-8 -*-
"""IFR internal module.

Module containing operation for IFR (infection fatality ratio).
IFR is modelled with a Uniform distribution, by default U(.004,.01).

Example:
    Module constructs the plot of IFR with
    
        ifr.plot()
        
    It is also possible to get simulations of IFR
    
        sim100 = ifr.rvs(100)
        

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform,bernoulli

def rvs(size=1e4, a=.004, b=.01):
    """Draw a random sample of IFR.
    
    Args:
        size (int): Number of samples.
        a,b (float): Parameters of the distribution U(a,b).
    Returns:
        np.array: IFR sample.
    """
    sample = uniform.rvs(a, b - a, size = int(size), random_state=12345)
    return sample

def pdf(size=1e4, a=.004, b=.01):
    """Make a grid of IFR for plotting.
    
    Args:
        size (int): Number of values in grid.
        a,b (float): Parameters of the distribution U(a,b).
    Returns:
        np.array: Grid of x-axis.
        np.array: Values of U(a,b) PDF for the grid.
    """
    xgrid = np.linspace(a - (b-a)/4, b + (b-a)/4, num=int(size))
    fx = uniform.pdf(xgrid, a, b - a)
    return xgrid, fx

def plot(save = False, name = 'img/sir/ifr.png'):
    """Plots IFR simulation together with the theoretical density.
    
    After call, use `plt.show()` to show the figure.
    
    Args:
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    # simulate
    draws = rvs()
    # get density
    xgrid, fx = pdf()
    # plot
    fig1, ax1 = plt.subplots()
    ax1.hist(draws, density = True, bins = 50, alpha = .3)
    ax1.plot(xgrid, fx)
    ax1.set_xlabel('IFR')
    ax1.set_ylabel('Density')
    if save: fig1.savefig(name)

