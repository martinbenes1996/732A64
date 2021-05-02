# -*- coding: utf-8 -*-
"""IFR internal module.

Module containing operation for IFR (infection fatality ratio).

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

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
    sample = uniform.rvs(a, b - a, size = size, random_state=12345)
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
    xgrid = np.linspace(a - (b-a)/4, b + (b-a)/4, num=size)
    fx = uniform.pdf(xgrid, a, b - a)
    return xgrid, fx

def plot_ifr(save = False, name = 'img/sir/ifr.png'):
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

if __name__ == "__main__":
    plot_ifr(save = True)
    plt.show()
