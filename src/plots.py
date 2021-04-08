
import numpy as np
from matplotlib import pyplot as plt

def linear_spline():
    xgrid = np.linspace(-6,6,1000)
    y = []
    for x in xgrid:
        if x <= -3: y.append(-3-x)
        elif x <= 0: y.append(x+3)
        elif x <= 3: y.append(3-2*x)
        else: y.append(.5*x - 4.5)
    plt.plot(xgrid,y)
    for x in [-6,-3,0,3,6]:
        plt.axvline(x = x, color = 'b', alpha=.1)
    plt.show()

def cubic_spline_deg2():
    """Comes from https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation."""
    xgrid = np.linspace(0,3,100)
    y = []
    for x in xgrid:
        if x <= 1: y.append(.5*x**3-.15*x**2+.15*x)
        elif x <= 2: y.append(-1.2*(x-1)**3+1.35*(x-1)**2+1.35*(x-1)+.5)
        else: y.append(1.3*(x-2)**3-2.25*(x-2)**2+.45*(x-2)+2)
    plt.plot(xgrid,y)
    for x in [0,1,2,3]:
        plt.axvline(x = x, color = 'b', alpha=.1)
    plt.show()

def cubic_spline_deg0():
    xgrid = np.linspace(-3,3,100)
    y = []
    for x in xgrid:
        if x <= -1: y.append((x+1)**2)
        elif x <= 1: y.append(.25*(x+1)**2)
        else: y.append(-(2/3*(x-2.5)**2) + 2.5)
    plt.plot(xgrid,y)
    for x in [-3,-1,1,3]:
        plt.axvline(x = x, color = 'b', alpha=.1)
    #plt.scatter(3,42/18)
    plt.show()

if __name__ == '__main__':
    cubic_spline_deg0()