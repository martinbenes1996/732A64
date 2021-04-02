
from datetime import datetime
from hmmlearn.hmm import GaussianHMM
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append('src')
import _src

def _data(country, dates):
    x = _src.get_data()
    x = x[x.iso_alpha_3 == country]
    
    # daily to cumsum and normalize by tests
    #x['cumtests'] = x.tests.cumsum()
    #x['confirmed'] = x.confirmed / x.tests
    x['recovered'] = x.recovered.cumsum()# / x.cumtests
    x['deaths'] = x.deaths.cumsum()# / x.cumtests
    # filter by dates
    x = x[(x.date >= dates[0]) & (x.date <= dates[1])]
    return x

def _startM(POP, E=0, I=1, R=0, D=0):
    return np.array([(POP - E-I-R-D)/POP, E/POP, I/POP, R/POP, D/POP])
def _transitionM(a, c, b, d):
    return np.array([[1-a,   0,     0, 0, 0],
                     [  a, 1-c,     0, 0, 0],
                     [  0,   c, 1-b-d, 0, 0],
                     [  0,   0,     b, 1, 0],
                     [  0,   0,     d, 0, 1]])

def _emissionM(eE, eI, eR, eD):
    return np.array([[1-eE-eI, 1-eR,    0],
                     [     eE,    0,    0],
                     [     eI,    0,    0],
                     [      0,   eR,    0],
                     [      0,    0,    1]])

pop = 1e7
sM = _startM(1000, I = 0)
tM = _transitionM(.002, .4, .005, .0002)
eM = _emissionM(.4, .7, .7, .9)

x = _data('CZE', (datetime(2020,3,15),datetime(2021,1,31)))
xm = x[['confirmed','recovered','deaths']].to_numpy() / pop

#print(tM)
#print(eM)
#print(sM)
#print(x)

# fit
model = GaussianHMM(5, algorithm='viterbi', verbose=True, params='c', covariance_type='full')
model.startprob=sM
model.transmat=tM
model.emissionprob=eM
model.fit(xm)

# predict
probs = model.sample(323)
print(probs)
#probs = model.predict(xm)
#print(probs)
df = pd.DataFrame(probs, columns = ['S','E','I','R','D'])
print(df)

# plot
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#ffffff', axisbelow=True)
ax.plot(x.date, df.S, 'gray', alpha=0.5, lw=2, label='Susceptible')
ax.plot(x.date, df.E, 'orange', alpha=0.5, lw=2, label='Exposed')
ax.plot(x.date, df.I, 'red', alpha=0.5, lw=2, label='Infected')
ax.plot(x.date, df.R, 'green', alpha=0.5, lw=2, label='Recovered')
ax.plot(x.date, df.D, 'black', alpha=0.5, lw=2, label='Deaths')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number')
ax.set_ylim(0,1.1)
#ax.yaxis.set_tick_params(length=0)
#ax.xaxis.set_tick_params(length=0)
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
#legend.get_frame().set_alpha(0.5)
#for spine in ('top', 'right', 'bottom', 'left'):
#    ax.spines[spine].set_visible(False)
plt.show()

