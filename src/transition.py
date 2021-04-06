
from datetime import datetime,timedelta
import functools as F
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.stats import beta
import sys

sys.path.append('src')
import _incubation
import _symptoms

def _transitional_probability(probs):
    """"""
    assert(len(probs) > 0)
    assert(abs(sum(probs) - 1) < 0.01)
    trans_probabilities = [probs[0]]
    for prob in probs[1:]:
        trans = prob / F.reduce(lambda i,j: i*j, [1-p for p in trans_probabilities])
        trans_probabilities.append(trans)
    trans_probabilities[-1] = 1
    return trans_probabilities #[t / trans_probabilities[-1] for t in trans_probabilities]

def incubation():
    """Incubation period distribution."""
    incubation = _incubation.discrete()\
        .rename({'x': 'day', 'Px': 'probability'}, axis = 1)
    incubation['transition'] = _transitional_probability(incubation.probability)
    return incubation

def symptoms():
    """Symptom period distribution."""
    symptoms = _symptoms.discrete()\
        .rename({'x': 'day', 'Px': 'probability'}, axis = 1)
    #symptoms = pd.read_csv('data/symptoms.csv', header = None, names = ['day','probability'])
    symptoms['transition'] = _transitional_probability(symptoms.probability)
    return symptoms

def write_distributions():
    """Write distributions."""
    # featch and save
    incubation()\
        .to_csv('data/distr/incubation.csv', index = False, header = False)
    symptoms()\
        .to_csv('data/distr/symptoms.csv', index = False, header = False)


def seird(y, t, POP, a, c, b, d):
    S, E, I, R, D = y
    dSdt = - a*S*I
    dEdt = a*S*I - c*E
    dIdt = c*E - b*I - d*I
    dRdt = b*(1-d)*I
    dDdt = b*d*I
    return dSdt, dEdt, dIdt, dRdt, dDdt

def parse_const_params(a, c, b, d):
    return a,c,b,d
def parse_random_params(prior_a, prior_c, prior_b, prior_d):
    a = beta.rvs(*prior_a, size = 1)[0]
    c = beta.rvs(*prior_c, size = 1)[0]
    b = beta.rvs(*prior_b, size = 1)[0]
    d = beta.rvs(*prior_d, size = 1)[0]
    return a,c,b,d
    
def transition(POP, initial_values, parameters, parse_params = None):
    """Transition sampler.
    
    Args:
        POP (int): Population size.
        initial_values (tuple): Initial values (S0, E0, I0, R0, D0).
        parameters (tuple): Dataframe of parameters: start, end, a, c, b, d.
    """
    assert(len(initial_values) == 5)
    parse_params = parse_const_params if parse_params is None else parse_params
    # iterate over time slots
    result = {'date': [], 'S': [], 'E': [], 'I': [], 'R': [], 'D': []}
    for i,row in enumerate(parameters.itertuples()):
        D = (row.end - row.start).days
        t = np.linspace(0, D, D+1)
        # integrate
        a,c,b,d = parse_params(row.a, row.c, row.b, row.d)
        r = odeint(seird, initial_values, t, args=(POP, a, c, b, d))
        # accumulate
        for dt in pd.date_range(row.start, row.end-timedelta(days=1)):
            result['date'].append(dt)
        result['S'] = [*result['S'], *r.T[0,:D]]
        result['E'] = [*result['E'], *r.T[1,:D]]
        result['I'] = [*result['I'], *r.T[2,:D]]
        result['R'] = [*result['R'], *r.T[3,:D]]
        result['D'] = [*result['D'], *r.T[4,:D]]
        # change initial value
        initial_values = r[D,:]
        # add last
        if i == (parameters.shape[0]-1):
            result['date'].append(row.end)
            result['S'].append(r.T[0,D])
            result['E'].append(r.T[1,D])
            result['I'].append(r.T[2,D])
            result['R'].append(r.T[3,D])
            result['D'].append(r.T[4,D])
    # R,D to diff
    result = pd.DataFrame(result)
    result['dR'] = result['R'].diff()
    result.loc[0,'dR'] = 0#result.loc[0,'R']
    result['dD'] = result['D'].diff()
    result.loc[0,'dD'] = 0#result.loc[0,'D']
    #print(result)
    # return result
    return result





