# -*- coding: utf-8 -*-
"""Model transition component.

Module containing operations of transition component in HMM.

Example:
    Transition model is executed with
    
        transition.transition(
            POP=1e4,
            initial_values=(1-.02,.01,.01,0,0),
            parameters=pd.DataFrame({
                'start': [datetime.datetime(2020,3,1)],
                'end': [datetime.datetime(2021,5,31)],
                'a':[.8],'c':[.3],'b':[.3],'d':[.05]
            })
        )
    
    Simulate single segment transition with
    
        simulate_epidemic1()
    
    Simulate pandemic with
    
        simulate_epidemic2()
        
"""
from datetime import datetime,timedelta
import functools as F
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.stats import beta,uniform
import sys
# suppress warnings
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')
from covid19 import incubation as _incubation, symptoms as _symptoms

#def _transitional_probability(probs):
#    """"""
#    assert(len(probs) > 0)
#    assert(abs(sum(probs) - 1) < 0.01)
#    trans_probabilities = [probs[0]]
#    for prob in probs[1:]:
#        trans = prob / F.reduce(lambda i,j: i*j, [1-p for p in trans_probabilities])
#        trans_probabilities.append(trans)
#    trans_probabilities[-1] = 1
#    return trans_probabilities #[t / trans_probabilities[-1] for t in trans_probabilities]

#def incubation():
#    """Incubation period distribution."""
#    incubation = _incubation.discrete()\
#        .rename({'x': 'day', 'Px': 'probability'}, axis = 1)
#    incubation['transition'] = _transitional_probability(incubation.probability)
#    return incubation

#def symptoms():
#    """Symptom period distribution."""
#    symptoms = _symptoms.discrete()\
#        .rename({'x': 'day', 'Px': 'probability'}, axis = 1)
#    #symptoms = pd.read_csv('data/symptoms.csv', header = None, names = ['day','probability'])
#    symptoms['transition'] = _transitional_probability(symptoms.probability)
#    return symptoms

#def write_distributions():
#    """Write distributions."""
#    # featch and save
#    incubation()\
#        .to_csv('data/distr/incubation.csv', index = False, header = False)
#    symptoms()\
#        .to_csv('data/distr/symptoms.csv', index = False, header = False)

def _seird(y, t, POP, a, c, b, d):
    """SEIRD model step.
    
    Args:
        y (float): States (S,E,I,R,D).
        t (float): Time.
        POP (int): Population size. 
        a,c,b,d (float): SEIRD parameters.
    Returns:
        (tuple (5) of floats): Difference steps for S,E,I,R,D.
    """
    # states
    S, E, I, R, D = y
    # model
    dSdt = - a*S*I
    dEdt = a*S*I - c*E
    dIdt = c*E - b*I - d*I
    dRdt = b*(1-d)*I
    dDdt = b*d*I
    return dSdt, dEdt, dIdt, dRdt, dDdt

def _parse_const_params(a, c, b, d):
    """Interprets parameters as constant.
    
    Args:
        a,c,b,d (float): Parameters' values.
    Returns:
        (tuple (4) of floats): Parameters a,c,b,d.
    """
    # return parameters
    return a,c,b,d

def _parse_random_params(prior_a, prior_c, prior_b, prior_d):
    """Interprets parameters as random variables of parameters.
    
    Args:
        prior_a,prior_c,prior_b (tuple of floats): Parameters of Beta random variables.
        prior_d (tuple of floats): Parameters of Uniform random variable.
    Returns:
        (tuple (4) of floats): Parameters a,c,b,d.
    """
    # random draws
    a = beta.rvs(*prior_a, size = 1)[0]
    c = beta.rvs(*prior_c, size = 1)[0]
    b = beta.rvs(*prior_b, size = 1)[0]
    d = uniform.rvs(*prior_d, size = 1)[0]
    # return parameters
    return a,c,b,d
    
def transition(POP, initial_values, parameters, random_params = False):
    """Transition sampler.
    
    Args:
        POP (int): Population size.
        initial_values (tuple): Initial values (S0, E0, I0, R0, D0).
        parameters (tuple): Dataframe of parameters: start, end, a, c, b, d.
    """
    assert(len(initial_values) == 5)
    parse_params = _parse_const_params if not random_params else _parse_random_params
    # iterate over time slots
    result = {'date': [], 'S': [], 'E': [], 'I': [], 'R': [], 'D': []}
    for i,row in enumerate(parameters.itertuples()):
        D = (row.end - row.start).days
        t = np.linspace(0, D, D+1)
        # integrate
        a,c,b,d = parse_params(row.a, row.c, row.b, row.d)
        r,_ = odeint(_seird, initial_values, t, args=(POP, a, c, b, d), full_output = 1)
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
    # return
    return pd.DataFrame(result)

def simulate_epidemic1(save=False, name = 'img/results/transition1.png'):
    """Simulate single-segment transition of epidemic.
    
    Args:
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    # transition
    parameters = pd.DataFrame({
        'start':[datetime(2020,3,1)],
        'end':[datetime(2020,5,31)],
        'a':[.8],'c':[.3],'b':[.3],'d':[.05]
    })
    x = transition(POP=1e4, parameters=parameters,
                   initial_values=(1-.02,.01,.01,0,0))
    x = pd.melt(x[['date','S','E','I','R','D']],
                id_vars='date', var_name='Variable', value_name='Value')
    # plot
    fig, ax = plt.subplots(figsize=(10,6))
    for label,df in x.groupby('Variable'):
        ax.plot(df.date, df.Value, label=label)
    ax.legend()
    if save: fig.savefig(name)
    
def simulate_epidemic2(save=False, name = 'img/results/transition2.png'):
    """Simulate two-waves transition of epidemic.
    
    Args:
        save (bool, optional): Whether to save the figure, defaultly not.
        name (str, optional): Path to save the plot to.
    """
    # transition
    parameters = pd.DataFrame({
        'start':[datetime(2020,3,1),datetime(2020,4,15),datetime(2020,6,1)],
        'end':[datetime(2020,4,15),datetime(2020,5,30),datetime(2020,8,31)],
        'a':[.4,.15,.6],'c':.4,'b':.2,'d':.05
    })
    x = transition(POP=1e4, parameters=parameters,
                   initial_values=(1-2/1000,1/1000,1/1000,0,0))
    x = pd.melt(x[['date','S','E','I','R','D']],
                id_vars='date', var_name='Variable', value_name='Value')
    # plot
    fig, ax = plt.subplots(figsize=(10,6))
    for label,df in x.groupby('Variable'):
        ax.plot(df.date, df.Value, label=label)
    ax.axvline(datetime(2020,4,15), color='grey', alpha=.4)
    ax.axvline(datetime(2020,5,30), color='grey', alpha=.4)
    ax.legend()
    if save: fig.savefig(name)
