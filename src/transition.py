
import functools as F
import pandas as pd
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
        .to_csv('data/incubation.csv', index = False, header = False)
    symptoms()\
        .to_csv('data/symptoms.csv', index = False, header = False)



#if __name__ == '__main__':
#    print(priors())
    
#write_distributions()

#x = _symptoms()
#print(x)





