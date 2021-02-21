
def prod(seq):
    _product = 1.0
    for item in seq:
        _product *= item
    return _product

def transitional_probability(probs):
    """"""
    assert(len(probs) > 0)
    assert((sum(probs) - 1) < 0.0001)
    trans_probabilities = [probs[0]]
    for index in range(1,len(probs)):
        trans = probs[index] / prod([1-p for p in trans_probabilities[:index]])
        trans_probabilities.append(trans)
    return trans_probabilities

if __name__ == "__main__":
    t1 = transitional_probability([1/3, 1/3, 1/3])
    t2 = transitional_probability([1/2, 1/4, 1/4])
    t3 = transitional_probability([1/3, 1/12, 1/6, 1/4, 1/6])
    
    print("t1 =", t1)
    print("t2 =", t2)
    print("t3 =", t3)
    
    

