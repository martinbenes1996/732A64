
import numpy as np
from scipy import stats
import sys
import unittest

sys.path.append('src')
import transition


class TestTransition(unittest.TestCase):
    def _plot_hists(self, y, y_ref):
        import matplotlib.pyplot as plt
        plt.hist(y, alpha = .25, bins = 20)
        plt.hist(y_ref, alpha = .25, bins = 20)
        plt.show()
        
    def test_transition_1(self):
        t1 = transition._transitional_probability([1/3, 1/3, 1/3])
        for res,ref in zip(t1, [1/3, 1/2, 1]):
            self.assertAlmostEqual(res, ref)
    def test_transition_2(self):
        t2 = transition._transitional_probability([1/2, 1/4, 1/4])
        for res,ref in zip(t2, [1/2, 1/2, 1]):
            self.assertAlmostEqual(res, ref)
    def test_transition_3(self):
        t3 = transition._transitional_probability([1/3, 1/12, 1/6, 1/4, 1/6])
        for res,ref in zip(t3, [1/3, 1/8, 2/7, 3/5, 1]):
            self.assertAlmostEqual(res, ref)
        
    def test_incubation(self):
        # get incubation
        inc = transition._incubation()
        K = inc.shape[0]
        
        # probability properties
        self.assertAlmostEqual(inc.probability.sum(), 1, 3)
        self.assertAlmostEqual(inc.transition[K - 1], 1)
        
        # K-S test
        # H0: X,Y id
        # HA: X,Y not id
        x,y = np.zeros(K + 1, dtype=int),np.zeros(K, dtype = int)
        x[0] = 10000
        y_ref = np.random.multinomial(x[0], inc.probability)
        y_sample,y_ref_sample = [],[]
        for d in range(K):
            y[d] = np.random.binomial(x[d], inc.transition[d])
            x[d + 1] = x[d] - y[d]
            for individual in range(y[d]): y_sample.append(d)
            for individual in range(y_ref[d]): y_ref_sample.append(d)

        ks_res = stats.ks_2samp(y_sample,y_ref_sample)
        self.assertGreater(ks_res.pvalue, 0.05)
        
        #self._plot_hists(y_sample, y_ref_sample)
        
    def test_symptoms(self):
        # get symptoms
        sym = transition._symptoms()
        K = sym.shape[0]
        
        # probability properties
        self.assertAlmostEqual(sym.probability.sum(), 1, 2)
        self.assertAlmostEqual(sym.transition[K - 1], 1)
        
        # K-S test
        # H0: X,Y id
        # HA: X,Y not id
        x,y = np.zeros(K + 1, dtype=int),np.zeros(K, dtype = int)
        x[0] = 10000
        y_ref = np.random.multinomial(x[0], sym.probability)
        y_sample,y_ref_sample = [],[]
        for d in range(K):
            y[d] = np.random.binomial(x[d], sym.transition[d])
            x[d + 1] = x[d] - y[d]
            for individual in range(y[d]): y_sample.append(d)
            for individual in range(y_ref[d]): y_ref_sample.append(d)

        ks_res = stats.ks_2samp(y_sample,y_ref_sample)
        self.assertGreater(ks_res.pvalue, 0.05)
        
        #self._plot_hists(y_sample, y_ref_sample)
        
        