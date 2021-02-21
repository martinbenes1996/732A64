
import torch
from torch import distributions
from torch.optim import Adam
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,10)
import numpy as np
from scipy.stats import gamma

class _QuantileDistributionOptimizer:
    def __init__(self, quantiles, parameters, cdf, optimizer):
        # save parameters
        self.parameters = parameters
        self.cdf = cdf
        self.optimizer = optimizer
        # initialize data
        self.quantile_tensors = [
            (k, torch.tensor(v)) for k,v in quantiles
        ]
        # run data
        self.losses = []
    def MSE(self):
        loss = .0
        for quantile,quantile_value in self.quantile_tensors:
            loss += (quantile + self.cdf(quantile_value)) ** 2
        #print(loss)
        return loss #torch.tensor(loss.item(), requires_grad=True)
    
    def optim(self, N = 2000, lr = .005):
        for i in range(N):
            self.optimizer.zero_grad()
            loss = self.MSE()
            #print(loss)
            loss.detach()
            self.losses.append(loss)
            loss.backward(retain_graph = True)
            self.optimizer.step()
            
            #if i % 100 == 0:
            #    print("%d) %f" % (i,loss.item()))

class LogNormal(_QuantileDistributionOptimizer):
    def __init__(self, quantiles):
        # parameters
        parameters = [torch.tensor(1.0, requires_grad=True) for _ in range(2)]
        # init parent
        super().__init__(
            quantiles = quantiles,
            parameters = parameters,
            cdf = distributions.LogNormal(*parameters).cdf,
            optimizer = Adam(parameters, lr = .01))
        # mean,sigma
        self.parameters = parameters
        self.mu,self.sigma = self.parameters
    
    def optim(self, *args, **kw):
        super().optim(*args, **kw)
        return [p.item() for p in self.parameters]

class Weibull(_QuantileDistributionOptimizer):
    def __init__(self, quantiles):
        # parameters
        parameters = [torch.tensor(1.0, requires_grad=True) for _ in range(2)]
        # init parent
        super().__init__(
            quantiles = quantiles,
            parameters = parameters,
            cdf = distributions.Weibull(*parameters).cdf,
            optimizer = Adam(parameters, lr = .01))
        # alpha,beta
        self.parameters = parameters
        self.alpha,self.beta = self.parameters
    
    def optim(self, *args, **kw):
        super().optim(*args, **kw)
        return [p.item() for p in self.parameters]
    
class Gamma(_QuantileDistributionOptimizer):
    def __init__(self, quantiles):
        raise NotImplementedError
        # parameters
        parameters = [torch.tensor(1.0, requires_grad=True) for _ in range(3)]
        def _cdf(x):
            with torch.no_grad():
                xx = x.numpy()
                cdf = gamma.cdf(xx, *parameters)
            return torch.tensor(cdf, requires_grad=True)
        # init parents
        super().__init__(
            quantiles = quantiles,
            parameters = parameters,
            cdf = _cdf,
            optimizer = Adam(parameters, lr = .01))
        # a,loc,scale
        self.parameters = parameters
        self.a,self.loc,self.scale = self.parameters
    
    def optim(self, *args, **kw):
        super().optim(*args, **kw)
        return [p.item() for p in self.parameters]

class Probit(_QuantileDistributionOptimizer):
    def __init__(self, quantiles):
        raise NotImplementedError
        # parameters
        parameters = [torch.tensor(1.0, requires_grad=True) for _ in range(3)]
        def _cdf(x):
            with torch.no_grad():
                xx = x.numpy()
                cdf = gamma.cdf(xx, *parameters)
            return torch.tensor(cdf, requires_grad=True)
        # init parents
        super().__init__(
            quantiles = quantiles,
            parameters = parameters,
            cdf = _cdf,
            optimizer = Adam(parameters, lr = .01))
        # a,loc,scale
        self.parameters = parameters
        self.a,self.loc,self.scale = self.parameters
    
    def optim(self, *args, **kw):
        super().optim(*args, **kw)
        return [p.item() for p in self.parameters]
    
if __name__ == "__main__":
    # quantiles
    quantiles = [(.05,1.64),(.5,4.8),(.95,14.04)]
    
    # fit lognormal
    #ln = LogNormal(quantiles)
    #ln.optim()
    #print("LN(%f, %f)" % (ln.mu,ln.sigma))
    #plt.plot(ln.losses)
    #plt.show()
    
    # fit weibull
    #w = Weibull(quantiles)
    #w.optim(N = 3000)
    #print("W(%f, %f)" % (w.alpha,w.beta))
    #plt.plot(w.losses)
    #plt.show()
    
    # fit gamma
    #g = Gamma(quantiles)
    #g.optim()
    #print("Gamma(%f, %f, %f)" % (g.a, g.loc, g.scale))