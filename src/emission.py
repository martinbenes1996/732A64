
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta
from statsmodels.tsa.arima_process import ArmaProcess

def emission(xbar, T, a, b):
    # parameters
    alpha_ = (a + T * xbar)
    beta_ = (b + T * (1 - xbar))
    # simulate
    D = T.shape[0]
    draw = np.zeros((D,))
    for i in range(D):
        try:
            draw[i] = beta.rvs(alpha_[i], beta_[i], size = 1)
        except:
            print(i, a, b, T[i], xbar[i], alpha_[i], beta_[i])
            raise
    # result
    return draw

def emission_objective(reported, xbar, T, a, b):
    # parameters
    alpha_ = (a + T * xbar)
    beta_ = (b + T * (1 - xbar))
    # compute loglik
    D = T.shape[0]
    logL = 0
    for i in range(D):
        logL += beta.logpdf(reported[i] + 1e-11, alpha_[i], beta_[i])
    # result
    return -logL

def plot_MA_emission(maxit = 100, N = 365, T = 1000):
    # create model    
    MA = ArmaProcess(ma = [.2,-.4,.2,-.7])
    # iterate
    z = np.zeros((maxit, N))
    x = np.zeros((maxit, N))
    for i in range(maxit):
        # create z[t] sample
        z[i,:] = MA.generate_sample(nsample=N) + 3*np.sin(np.array(range(N))/N*2*np.pi)
        z[i,:] = (z[i,:] - z[i,:].min()) / (z[i,:].max() - z[i,:].min())
        # create x[t] sample
        x[i,:] = emission(z[i,:], np.array([T for i in range(N)]), 1, 50)
    def get_mu_ci(ts):
        mu = ts.mean(axis = 0)
        ci = np.quantile(ts, [.025,.975],axis=0)
        return mu,ci
    z_mu,z_ci = get_mu_ci(z)
    x_mu,x_ci = get_mu_ci(x)
    # plot
    fig1, ax1 = plt.subplots()
    ax1.plot(range(N), z_mu, color='red', label='z[t]')
    ax1.fill_between(range(N), z_ci[0,:], z_ci[1,:], color = 'red', alpha = .1)
    ax1.plot(range(N), x_mu, color='blue', label='x[t]')
    ax1.fill_between(range(N), x_ci[0,:], x_ci[1,:], color = 'blue', alpha = .1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    plt.legend()
    plt.show()

