
import numpy as np
import pandas as pd

POP = 10629928

# tests
tests = pd.read_csv('results/14_03-prior2/tests.csv', names=['days','tests'], header=0).tests

# HMM parameters
colnames_window = ['index', *range(27)]
a_sir = pd.read_csv('results/14_03-prior2/a_sir.csv', names=colnames_window, header=0)
c_sir = pd.read_csv('results/14_03-prior2/c_sir.csv', names=colnames_window, header=0)
b_sir = pd.read_csv('results/14_03-prior2/b_sir.csv', names=colnames_window, header=0)
d_sir = pd.read_csv('results/14_03-prior2/d_sir.csv', names=colnames_window, header=0)

# output
colnames_day = ['index', *range(27)]
R0 = pd.read_csv('results/14_03-prior2/R0.csv', names=colnames_day, header=0)
recov = pd.read_csv('results/14_03-prior2/recovery_time.csv', names=colnames_day, header=0)

# latent
def cname(limit, prefix):
    return [prefix + str(i) for i in range(limit)]
colnames_latent = ['index', *cname(274,'S'), *cname(274,'E'), *cname(274,'I'), *cname(274,'R'), *cname(274,'D')]
y = pd.read_csv('results/14_03-prior2/y.csv', names=colnames_latent, header=0)
y_S = y[cname(274,'S')] * POP
y_E = y[cname(274,'E')] * POP
y_I = y[cname(274,'I')] * POP
y_R = y[cname(274,'R')]
y_D = y[cname(274,'D')] * POP

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
def sim_to_mean(x):
    # latent simulations to mean
    x_mean = x.mean(axis=0)\
        .to_numpy()#\
        #.reshape((-1,1))
    return x_mean
def mean_to_smooth(x_mean):
    # axis
    xaxis = np.array(range(x_mean.shape[0]))\
        .reshape((-1,1))
    #print("xaxis:", xaxis.shape)
    # smoothing
    neigh = KNeighborsRegressor(n_neighbors=3)
    neigh.fit(xaxis, x_mean)
    x_smooth = neigh.predict(xaxis)
    return x_smooth

# axis
from datetime import datetime
dt = pd.date_range(datetime(2020,4,2),datetime(2020,12,31))
# plotting
import matplotlib.pyplot as plt

# parameters
dt_param = pd.date_range(datetime(2020,4,1),datetime(2020,12,31), freq = '10D')
a_mean = sim_to_mean(a_sir)
c_mean = sim_to_mean(c_sir)
b_mean = sim_to_mean(b_sir)
d_mean = sim_to_mean(d_sir)
plt.plot(dt_param[1:], a_mean[1:], label="S-E")
plt.plot(dt_param[1:], c_mean[1:], label="E-I")
plt.plot(dt_param[1:], b_mean[1:], label="I-R")
plt.plot(dt_param[1:], d_mean[1:], label="I-D")
plt.legend()
plt.show()

#print(y_E.shape)
#print(sim_to_mean(y_E).shape)
#print(mean_to_smooth(sim_to_mean(y_E)).shape)
#print("tests", tests.shape)
#print("dt", dt.shape)
#exit()

# eird
y_E_smooth = mean_to_smooth(sim_to_mean(y_E))
y_I_smooth = mean_to_smooth(sim_to_mean(y_I))
y_R_smooth = mean_to_smooth(sim_to_mean(y_R) * tests)
y_D_smooth = mean_to_smooth(sim_to_mean(y_D))
plt.plot(dt[5:], y_E_smooth[5:], label="E")
plt.plot(dt[5:], y_I_smooth[5:], label="I")
plt.plot(dt[5:], y_R_smooth[5:], label="R")
plt.plot(dt[5:], y_D_smooth[5:], label="D")
plt.legend()
plt.show()

# s
y_S_smooth = mean_to_smooth(sim_to_mean(y_S))  
plt.plot(dt, y_S_smooth, label="S")
plt.legend()
plt.show()

# r0
R0_smooth = mean_to_smooth(sim_to_mean(R0))
plt.plot(dt_param, R0_smooth, label="R0")
plt.axhline(1, color = 'gray', linestyle='-')
plt.legend()
plt.show()

# recovery
#recov_smooth = mean_to_smooth(sim_to_mean(recov))
#plt.plot(dt[5:], recov_smooth[5:-1], label="Recovery Interval")
#plt.legend()
#plt.show()
