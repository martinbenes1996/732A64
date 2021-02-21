
import pandas as pd
df = pd.read_csv('misc/parameters/shedding.csv')
df = df.drop(['Unnamed: 0'], axis = 1)

# weighted mean
mu_w = (df.mu * df.w).sum() / df.w.sum()
print(mu_w)

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

mu_sample = np.random.choice(df.mu, size=50000, p = df.w / df.w.sum())
sns.histplot(data = pd.DataFrame({'mu': mu_sample}), x = 'mu', stat = 'density', bins = 25)
plt.show()