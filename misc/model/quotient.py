
import pystan

with open('model/quotient.stan') as fp:
    quotient_code = fp.read()

import pandas as pd
cz_incidence = pd.read_csv('data/cz_incidence.csv')

quotient_data = {
    'a': 1.5,
    'b': 1.5,
    'incidence': cz_incidence.I,
    'N': cz_incidence.shape[0]
}

posterior = pystan.StanModel(model_code=quotient_code)
fit = posterior.sampling(data=quotient_data, num_chains=4, num_samples=1000)
R0 = fit.extract()["reproduction_number"]
print(R0)

import matplotlib.pyplot as plt
fit.plot()
plt.show

#df = fit.to_frame()
