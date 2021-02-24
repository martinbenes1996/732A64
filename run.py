import matplotlib.pyplot as plt
import sys
sys.path.append('src')

# === prior ===
import prior
# prior c (EI)
prior.plot_EI(save = True)
plt.show()
# prior b (IR)
prior.plot_IR(save = True)
plt.show()
# plot test ratio (base for prior)
prior.test_prior(save = True)
prior.plot_test_ratio_all(save = True)
plt.show()
# parameters for the prior distributions
prior_parameters = prior.priors(save = True)
print(prior_parameters)

# === lethality ===
import lethality
# age-gender distribution
lethality.plot_violin(save = True)
plt.show()
# test that deaths are > 60 years
over60 = lethality.test_over60()
print(over60)

# === population ===
import population
# per country / per region populations (2020)
pop_countries = population.countries()
pop_regions = population.regions()
# save populations (both country and regions)
population.population(save = True)

# === infected ===
import infected
# plot ratio of confirmed tests
infected.plot_test_ratio_all()
# export tests
infected.export_tests()

# === 

