import matplotlib.pyplot as plt
import sys
sys.path.append('src')

# # === prior ===
# import prior
# # prior c (EI)
# prior.plot_EI(save = True)
# plt.show()
# # prior b (IR)
# prior.plot_IR(save = True)
# plt.show()
# # plot test ratio (base for prior)
# prior.test_prior(save = True)
# prior.plot_test_ratio_all(save = True)
# plt.show()
# # parameters for the prior distributions
# prior_parameters = prior.priors(save = True)
# print(prior_parameters)

# # === lethality ===
# import lethality
# # age-gender distribution
# lethality.plot_violin(save = True)
# plt.show()
# # test that deaths are > 60 years
# over60 = lethality.test_over60()
# print(over60)

# # === population ===
import population
# # per country / per region populations (2020)
# pop_countries = population.countries()
# pop_regions = population.regions()
# # save populations (both country and regions)
pop = population.population()

# # === infected ===
# import infected
# # plot ratio of confirmed tests
# infected.plot_test_ratio_all()
# # export tests
# infected.export_tests()

# === emission ===
#import emission
# plot emission with transition MA
#emission.plot_MA_emission(T = 100)



import _src
x = _src._se_regional_tests()

x = x\
    .merge(pop.rename({'region':'Region'},axis=1), how='left', on=['Region'])
x['Tests_per1K'] = x.Tests / x.population * 1000
print(x)

fig, ax = plt.subplots(figsize=(8,6))
#x['tests'] = x.groupby('region')['tests'].cumsum()
for label, df in x.groupby('Region'):
    df.plot(x = 'Monday', y = 'Tests_per1K', ax=ax, label=label)
plt.xlabel('Time')
plt.ylabel('Tests per 1000 people')
plt.set_cmap('plasma')
plt.show()