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

print("CZ level 1")
x = _src._CZ_data(level = 1)
x.to_csv('CZ_data_1.csv', index = False)

print("PL level 1")
x = _src._PL_data(level = 1)
x.to_csv('PL_data_1.csv', index = False)

print("IT level 1")
x = _src._IT_data(level = 1)
x.to_csv('IT_data_1.csv', index = False)

print("SE level 1")
x = _src._SE_data(level = 1)
x.to_csv('SE_data_1.csv', index = False)

print("CZ level 2")
x = _src._CZ_data(level = 2)
x.to_csv('CZ_data_2.csv', index = False)

print("PL level 2")
x = _src._PL_data(level = 2)
x = x\
    .merge(pop, how='left', on=['region'])
x.to_csv('PL_data_2.csv', index = False)

print("IT level 2")
x = _src._IT_data(level = 2)
x = x\
    .merge(pop, how='left', on=['region'])
x.to_csv('IT_data_2.csv', index = False)

print("SE level 2")
x = _src._SE_data(level = 2)
x = x\
    .merge(pop, how='left', on=['region'])
x.to_csv('SE_data_2.csv', index = False)

#x['Tests_per1K'] = x.tests / x.population * 1000
#print(x['Tests_per1K'])
#fig, ax = plt.subplots(figsize=(8,6))
#for label, df in x.groupby('region'):
#    print(label)
#    df.plot(x = 'date', y = 'recovered', ax=ax, label=label)
#plt.xlabel('Time')
#plt.ylabel('Tests_per1K')
#plt.set_cmap('plasma')
#plt.show()