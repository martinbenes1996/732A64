import matplotlib.pyplot as plt
import sys
sys.path.append('src')

# settings
save_plots = False
show_plots = True

# === covid19 parameters ===
import covid19
# IFR
print("Covid-19 IFR.")
#if show_plots or save_plots:
#    covid19.ifr.plot(save=save_plots)
#    print("- Plot of simulation.")
#    if show_plots: plt.show()
print("")
# Incubation period
print("Covid-19 incubation period.")
print("- Distributions.")
distr = covid19.incubation.continuous()
print(distr)
print("- Distributions' MSEs.")
MSEs = covid19.incubation.mse()
print(MSEs)
#if show_plots or save_plots:
#    covid19.incubation.plot.continuous(save=save_plots)
#    print("- Plot of continuous distributions.")
#    if show_plots: plt.show()
#    covid19.incubation.plot.discrete(save=save_plots)
#    print("- Plot of discrete Gamma.")
#    if show_plots: plt.show()
print("")
# Symptoms duration
print("Covid-19 symptoms' duration")
print("- Distributions.")
distr = covid19.symptoms.continuous()
print(distr)
print("- Data summary.")
data_summary = covid19.symptoms.data_summary()
print(data_summary)
print("- AIC.")
aic = covid19.symptoms.aic()
print(aic)
#if show_plots or save_plots:
#    print("- Plot of continuous distributions.")
#    covid19.symptoms.plot.continuous(save=save_plots)
#    if show_plots: plt.show()
#    print("- Plot of discrete Gamma.")
#    covid19.symptoms.plot.discrete(save=save_plots)
#    if show_plots: plt.show()
# Tests
print("Covid-19 tests.")
print("- Fetch tests data.")
tests = covid19.tests.get()
print(tests)
#if show_plots or save_plots:
#    print("- Plot of positive tests' ratio.")
#    covid19.tests.plot_positive_test_ratio()
#    if show_plots: plt.show()

# === demographics ===
# Mortality

exit()

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
print("PL level 1")
x = _src._PL_data(level = 1)
print("IT level 1")
x = _src._IT_data(level = 1)
print("SE level 1")
x = _src._SE_data(level = 1)
print("CZ level 2")
x = _src._CZ_data(level = 2)
print("PL level 2")
x = _src._PL_data(level = 2)
print("IT level 2")
x = _src._IT_data(level = 2)
print("SE level 2")
x = _src._SE_data(level = 2)

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