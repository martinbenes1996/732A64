
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# Incubation period
print("\nCovid-19 incubation period.")

print("- Distributions.")
distr = covid19.incubation.continuous()
print(distr)

print("- Distributions' MSEs.")
MSEs = covid19.incubation.mse()
print(MSEs)

#if show_plots or save_plots:
#    print("- Plot of continuous distributions.")
#    covid19.incubation.plot.continuous(save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot of discrete Gamma.")
#    covid19.incubation.plot.discrete(save=save_plots)
#    if show_plots: plt.show()

# Symptoms duration
print("\nCovid-19 symptoms' duration")

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

#if show_plots or save_plots:
#    print("- Plot of discrete Gamma.")
#    covid19.symptoms.plot.discrete(save=save_plots)
#    if show_plots: plt.show()

# Tests
print("\nCovid-19 tests.")

print("- Fetch tests data.")
tests = covid19.tests.get()
print(tests)

#if show_plots or save_plots:
#    print("- Plot of positive tests' ratio.")
#    covid19.tests.plot_positive_test_ratio(save=save_plots)
#    if show_plots: plt.show()

# === demographics ===
import demographic

# Mortality
print("\nMortality.")
print("- Fetching mortality data.")
mortality = demographic.mortality.data()
print(mortality)

#if show_plots or save_plots:
#    print("- Violinplot of mortality.")
#    demographic.mortality.plot_violin(save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot Poland mortality over years.")
#    demographic.mortality.plot_poland_years(range(2010,2021), save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot Poland mortality (age 0-4y) over years.")
#    demographic.mortality.plot_poland_0_4(save=save_plots)
#    if show_plots: plt.show()

print("- Test of equal countries' mortalities.")
countries = ['CZ','IT','PL','SE']
for i in range(4):
    for j in range(i+1,4):
        c1,c2 = countries[i],countries[j]
        countries_equal = demographic.mortality.test_countries_equal(c1,c2)
        print(countries_equal)
        
print("- Test of equal mortalities of genders in country.")
for i in range(4):
    c1 = countries[i]
    country_age_equal = demographic.mortality.test_country_gender_equal(c1)
    print(country_age_equal)
    
#if show_plots or save_plots:
#    print("- Plot Czech mortality.")
#    demographic.mortality.plot_CZ(save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    for c1 in countries:
#        print(f"- Plot 0-4y trace plot for {c1}.")
#        demographic.mortality.plot_children(c1, save=save_plots)
#        plt.show()

print("- Test of Poland greater in age group 0-4 and other countries equal.")
_0_4_greater = demographic.mortality.test_0_4_greater()
print(_0_4_greater)

# Population
print("\nPopulations.")

print("- Fetching population country data.")
pops_countries = demographic.population._countries()
print(pops_countries)

print("- Fetching population regional data.")
pops_regions = demographic.population._regions()
print(pops_regions)

print("- Fetching population data.")
pops = demographic.population.population(save=True)
print(pops)

print("- Get population of a regions (CZ010, SE110, PL51, ITC4).")
pop_CZ010 = demographic.population.get_population('CZ010')
pop_SE110 = demographic.population.get_population('SE110')
pop_PL51 = demographic.population.get_population('PL51')
pop_ITC4 = demographic.population.get_population('ITC4')
print(pop_CZ010, pop_SE110, pop_PL51, pop_ITC4)

#if show_plots or save_plots:
#    print("- Plot population violin.")
#    demographic.population.plot_violin(save=save_plots)
#    if show_plots: plt.show()
    
print("- Test populations are equal.")
for i in range(4):
    for j in range(i+1,4):
        c1,c2 = countries[i],countries[j]
        countries_equal = demographic.population.test_countries_equal(c1,c2)
        print(countries_equal)

# === Plots ===
import plots
print("\nPlots.")

#if show_plots or save_plots:
#    print("- Plot linear spline.")
#    plots.linear_spline(save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot cubic spline with deg2.")
#    plots.cubic_spline_deg2(save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot cubic spline with deg0.")
#    plots.cubic_spline_deg0(save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Trace plot of confirmed.")
#    plots.covid_confirmed(save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Trace plot of deaths.")
#    plots.covid_deaths(save=save_plots)
#    if show_plots: plt.show()
    
#if show_plots or save_plots:
#    print("- Trace plot of recovered.")
#    plots.covid_recovered(save=save_plots)
#    if show_plots: plt.show()

# === Transition ===
import transition
print("\Transition.")

print("- Execute transition procedure.")
df_transition = transition.transition(
    POP=1e4,
    initial_values=(1-.02,.01,.01,0,0),
    parameters=pd.DataFrame({
        'start': [datetime(2020,3,1)],
        'end': [datetime(2021,5,31)],
        'a':[.8],'c':[.3],'b':[.3],'d':[.05]
    })
)
print(df_transition)

#if show_plots or save_plots:
#    print("- Simulate single-segment transition.")
#    transition.simulate_epidemic1(save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Simulate transition.")
#    transition.simulate_epidemic2(save=save_plots)
#    if show_plots: plt.show()

# === Emission ===
import emission
print("\nEmission.")

print("- Apply emission posterior.")
df_emission = emission.emission(
    xbar = np.array([.3,.4,.4,.3,.3]),
    T = np.array([20,30,35,30,35]),
    a = 2,
    b = 3
)
print(df_emission)

print("- Get emission posterior nlogL.")
score_emission = emission.emission_objective(
    reported = np.array([.3,.4,.4,.3,.3]),
    xbar = np.array([.3,.4,.4,.3,.3]),
    T = np.array([20,30,35,30,35]),
    a = 2,
    b = 3
)
print(score_emission)

#if show_plots or save_plots:
#    print("- Plot of emission with MA transition.")
#    emission.plot_MA(save=save_plots)
#    if show_plots: plt.show()


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