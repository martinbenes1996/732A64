
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append('src')

# settings
save_plots = False
show_plots = True
countries = ['CZ','IT','PL','SE']

# === covid19 parameters ===
import covid19

# IFR
print("Covid-19 IFR.")

#if show_plots or save_plots:
#    covid19.ifr.plot(save=save_plots)
#    print("- Plot of simulation.")
#    if show_plots: plt.show()

# Fatality
print("\nCovid-19 deaths.")

#print("- Fetch Covid-19 deaths for Italy.")
#covid_deaths_it = covid19.deaths.covid19italy.covid_deaths()
#print(covid_deaths_it)

#print("- Fetch Covid-19 deaths.")
#covid_deaths = covid19.deaths.get_data()
#print(covid_deaths)

#if show_plots or save_plots:
#    print("- Construct violin plot of Covid-19 deaths.")
#    covid19.deaths.plot_violin(save=save_plots)
#    if show_plots: plt.show()

#print("- Test that age of death are significantly greater than > 60.")
#result_over60 = covid19.deaths.test_over60()
#print(result_over60)

# Incubation period
print("\nCovid-19 incubation period.")

#print("- Distributions.")
#distr = covid19.incubation.continuous()
#print(distr)

#print("- Distributions' MSEs.")
#MSEs = covid19.incubation.mse()
#print(MSEs)

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

#print("- Distributions.")
#distr = covid19.symptoms.continuous()
#print(distr)

#print("- Data summary.")
#data_summary = covid19.symptoms.data_summary()
#print(data_summary)

#print("- AIC.")
#aic = covid19.symptoms.aic()
#print(aic)

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

#print("- Fetch tests data.")
#tests = covid19.tests.get()
#print(tests)

#if show_plots or save_plots:
#    print("- Plot of positive tests' ratio.")
#    covid19.tests.plot_positive_test_ratio(save=save_plots)
#    if show_plots: plt.show()

# === demographics ===
import demographic

# Mortality
print("\nMortality.")

#print("- Fetching mortality data.")
#mortality = demographic.mortality.data()
#print(mortality)

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

#print("- Test of equal countries' mortalities.")
#for i in range(4):
#    for j in range(i+1,4):
#        c1,c2 = countries[i],countries[j]
#        countries_equal = demographic.mortality.test_countries_equal(c1,c2)
#        print(countries_equal)

#print("- Test of equal mortalities of genders in country.")
#for i in range(4):
#    c1 = countries[i]
#    country_age_equal = demographic.mortality.test_country_gender_equal(c1)
#    print(country_age_equal)

#if show_plots or save_plots:
#    print("- Plot Czech mortality.")
#    demographic.mortality.plot_CZ(save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    for c1 in countries:
#        print(f"- Plot 0-4y trace plot for {c1}.")
#        demographic.mortality.plot_children(c1, save=save_plots)
#        plt.show()

#print("- Test of Poland greater in age group 0-4 and other countries equal.")
#_0_4_greater = demographic.mortality.test_0_4_greater()
#print(_0_4_greater)

# Population
print("\nPopulations.")

#print("- Fetching population country data.")
#pops_countries = demographic.population._countries()
#print(pops_countries)

#print("- Fetching population regional data.")
#pops_regions = demographic.population._regions()
#print(pops_regions)

#print("- Fetching population data.")
#pops = demographic.population.population(save=True)
#print(pops)

#print("- Get population of a regions (CZ010, SE110, PL51, ITC4).")
#pop_CZ010 = demographic.population.get_population('CZ010')
#pop_SE110 = demographic.population.get_population('SE110')
#pop_PL51 = demographic.population.get_population('PL51')
#pop_ITC4 = demographic.population.get_population('ITC4')
#print(pop_CZ010, pop_SE110, pop_PL51, pop_ITC4)

#if show_plots or save_plots:
#    print("- Plot population violin.")
#    demographic.population.plot_violin(save=save_plots)
#    if show_plots: plt.show()

#print("- Test populations are equal.")
#for i in range(4):
#    for j in range(i+1,4):
#        c1,c2 = countries[i],countries[j]
#        countries_equal = demographic.population.test_countries_equal(c1,c2)
#        print(countries_equal)

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
print("\nTransition.")

#print("- Execute transition procedure.")
#df_transition = transition.transition(
#    POP=1e4,
#    initial_values=(1-.02,.01,.01,0,0),
#    parameters=pd.DataFrame({
#        'start': [datetime(2020,3,1)],
#        'end': [datetime(2021,5,31)],
#        'a':[.8],'c':[.3],'b':[.3],'d':[.05]
#    })
#)
#print(df_transition)

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

#print("- Apply emission posterior.")
#df_emission = emission.emission(
#    xbar = np.array([.3,.4,.4,.3,.3]),
#    T = np.array([20,30,35,30,35]),
#    a = 2,
#    b = 3
#)
#print(df_emission)

#print("- Get emission posterior nlogL.")
#score_emission = emission.emission_objective(
#    reported = np.array([.3,.4,.4,.3,.3]),
#    xbar = np.array([.3,.4,.4,.3,.3]),
#    T = np.array([20,30,35,30,35]),
#    a = 2,
#    b = 3
#)
#print(score_emission)

#if show_plots or save_plots:
#    print("- Plot of emission with MA transition.")
#    emission.plot_MA(save=save_plots)
#    if show_plots: plt.show()

# === Calendar ===
import event_calendar
print("\nCalendar.")

#print("- Load calendar.")
#df_calendar = event_calendar.load_calendar()
#print(df_calendar)

#if show_plots or save_plots:
#    for country in countries:
#        print(f"- Plot confirmed of {country} with calendar.")
#        event_calendar.plot_confirmed(
#            country,save=save_plots,
#            name=f'img/discussion/restrictions_{country}.png')
#        if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot of 1st CZ lockdown 2020-10-28.")
#    event_calendar.plot_segment(
#        'CZ',(datetime(2020,10,10),datetime(2020,11,30)),event=datetime(2020,10,28),
#        save=save_plots,name=f'img/discussion/restrictions_CZ1.png')
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot of 2nd CZ lockdown 2020-12-27.")
#    event_calendar.plot_segment(
#        'CZ',(datetime(2020,12,15),datetime(2021,1,31)),event=datetime(2020,12,27),
#        save=save_plots,name=f'img/discussion/restrictions_CZ2.png')
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot of 3rd CZ lockdown 2021-02-26.")
#    event_calendar.plot_segment(
#        'CZ',(datetime(2021,2,15),datetime(2021,3,31)),event=datetime(2021,2,26),
#        save=save_plots,name=f'img/discussion/restrictions_CZ3.png')
#    if show_plots: plt.show()

# === Results ===
import results
print("\nResults.")

#print("- Load results.")
#sim,dates,region,params = results.load(
#    dates=(datetime(2020,8,1),datetime(2021,3,15)),
#    region='CZ',
#    now=datetime(2021,4,25)
#)
#print([i.shape for i in sim], dates.shape, region, params.shape)

#print("- Save results.")
# TODO

#if show_plots or save_plots:
#    print("- Plot simulation parameters.")
#    results.plot_params(
#        dates=(datetime(2020,3,10),datetime(2020,5,31)),
#        region='CZ',
#        now=datetime(2021,4,12)
#    )
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    for charac in ['r0','ifr','symptom duration']:
#        print(f"- Plot Covid-19 {charac} estimate.")
#        results.plot_characteristics(
#            dates=(datetime(2020,3,10),datetime(2020,5,31)),
#            region='CZ',
#            now=datetime(2021,4,12),
#            par=charac
#        )
#        if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot weekly results.")
#    results.plot_weekly(
#        dates=(datetime(2020,3,3),datetime(2021,4,16)),
#        region='PL',
#        now=datetime(2021,4,18),
#        weekly=False
#    )
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot susceptible SE224 weekly.")
#    results.plotSusceptible_SE224_Weekly(save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot susceptible PL weekly.")
#    results.plotSusceptible_PL_Weekly(save=save_plots)
#    if show_plots: plt.show()

# === Prior ===
import prior
print("Prior.")

#print("- Fit EI.")
#fit_EI = prior.EI()
#print(fit_EI)

#print("- Fit IR.")
#fit_IR = prior.IR()
#print(fit_IR)

#print("- Fit ID.")
#fit_ID = prior.ID()
#print(fit_ID)

#print("- Draw R0.")
#draws_R0 = prior.draw_R0(1000)
#print(draws_R0)

#print("- Fit SI.")
#fit_SI = prior.SI()
#print(fit_SI)

#if show_plots or save_plots:
#    print("- Plot R0.")
#    prior.plot_R0(save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot SI.")
#    prior.plot_SI(save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot EI.")
#    prior.plot_EI(save=save_plots)
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot IR.")
#    prior.plot_IR(save=save_plots)
#    if show_plots: plt.show()
    
#if show_plots or save_plots:
#    print("- Plot ID.")
#    prior.plot_ID(save=save_plots)
#    if show_plots: plt.show()
    
#print("- Fit priors.")
#fit_priors = prior.priors()
#print(fit_priors)

#print("- Get test ratio.")
#country_tests = prior.test_prior()
#print(country_tests)

#print("- Fit distribution to test ratio.")
#fit_tests = prior.tested()
#print(fit_tests)

#print("- Get confirmed test ratio.")
#country_confirmed = prior.confirmed_prior()
#print(country_confirmed)

#if show_plots or save_plots:
#    print("- Plot test prior.")
#    prior.plot_test_prior()
#    if show_plots: plt.show()

#if show_plots or save_plots:
#    print("- Plot test ratio for countries.")
#    prior.plot_test_ratio_all(save=save_plots)
#    if show_plots: plt.show()

# === Regional ===
import regional
print("Regional.")

print("- Plot confirmed.")




exit()



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