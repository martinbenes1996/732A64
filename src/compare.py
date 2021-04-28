
from datetime import datetime,timedelta
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.path.append('src')
import population
import posterior

CZ_regions = ['CZ010','CZ020','CZ031','CZ032','CZ041','CZ042','CZ051',
              'CZ052','CZ053','CZ063','CZ064','CZ071','CZ072','CZ080']
IT_regions = ['ITC1','ITC2','ITC3','ITC4','ITF1','ITF2','ITF3','ITF4',
              'ITF5','ITF6','ITG1','ITG2','ITH10','ITH20','ITH3','ITH4',
              'ITH5','ITI1','ITI2','ITI3','ITI4']
PL_regions = ['PL71','PL72','PL21','PL22','PL81','PL82','PL84','PL41',
              'PL42','PL43','PL51','PL52','PL61','PL62','PL63','PL9']
SE_regions = ['SE','SE110','SE121','SE122','SE123','SE124','SE125','SE211',
              'SE212','SE213','SE214','SE221','SE224','SE231','SE232','SE311',
              'SE312','SE313','SE321','SE322','SE331','SE332']
def get_path(region):
    return f'results/{region[:2]}w/{region}'

def load_result(region):
    path = get_path(region)
    # load
    x = pd.read_csv(f'{path}/data.csv')
    x['date'] = x.date.apply(lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
    # parse
    lat = x[['latent_S','latent_E','latent_I','latent_R','latent_D']].to_numpy().T
    obs = np.zeros((5,x.shape[0]))
    obs[2:5,:] = x[['observed_I','observed_R','observed_D']].to_numpy().T
    #region = x.loc[0,'region']
    dates = (x.date.min(), x.date.max())
    params = x[['param_a','param_c','param_b','param_d']].to_numpy().T
    return lat,x.date,params

def prediction_data_correlation(regions, components, delta=None, weekly=False):
    # initialize
    corrs = {'region': regions}
    for c in components: corrs[c] = []
    # regions
    for region in regions:
        # load and crop
        lat,dt,_ = load_result(region=region)
        if delta is None:
            dates = [dt.min(), dt.max()]
        else:
            dates = [dt.min(), dt.min()+delta]
        lat = lat[:,(dates[0] <= dt) & (dates[1] >= dt)]
        dt = dt[(dates[0] <= dt) & (dates[1] >= dt)]
        x = posterior._posterior_data(region, (max(dt.min(),dates[0]),min(dt.max(),dates[1])), weekly=weekly)
        # compute correlation
        if 'I' in components:
            corrs['I'].append(np.corrcoef(lat[2,:], x.confirmed.to_numpy())[1,0])
        if 'R' in components:
            corrs['R'].append(np.corrcoef(lat[3,:], x.recovered.to_numpy())[1,0])
        if 'D' in components:
            corrs['D'].append(np.corrcoef(lat[4,:], x.deaths.to_numpy())[1,0])
    corrs = pd.DataFrame(corrs)
    return corrs

def parameters(regions, delta=None, weekly=False):
    # initialize
    pars = {'Region': [], 'Year': np.array([]), 'Month': np.array([]),
            'a': np.array([]), 'c': np.array([]), 'b': np.array([]), 'd': np.array([])}
    # regions
    for region in regions:
        # load and crop
        _,dt,params = load_result(region=region)
        if delta is None:
            dates = [dt.min(), dt.max()]
        else:
            dates = [dt.min(), dt.min()+delta]
        params = params[:,(dates[0] <= dt) & (dates[1] >= dt)]
        dt = dt[(dates[0] <= dt) & (dates[1] >= dt)]
        x = posterior._posterior_data(region, (max(dt.min(),dates[0]),min(dt.max(),dates[1])), weekly=weekly)
        # dates
        months = dt.apply(lambda d: int(datetime.strftime(d,'%m')))
        years = dt.apply(lambda d: int(datetime.strftime(d,'%Y')))
        for _ in range(dt.shape[0]):
            pars['Region'].append(region)
        pars['Year'] = np.concatenate([pars['Year'], years.to_numpy()])
        pars['Month'] = np.concatenate([pars['Month'], months.to_numpy()])
        pars['a'] = np.concatenate([pars['a'], params[0,:]])
        pars['c'] = np.concatenate([pars['c'], params[1,:]])
        pars['b'] = np.concatenate([pars['b'], params[2,:]])
        pars['d'] = np.concatenate([pars['d'], params[3,:]])
    pars = pd.DataFrame(pars)
    pars['Date'] = pars.apply(lambda r: '%04d-%02d' % (r.Year,r.Month), axis=1)
    return pars

def get_R0(countries, log=True):
    # get parameters
    x = None
    regions = []
    if 'CZ' in countries: regions.append(['CZ',*CZ_regions])
    if 'PL' in countries: regions.append(['PL',*PL_regions])
    if 'SE' in countries: regions.append(['SE',*SE_regions])
    if 'IT' in countries: regions.append(['IT',*IT_regions])
    for country in regions: # TODO
        pars = parameters(country)
        susc = pd.DataFrame({region: load_result(region)[0][0,:]
                             for region in country})
        susc = pd.melt(susc,var_name='region',value_name='S')
        pars = pd.concat([pars, susc[['S']]], axis=1)
        pars['Country'] = country[0]
        pars['POP'] = pars.Region.apply(population.get_population)
        if x is None: x = pars
        else: x = pd.concat([x,pars])
    # compute reproduction number
    x['R0'] = x.a / x.b * x.S
    
    res = {'Country': [], 'Year': [], 'Month': [], 'Date': [],
           'Mean': [], 'Low': [], 'High': []}
    for (c,y,m,dt),df in x.groupby(['Country','Year','Month','Date']):
        res['Country'].append(c)
        res['Year'].append(y)
        res['Month'].append(m)
        res['Date'].append(dt)
        #print(df)
        res['Mean'].append(df.R0.mean())
        res['Low'].append(df.R0.quantile(.05))
        res['High'].append(df.R0.quantile(.95))
        #break
    
    res = pd.DataFrame(res)
    fig, ax = plt.subplots(figsize=(10, 6))
    if log is True: ax.set(yscale="log")
    #sns.lineplot(x="Date", y="mean", hue='Country', data=x, ax=ax)
    print(res)
    colors=['black','green','blue','red']
    for (g,d),col in zip(res.groupby('Country'),colors):
        ax.plot(d.Date, d.Mean, label=g, color=col)
        #ax.fill_between(d.Date, d.Low, d.High, alpha=.1, color=col)
    #ax.axhline(1, ls='--', color='grey', alpha=.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('R0')
    ax.legend()
    plt.show()

def plot_R0(countries, log=True):
    # get parameters
    x = None
    regions = []
    if 'CZ' in countries: regions.append(['CZ',*CZ_regions])
    if 'PL' in countries: regions.append(['PL',*PL_regions])
    if 'SE' in countries: regions.append(['SE',*SE_regions])
    if 'IT' in countries: regions.append(['IT',*IT_regions])
    for country in regions: # TODO
        pars = parameters(country)
        susc = pd.DataFrame({region: load_result(region)[0][0,:]
                             for region in country})
        susc = pd.melt(susc,var_name='region',value_name='S')
        pars = pd.concat([pars, susc[['S']]], axis=1)
        pars['Country'] = country[0]
        pars['POP'] = pars.Region.apply(population.get_population)
        if x is None: x = pars
        else: x = pd.concat([x,pars])
    # compute reproduction number
    x['R0'] = x.a / x.b * x.S
    x.loc[x.R0 < .001, 'R0'] = .001
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    if log is True: ax.set(yscale="log")
    ax = sns.boxplot(x="Date", y="R0", hue='Country', data=x, ax=ax)
    ax.axhline(1, ls='--', color='grey', alpha=.5)
    plt.show()

def get_IFR(countries, log=True):
    # get parameters
    x = None
    regions = []
    if 'CZ' in countries: regions.append(['CZ',*CZ_regions])
    if 'PL' in countries: regions.append(['PL',*PL_regions])
    if 'SE' in countries: regions.append(['SE',*SE_regions])
    if 'IT' in countries: regions.append(['IT',*IT_regions])
    for country in regions: # TODO
        pars = parameters(country)
        pars['Country'] = country[0]
        if x is None: x = pars
        else: x = pd.concat([x,pars])
    # compute reproduction number
    x['IFR'] = x.d
    
    res = {'Country': [], 'Year': [], 'Month': [], 'Date': [],
           'Mean': [], 'Low': [], 'High': []}
    for (c,y,m,dt),df in x.groupby(['Country','Year','Month','Date']):
        res['Country'].append(c)
        res['Year'].append(y)
        res['Month'].append(m)
        res['Date'].append(dt)
        #print(df)
        res['Mean'].append(df.IFR.mean())
        res['Low'].append(df.IFR.quantile(.05))
        res['High'].append(df.IFR.quantile(.95))
        #break
    
    res = pd.DataFrame(res)
    fig, ax = plt.subplots(figsize=(10, 6))
    if log is True: ax.set(yscale="log")
    #sns.lineplot(x="Date", y="mean", hue='Country', data=x, ax=ax)
    
    print(res)
    colors=['black','green','blue','red']
    for (g,d),col in zip(res.groupby('Country'),colors):
        ax.plot(d.Date, d.Mean, label=g, color=col)
        #ax.fill_between(d.Date, d.Low, d.High, alpha=.1, color=col)
    #ax.axhline(1, ls='--', color='grey', alpha=.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('IFR')
    ax.legend()
    plt.show()

def plot_IFR(countries, log=True):
    # get parameters
    x = None
    regions = []
    if 'CZ' in countries: regions.append(['CZ',*CZ_regions])
    if 'PL' in countries: regions.append(['PL',*PL_regions])
    if 'SE' in countries: regions.append(['SE',*SE_regions])
    if 'IT' in countries: regions.append(['IT',*IT_regions])
    for country in regions: # TODO
        pars = parameters(country)
        pars['Country'] = country[0]
        if x is None: x = pars
        else: x = pd.concat([x,pars])
    # compute reproduction number
    x['IFR'] = x.d
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    if log is True: ax.set(yscale="log")
    ax = sns.boxplot(x="Date", y="IFR", hue='Country', data=x, ax=ax)
    #ax.axhline(1, ls='--', color='grey', alpha=.5)
    plt.show()

def plot_symptoms(countries, log=True):
    # get parameters
    x = None
    regions = []
    if 'CZ' in countries: regions.append(['CZ',*CZ_regions])
    if 'PL' in countries: regions.append(['PL',*PL_regions])
    if 'SE' in countries: regions.append(['SE',*SE_regions])
    if 'IT' in countries: regions.append(['IT',*IT_regions])
    for country in regions: # TODO
        pars = parameters(country)
        pars['Country'] = country[0]
        if x is None: x = pars
        else: x = pd.concat([x,pars])
    # compute reproduction number
    x['Duration of symptoms'] = 1 / x.b
    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    if log is True: ax.set(yscale="log")
    ax = sns.boxplot(x="Date", y="Duration of symptoms", hue='Country', data=x, ax=ax)
    #ax.axhline(1, ls='--', color='grey', alpha=.5)
    plt.show()

def plot_correlation_heatmap(countries, delta=timedelta(days=60)):
    # initialize
    pars = {'Country': [], 'Date': [], 'S': [], 'E': [], 'I': [], 'R': [], 'D': []}
    regions = []
    if 'CZ' in countries: regions = [*regions, 'CZ', *CZ_regions]
    if 'PL' in countries: regions = [*regions, 'PL', *PL_regions]
    if 'SE' in countries: regions = [*regions, 'SE', *SE_regions]
    if 'IT' in countries: regions = [*regions, 'IT', *IT_regions]
    for region in regions: # TODO
        # load and crop
        lat,dt,_ = load_result(region=region)
        dates = [dt.min(), dt.max() if delta is None else dt.min()+delta]
        lat = lat[:,(dates[0] <= dt) & (dates[1] >= dt)]
        dt = dt[(dates[0] <= dt) & (dates[1] >= dt)]
        # append
        for _ in range(lat.shape[1]):
            pars['Country'].append(region)
        pars['Date'] = [*pars['Date'], *dt.to_list()]
        pars['S'] = [*pars['S'], *lat[0,:].tolist()]
        pars['E'] = [*pars['E'], *lat[1,:].tolist()]
        pars['I'] = [*pars['I'], *lat[2,:].tolist()]
        pars['R'] = [*pars['R'], *lat[3,:].tolist()]
        pars['D'] = [*pars['D'], *lat[4,:].tolist()]
    pars = pd.DataFrame(pars)
    reg_map = {r: i for i,r in enumerate(regions)}
    corM = np.zeros((len(regions), len(regions)))
    for reg1 in pars.Country.unique():
        for reg2 in pars.Country.unique():
            pars1 = pars[pars.Country == reg1]
            pars2 = pars[pars.Country == reg2]
            corS = np.corrcoef(pars1.S, pars2.S)[1,0]
            corE = np.corrcoef(pars1.E, pars2.E)[1,0]
            corI = np.corrcoef(pars1.I, pars2.I)[1,0]
            corR = np.corrcoef(pars1.R, pars2.R)[1,0]
            corD = np.corrcoef(pars1.D, pars2.D)[1,0]
            i,j = reg_map[reg1],reg_map[reg2]
            corM[i,j] = (corS+corE+corI+corR+corD)/5
    print(corM)
    return        
            #print(reg1, reg2, '=', corS, corE, corI, corR, corD)
        #break
    #print(pars.Country.unique())
    #print(pars)

def plot_correlation_distribution(countries, delta=timedelta(days=60), weekly=True):
    # regions
    x = None
    regions = []
    if 'CZ' in countries: regions.append(['CZ',*CZ_regions])
    if 'PL' in countries: regions.append(['PL',*PL_regions])
    if 'SE' in countries: regions.append(['SE',*SE_regions])
    if 'IT' in countries: regions.append(['IT',*IT_regions])
    # compute correlations
    for country in regions:
        components = 'ID' if country[0] in {'PL','SE'} else 'IRD'
        corrs = prediction_data_correlation(country, components, delta=delta, weekly=weekly)
        corrs['Country'] = country[0]
        if x is None: x = corrs
        else: x = pd.concat([x,corrs])
    #
    #xgrid = np.linspace(-1,1,41)
    #bars = np.array([[xgrid[i],xgrid[i+1]] for i in range(xgrid.shape[0]-1)])
    #x['I'] = x.I.apply()
    #xx = .99
    #b = (bars[:,0] < xx) & (bars[:,1] >= xx)
    #print(b)
    #print(bars[np.argmax(b),:])
    #return
    # plot
    #fig, ax = plt.subplots(figsize=(8, 6))
    sns.displot(x, x="D", hue="Country", element="step", multiple="stack", bins=20)
    #ax.hist(corrs['I'], bins=40, range=[-1,1], density=False)
    plt.xlim([-1,1])

def compare_60d():
    prediction_data_correlation(['CZ',*CZ_regions],
                                'IRD', timedelta(days=60))#(datetime(2020,8,1),datetime(2020,9,30)))
    prediction_data_correlation(['SE',*SE_regions],
                                'ID', timedelta(days=60))#(datetime(2020,7,6),datetime(2020,12,31)))
    prediction_data_correlation(['PL',*PL_regions],
                                'ID', timedelta(days=60))#(datetime(2020,3,4),datetime(2020,4,15)))
    prediction_data_correlation(['IT',*IT_regions],
                                'IRD', timedelta(days=60))#(datetime(2020,3,1),datetime(2020,4,30)))
def compare_all():
    prediction_data_correlation(['CZ',*CZ_regions], 'IRD')
    prediction_data_correlation(['SE',*SE_regions], 'ID')
    prediction_data_correlation(['PL',*PL_regions], 'ID')
    prediction_data_correlation(['IT',*IT_regions], 'IRD')

if __name__ == '__main__':
    #plot_correlation_distribution(['CZ','SE','PL','IT'])
    #plt.show()
    
    plot_R0(['CZ','PL','IT','SE'], log=True)
    #get_symptoms(['IT','CZ'],#['CZ','PL','IT','SE'],
    #             log=False)
    plot_symptoms(['CZ','PL','IT','SE'], log=True)
    plot_IFR(['CZ','PL','IT','SE'], log=True)
    #plot_symptoms(['CZ','PL','IT','SE'], log=True)
    
    #plot_correlation_heatmap(['CZ','PL','IT','SE'])