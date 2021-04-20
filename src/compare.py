
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import sys
sys.path.append('src')
import posterior

def get_path(region):
    return f'results/{region[:2]}/{region}'

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
    #params = x[['param_a','param_c','param_b','param_d']].to_numpy().T
    return lat,x.date

def prediction_data_correlation(regions, components, delta=None, weekly=False):
    # initialize
    corrs = {'region': regions}
    for c in components: corrs[c] = []
    # regions
    for region in regions:
        # load and crop
        lat,dt = load_result(region=region)
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
    print(corrs)
    return corrs

def compare_60d():
    prediction_data_correlation(['CZ','CZ010','CZ020','CZ031','CZ032','CZ041','CZ042','CZ051',
                                 'CZ052','CZ053','CZ063','CZ064','CZ071','CZ072','CZ080'],
                                'IRD', timedelta(days=60))#(datetime(2020,8,1),datetime(2020,9,30)))
    prediction_data_correlation(['SE','SE110','SE121','SE122','SE123','SE124','SE125','SE211',
                                 'SE212','SE213','SE214','SE221','SE224','SE231','SE232',
                                 'SE311','SE312','SE313','SE321','SE322','SE331','SE332'],
                                'ID', timedelta(days=60))#(datetime(2020,7,6),datetime(2020,12,31)))
    prediction_data_correlation(['PL','PL71','PL72','PL21','PL22','PL81','PL82','PL84','PL41','PL42',
                                 'PL43','PL51','PL52','PL61','PL62','PL62','PL63','PL9'],
                                'ID', timedelta(days=60))#(datetime(2020,3,4),datetime(2020,4,15)))
    prediction_data_correlation(['IT','ITC1','ITC2','ITC3','ITC4','ITF1','ITF2','ITF3','ITF4','ITF5','ITF6',
                                 'ITG1','ITG2','ITH10','ITH20','ITH3','ITH4','ITH5','ITI1','ITI2','ITI3','ITI4'],
                                'IRD', timedelta(days=60))#(datetime(2020,3,1),datetime(2020,4,30)))
def compare_all():
    prediction_data_correlation(['CZ','CZ010','CZ020','CZ031','CZ032','CZ041','CZ042','CZ051',
                                 'CZ052','CZ053','CZ063','CZ064','CZ071','CZ072','CZ080'], 'IRD')
    prediction_data_correlation(['SE','SE110','SE121','SE122','SE123','SE124','SE125','SE211',
                                 'SE212','SE213','SE214','SE221','SE224','SE231','SE232',
                                 'SE311','SE312','SE313','SE321','SE322','SE331','SE332'], 'ID')
    prediction_data_correlation(['PL','PL71','PL72','PL21','PL22','PL81','PL82','PL84','PL41','PL42',
                                 'PL43','PL51','PL52','PL61','PL62','PL62','PL63','PL9'], 'ID')
    prediction_data_correlation(['IT','ITC1','ITC2','ITC3','ITC4','ITF1','ITF2','ITF3','ITF4',
                                 'ITF5','ITF6','ITG1','ITG2','ITH10','ITH20','ITH3','ITH4',
                                 'ITH5','ITI1','ITI2','ITI3','ITI4'], 'IRD')

if __name__ == '__main__':
    compare_all()