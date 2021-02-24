
from datetime import datetime,timedelta
import pandas as pd
import sys
sys.path.append('src')

import _src

def tests(country = None, date_min = None, date_max = None):
    # fetch data
    x = _src.get_data()
    if country is None:
        country = ['CZE','ITA','SWE','POL']
    else:
        try: country = [*country]
        except: country = [country]
    x = x[x.iso_alpha_3.isin(country)]
    # filter segment
    if date_min is not None:
        date_min = datetime.strptime(date_min,'%Y-%m-%d')-timedelta(days=1)
        x = x[(x.date >= date_min)]
    if date_max is not None:
        date_max = datetime.strptime(date_max,'%Y-%m-%d')
        x = x[(x.date < date_max)]
    # parse
    res = pd.DataFrame({
        'country': x.iso_alpha_3,
        'date': x.date,
        'tests': x.tests,
        'pos': x.confirmed
    })
    return res.iloc[1:]
def _get_test_ratios(*args, **kw):
    # data
    x = tests(*args, **kw)
    # compute distribution
    x['ratio'] = x.pos / x.tests
    x['ratio'] = x.ratio.fillna(0.0)
    x['ratio'] = x.ratio.apply(lambda i: i if abs(i) != float('inf') else 0)
    # result
    return x