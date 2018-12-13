# from option_utilities import read_feather, write_feather
from spx_data_update import UpdateSP500Data, quandle_api
import numpy as np
# from arch import arch_model
import pyfolio as pf
import statsmodels.formula.api as sm
# import matplotlib
# matplotlib.use('MacOSX')

import pandas_datareader.data as web
import pandas as pd
import quandl
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from implied_to_realized import SPX5MinuteBars

bars = SPX5MinuteBars()
vol = bars.realized_vol()
vrp = bars.vol_risk_premium
evol = bars.expected_vol()
rv = bars.realized_variance() * 10000
daily_ret = bars.daily_return()

for i in range(0, 10):
    c = cm.viridis(i / 10, 1)
    evol.iloc[:, -i].plot(color=c)

[sp500, vix] = [web.get_data_yahoo(item, 'JAN-01-90') for item in ['^GSPC', '^VIX']]

vix = vix['Close']
IV = vix**2 / 12
# Check IV is same as vpr['IV']
# diff = vrp['IV'] - IV.resample('BM', closed='left').ffill().dropna()
# diff.plot()
# diff = vrp['RV'] - rv.resample('BM', closed='left').ffill().dropna()
# diff.plot()






# # realized_quadratic_variation = squared_diff.rolling(1716).sum().dropna() * 10000
# RV_calc = rv_22.resample('BM').bfill()
# RV_calc = RV_calc.rename('RV_calc')

#
# [sp500, vix] = [web.get_data_yahoo(item, 'JAN-01-90') for item in ['^GSPC', '^VIX']]
# sp_monthly_ret = sp500['Adj Close'].resample('BM').bfill().dropna().pct_change().dropna()
#
# sp_quarterly_ret = sp500['Adj Close'].resample('BQ').bfill().dropna().pct_change().dropna()
# [sp_monthly_ret, sp_quarterly_ret] = [df.rename('sp5_ret') for df in [sp_monthly_ret, sp_quarterly_ret]]
#
# quandl.ApiConfig.api_key = quandle_api()
# cape = quandl.get('MULTPL/SHILLER_PE_RATIO_MONTH', collapse='monthly')
#
# cape = np.log(cape['Value'].rename('cape'))
#
# IV_calc = vix['Close']**2/12
# IV_calc = IV_calc.rename('IV_calc')
# IV_calc = IV_calc.resample('BM').bfill()
# VRP_calc = IV_calc - RV_calc
# VRP_calc = VRP_calc.rename('VRP_calc')
#
# VRP_combo = VRP_calc.combine_first(vrp_data['VRP'])
# VRP_combo = VRP_combo.rename('VRP_combo')
#
# regression_data = pd.concat([sp_monthly_ret, VRP_combo.shift(1), vrp_data['VRP'].shift(1),
#                              cape.shift(1).resample('BM').bfill()], axis=1)
#
# regression_data = regression_data.dropna(axis=0, how='any')
#
# regression_string = 'sp5_ret ~ VRP_combo'
# results = sm.ols(formula=regression_string, data=regression_data).fit()
# results.summary()
# sns.lmplot(x='VRP_combo', y='sp5_ret', data=regression_data , height=10, aspect=2)
#
# [VRP_combo_q, vrp_data_q, cape_q] = [df.resample('BQ').bfill() for df in [VRP_combo, vrp_data, cape]]
#
#
# regression_data_quarterly = pd.concat([sp_quarterly_ret, VRP_combo_q.shift(1), cape_q.shift(1)],
#                                       axis=1)
# regression_data_quarterly = regression_data_quarterly.dropna(axis=0, how='any')
#
# regression_string = 'sp5_ret ~ VRP_combo'
# results = sm.ols(formula=regression_string, data=regression_data_quarterly).fit()
# results.summary()
# sns.lmplot(x='VRP_combo', y='sp5_ret', data=regression_data_quarterly, height=10, aspect=2)
#
#

#
#
#
#
#
# e_vol = e_vol.transpose()
# mask = e_vol > 1
# e_vol[mask] = 1
#
#
# for i in range(1, len(foo)):
#     print(np.sqrt(foo[i]))