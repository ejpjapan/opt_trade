from option_utilities import read_feather, write_feather
from spx_data_update import UpdateSP500Data, quandle_api
import numpy as np
# from arch import arch_model
import pyfolio as pf
import statsmodels.formula.api as sm
import pandas_datareader.data as web
import pandas as pd
import quandl
import datetime
import seaborn as sns
from implied_to_realized import spx_bar_history
import matplotlib.pyplot as plt

full_hist = spx_bar_history(update_bars=False)

squared_diff = (np.log(full_hist['close'] / full_hist['close'].shift(1)))**2
realized_quadratic_variation = squared_diff.groupby(squared_diff.index.date).sum()
realized_quadratic_variation = realized_quadratic_variation.reindex(pd.to_datetime(realized_quadratic_variation.index))

daily_vol = np.sqrt(realized_quadratic_variation * 252)

[sp500, vix] = [web.get_data_yahoo(item, 'JAN-01-90') for item in ['^GSPC', '^VIX']]
# rv_22 = realized_quadratic_variation.rolling(22).sum()

sp_daily_ret = sp500['Adj Close']



# # realized_quadratic_variation = squared_diff.rolling(1716).sum().dropna() * 10000
# RV_calc = rv_22.resample('BM').bfill()
# RV_calc = RV_calc.rename('RV_calc')
# vrp_data = pd.read_csv(UpdateSP500Data.DATA_BASE_PATH / 'xl' / 'vol_risk_premium.csv',
#                        usecols=['VRP', 'EVRP', 'IV', 'RV', 'ERV'])
# vrp_data = vrp_data.set_index(pd.date_range('31-jan-1990', '31-dec-2017', freq='BM'))
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
# realized_volatility = np.sqrt(realized_quadratic_variation)
# series_list = []
# for i in range(500, len(realized_volatility) + 1):
#     am = arch_model(realized_volatility[i-500:i], mean='HAR', lags=[1, 5, 22],  vol='Constant')
#     res = am.fit()
#     forecasts = res.forecast(horizon=50)
#     np_vol = forecasts.mean.iloc[-1]
#     series_list.append(np_vol)
#
# e_vol = pd.concat(series_list, axis=1) * np.sqrt(252) * 100  # annualize
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