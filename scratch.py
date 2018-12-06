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
from ib_insync import *

# Get history
file_name = 'sp500_5min_bars'
df_hist = read_feather(UpdateSP500Data.DATA_BASE_PATH / file_name)
update_bars = True
# Download latest
if update_bars:
    ib = IB()
    ib.connect('127.0.0.1', port=4001, clientId=40)

    contract = Index('SPX', 'CBOE', 'USD')

    end = datetime.datetime(2006, 12, 6, 9, 30)

    bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='1 M',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1)

    ib.disconnect()
    df = util.df(bars)
    df = df.set_index('date')
    full_hist = pd.concat([df_hist, df], axis=0, sort=True)
    full_hist = full_hist.drop_duplicates()
    write_feather(full_hist.drop_duplicates(), UpdateSP500Data.DATA_BASE_PATH / file_name)
else:
    full_hist = df_hist.copy()

squared_diff = (np.log(full_hist['close'] / full_hist['close'].shift(1)))**2

realized_quadratic_variation = squared_diff.rolling(1716).sum().dropna()

# annualizedVol =np.sqrt(rv*12) * 100
#
#
# realized_quadratic_variation = squared_diff.resample('B').sum()
# remove zero return days
realized_quadratic_variation = realized_quadratic_variation[realized_quadratic_variation != 0]

realized_volatility = np.sqrt(realized_quadratic_variation * 252)

series_list = []
for i in range(500, len(realized_volatility) + 1):
    am = arch_model(realized_volatility[i-500:i], mean='HAR', lags=[1, 5, 22],  vol='Constant')
    res = am.fit()
    forecasts = res.forecast(horizon=50)
    np_vol = forecasts.mean.iloc[-1] * np.sqrt(252)
    series_list.append(np_vol)

e_vol = pd.concat(series_list, axis=1)
e_vol = e_vol.transpose()
mask = e_vol > 1
e_vol[mask] = 1


[sp500, vix] = [web.get_data_yahoo(item, 'JAN-01-90') for item in ['^GSPC', '^VIX']]

sp_monthly_ret = sp500['Adj Close'].resample('BM').bfill().dropna().pct_change().dropna()
# sp_monthly_ret  = np.log(sp500['Adj Close'].resample('BM').bfill().dropna() / sp500['Adj Close'].resample('BM').bfill().dropna().shift(1))
sp_monthly_ret = sp_monthly_ret.rename('sp5_ret')

vrp_data = pd.read_csv(UpdateSP500Data.DATA_BASE_PATH / 'xl' / 'vol_risk_premium.csv',
                       usecols=['VRP', 'EVRP', 'IV', 'RV', 'ERV'])
vrp_data = vrp_data.set_index(pd.date_range('31-jan-1990', '31-dec-2017', freq='BM'))


# realized_quadratic_variation_22 = realized_quadratic_variation.rolling(22).sum()
rv_calc = realized_volatility.resample('BM').bfill().dropna()
rv_calc = rv_calc.rename('rv_calc')

quandl.ApiConfig.api_key = quandle_api()
cape = quandl.get('MULTPL/SHILLER_PE_RATIO_MONTH', collapse='monthly')

regression_data = pd.concat([sp_monthly_ret, vrp_data.shift(1) / 100, rv_calc.shift(1), cape.apply(np.log).shift(1).resample('BM').bfill()], axis=1)
regression_data = regression_data.dropna(axis=0, how='any')

regression_string = 'sp5_ret ~ rv_calc'
results = sm.ols(formula=regression_string, data=regression_data).fit()
results.summary()
