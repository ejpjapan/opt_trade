#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:48:37 2018

@author: esbMac
"""
import numpy as np

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd


from abc import ABC, abstractmethod


class Volatility(ABC):
    def __init__(self, window: int, sigma_target: float):
        self.rolling_window = window
        self.sigma_target = sigma_target

    @abstractmethod
    def compute(self):
        pass


class VolatilitySD(Volatility):
    def __init__(self, window: int, sigma_target: float, asset_data: pd.DataFrame):
        super().__init__(window, sigma_target)
        self.price_data = asset_data

    def compute(self):
        # volatility = self.daily_ret.rolling(self.rolling_window).std() * np.sqrt(252)
        volatility = self.price_data['Close'].pct_change().rolling(self.rolling_window).std() * np.sqrt(252)
        # volatility[volatility < self.sigma_target / 10.0] = self.sigma_target / 10.0
        volatility.fillna(np.nan, inplace=True)

        return volatility


class VolatilityYZ(Volatility):
    def __init__(self, window: int, sigma_target: float, asset_data: pd.DataFrame):
        super().__init__(window, sigma_target)
        self.data = asset_data

    def compute(self):
        volatility = self.__get_estimator(self.data, self.rolling_window)
        # volatility[volatility < self.sigma_target / 10.0] = self.sigma_target / 10.0
        volatility.fillna(np.nan, inplace=True)

        return volatility

    @staticmethod
    def __get_estimator(price_data: pd.DataFrame, window, trading_periods=252):
        log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
        log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
        log_co = (price_data['Close'] / price_data['Open']).apply(np.log)

        log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
        log_oc_sq = log_oc ** 2

        log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
        log_cc_sq = log_cc ** 2

        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        close_vol = log_cc_sq.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))
        open_vol = log_oc_sq.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))
        window_rs = rs.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))

        k = 0.34 / (1 + (window + 1) / (window - 1))
        result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

        # if clean:
        #     return result.dropna()
        # else:
        return result

########################################

# from option_utilities import read_feather, write_feather
from spx_data_update import UpdateSP500Data
import pandas as pd
import numpy as np
import feather
import quandl


# from ib_insync import *
# ib = IB()
# ib.connect('127.0.0.1', port=4001, clientId=40)
#
# contract = Index('SPX', 'CBOE', 'USD')
#
#
# bars = ib.reqHistoricalData(
#         contract,
#         endDateTime='',
#         durationStr='4 Y',
#         barSizeSetting='5 mins',
#         whatToShow='TRADES',
#         useRTH=True,
#         formatDate=1)
#
# ib.disconnect()
# df = util.df(bars)
# feather.write_dataframe(df, UpdateSP500Data.DATA_BASE_PATH / 'sp5_bars')

df = feather.read_dataframe(UpdateSP500Data.DATA_BASE_PATH / 'sp5_bars')

df = df.set_index('date')
squared_diff = (np.log(df['close'] / df['close'].shift(1)))**2
rv = squared_diff.rolling(78*22).sum()
annualizedVol =np.sqrt(rv*12) * 100
rv_month_end = annualizedVol.resample('BM').bfill().dropna()

vrp_data = pd.read_csv(UpdateSP500Data.DATA_BASE_PATH / 'xl' / 'vol_risk_premium.csv',
                       usecols=['VRP', 'EVRP', 'IV', 'RV', 'ERV'])
vrp_data = vrp_data.set_index(pd.date_range('31-jan-1990', '31-dec-2017',freq='BM'))

compare_data = pd.concat([vrp_data['RV'].reindex(rv_month_end.index), rv_month_end], axis=1).dropna(how='any').corr()

from option_simulation import OptionSimulation, OptionTrades
from time import time
import pandas_datareader.data as web
import pyfolio as pf

before = time()
optsim = OptionSimulation(update_simulation_data=False)

dtfs = optsim.trade_sim(-1, 1, trade_type='EOM', option_type='P')

# variable_leverage = pd.Series(np.linspace(1,2,len(dtfs[2])), index=dtfs[2])
opt_trade = OptionTrades(dtfs, 2)
opt_idx =pf.timeseries.cum_returns(opt_trade.returns[1],100)

opt_idx_ret = opt_idx.resample('BM').bfill().pct_change()


[sp500, vix] = [web.get_data_yahoo(item, 'JAN-01-90') for item in ['^GSPC', '^VIX']]
sp_monthly_ret = sp500['Adj Close'].resample('BM').bfill().dropna().pct_change().dropna()

################################################################
from spx_data_update import UpdateSP500Data
import pandas_datareader.data as web
# from implied_to_realized import VolatilityYZ, VolatilitySD
import pandas as pd
import pyfolio as pf
import statsmodels.formula.api as sm
import numpy as np

[sp500, vix] = [web.get_data_yahoo(item, 'JAN-01-90') for item in ['^GSPC', '^VIX']]


sp_monthly_ret  = sp500['Adj Close'].resample('BM').bfill().dropna().pct_change().dropna()
# sp_monthly_ret  = np.log(sp500['Adj Close'].resample('BM').bfill().dropna() / sp500['Adj Close'].resample('BM').bfill().dropna().shift(1))
sp_monthly_ret = sp_monthly_ret.rename('sp5_ret')


vrp_data = pd.read_csv(UpdateSP500Data.DATA_BASE_PATH / 'xl' / 'vol_risk_premium.csv',
                       usecols=['VRP', 'EVRP', 'IV','RV','ERV'])

vrp_data = vrp_data.set_index(pd.date_range('31-jan-1990', '31-dec-2017',freq='BM'))
cape = quandl.get('MULTPL/SHILLER_PE_RATIO_MONTH', collapse='monthly')
regression_data = pd.concat([sp_monthly_ret, vrp_data.shift(1) / 100, cape.apply(np.log).shift(1).resample('BM').bfill()], axis=1)
regression_data = regression_data.dropna(axis=0, how='any')


annual_returns_dict = {}
for col_name in ['sp5_ret']:
    annual_returns_dict[col_name] = pf.timeseries.annual_return(regression_data[col_name], period='monthly')

regression_string = 'sp5_ret ~ VRP + RV + Value'
#
results = sm.ols(formula=regression_string, data=regression_data).fit()

results.summary()



# vol_yz= VolatilityYZ(22, 10, sp500)
# vol_sd= VolatilitySD(22, 10, sp500)
#
# vol_rf = (vix['Close']/100 - vol_sd.compute())
# vol_rf.plot()
# vol_rf_yz = (vix['Close']/100 - vol_yz.compute())
# vol_rf_yz.mean()



from option_simulation import OptionSimulation, OptionTrades
from time import time
before = time()
optsim = OptionSimulation(update_simulation_data=False)

dtfs = optsim.trade_sim(-1, 1, trade_type='EOM', option_type='P')

variable_leverage = pd.Series(np.linspace(1,2,len(dtfs[2])), index=dtfs[2])
opt_trade = OptionTrades(dtfs, 2)

opt_sim_index = pf.timeseries.cum_returns(opt_trade.returns[1], starting_value=100)

opt_sim_index = opt_sim_index.resample('BM').bfill().dropna().pct_change().dropna()
opt_sim_index = opt_sim_index.rename('options')


regression_data_2 = pd.concat([regression_data, opt_sim_index], axis=1)
regression_data_2 = regression_data_2.dropna(axis=0, how='any')


regression_string = 'options ~ ERV'
#
results = sm.ols(formula=regression_string, data=regression_data_2).fit()

results.summary()