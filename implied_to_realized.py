from option_utilities import read_feather, write_feather
from spx_data_update import UpdateSP500Data, ImpliedVolatilityHistory
from ib_insync import IB, Index, util
import numpy as np
import pandas as pd
from arch import arch_model


class SPX5MinuteBars:

    def __init__(self, update_bars=True):
        self.bars = self.spx_bar_history(update_bars)
        self.vol_risk_premium = self.vrp()

    @staticmethod
    def spx_bar_history(update_bars=True):
        file_name = 'sp500_5min_bars'
        df_hist = read_feather(UpdateSP500Data.DATA_BASE_PATH / file_name)
        # Download latest
        if update_bars:
            ib = IB()
            ib.connect('127.0.0.1', port=4001, clientId=40)
            contract = Index('SPX', 'CBOE', 'USD')
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
            full_hist = df.combine_first(df_hist)
            write_feather(full_hist, UpdateSP500Data.DATA_BASE_PATH / file_name)
        else:
            full_hist = df_hist.copy()
        return full_hist

    @staticmethod
    def vrp():
        vrp = pd.read_csv(UpdateSP500Data.DATA_BASE_PATH / 'xl' / 'vol_risk_premium.csv',
                               usecols=['VRP', 'EVRP', 'IV', 'RV', 'ERV'])
        vrp = vrp.set_index(pd.date_range('31-jan-1990', '31-dec-2017', freq='BM'))
        return vrp

    def realized_vol(self):
        """Annualized daily volatility calculated as sum of squared 5 minute returns"""
        squared_diff = (np.log(self.bars['close'] / self.bars['close'].shift(1))) ** 2
        realized_quadratic_variation = squared_diff.groupby(squared_diff.index.date).sum()
        realized_quadratic_variation = realized_quadratic_variation.reindex(
            pd.to_datetime(realized_quadratic_variation.index))
        daily_vol = np.sqrt(realized_quadratic_variation * 252)
        daily_vol = daily_vol.rename('rv_daily')
        return daily_vol

    def expected_vol(self, window=500):
        """Expected volatility out to 50 days using HAR model"""
        daily_vol = self.realized_vol()
        series_list = []
        for i in range(window, len(daily_vol) + 1):
            am = arch_model(daily_vol[i-window:i], mean='HAR', lags=[1, 5, 22],  vol='Constant')
            res = am.fit()
            forecasts = res.forecast(horizon=50)
            np_vol = forecasts.mean.iloc[-1]
            series_list.append(np_vol)
        e_vol = pd.concat(series_list, axis=1)
        return e_vol

    def realized_variance(self, window=22):
        """Realized variance see VRP literature"""
        realized_quadratic_variation = (self.realized_vol()**2) / 252
        rv = realized_quadratic_variation.rolling(window).sum()
        rv = rv.rename('RV_CALC')
        return rv

    def daily_return(self):
        daily_ret = self.bars['close'].groupby(self.bars.index.date).last().pct_change()
        return daily_ret

