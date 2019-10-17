from option_utilities import read_feather, write_feather
from spx_data_update import UpdateSP500Data, IbWrapper
from ib_insync import IB, Index, util
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class SPX5MinuteBars:

    def __init__(self, update_bars=True, window=500, horizon=50, realized_window=22):
        self.bars = self.spx_bar_history(update_bars)
        self.vol_risk_premium = self.vrp()
        self.har_vol = pd.DataFrame()
        self.window = window
        self.horizon = horizon
        self.realized_window = realized_window

    @staticmethod
    def spx_bar_history(update_bars=True):
        file_name = 'sp500_5min_bars'
        df_hist = read_feather(UpdateSP500Data.DATA_BASE_PATH / file_name)
        # Download latest
        if update_bars:
            ibw = IbWrapper()
            ib = ibw.ib
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

    def plot_vol_forecast(self, num_days=10):
        expected_volatility = self.expected_vol
        fig, ax = plt.subplots(figsize=(12, 5), dpi=80, facecolor='w', edgecolor='k')

        for i in range(-1, -(num_days + 1), -1):
            if i == -1:
                expected_volatility.iloc[:, -1].plot(color='r')
            else:
                c = cm.viridis(-i / num_days, 1)
                expected_volatility.iloc[:, i].plot(color=c)

        plt.autoscale(enable=True, axis='x', tight=True)
        legend_labels = expected_volatility.iloc[:, -num_days:].columns.strftime('%d-%b')
        _ = plt.legend(legend_labels[::-1])
        _ = plt.title('HAR Volatity Forecast')
        _ = ax.set_ylabel('Annualized Vol %')
        return ax

    @property
    def realized_vol(self):
        """Annualized daily volatility calculated as sum of squared 5 minute returns"""
        squared_diff = (np.log(self.bars['close'] / self.bars['close'].shift(1))) ** 2
        realized_quadratic_variation = squared_diff.groupby(squared_diff.index.date).sum()
        realized_quadratic_variation = realized_quadratic_variation.reindex(
            pd.to_datetime(realized_quadratic_variation.index))
        daily_vol = np.sqrt(realized_quadratic_variation * 252)
        daily_vol = daily_vol.rename('rv_daily')
        return daily_vol

    @property
    def expected_vol(self):
        """Expected volatility out to 50 days using HAR model"""
        if self.har_vol.empty:
            daily_vol = self.realized_vol
            series_list = []
            for i in range(self.window, len(daily_vol) + 1):
                am = arch_model(daily_vol[i - self.window:i], mean='HAR', lags=[1, 5, 22], vol='Constant')
                res = am.fit()
                forecasts = res.forecast(horizon=self.horizon)
                np_vol = forecasts.mean.iloc[-1]
                series_list.append(np_vol)
            e_vol = pd.concat(series_list, axis=1)
            self.har_vol = e_vol
        else:
            e_vol = self.har_vol
        return e_vol

    @property
    def realized_variance(self):
        """Realized variance see VRP literature"""
        realized_quadratic_variation = (self.realized_vol**2) / 252
        rv = realized_quadratic_variation.rolling(self.realized_window).sum()
        rv = rv.rename('RV_CALC')
        return rv

    @property
    def daily_return(self):
        daily_ret = self.bars['close'].groupby(self.bars.index.date).last().pct_change()
        return daily_ret

