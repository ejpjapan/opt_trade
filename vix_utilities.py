# from IPython.display import display_html, HTML
import pyfolio as pf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARMA
# import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
# from arch import arch_model
import matplotlib.gridspec as gridspec
from ib_insync import Future, util
# import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import re
import zipfile
from option_utilities import read_feather
from pyfolio.timeseries import cum_returns


# import nest_asyncio
# from time import time
# import plistlib
# import sys

from spx_data_update import UpdateSP500Data, ImpliedVolatilityHistory, SP500Index, IbWrapper, GetRawCBOEOptionData

from option_utilities import PlotConstants, chart_format


class VixForecast:

    def __init__(self):
        self.vix_full_hist = self.vix_history
        self.yhat = None
        self.model_fit = None
        self.vix = None

    def forecast_vix(self, history=None, steps=300):
        if history is None:
            history = self.vix_full_hist
        model = ARMA(history.values, order=(2, 2))
        # start_params for ARMA model are from VIX Premium paper
        self.model_fit = model.fit(disp=0, start_params=[20.083, 1.651, -0.654, -0.714, -0.064])

        output = self.model_fit.forecast(steps=steps)
        self.yhat = output[0]
        self.vix = history
        return

    @property
    def vix_history(self):
        iv_hist = ImpliedVolatilityHistory()
        vix = iv_hist.implied_vol_index
        if vix.index[-1].date() == pd.to_datetime('today').date():
            # remove last observation if today
            vix = vix[:-1]
        return vix

    @property
    def time_series_plots(self):
        self._tsplot(self.vix, lags=50)

    def vix_beta(self, rolling_window=21 * 60):
        sp5 = SP500Index()
        if self.vix is None:
            vix = self.vix_full_hist
        else:
            vix = self.vix
        shared_dates = vix.index.intersection(sp5.excess_return_index.index)
        beta = pf.timeseries.rolling_beta(vix.reindex(shared_dates).pct_change().dropna(how='any'),
                                          sp5.excess_return_index.reindex(shared_dates).pct_change().dropna(how='any'),
                                          rolling_window=rolling_window)
        return beta, rolling_window

    def plot_rolling_beta(self, **kwargs):
        beta, rolling_window = self.vix_beta(**kwargs)
        pc = PlotConstants()
        with plt.style.context('bmh'):
            _ = plt.figure(figsize=pc.fig_size,
                           dpi=600,
                           facecolor='None',
                           edgecolor='None')
            gs = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.25)
            ax_beta = plt.subplot(gs[:])
            ax_beta = beta.plot(lw=1.5,
                                ax=ax_beta,
                                grid=True,
                                alpha=0.4,
                                color=pc.color_yellow,
                                title='VIX beta to S&P500 - {} days rolling window'.format(rolling_window))
            ax_beta.set_ylabel('Beta')
            ax_beta.axhline(beta.mean(),
                            color='k',
                            ls='--',
                            lw=0.75,
                            alpha=1.0)
            chart_format([ax_beta], pc.color_light)
            plt.autoscale(enable=True,
                          axis='x',
                          tight=True)
        return ax_beta

    @staticmethod
    def _tsplot(y, lags=None, figsize=(16, 9), style='bmh'):
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        with plt.style.context(style):
            _ = plt.figure(figsize=figsize)
            mpl.rcParams['font.sans-serif'] = 'Roboto Condensed'
            mpl.rcParams['font.family'] = 'sans-serif'
            layout = (3, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))
            qq_ax = plt.subplot2grid(layout, (2, 0))
            pp_ax = plt.subplot2grid(layout, (2, 1))

            y.plot(ax=ts_ax)
            ts_ax.set_title('Time Series Analysis Plots')
            smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
            smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
            sm.qqplot(y, line='s', ax=qq_ax)
            qq_ax.set_title('QQ Plot')
            scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

            plt.tight_layout()
        return


def get_futures(contract_str, remove_weekly=False):
    ibw = IbWrapper()
    ib = ibw.ib
    vix = Future(contract_str, includeExpired=False)
    cds = ib.reqContractDetails(vix)

    contracts = [cd.contract for cd in cds]
    if remove_weekly:
        contracts = [contract for contract in contracts if len(contract.localSymbol) <= 4]

    bars_list = []
    for contract in contracts:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='5 D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1)
        if bars:
            bars_list.append(util.df(bars))

    ib.disconnect()
    contract_df = util.df(contracts)
    close_list = [item.loc[:, ['date', 'close']] for item in bars_list]
    close_list = [item.set_index('date') for item in close_list]

    close_list = [item.rename(index=str, columns={'close': name})
                  for item, name in zip(close_list,
                                        pd.to_datetime(contract_df['lastTradeDateOrContractMonth']))]
    future_series = pd.concat(close_list, axis=1, sort=False)
    future_series = future_series.transpose().sort_index()
    future_series.columns = pd.to_datetime(future_series.columns)
    return future_series, contract_df


class UpdateVIXData:
    DIRECTORY = UpdateSP500Data.DATA_BASE_PATH / 'CBOERawVixData'

    def __init__(self):
        self.order_string = '/order_000008108/item_000010804/'
        self.zip_directory, self.csv_directory, self.feather_directory = [self.DIRECTORY / item for item in
                                                                          ['zip', 'csv', 'feather']]
        self.price_columns = ['open_bid', 'open_ask', 'avg_bid', 'avg_ask', 'close_bid',
                              'close_ask', 'open_px', 'close_px', 'low_px', 'high_px']

    def download_cboe_vix(self):
        """Retrieve raw zip files from CBOE Datashop FTP server
        Will return error once data is removed from FTP server"""
        ftp = GetRawCBOEOptionData.open_ftp()
        # Download zip files
        ftp.cwd(self.order_string)
        ftp_file_list = ftp.nlst()
        for file in ftp_file_list:
            if file.endswith('.zip'):
                print("Downloading..." + file)
                ftp.retrbinary("RETR " + file, open(self.zip_directory / file, 'wb').write)
        ftp.close()

        # Unzip to csv
        for item in os.listdir(self.zip_directory):  # loop through items in dir
            if item.endswith('.zip'):
                file_name = self.zip_directory / item  # get full path of files
                zip_ref = zipfile.ZipFile(file_name)  # create zipfile object
                try:
                    zip_ref.extractall(self.csv_directory)  # extract file to dir
                except zipfile.BadZipFile as err:
                    print("Zipfile error: {0} for {1}".format(err, item))
                zip_ref.close()  # close file
        num_csv, num_zip = len(os.listdir(self.csv_directory)), len(os.listdir(self.zip_directory))
        assert (num_csv == num_zip)

    @property
    def raw_df(self):
        dataframe_list = []
        for item in os.listdir(self.csv_directory):
            if item.endswith('.csv'):
                future_df = pd.read_csv(self.csv_directory / item)
                dataframe_list.append(future_df)
        raw_df = pd.concat(dataframe_list, axis=0, ignore_index=True)
        return raw_df


class VixCBOE:
    def __init__(self, expiry_type=0):
        """Class to manipulate and provide methods on raw CBOE vix futures dataframe"""

        raw_cboe_df = read_feather(UpdateVIXData.DIRECTORY / 'feather' / 'vix_data.feather')
        raw_cboe_df = raw_cboe_df.set_index('quote_date')

        # Fill missing close price with average of bid & ask
        matching_exp = [s for s in raw_cboe_df.columns if "expiry" in s]
        close_digits = []
        for string in matching_exp:
            digit = int(re.findall("\d+", string)[0])
            close_digits.append(digit)

        close_digits.sort()
        for item in close_digits:
            raw_cboe_df.loc[:, 'close' + str(item)] = raw_cboe_df[['close_ask' + str(item),
                                                                   'close_bid' + str(item)]].mean(axis=1)
            raw_cboe_df.loc[:, 'close' + str(item)] = raw_cboe_df.loc[:, 'close' + str(item)].fillna(
                raw_cboe_df.loc[:, 'close_px'
                                + str(item)])
        self.closing_prices = raw_cboe_df[['close' + str(digit) for digit in close_digits]]
        self.expirations = raw_cboe_df[['expiry' + str(digit) for digit in close_digits]]

        self.start_date = self.closing_prices.index[0]
        self.expiry_type = expiry_type  # expiry_type is either string or positive integer

    @property
    def _expiry_returns(self):
        """ Returns continous future arithmetic return if contracts are held to expiry"""
        roll_rows = self.expirations['expiry1'].diff(1) > pd.Timedelta('0 Days')  # Day before expiry
        returns = self.closing_prices.pct_change()
        close = self.closing_prices.copy()
        # Cross the columns on the day after expiry
        column_shift_ret = close.divide(close.shift(periods=-1, axis='columns').shift(periods=1, axis='rows')) - 1
        returns[roll_rows] = column_shift_ret[roll_rows]
        return returns

    @property
    def _business_days_2_expiry(self):
        """ Returns number of business days to expiration """
        expirations = self.expirations.copy()
        column_names = self.expirations.columns
        business_day_list = []
        _notNaT = np.datetime64(pd.datetime.today())
        for col in column_names:
            expiry = expirations[col]
            begin_dates = expiry.index
            end_dates = pd.DatetimeIndex(expiry.values)
            end_dates_mask = end_dates.to_series().isna().values
            bus_days = np.busday_count(list(begin_dates.date),
                                       list(pd.DatetimeIndex(np.where(end_dates_mask, _notNaT, end_dates)).date))
            out_bus_days = [np.nan if x in bus_days[end_dates_mask] else x for x in bus_days]
            business_day_list.append(pd.Series(data=out_bus_days, index=expiry.index, name=col))
        return pd.concat(business_day_list, axis=1)

    def rolled_future(self):
        """Returns continuous return, price index, expiries and days 2 expiry for vix future rolled according to
        expiry type"""
        expiry_dates = self.expirations['expiry1']
        returns = self._expiry_returns
        business_days_2_exp = self._business_days_2_expiry
        eom_dates = returns.loc[returns.groupby(returns.index.to_period('M')).apply(lambda x: x.index.max())].index

        last_month_end = eom_dates[-1] + pd.offsets.MonthEnd(0)
        eom_dates = eom_dates[:-1]
        eom_dates = eom_dates.insert(-1, last_month_end)
        roll_dates = eom_dates.sort_values()

        expiry_for_roll = []
        for dts in expiry_dates:
            idx = roll_dates.get_loc(dts, method='ffill')
            expiry_for_roll.append(roll_dates[idx])
        day_diff = expiry_dates.index - pd.DatetimeIndex(expiry_for_roll)
        front_month_bool = day_diff.days <= 0
        back_month_bool = ~front_month_bool

        rolled_return, rolled_future_price = \
            [pd.concat([item['close2'][back_month_bool], item['close1'][front_month_bool]], axis=0).sort_index()
             for item in [returns, self.closing_prices]]

        rolled_expiries, days_2_exp = \
            [pd.concat([item['expiry2'][back_month_bool], item['expiry1'][front_month_bool]], axis=0).sort_index()
             for item in [self.expirations, business_days_2_exp]]

        rolled_return[0] = np.nan  # replace first empty observation with NaN
        return rolled_return, rolled_expiries, days_2_exp, rolled_future_price

    @property
    def rolled_return(self):
        """Returns daily return from long position in vix future"""
        return self.rolled_future()[0].rename('long_vix')

    @property
    def rolled_idx(self):
        """Returns cumulative return index from long position in vix future"""
        start_idx = 100
        cumulative_returns = cum_returns(self.rolled_return, start_idx)
        # Add back start of index
        cumulative_returns[self.start_date] = start_idx
        idx = cumulative_returns.reindex(cumulative_returns.index.sort_values())
        return idx.rename('long_vix')

    @property
    def rolled_idx_short(self):
        """Returns cumulative return index from short position in vix future"""
        idx = 1 / self.rolled_idx
        idx = idx / idx[0] * 100
        return idx.rename('short_vix')

    @property
    def rolled_return_short(self):
        """Returns arithmetic return from short position in vix future"""
        return self.rolled_idx_short.pct_change()
