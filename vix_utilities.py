# from IPython.display import display_html, HTML
import pyfolio as pf
# import numpy as np
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
# import nest_asyncio
# from time import time
# import plistlib
# import sys

from spx_data_update import ImpliedVolatilityHistory, SP500Index, IbWrapper

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
    future_series.columns = pd.to_datetime(future_series.columns )
    return future_series, contract_df
