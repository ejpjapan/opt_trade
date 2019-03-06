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

# import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
# import nest_asyncio
# from time import time
# import plistlib
# import sys

from spx_data_update import ImpliedVolatilityHistory
# from option_utilities import USZeroYieldCurve, write_feather, read_feather, get_asset, chart_format, PlotConstants


class VixForecast:

    def __init__(self):
        self.vix_full_hist = self.vix_history
        self.yhat = None
        self.model_fit = None
        self.vix = None

    def forecast_vix(self, history=None, steps=300):
        if history is None:
            history = self.vix_full_hist
        model = ARMA(history, order=(2, 2))
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
    def time_series_plot(self):
        self._tsplot(self.vix, lags=50)

    @staticmethod
    def _tsplot(y, lags=None, figsize=(16, 9), style='bmh'):
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
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


pf.plot_rolling_beta