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

import seaborn as sns
from option_simulation import OptionSimulation


def spx_bar_history(update_bars=True):
    file_name = 'sp500_5min_bars'
    df_hist = read_feather(UpdateSP500Data.DATA_BASE_PATH / file_name)
    # Download latest
    if update_bars:
        ib = IB()
        ib.connect('127.0.0.1', port=4001, clientId=40)
        contract = Index('SPX', 'CBOE', 'USD')

        # end = datetime.datetime(2006, 12, 6, 9, 30)

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