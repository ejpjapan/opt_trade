"""
Created on Sat Mar 10 14:50:38 2018

@author: macbook2
"""

import calendar
from time import time
import datetime as dt
from pathlib import Path
import numpy as np
import pandas_datareader.data as web
from dateutil.relativedelta import relativedelta
import feather
from XmlConverter import XmlConverter
from urllib.request import urlretrieve
import pandas as pd
import pyfolio as pf
import matplotlib.transforms as bbox
from matplotlib import rcParams
# from spx_data_update import UpdateSP500Data


def time_it(method):

    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        print('Function: {} took {:.5f} sec'.format(method.__name__, te - ts))
        return result

    return timed


def chart_format(ax_list, txt_color):
    grid_ticks_format(ax_list)
    for item in ax_list:
        color_axis(item, txt_color)
        invisible_spines(item)


def color_axis(ax, txt_color):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_color(txt_color)


def invisible_spines(ax):
    """Hide axis spines"""
    for key, value in ax.spines.items():
        ax.spines[key].set_visible(False)


def grid_ticks_format(ax_list):
    """Hide x & y ticks and format grid lines"""
    [ax.grid(color='grey',
             linestyle=':',
             linewidth=1,
             alpha=0.5) for ax in ax_list]

    [ax.tick_params(axis='both',  # changes apply to the both axis 'x', 'y'
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,
                    left=False) for ax in ax_list]


def next_third_friday(dts):
    """ Given a third friday find next third friday"""
    dts += dt.timedelta(weeks=4)
    return dts if dts.day >= 15 else dts + dt.timedelta(weeks=1)


def third_fridays(dts, num_dts):
    """Given a date, calculates num_dts next third fridays"""
    # Find closest friday to 15th of month
    middle_month = dt.date(dts.year, dts.month, 15)
    result = [middle_month + dt.timedelta(days=(calendar.FRIDAY - middle_month.weekday()) % 7)]
    # This month's third friday is today or has passed. Find next.
    if result[0] <= dts:
        result[0] = next_third_friday(result[0])
    for _ in range(num_dts - 1):
        result.append(next_third_friday(result[-1]))
    return result


# @time_it
def get_actual_option_expiries(expiry_dates_theo, trade_dates, in_dir):
    """retrieve available option expiries given theoretical expiries and trade_dates"""
    expiry_dates_actual = []
    all_available_expiry = []
    for i, item in enumerate(expiry_dates_theo):
        dtf = feather.read_dataframe(in_dir + trade_dates[i].strftime(format='%Y-%m-%d') + '_P' + '.feather')
        all_expiration_dates = pd.DatetimeIndex(dtf['expiration'].unique())
        all_expiration_dates = all_expiration_dates.sort_values()
        all_available_expiry.append(all_expiration_dates)
        expiry_index = all_expiration_dates.get_loc(item, method='ffill')
        if trade_dates[i] == trade_dates[-1]:
            expiration_date = all_expiration_dates[expiry_index]
        else:
            while all_expiration_dates[expiry_index] <= trade_dates[i + 1]:
                expiry_index = expiry_index + 1
            expiration_date = all_expiration_dates[expiry_index]
        expiry_dates_actual.append(expiration_date)
    return pd.DatetimeIndex(expiry_dates_actual), all_available_expiry


def get_theoretical_strike(trade_dates, expiry_dates, spot_price, risk_free, z_score, dividend_yield, sigma,
                           listing_spread=''):
    """Returns option strike with constant delta"""
    num_exp = np.size(expiry_dates)
    if len(trade_dates) == 1:  # Daily run use case
        trade_dates = trade_dates.repeat(num_exp)
        sigma = np.tile(sigma, num_exp)
        dividend_yield = np.tile(dividend_yield, num_exp)
        spot_price = np.tile(spot_price, num_exp)

    option_life = np.array([timeDelta.days / 365 for timeDelta in
                            [expiryDate - tradeDate for expiryDate,
                             tradeDate in zip(expiry_dates, trade_dates)]])
    time_discount = np.multiply((risk_free - dividend_yield + (sigma ** 2) / 2), option_life)
    time_scale = np.multiply(sigma, np.sqrt(option_life))
    theoretical_strike = [np.multiply(spot_price, np.exp(time_discount + np.multiply(time_scale, score))) for score in
                          z_score]
    theoretical_strike = np.column_stack(tuple(theoretical_strike))
    if listing_spread != '':
        theoretical_strike = np.transpose(np.round(theoretical_strike) -
                                          np.mod(np.round(theoretical_strike), listing_spread))
    return theoretical_strike


def write_feather(dataframe: pd.DataFrame, path):
    """ Wrapper function for feather.write_dataframe adds row index as column and saves as feather"""
    dataframe['index'] = dataframe.index
    feather.write_dataframe(dataframe, path)


def read_feather(path):
    """ Wrapper function feather.read_dataframe adds date columns from index"""
    out_df = feather.read_dataframe(path)
    out_df = out_df.set_index(['index'])
    return out_df


def perf_stats(returns: pd.Series, **kwargs):
    """ Wrapper function for pf.timeseries.performance"""
    performance = pf.timeseries.perf_stats(returns, **kwargs)
    perf_index = list(performance.index)
    performance['StartDate'], performance['EndDate'] = list(returns.index[[0, -1]]
                                                            .strftime('%b %d, %Y'))
    performance = performance.reindex(['StartDate', 'EndDate'] + perf_index)
    performance = performance.rename(returns.name)
    performance = performance.drop('common_sense_ratio', axis=0)
    return performance


def get_asset(fund_dict, update=True):
    """Wrapper function to return Adjusted close from Yahoo - Use with care as fund dictionary value will over-write
    name"""
    db_directory = Path.home() / 'Library' / 'Mobile Documents' / 'com~apple~CloudDocs' / 'localDB' / 'feather'
    if update:
        all_funds = [web.get_data_yahoo(key, 'DEC-31-70') for key, _ in fund_dict.items()]
        all_funds = [fund['Adj Close'] for fund in all_funds]
        all_funds = [fund.rename(fund_name) for fund, fund_name in zip(all_funds, fund_dict.values())]
        [write_feather(fund.to_frame(), db_directory / (name + '.feather')) for fund, name in zip(all_funds,
                                                                                                  fund_dict.keys())]
    all_funds = [read_feather(db_directory / (key + '.feather')) for key, _ in fund_dict.items()]
    return all_funds


def matlab2datetime(matlab_datenum):
    def matlab_convert_2_datetime(single_date):
        day = dt.datetime.fromordinal(int(single_date))
        dayfrac = dt.timedelta(days=single_date % 1) - dt.timedelta(days=366)
        return day + dayfrac

    try:
        python_dates = [matlab_convert_2_datetime(int(dts)) for dts in matlab_datenum]
    except TypeError:
        print(matlab_datenum, 'is not iterable')
    return pd.DatetimeIndex(python_dates)


class PlotConstants:
    FONT_SIZE = 9
    FIG_PATH = Path.home() / 'Dropbox' / 'outputDev' / 'fig'
    BB = bbox.Bbox([[0.25, 0.25], [7.46, 4.2]])
    FIG_SIZE = (8, 4.5)  # 16/9 Aspect ratio
    COLOR_LIGHT = '#3f5378'
    COLOR_DARK = '#263248'
    COLOR_YELLOW = '#ff9800'

    def __init__(self, font_size=FONT_SIZE, fig_path=FIG_PATH, fig_size=FIG_SIZE,
                 bb=BB, color_light=COLOR_LIGHT, color_dark=COLOR_DARK, color_yellow=COLOR_YELLOW):
        rcParams['font.sans-serif'] = 'Roboto Condensed'
        rcParams['font.family'] = 'sans-serif'

        self.font_size = font_size
        self.fig_path = fig_path

        self.bb = bb
        self.fig_size = fig_size
        self.color_light = color_light
        self.color_dark = color_dark
        self.color_yellow = color_yellow


class USSimpleYieldCurve:
    """Simple US Zero coupon yield curve for today up to one year"""
    # Simple Zero yield curve built from TBill discount yields and effective fund rate
    # This is a simplified approximation for a full term structure model
    # Consider improving by building fully specified yield curve model using
    # Quantlib
    def __init__(self):
        end = dt.date.today()
        start = end - dt.timedelta(days=10)
        zero_rates = web.DataReader(['DFF', 'DTB4WK', 'DTB3', 'DTB6', 'DTB1YR'], 'fred', start, end)
        zero_rates = zero_rates.dropna(axis=0)
        zero_yld_date = zero_rates.index[-1]
        new_index = [zero_yld_date + relativedelta(days=1),
                     zero_yld_date + relativedelta(weeks=4),
                     zero_yld_date + relativedelta(months=3),
                     zero_yld_date + relativedelta(months=6),
                     zero_yld_date + relativedelta(years=1)]
        zero_curve = pd.DataFrame(data=zero_rates.iloc[-1].values, index=new_index, columns=[end])
        self.zero_curve = zero_curve.resample('D').interpolate(method='polynomial', order=2)

    def get_zero4_date(self, input_date):
        """Retrieve zero yield maturity for input_date"""
        return self.zero_curve.loc[input_date]


class USZeroYieldCurve:
    """US Zero coupon overnnight to 30 year interpolated yield curve"""
    ZERO_URL = 'http://www.federalreserve.gov/econresdata/researchdata/feds200628.xls'
    DB_PATH = Path.home() / 'Library'/ 'Mobile Documents' / 'com~apple~CloudDocs' / 'localDB' / 'xl'

    def __init__(self, update_data=True):
        self.relative_dates = [relativedelta(days=1), relativedelta(months=3), relativedelta(months=6)] + \
                              [relativedelta(years=x) for x in range(1, 31)]
        fed_zero_feather = Path(self.DB_PATH / 'fedzero.feather')
        if update_data:
            if fed_zero_feather.is_file():
                # load old file
                seconds_since_update = time() - fed_zero_feather.stat().st_mtime
                zero_yields_old = read_feather(str(fed_zero_feather))
                latest_business_date = pd.to_datetime('today') - pd.tseries.offsets.BDay(1)
                if zero_yields_old.index[-1].date() != latest_business_date.date():
                    # Check file was updated in last 8 hours
                    if seconds_since_update > (3600 * 8):
                        self.get_raw_zeros()
            else:
                self.get_raw_zeros()

        self.zero_yields = read_feather(str(fed_zero_feather))

    def get_zero_4dates(self, as_of_dates, maturity_dates, date_adjust):
        """Retrieve zero yield maturities for maturity dates"""
        if isinstance(as_of_dates, pd.Timestamp):
            return self.__get_zero_4date(as_of_dates, maturity_dates, date_adjust)
        elif isinstance(as_of_dates, pd.DatetimeIndex):
            assert as_of_dates.shape == maturity_dates.shape
            zeros = []
            for each_date, each_maturity in zip(as_of_dates, maturity_dates):
                zeros.append(self.__get_zero_4date(each_date, each_maturity, date_adjust))
            return zeros

    def __get_zero_4date(self, as_of_date, maturity_date, date_adjust):
        """Interpolate yield curve between points"""
        maturities = pd.DatetimeIndex([as_of_date + x for x in self.relative_dates])
        # Bond market sometimes closed means we will have missing dates e.g columbus day
        if date_adjust:
            try:
                zero_yld_curve = self.zero_yields.loc[[as_of_date]]
            except:
                dt_idx = self.zero_yields.index.get_loc(as_of_date, method='pad')
                tmp_zero_dts = self.zero_yields.index[dt_idx]
                zero_yld_curve = self.zero_yields.loc[[tmp_zero_dts]]
        else:
            zero_yld_curve = self.zero_yields.loc[[as_of_date]]

        zero_yld_series = pd.Series(data=zero_yld_curve.values.squeeze(), index=maturities)
        if not(maturity_date in maturities):
            zero_yld_series.loc[pd.to_datetime(maturity_date)] = float('nan')
            zero_yld_series = zero_yld_series.sort_index()
        zero_yld_series = zero_yld_series.interpolate(method='polynomial', order=2)
        return zero_yld_series[maturity_date]
        # zero_yld_curve = pd.DataFrame(data=np.transpose(zero_yld_curve.values),
        #                               index=maturities, columns=[as_of_date])
        # # TODO check 2nd order polynomial yield curve interpolation
        # zero_yld_curve = zero_yld_curve.resample('D').interpolate(method='polynomial', order=2)
        # return zero_yld_curve.loc[maturity_date]

    def get_raw_zeros(self):
        """Update zero coupon yields from FED and FRED"""
        try:
            print('Updating zero coupon yields')
            start_time = time()
            urlretrieve(self.ZERO_URL, self.DB_PATH / 'feds200628.xls')
            converter = XmlConverter(input_path=str(self.DB_PATH) + '/feds200628.xls',
                                     first_header='SVENY01', last_header='TAU2')
            converter.parse()
            gsw_zero = converter.build_dataframe()
            gsw_zero = gsw_zero.iloc[:, 0:30].copy()
            # Reverse dates
            gsw_zero = gsw_zero.reindex(index=gsw_zero.index[::-1])
            start_date = gsw_zero.index[0]
            fred_data = web.DataReader(['DFF', 'DTB3', 'DTB6'], 'fred', start_date)
            zero_yld_matrix = pd.concat([fred_data.dropna(), gsw_zero], axis=1)
            zero_yld_matrix = zero_yld_matrix.fillna(method='ffill')
            write_feather(zero_yld_matrix, str(self.DB_PATH / 'fedzero.feather'))
            end_time = time()
            print('File updated in ' + str(round(end_time-start_time)) + ' seconds')
        except:
            print('Zero curve update failed - Zero curve not updated')

    @property
    def cash_index(self):
        """Daily Cash return index based on monthly investment in a 3-month t-bill"""
        discount_yield = self.zero_yields['DTB3'].resample('BM').ffill()
        face_value = 10000
        tbill_price = face_value - (discount_yield / 100 * 91 * face_value) / 360
        investment_yield = (face_value - tbill_price) / face_value * (365 / 91)
        return_per_day_month = (investment_yield.shift(1) / 12) / investment_yield.shift(1).index.days_in_month
        return_per_day = return_per_day_month.resample('D').bfill()
        cash_idx = pf.timeseries.cum_returns(return_per_day, 100)
        return cash_idx




