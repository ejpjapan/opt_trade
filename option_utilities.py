"""
Created on Sat Mar 10 14:50:38 2018

@author: macbook2
"""

import calendar
from time import time
from datetime import timedelta, date
from urllib.request import urlretrieve
from pathlib import Path
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from dateutil.relativedelta import relativedelta
import feather
from XmlConverter import XmlConverter


def next_third_friday(dts):
    """ Given a third friday find next third friday"""
    dts += timedelta(weeks=4)
    return dts if dts.day >= 15 else dts + timedelta(weeks=1)


def third_fridays(dts, num_dts):
    """Given a date, calculates num_dts next third fridays"""
    # Find closest friday to 15th of month
    middle_month = date(dts.year, dts.month, 15)
    result = [middle_month + timedelta(days=(calendar.FRIDAY - middle_month.weekday()) % 7)]
    # This month's third friday is today or has passed. Find next.
    if result[0] <= dts:
        result[0] = next_third_friday(result[0])
    for _ in range(num_dts - 1):
        result.append(next_third_friday(result[-1]))
    return result


def get_live_option_expiries(expiry_dates_theo, trade_dates, in_dir):
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
    if np.size(trade_dates) != num_exp:
        trade_dates = np.tile(trade_dates, num_exp)
    if np.isscalar(spot_price):
        spot_price = np.tile(spot_price, (num_exp, np.size(z_score)))
    if np.isscalar(sigma):
        sigma = np.tile(sigma, (1, num_exp))
    if np.isscalar(dividend_yield):
        dividend_yield = np.tile(dividend_yield, (1, num_exp))
    if np.isscalar(risk_free):
        risk_free = np.tile(risk_free, (1, num_exp))

    option_life = np.array([timeDelta.days / 365 for timeDelta in
                            [expiryDate - tradeDate for expiryDate,
                             tradeDate in zip(expiry_dates, trade_dates)]])
    time_discount = np.tile((np.transpose(risk_free / 100) - dividend_yield + (sigma ** 2) / 2) *
                            option_life, (np.size(z_score), 1))
    time_scale = np.tile(sigma * np.sqrt(option_life), (np.size(z_score), 1))
    z_score_tile = np.transpose(np.tile(z_score, (num_exp, 1)))
    theoretical_strike = spot_price * np.exp(time_discount +
                                             np.multiply(time_scale, z_score_tile))
    if listing_spread != '':
        theoretical_strike = np.transpose(np.round(theoretical_strike) -
                                      np.mod(np.round(theoretical_strike), listing_spread))
    return theoretical_strike


def write_feather(dataframe: pd.DataFrame, source: str):
    """ Wrapper function for feather.write_dataframe adds row index as column and saves as feather"""
    dataframe['index'] = dataframe.index
    feather.write_dataframe(dataframe, source)


def read_feather(source: str):
    """ Wrapper function feather.read_dataframe adds row index as column and saves as feather"""
    out_df = feather.read_dataframe(source)
    out_df = out_df.set_index(['index'])
    return out_df


class USSimpleYieldCurve:
    """Simple US Zero coupon yield curve for today up to one year"""
    # Simple Zero yield curve built from TBill discount yields and effective fund rate
    # This is a simplified approximation for a full term structure model
    # Consider improving by building fully specified yield curve model using
    # Quantlib
    def __init__(self):
        end = date.today()
        start = end - timedelta(days=10)
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
    """US Zero coupon 30 year interpolated yield curve"""
    def __init__(self):
        top_level_directory = Path.home() / 'Library'/ 'Mobile Documents' / 'com~apple~CloudDocs'
        self.url = 'http://www.federalreserve.gov/econresdata/researchdata/feds200628.xls'
        self.db_path = top_level_directory / 'localDB' / 'xl'
        self.relative_dates = [relativedelta(days=1), relativedelta(months=3), relativedelta(months=6)] + \
                              [relativedelta(years=x) for x in range(1, 31)]
        fed_zero_feather = Path(self.db_path / 'fedzero.feather')
        if fed_zero_feather.is_file():
            # load old file
            seconds_since_upate = time() - fed_zero_feather.stat().st_mtime
            zero_yields_old = read_feather(str(fed_zero_feather))
            latest_business_date = pd.to_datetime('today') - pd.tseries.offsets.BDay(1)
            if zero_yields_old.index[-1].date() != latest_business_date.date():
                if seconds_since_upate > 86400:
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
            return pd.concat(zeros)

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

        zero_yld_curve = pd.DataFrame(data=np.transpose(zero_yld_curve.values),
                                      index=maturities, columns=[as_of_date])
        # TODO check 2nd order polynomial yield curve interpolation
        zero_yld_curve = zero_yld_curve.resample('D').interpolate(method='polynomial', order=2)
        return zero_yld_curve.loc[maturity_date]

    def get_raw_zeros(self):
        """Update zero coupon yields from FED and FRED"""
        try:
            print('Updating zero coupon yields')
            start_time = time()
            urlretrieve(self.url, self.db_path / 'feds200628.xls')
            converter = XmlConverter(input_path=str(self.db_path) + '/feds200628.xls',
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
            write_feather(zero_yld_matrix, str(self.db_path / 'fedzero.feather'))
            end_time = time()
            print('File updated in ' + str(round(end_time-start_time)) + ' seconds')
        except:
            print('Zero curve update failed - Zero curve no updated')
