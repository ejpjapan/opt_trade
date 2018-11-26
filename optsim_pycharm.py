#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:19:37 2018

@author: macbook2
"""

import os
import re
import pandas as pd
import numpy as np
import feather
from time import time
from pathlib import Path
from option_utilities import get_live_option_expiries, _usZeroYldCurve, get_daily_close, \
                            get_theoretical_strike, _getRawCBOEOptionData, get_sp5_div_yield
import pandas_datareader.data as web


class OptionSimulation:
    COL_NAMES = ['strike_traded', 'strike_theo', 'days_2_exp', 'zero', 'bid_eod', 'ask_eod']
    GREEK_COL_NAMES = ['delta_1545', 'gamma_1545', 'theta_1545', 'vega_1545', 'rho_1545', 'implied_volatility_1545']

    def __init__(self, listing_spread=5):
        daily_file_directory = Path.home() / 'Library' / 'Mobile Documents' / 'com~apple~CloudDocs' / 'localDB' / 'cboeRawVolData' / 'test'
        self.listing_spread = listing_spread
        self.raw_option_container = _getRawCBOEOptionData()
        self.feather_directory = self.raw_option_container.feather_directory
        # Simulation dates depend on availability of feather files
        opt_dates_all = self.get_dates
        if self.update_data_files(opt_dates_all, daily_file_directory):
            opt_dates_all = self.get_dates

        # Simulation dates depend depend on availability of zero rates
        self.usZeroYldCurve = _usZeroYldCurve()
        last_zero_date = self.usZeroYldCurve.zeroyldmatrix.index[-1]
        self.sim_dates_all = opt_dates_all[opt_dates_all <= last_zero_date]
        self.dtf_trades = None

    def update_data_files(self, opt_dates_all, daily_file_directory):
        """ Download zip files from CBOE, unzip to csv, process and turn into feather
        TO DO: Should be in separate simulation data update & fetch class that creates/updates database
         :rtype: Bool"""
        files_updated = False
        latest_business_date = pd.to_datetime('today') - pd.tseries.offsets.BDay(1)
        try:
            if opt_dates_all[-1].date() != latest_business_date.date():
                start_time = time()
                print('Downloading Option data from CBOE')
                self.raw_option_container.get_daily_history(daily_file_directory)
                self.raw_option_container.unzip2csv(daily_file_directory, daily_file_directory)
                self.raw_option_container.csv2feather(daily_file_directory,
                                                      self.raw_option_container.feather_directory)
                end_time = time()
                files_updated = True
                print('Option files updated in: ' + str(round(end_time - start_time)) + ' seconds')
        except:
            print('Option files not updated')
        return files_updated

    @property
    def get_dates(self):
        """ Fetch dates from feather file names
        :rtype: pd.DatetimeIndex
        """
        regex_pattern = r'\d{4}-\d{2}-\d{2}'  # this will fail if month>12 or days>31
        opt_dates_list = []
        for item in os.listdir(self.feather_directory):  # loop through items in dir
            if item.endswith('.feather'):
                date_string = re.search(regex_pattern, item)
                if date_string:
                    opt_dates_list.append(date_string.group())
        opt_dates_list = list(set(opt_dates_list))
        opt_dates_all = pd.DatetimeIndex([pd.to_datetime(date, yearfirst=True, \
                                                         format='%Y-%m-%d') \
                                          for date in opt_dates_list])
        opt_dates_all = opt_dates_all.sort_values()
        return opt_dates_all

    def load_data(self):
        """" Fetch vix, dividend yield, S&P 500 close and create yield curb object """
        # Get vix data
        vix = self.__get_vix()
        #Simulation dates depend on availability of VIX data from fred
        self.sim_dates_all = self.sim_dates_all.intersection(vix.index)
        # get S&P500 close
        daily_close = get_daily_close(self.sim_dates_all, self.feather_directory\
                                       / 'UnderlyingOptionsEODCalcs_')
        # get S&P 500 dividend yield
        dy_monthly = self.__get_div_yield()
        dy = dy_monthly.resample('B').backfill()
#        dy = dy.loc[self.sim_dates_all].fillna(method='pad').fillna(method='backfill')
        dy = dy.reindex(self.sim_dates_all).fillna(method='pad').fillna(method='backfill')
        mkt_data_daily = vix.copy()
        mkt_data_daily['daily_close'] = daily_close
        mkt_data_daily['dy'] = dy.loc[mkt_data_daily.index]
        self.option_model_inputs = mkt_data_daily

    def __get_vix(self):
        ''' Fetch vix from FRED'''
        fred_vix = web.DataReader(['VIXCLS'], 'fred', self.sim_dates_all[0], \
                                   self.sim_dates_all[-1])
        fred_vix = fred_vix.copy().dropna()
        return fred_vix

    def __get_div_yield(self):
        ''' Fetch dividend yield from quandl or scrape from http://www.multpl.com'''
        try:
            dy_monthly = web.DataReader('MULTPL/SP500_DIV_YIELD_MONTH', \
                                    'quandl', self.sim_dates_all[0], self.sim_dates_all[-1])
        except:
            dy_monthly = get_sp5_div_yield()
        return dy_monthly

    def __get_trade_dates(self, trade_dates=None, trade_type='EOM'):
        '''Create trade dates datetime index'''
        if trade_dates is None:
            # Add pre-cooked trade date recipes here
            month_diff = self.sim_dates_all.month[1:] - self.sim_dates_all.month[0:-1]
            eom_trade_dates = self.sim_dates_all[np.append(month_diff.values.astype(bool), False)]
            thu3_trade_dates = pd.date_range(self.sim_dates_all[0], self.sim_dates_all[-1], freq='WOM-3THU')

            # Use dictionnary as switch case
            def select_trade_date_type(x):
                return {
                    'EOM': eom_trade_dates,
                    'THU3': thu3_trade_dates
                    }.get(x, 9)
            trade_dates = select_trade_date_type(trade_type)
            trade_dates = pd.DatetimeIndex(trade_dates.date)

        # Check all trade dates are part of self.sim_dates_all
        missing_dates = []

        for dts in trade_dates:
            if dts not in self.sim_dates_all:
                missing_dates.append(dts)

        assert not missing_dates, 'Trade dates are not a subset of simulation dates'
        return trade_dates

    def __get_expiration_dates(self, option_duration_months):
        '''Create expiration dates based on trade dates and number of expiry months'''
        # TODO: Generalize for option_duration_days
        expiration_theoretical = self.trade_dates + \
                                  pd.Timedelta(option_duration_months, unit='M')
        # NEED to remove hours
        expiration_theoretical = pd.DatetimeIndex(expiration_theoretical.date)
        expiration_actual, available_expiries = get_live_option_expiries(expiration_theoretical,
                                                                          self.trade_dates,
                                                                          str(self.feather_directory) +
                                                                         '/UnderlyingOptionsEODCalcs_')
        return expiration_actual

    def trade_sim(self, zscore, option_duration_months,
                         trade_dates=None, trade_type='EOM'):
        '''Run option simulation'''
        print('Running Simulation:Zscore ' + str(zscore) +' Duration '
              + str(option_duration_months))
        # TODO Check load_data has run
        self.trade_dates = self.__get_trade_dates(trade_dates,
                                                  trade_type)

        try:
            trade_model_inputs = self.option_model_inputs.loc[self.trade_dates]
        except:
            self.load_data()
            trade_model_inputs = self.option_model_inputs.loc[self.trade_dates]

        self.expiration_actual = self.__get_expiration_dates(option_duration_months)

        zero_ylds = self.usZeroYldCurve.getzero4dates(asofdate=self.trade_dates,
                                                      maturitydate=self.expiration_actual,
                                                      dateadjust=True)
        zero_ylds = zero_ylds.rename('zeros')
        zero_ylds = pd.concat([zero_ylds,
                               pd.Series(data=self.expiration_actual, index=zero_ylds.index,
                                         name='expiration_date')], axis=1)
        trade_model_inputs[zero_ylds.columns] = zero_ylds
        spot_price = trade_model_inputs.loc[:, 'daily_close'].values
        dividend_yield = trade_model_inputs.loc[:, 'dy'].values / 100
        sigma = trade_model_inputs.loc[:, 'VIXCLS'].values / 100
        risk_free = trade_model_inputs.loc[:, 'zeros'].values / 100
        option_strikes_theoretical = get_theoretical_strike(self.trade_dates,
                                                            self.expiration_actual,
                                                            spot_price, risk_free, zscore,
                                                            dividend_yield, sigma)
        self.trade_model_inputs = trade_model_inputs
        self.trade_model_inputs['strike_theoretical'] = option_strikes_theoretical

        self.sim_dates_live = pd.date_range(self.trade_dates[0], self.sim_dates_all[-1], freq='B')
        self.sim_dates_live = self.sim_dates_live.intersection(self.sim_dates_all)

        # Simulation date cannot go beyond last expiry
        if self.sim_dates_live[-1] >= self.expiration_actual[-1]:
            last_sim_date_idx = self.sim_dates_live.get_loc(self.expiration_actual[-1])
            self.sim_dates_live = self.sim_dates_live[:last_sim_date_idx]

        dtf_trades = []

        for i, trade_dt in enumerate(self.trade_dates):
            # Get date slice between two trading dates
            start_idx = self.sim_dates_live.get_loc(self.trade_dates[i])

            if trade_dt == self.trade_dates[-1]:
                # last date slice is to end of simulation period
                date_slice = self.sim_dates_live[start_idx:]
            else:
                end_idx = self.sim_dates_live.get_loc(self.trade_dates[i+1]) + 1
                date_slice = self.sim_dates_live[start_idx:end_idx]
            # Create empty data frame
            df_out = pd.DataFrame(np.nan, index=date_slice, columns=self.COL_NAMES
                                  + self.GREEK_COL_NAMES)

            # loop through date_slice
            for dts in date_slice:
                dtf = feather.read_dataframe(str(self.feather_directory) +
                                             '/UnderlyingOptionsEODCalcs_' +
                                             dts.strftime(format='%Y-%m-%d')
                                             + '_P' + '.feather')

                # First trade date find traded strike from available strikes based on
                # theoretical strike
                if dts == date_slice[0]:
                    expiry_date = self.trade_model_inputs.loc[dts]['expiration_date']
                    strike_theo = self.trade_model_inputs.loc[dts]['strike_theoretical']
                    option_trade_data = dtf[dtf['expiration'] == expiry_date]
                    available_strikes = option_trade_data['strike']

                # Look for strike in available strikes
                    while not available_strikes.isin([strike_theo]).any():
                        strike_theo = strike_theo - self.listing_spread

                    strike_traded = strike_theo
                else:
                    option_trade_data = dtf[dtf['expiration'] == expiry_date]

                days2exp = expiry_date - dts
                zero_rate = self.usZeroYldCurve.getzero4dates(asofdate=dts,
                                                          maturitydate=expiry_date,
                                                          dateadjust=True) / 100
#                df_out.loc[dts]['zero'] = zero_rate.iloc[0]
#                df_out.loc[dts]['strike_traded'] = strike_traded
#                df_out.loc[dts]['days_2_exp'] = days2exp.days
#                df_out.loc[dts]['strike_theo'] = strike_theo
#                df_out.loc[dts]['bid_eod'] = option_trade_data[option_trade_data['strike'] ==
#                                              strike_traded]['bid_eod'].iloc[0]
#                df_out.loc[dts]['ask_eod'] = option_trade_data[option_trade_data['strike'] ==
#                                              strike_traded]['ask_eod'].iloc[0]
#                
                df_out.loc[dts, 'zero'] = zero_rate.iloc[0]
                df_out.loc[dts, 'strike_traded'] = strike_traded
                df_out.loc[dts, 'days_2_exp'] = days2exp.days
                df_out.loc[dts, 'strike_theo'] = strike_theo
                df_out.loc[dts, 'bid_eod'] = option_trade_data[option_trade_data['strike'] ==
                                              strike_traded]['bid_eod'].iloc[0]
                df_out.loc[dts, 'ask_eod'] = option_trade_data[option_trade_data['strike'] ==
                                              strike_traded]['ask_eod'].iloc[0]

                df_out.loc[dts, self.GREEK_COL_NAMES] = option_trade_data[option_trade_data['strike'] ==
                                              strike_traded][self.GREEK_COL_NAMES].iloc[0]
            dtf_trades.append(df_out)
        self.dtf_trades = dtf_trades

    def sell_put(self, leverage, trade_mid=True):
        assert not self.dtf_trades is None, 'No trades loaded - run trade_sim first'
        self.leverage = leverage
        for i, item in enumerate(self.dtf_trades):
            item['discount'] = item['days_2_exp'] / 365 * - item['zero']
            item['discount'] = item['discount'].map(np.exp)
            if trade_mid:
                item['premium_sold'] = pd.concat([item['ask_eod'],
                                         item['bid_eod']], axis=1).mean(axis=1)
            else:
                # Option sold at bid and then valued @ ask
                item['premium_sold'] = item['ask_eod']
                item.loc[item.index[0], ('premium_sold')] = item.iloc[0]['bid_eod'].astype(float)
            # Option bought at bid and then valued @ bid
            #item['premium_bought'] = item['bid_eod'].astype(float)
            #item.loc[item.index[0],('premium_bought')] = item.iloc[0]['ask_eod'].astype(float)
            item['asset_capital'] = item['strike_traded'] * item['discount'] - item['premium_sold']
            item['equity_capital'] = item['asset_capital'] / self.leverage
            premium_diff = item[['premium_sold']].diff(axis=0) * -1
            # Continous return
            #nav = premium_diff.fillna(value=0).add(item.loc[:,'equity_capital'], axis=0)
            #nav/nav.shift(1)
            item['return_arithmetic'] = premium_diff.divide(item['equity_capital'].shift(1),
                                                            axis=0).astype(np.float64)
            premium_diff.iloc[0] = item['equity_capital'].iloc[0]
            item['return_geometric'] = np.log(item['return_arithmetic'].astype(np.float64) + 1)
        self.dtf_trades[i] = item

        return_list_geometric = []
        return_list_arithmetic = []
        for item in self.dtf_trades:
            return_list_geometric.append(item['return_geometric'].dropna())
            return_list_arithmetic.append(item['return_arithmetic'].dropna())

        returns_geometric = pd.concat(return_list_geometric)
        returns_arithmetic = pd.concat(return_list_arithmetic)
        return returns_geometric, returns_arithmetic

    def get_greeks(self):
        '''Get trade simlution greeks'''
        assert not self.dtf_trades is None, 'No trades loaded - run trade_sim first'
        greeks_list = []
        for item in self.dtf_trades:
            # Greeks are 1 to n-1
            greeks_list.append(item[self.GREEK_COL_NAMES].iloc[:-1].astype(np.float64))
        return  pd.concat(greeks_list)
    
    def get_strikes(self):
        '''Get trade simlution strikes'''
        assert not self.dtf_trades is None, 'No trades loaded - run trade_sim first'
        strike_list = []
        for item in self.dtf_trades:
            # Greeks are 1 to n-1
            strike_list.append(item['strike_traded'].iloc[:-1].astype(np.float64))
        return  pd.concat(strike_list)    
    
    @property
    def get_days_2_expiry(self):
        '''Get trade days 2 expiry'''
        assert not self.dtf_trades is None, 'No trades loaded - run trade_sim first'
        days_list = []
        for item in self.dtf_trades:
            # Greeks are 1 to n-1
            days_list.append(item['days_2_exp'].iloc[:-1].astype(np.float64))
        return  pd.concat(days_list)
    