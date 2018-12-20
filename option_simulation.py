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
from option_utilities import get_live_option_expiries, USZeroYieldCurve, get_theoretical_strike, read_feather
from spx_data_update import UpdateSP500Data, get_dates
import pyfolio as pf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter


class OptionSimulation:
    COL_NAMES = ['strike_traded', 'strike_theo', 'days_2_exp', 'zero', 'bid_1545', 'ask_1545']
    GREEK_COL_NAMES = ['delta_1545', 'gamma_1545', 'theta_1545', 'vega_1545', 'rho_1545', 'implied_volatility_1545']

    def __init__(self, update_simulation_data=False):
        if update_simulation_data:
            _ = UpdateSP500Data()
        self.feather_directory = UpdateSP500Data.TOP_LEVEL_PATH / 'feather'
        self.usZeroYldCurve = USZeroYieldCurve()
        file_names = {'spot': 'sp500_close', 'sigma': 'vix_index', 'dividend_yield': 'sp500_dividend_yld'}
        self.sim_param = self.get_simulation_parameters(UpdateSP500Data.TOP_LEVEL_PATH, file_names)
        self.trade_dates = None
        # Simulation dates depend depend on availability of zero rates
        last_zero_date = self.usZeroYldCurve.zero_yields.index[-1]
        self.sim_dates_all = self.sim_param.index[self.sim_param.index <= last_zero_date]

    def _get_trade_dates(self, trade_dates=None, trade_type='EOM'):
        """Create trade dates datetime index"""
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

    def _get_expiration_dates(self, option_duration_months):
        '''Create expiration dates based on trade dates and number of expiry months'''
        # TODO: Generalize for option_duration_days
        expiration_theoretical = self.trade_dates + \
                                  pd.Timedelta(option_duration_months, unit='M')
        expiration_theoretical = pd.DatetimeIndex(expiration_theoretical.date)
        expiration_actual, available_expiries = get_live_option_expiries(expiration_theoretical,
                                                                          self.trade_dates,
                                                                          str(self.feather_directory) +
                                                                         '/UnderlyingOptionsEODCalcs_')
        return expiration_actual

    def trade_sim(self, zscore, option_duration_months, option_type='P',
                  trade_dates=None, trade_type='EOM'):
        '''Run option simulation'''
        print('Running Simulation:Zscore ' + str(zscore) +' Duration '
              + str(option_duration_months))

        self.trade_dates = self._get_trade_dates(trade_dates,
                                                 trade_type)

        trade_model_inputs = self.sim_param.loc[self.trade_dates]


        self.expiration_actual = self._get_expiration_dates(option_duration_months)

        zero_yields = self.usZeroYldCurve.get_zero_4dates(as_of_dates=self.trade_dates,
                                                          maturity_dates=self.expiration_actual,
                                                          date_adjust=True)
        zero_yields = zero_yields.rename('zeros')
        zero_yields = pd.concat([zero_yields,
                                 pd.Series(data=self.expiration_actual, index=zero_yields.index,
                                           name='expiration_date')], axis=1)
        trade_model_inputs[zero_yields.columns] = zero_yields
        spot_price = trade_model_inputs.loc[:, 'sp500_close'].values
        dividend_yield = trade_model_inputs.loc[:, 'Yield Value'].values / 100
        sigma = trade_model_inputs.loc[:, 'vix_index'].values / 100
        risk_free = trade_model_inputs.loc[:, 'zeros'].values / 100
        option_strikes_theoretical = get_theoretical_strike(self.trade_dates,
                                                            self.expiration_actual,
                                                            spot_price, risk_free, zscore,
                                                            dividend_yield, sigma)
        # self.trade_model_inputs = trade_model_inputs
        trade_model_inputs['strike_theoretical'] = np.transpose(option_strikes_theoretical)

        sim_dates_live = pd.date_range(self.trade_dates[0], self.sim_dates_all[-1], freq='B')
        sim_dates_live = sim_dates_live.intersection(self.sim_dates_all)

        # Simulation date cannot go beyond last expiry
        if sim_dates_live[-1] >= self.expiration_actual[-1]:
            last_sim_date_idx = sim_dates_live.get_loc(self.expiration_actual[-1])
            sim_dates_live = sim_dates_live[:last_sim_date_idx]

        dtf_trades = []

        for i, trade_dt in enumerate(self.trade_dates):
            # Get date slice between two trading dates
            start_idx = sim_dates_live.get_loc(self.trade_dates[i])

            if trade_dt == self.trade_dates[-1]:
                # last date slice is to end of simulation period
                date_slice = sim_dates_live[start_idx:]
            else:
                end_idx = sim_dates_live.get_loc(self.trade_dates[i+1]) + 1
                date_slice = sim_dates_live[start_idx:end_idx]
            # Create empty data frame
            df_out = pd.DataFrame(np.nan, index=date_slice, columns=self.COL_NAMES
                                  + self.GREEK_COL_NAMES)

            # loop through date_slice
            for dts in date_slice:
                dtf = feather.read_dataframe(str(self.feather_directory) +
                                             '/UnderlyingOptionsEODCalcs_' +
                                             dts.strftime(format='%Y-%m-%d')
                                             + '_' + option_type + '.feather')

                # First trade date find traded strike from available strikes based on
                # theoretical strike
                if dts == date_slice[0]:
                    expiry_date = trade_model_inputs.loc[dts]['expiration_date']
                    strike_theo = trade_model_inputs.loc[dts]['strike_theoretical']
                    option_trade_data = dtf[dtf['expiration'] == expiry_date]
                    available_strikes = option_trade_data['strike']
                    # Look for strike in available strikes
                    idx = (np.abs(available_strikes.values - strike_theo)).argmin()
                    strike_traded = available_strikes.iloc[idx]
                else:
                    option_trade_data = dtf[dtf['expiration'] == expiry_date]

                days2exp = expiry_date - dts
                zero_rate = self.usZeroYldCurve.get_zero_4dates(as_of_dates=dts,
                                                                maturity_dates=expiry_date,
                                                                date_adjust=True) / 100

                df_out.loc[dts, 'zero'] = zero_rate.iloc[0]
                df_out.loc[dts, 'strike_traded'] = strike_traded
                df_out.loc[dts, 'days_2_exp'] = days2exp.days
                df_out.loc[dts, 'strike_theo'] = strike_theo
                df_out.loc[dts, 'bid_1545'] = option_trade_data[option_trade_data['strike'] ==
                                              strike_traded]['bid_1545'].iloc[0]
                df_out.loc[dts, 'ask_1545'] = option_trade_data[option_trade_data['strike'] ==
                                              strike_traded]['ask_1545'].iloc[0]

                df_out.loc[dts, self.GREEK_COL_NAMES] = option_trade_data[option_trade_data['strike'] ==
                                              strike_traded][self.GREEK_COL_NAMES].iloc[0]
            dtf_trades.append(df_out)
        return dtf_trades, zscore, sim_dates_live

    @staticmethod
    def get_simulation_parameters(input_path, file_names):
        """ Returns closing spot, implied vol and dividend yield for instrument
        :param input_path:
        :param file_names:
        :return :
        """
        file_strings = [str(input_path / file_name) for file_name in file_names.values()]
        list_df = [read_feather(file_string) for file_string in file_strings]
        out_df = pd.concat(list_df, axis=1)
        # Forward fill first monthly dividend yield
        # TODO remove dependency on 'Yield Value' column name
        out_df['Yield Value'] = out_df[['Yield Value']].fillna(method='ffill')
        out_df = out_df.dropna(axis=0, how='any')

        # Double check dates from feather files are identical to out_df
        opt_dates = get_dates(input_path / 'feather')
        assert all(opt_dates == out_df.index)

        return out_df


class OptionTrades:
    def __init__(self, dtf_trades, zscore, sim_dates_live, leverage: float):
        self.dtf_trades = dtf_trades
        self.zscore = zscore
        self.sim_dates_live = sim_dates_live

        if np.isscalar(leverage):
            self.leverage = pd.Series(leverage, self.sim_dates_live)
        else:
            self.leverage = leverage
        self.returns = self.sell_option()

    def sell_option(self, trade_mid=True):
        for i, item in enumerate(self.dtf_trades):
            item['discount'] = item['days_2_exp'] / 365 * - item['zero']
            item['discount'] = item['discount'].map(np.exp)
            if trade_mid:
                item['premium_sold'] = pd.concat([item['ask_1545'],
                                                  item['bid_1545']], axis=1).mean(axis=1)
            else:
                # Option sold at bid and then valued @ ask
                item['premium_sold'] = item['ask_1545']
                item.loc[item.index[0], 'premium_sold'] = item.iloc[0]['bid_1545'].astype(float)

            item['asset_capital'] = item['strike_traded'] * item['discount'] - item['premium_sold']
            item['equity_capital'] = item['asset_capital'] / self.leverage
            premium_diff = item[['premium_sold']].diff(axis=0) * -1
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
        greeks_list = []
        for item in self.dtf_trades:
            # Greeks are 1 to n-1
            greeks_list.append(item[OptionSimulation.GREEK_COL_NAMES].iloc[:-1].astype(np.float64))
        # Need to add greeks for last day of simulation
        greeks_list[-1] = item[OptionSimulation.GREEK_COL_NAMES].astype(np.float64)

        greeks = pd.concat(greeks_list)
        # delta, gamma, theta, vega, rho need to be multiplied by -1 * leverage
        greek_col_bool = sum([greeks.columns.str.contains(item)
                              for item in ['delta', 'gamma', 'theta', 'vega', 'rho']]) > 0
        greek_columns = greeks.loc[:, greek_col_bool]
        greek_columns = greek_columns.multiply(-1 * self.leverage, axis=0)
        greeks.loc[:, greek_col_bool] = greek_columns
        return greeks

    def get_strikes(self):
        '''Get trade simulation strikes'''
        strike_list = []
        for item in self.dtf_trades:
            strike_list.append(item['strike_traded'].iloc[:-1].astype(np.float64))
        return pd.concat(strike_list)

    def get_days_2_expiry(self):
        '''Get trade days 2 expiry'''
        days_list = []
        for item in self.dtf_trades:
            # Greeks are 1 to n-1
            days_list.append(item['days_2_exp'].iloc[:-1].astype(np.float64))
        return pd.concat(days_list)

    def performance(self):
        performance = pf.timeseries.perf_stats(self.returns[1])
        performance['Leverage'] = self.leverage.mean()
        performance['ZScore'] = self.zscore

        return performance

    @staticmethod
    def plot_performance_quad(returns, fig_path=None, font_size=20):

        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 2, wspace=0.5, hspace=0.5)
        ax_heatmap = plt.subplot(gs[0, 0])
        ax_monthly = plt.subplot(gs[0, 1])
        ax_box_plot = plt.subplot(gs[1, 0])
        ax_yearly = plt.subplot(gs[1, 1])

        #   Chart 1: Heatmap
        pf.plotting.plot_monthly_returns_heatmap(returns, ax=ax_heatmap)
        ax_heatmap.set_xticklabels(np.arange(0.5, 12.5, step=1))
        ax_heatmap.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        #   Chart 2: Monthly return distribution
        pf.plotting.plot_monthly_returns_dist(returns, ax=ax_monthly)
        ax_monthly.xaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))
        leg1 = ax_monthly.legend(['mean'], framealpha=0.0, prop={'size': font_size})
        for text in leg1.get_texts():
            # text.set_color('white')
            text.set_label('mean')

        #   Chart 3: Return quantiles
        pf.plotting.plot_return_quantiles(returns, ax=ax_box_plot)

        #   Chart 4: Annual returns
        pf.plotting.plot_annual_returns(returns, ax=ax_yearly)
        _ = ax_yearly.legend(['mean'], framealpha=0.0, prop={'size': font_size})
        ax_yearly.xaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))

        for ax in [ax_box_plot, ax_heatmap, ax_monthly, ax_yearly]:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(font_size)

        for items in (ax_yearly.get_yticklabels() + ax_heatmap.get_yticklabels()):
            items.set_fontsize(font_size - 5)
        if fig_path.is_dir():
            plt.savefig(fig_path + 'heat_map', dpi=600, bbox_inches='tight', transparent=True)
        return fig
