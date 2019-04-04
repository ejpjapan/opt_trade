#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:19:37 2018

@author: macbook2
"""
import feather
import pandas as pd
import numpy as np
import pyfolio as pf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
from option_utilities import get_actual_option_expiries, USZeroYieldCurve, get_theoretical_strike, read_feather
from spx_data_update import UpdateSP500Data, get_dates


class OptionSimulation:
    COL_NAMES = ['strike_traded', 'strike_theo', 'days_2_exp', 'zero', 'bid_1545', 'ask_1545']
    GREEK_COL_NAMES = ['delta_1545', 'gamma_1545', 'theta_1545', 'vega_1545', 'rho_1545', 'implied_volatility_1545', 'active_underlying_price_1545']

    def __init__(self, update_simulation_data=False):
        if update_simulation_data:
            updater = UpdateSP500Data()
            self.usZeroYldCurve = updater.usZeroYldCurve
        else:
            self.usZeroYldCurve = USZeroYieldCurve(update_data=False)
        self.feather_directory = UpdateSP500Data.TOP_LEVEL_PATH / 'feather'
        file_names = {'spot': 'sp500_close', 'sigma': 'vix_index', 'dividend_yield': 'sp500_dividend_yld'}
        self.sim_param = self.get_simulation_parameters(UpdateSP500Data.TOP_LEVEL_PATH, file_names)
        self.expiration_actual = None
        # Simulation dates depend depend on availability of zero rates
        last_zero_date = self.usZeroYldCurve.zero_yields.index[-1]
        self.sim_dates_all = self.sim_param.index[self.sim_param.index <= last_zero_date]
        self.option_type = None

    @staticmethod
    def get_trade_dates(sim_dates_all, trade_type='EOM'):
        # Add pre-cooked trade date recipes here
        month_diff = sim_dates_all.month[1:] - sim_dates_all.month[0:-1]
        eom_trade_dates = sim_dates_all[np.append(month_diff.values.astype(bool), False)]
        # mon3_trade_dates = pd.date_range(self.sim_dates_all[0], self.sim_dates_all[-1], freq='WOM-3MON')
        # tue3_trade_dates = pd.date_range(self.sim_dates_all[0], self.sim_dates_all[-1], freq='WOM-3TUE')
        # wed3_trade_dates = pd.date_range(self.sim_dates_all[0], self.sim_dates_all[-1], freq='WOM-3WED')
        # thu3_trade_dates = pd.date_range(self.sim_dates_all[0], self.sim_dates_all[-1], freq='WOM-3THU')
        # fri3_trade_dates = pd.date_range(self.sim_dates_all[0], self.sim_dates_all[-1], freq='WOM-3FRI')
        # mon1_trade_dates = pd.date_range(self.sim_dates_all[0], self.sim_dates_all[-1], freq='WOM-1MON')
        # tue1_trade_dates = pd.date_range(self.sim_dates_all[0], self.sim_dates_all[-1], freq='WOM-1TUE')
        # wed1_trade_dates = pd.date_range(self.sim_dates_all[0], self.sim_dates_all[-1], freq='WOM-1WED')
        # thu1_trade_dates = pd.date_range(self.sim_dates_all[0], self.sim_dates_all[-1], freq='WOM-1THU')
        # fri1_trade_dates = pd.date_range(self.sim_dates_all[0], self.sim_dates_all[-1], freq='WOM-1FRI')
        if isinstance(trade_type, str):
            def select_trade_date_type(x):
                return {
                    'EOM': eom_trade_dates,
                    # '3MON': mon3_trade_dates,
                    # '3TUE': tue3_trade_dates,
                    # '3WED': wed3_trade_dates,
                    # '3THU': thu3_trade_dates,
                    # '3FRI': fri3_trade_dates,
                    # '1MON': mon1_trade_dates,
                    # '1TUE': tue1_trade_dates,
                    # '1WED': wed1_trade_dates,
                    # '1THU': thu1_trade_dates,
                    # '1FRI': fri1_trade_dates
                }.get(x, 9)
            trade_dates = select_trade_date_type(trade_type)
            trade_dates = pd.DatetimeIndex(trade_dates.date)
        elif isinstance(trade_type, tuple):
            assert len(trade_type) == 2
            assert trade_type[0] < trade_type[-1]
            trade_dates = sim_dates_all[trade_type[0]::trade_type[-1]]

        assert any(trade_dates.intersection(sim_dates_all) == trade_dates), \
            'Trade dates are not a subset of simulation dates'
        return trade_dates

    def _get_expiration_dates(self, option_duration_months, trade_dates):
        '''Create expiration dates based on trade dates and number of expiry months'''
        # TODO: Generalize for option_duration_days
        expiration_theoretical = trade_dates + pd.Timedelta(option_duration_months, unit='M')
        expiration_theoretical = pd.DatetimeIndex(expiration_theoretical.date)
        expiration_actual, available_expiries = get_actual_option_expiries(expiration_theoretical,
                                                                           trade_dates,
                                                                           str(self.feather_directory) +
                                                                           '/UnderlyingOptionsEODCalcs_')
        return expiration_actual

    def trade_sim(self, zscore, option_duration_months, option_type='P',
                  trade_day_type='EOM'):
        self.option_type = option_type

        '''Run option simulation'''
        print('Running Simulation - trade_day_type:' + str(trade_day_type) + ' | Z-score ' + str(zscore) +
              ' | Duration ' + str(option_duration_months) + ' | Option Type:{}'.format(option_type))

        trade_dates = self.get_trade_dates(self.sim_dates_all, trade_day_type)

        trade_model_inputs = self.sim_param.loc[trade_dates]

        self.expiration_actual = self._get_expiration_dates(option_duration_months, trade_dates)

        zero_yields = self.usZeroYldCurve.get_zero_4dates(as_of_dates=trade_dates,
                                                          maturity_dates=self.expiration_actual,
                                                          date_adjust=True)

        zero_yields = pd.Series(data=zero_yields, index=trade_dates, name='zeros')
        zero_yields = pd.concat([zero_yields,
                                 pd.Series(data=self.expiration_actual, index=zero_yields.index,
                                           name='expiration_date')], axis=1)

        trade_model_inputs[zero_yields.columns] = zero_yields
        spot_price = trade_model_inputs.loc[:, 'sp500_close'].values
        dividend_yield = trade_model_inputs.loc[:, 'Yield Value'].values / 100
        sigma = trade_model_inputs.loc[:, 'vix_index'].values / 100
        risk_free = trade_model_inputs.loc[:, 'zeros'].values / 100
        option_strikes_theoretical = get_theoretical_strike(trade_dates,
                                                            self.expiration_actual,
                                                            spot_price, risk_free, [zscore],
                                                            dividend_yield, sigma)

        trade_model_inputs['strike_theoretical'] = option_strikes_theoretical

        sim_dates_live = pd.date_range(trade_dates[0], self.sim_dates_all[-1], freq='B')
        sim_dates_live = sim_dates_live.intersection(self.sim_dates_all)

        # Simulation date cannot go beyond last expiry
        if sim_dates_live[-1] >= self.expiration_actual[-1]:
            last_sim_date_idx = sim_dates_live.get_loc(self.expiration_actual[-1])
            sim_dates_live = sim_dates_live[:last_sim_date_idx]

        dtf_trades = self.simulation_loop(option_type, sim_dates_live, trade_dates, trade_model_inputs,
                                          self.usZeroYldCurve,
                                          self.feather_directory)

        sim_output = SimulationParameters(dtf_trades, zscore, sim_dates_live, option_type, str(trade_day_type))
        return sim_output

    @staticmethod
    def simulation_loop(option_type, sim_dates_live, trade_dates, trade_model_inputs, zero_curve,
                        feather_input=None):
        dtf_trades = []
        for i, trade_dt in enumerate(trade_dates):
            # Get date slice between two trading dates
            start_idx = sim_dates_live.get_loc(trade_dates[i])

            if trade_dt == trade_dates[-1]:
                # last date slice is to end of simulation period
                date_slice = sim_dates_live[start_idx:]
            else:
                end_idx = sim_dates_live.get_loc(trade_dates[i + 1]) + 1
                date_slice = sim_dates_live[start_idx:end_idx]
            # Create empty data frame
            df_out = pd.DataFrame(np.nan, index=date_slice, columns=OptionSimulation.COL_NAMES
                                  + OptionSimulation.GREEK_COL_NAMES)
            # loop through each day within a date_slice
            for dts in date_slice:
                try:
                    dtf = feather_input[feather_input['quote_date'] == dts]
                except TypeError:
                    dtf = feather.read_dataframe(str(feather_input) +
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
                zero_rate = zero_curve.get_zero_4dates(as_of_dates=dts,
                                                       maturity_dates=expiry_date,
                                                       date_adjust=True) / 100
                df_out.loc[dts, 'zero'] = zero_rate
                df_out.loc[dts, 'strike_traded'] = strike_traded
                df_out.loc[dts, 'days_2_exp'] = days2exp.days
                df_out.loc[dts, 'strike_theo'] = strike_theo

                df_out.loc[dts, 'bid_1545'] = option_trade_data[option_trade_data['strike'] ==
                                                                strike_traded]['bid_1545'].iloc[0]

                df_out.loc[dts, 'ask_1545'] = option_trade_data[option_trade_data['strike'] ==
                                                                strike_traded]['ask_1545'].iloc[0]

                df_out.loc[dts, OptionSimulation.GREEK_COL_NAMES] = option_trade_data[option_trade_data['strike'] ==
                                                        strike_traded][OptionSimulation.GREEK_COL_NAMES].iloc[0]
            dtf_trades.append(df_out)
        return dtf_trades

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

    @staticmethod
    def get_previous_business_day(super_set: pd.DatetimeIndex, sub_set: pd.DatetimeIndex):
        diff = sub_set.difference(super_set)
        while len(diff) > 0:
            new_dates = diff - pd.tseries.offsets.BDay(1)
            sub_set = new_dates.union(sub_set.intersection(super_set))
            diff = sub_set.difference(super_set)
        return sub_set


class SimulationParameters:
    def __init__(self, dtf_trades, zscore, sim_dates_live, option_type: str, trade_day_type: str):
        self.dtf_trades = dtf_trades
        self.zscore = zscore
        self.sim_dates_live = sim_dates_live
        self.option_type = option_type
        self.trade_day_type = trade_day_type


class OptionTrades:
    def __init__(self, sim_output: SimulationParameters, leverage: float):
        self.simulation_parameters = sim_output
        if np.isscalar(leverage):
            self.leverage = pd.Series(leverage, self.simulation_parameters.sim_dates_live)
        else:
            self.leverage = leverage
        self.all_returns = self.sell_option()

    def sell_option(self, trade_mid=True):
        dtf_trades = self.simulation_parameters.dtf_trades
        for i, item in enumerate(dtf_trades):
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
        dtf_trades[i] = item

        return_list_geometric = []
        return_list_arithmetic = []
        for item in dtf_trades:
            return_list_geometric.append(item['return_geometric'].dropna())
            return_list_arithmetic.append(item['return_arithmetic'].dropna())

        returns_geometric = pd.concat(return_list_geometric)
        returns_arithmetic = pd.concat(return_list_arithmetic)
        return returns_geometric, returns_arithmetic

    @property
    def greeks(self):
        """Get trade simulation greeks"""
        greeks_list = []
        for item in self.simulation_parameters.dtf_trades:
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

    @property
    def strikes(self):
        """Get trade simulation strikes"""
        strike_list = []
        for item in self.simulation_parameters.dtf_trades:
            strike_list.append(item['strike_traded'].iloc[:-1].astype(np.float64))
        return pd.concat(strike_list)

    @property
    def spot(self):
        """Get trade simulation strikes"""
        spot_list = []
        for item in self.simulation_parameters.dtf_trades:
            spot_list.append(item['strike_traded'].iloc[:-1].astype(np.float64))
        return pd.concat(spot_list)

    @property
    def days_2_expiry(self):
        """Get trade simulation days 2 expiry"""
        days_list = []
        for item in self.simulation_parameters.dtf_trades:
            # Greeks are 1 to n-1
            days_list.append(item['days_2_exp'].iloc[:-1].astype(np.float64))
        return pd.concat(days_list)

    @property
    def returns(self):
        """Return daily arithmetic returns"""
        returns_out = self.all_returns[-1].rename(self.strategy_name)
        return returns_out

    @property
    def return_index(self):
        """Return daily arithmetic returns"""
        index_out = pf.timeseries.cum_returns(self.returns, 100)
        index_out[self.simulation_parameters.sim_dates_live[0]] = 100
        index_out = index_out.reindex(index_out.index.sort_values())
        return index_out

    @property
    def strategy_name(self):
        strategy_name = '{}{}{}L{}'.format(self.simulation_parameters.trade_day_type,
                                           self.simulation_parameters.option_type,
                                           self.simulation_parameters.zscore,
                                           self.leverage.mean())
        return strategy_name

    @property
    def trade_dates(self):
        simulation_trade_dates = [item.index[0] for item in self.simulation_parameters.dtf_trades]
        return pd.DatetimeIndex(simulation_trade_dates)

    @property
    def performance_summary(self):
        """Get simulation performance"""
        # convert returns to series for pyfolio function
        performance = pf.timeseries.perf_stats(self.returns)
        perf_index = list(performance.index)
        performance['StartDate'], performance['EndDate'] = list(self.simulation_parameters.sim_dates_live[[0, -1]]
                                                                .strftime('%b %d, %Y'))
        performance['Leverage'], performance['ZScore'], performance['Avg_Days'] = [self.leverage.mean(),
                                                                                   self.simulation_parameters.zscore,
                                                                                   self.days_2_expiry.mean()]
        performance = performance.reindex(['StartDate', 'EndDate', 'Leverage', 'ZScore', 'Avg_Days'] + perf_index)
        performance = performance.append(self.greeks.mean())
        performance = performance.rename(self.strategy_name)
        performance = performance.to_frame()
        performance = performance.drop(['active_underlying_price_1545'], axis=0)

        return performance


def plot_performance_quad(returns, fig_path=None, fig_name='heat_map_quad', font_size=20):

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(returns.name, fontsize=16)
    gs = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.3)
    ax_heatmap = plt.subplot(gs[0, 0])
    ax_monthly = plt.subplot(gs[0, 1])
    ax_box_plot = plt.subplot(gs[1, 0])
    ax_yearly = plt.subplot(gs[1, 1])

    #   Chart 1: Heatmap
    pf.plotting.plot_monthly_returns_heatmap(returns, ax=ax_heatmap)
    ax_heatmap.set_xticklabels(np.arange(0.5, 12.5, step=1))
    ax_heatmap.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                               rotation=45)
    ax_heatmap.set_xlabel('')
    ax_heatmap.set_ylabel('')
    # ax_heatmap.set_label(rotation=90)

    #   Chart 2: Monthly return distribution
    pf.plotting.plot_monthly_returns_dist(returns, ax=ax_monthly)
    ax_monthly.xaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))
    ax_monthly.set_xlabel('')
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
    plt.xticks(rotation=45)
    ax_yearly.set_xlabel('')
    ax_yearly.set_ylabel('')
    for ax in [ax_box_plot, ax_heatmap, ax_monthly, ax_yearly]:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(font_size)

    for items in (ax_yearly.get_yticklabels() + ax_heatmap.get_yticklabels()):
        items.set_fontsize(font_size - 5)
    if fig_path is not None:
        if Path.is_dir(fig_path):
            plt.savefig(fig_path / fig_name, dpi=600, bbox_inches='tight', transparent=True)
        return fig


class OptionWeeklySimulation:
    COL_NAMES = OptionSimulation.COL_NAMES
    GREEK_COL_NAMES = OptionSimulation.GREEK_COL_NAMES

    def __init__(self, update_simulation_data=False):
        if update_simulation_data:
            updater = UpdateSP500Data()
            self.usZeroYldCurve = updater.usZeroYldCurve
        else:
            self.usZeroYldCurve = USZeroYieldCurve(update_data=False)
        self.csv_directory = UpdateSP500Data.TOP_LEVEL_PATH / 'csv'
        file_names = {'spot': 'sp500_close', 'sigma': 'vix_index', 'dividend_yield': 'sp500_dividend_yld'}
        self.sim_param = OptionSimulation.get_simulation_parameters(UpdateSP500Data.TOP_LEVEL_PATH, file_names)
        self.expiration_actual = None
        # self.raw_df = feather.read_dataframe(UpdateSP500Data.TOP_LEVEL_PATH / 'raw_df.feather')
        self.raw_df = csv_2_feather(UpdateSP500Data.TOP_LEVEL_PATH / 'csv')
        # Simulation dates depend depend on availability of zero rates
        last_zero_date = self.usZeroYldCurve.zero_yields.index[-1]
        sim_dates_all = pd.DatetimeIndex(self.raw_df['quote_date'].unique())
        sim_dates_all = sim_dates_all[sim_dates_all <= last_zero_date]
        self.sim_dates_all = sim_dates_all.sort_values()
        self.zscore = None
        # self.option_data = self.raw_df()

    def trade_sim(self, zscore, option_duration, option_type='P'):
        raw_df = self.raw_df
        raw_df.loc[:, 'option_type'] = raw_df['option_type'].apply(str.upper)
        raw_df = raw_df[raw_df['option_type'] == option_type]
        '''Run option simulation'''
        print('Running Simulation - Weekly Options - | Z-score ' +
              str(zscore) + ' | Duration ' + str(option_duration.days) + ' Days | Option Type:{}'.format(option_type))
        self.zscore = zscore
        # trade_dates = OptionSimulation.get_trade_dates(self.sim_dates_all, trade_type=trade_day_type)
        # trade_model_inputs = self.sim_param.loc[trade_dates]
        # self.expiration_actual = self._get_expiration_dates(option_duration, trade_dates, raw_df)

        return raw_df

    # @staticmethod
    # def _get_expiration_dates(option_duration_weeks, trade_dates, raw_df):
    #     expiration_theoretical = OptionWeeklySimulation.theoretical_expiration_dates(option_duration_weeks, trade_dates)
    #
    #     # expiration_theoretical = pd.DatetimeIndex(expiration_theoretical)
    #     # expiration_actual, available_expiries = get_actual_option_expiries(expiration_theoretical,
    #     #                                                                    trade_dates,
    #     #                                                                    str(self.feather_directory) +
    #     #                                                                    '/UnderlyingOptionsEODCalcs_')
    #     return expiration_theoretical

    # def actual_expiration_dates(self):
    #     self.raw_df.groupby('quote_date')
    #     all_expiration_dates = pd.DatetimeIndex(dtf['expiration'].unique())
    #     all_expiration_dates = all_expiration_dates.sort_values()
    #     all_available_expiry.append(all_expiration_dates)
    #     expiry_index = all_expiration_dates.get_loc(item, method='ffill')
    #     if trade_dates[i] == trade_dates[-1]:
    #         expiration_date = all_expiration_dates[expiry_index]
    #     else:
    #         while all_expiration_dates[expiry_index] <= trade_dates[i + 1]:
    #             expiry_index = expiry_index + 1
    #         expiration_date = all_expiration_dates[expiry_index]
    #     expiry_dates_actual.append(expiration_date)
    #
    #     return pd.DatetimeIndex(expiry_dates_actual), all_available_expiry



    # @staticmethod
    # def theoretical_expiration_dates(option_duration, trade_dates):
    #     """Return DatetimeIndex of theoretical expiration dates"""
    #     expiration_theoretical = trade_dates + option_duration
    #     # Check that theoretical every expiration except last is after following trade_date
    #     bool_idx = expiration_theoretical[:-1] >= trade_dates[1:]
    #     if any(~bool_idx):
    #         print('Some expiration dates are before following trade date - shifting expirations')
    #         expiration_theoretical_series = expiration_theoretical[:-1].to_series()
    #         trade_dates_series = trade_dates[1:].to_series()
    #         expiration_theoretical_series[~bool_idx] = np.NaN  # Replace old values with nan
    #         expiration_theoretical_series = pd.concat([expiration_theoretical_series.dropna(),
    #                                                    trade_dates_series[~bool_idx]], axis=0)
    #         expiration_theoretical_series = expiration_theoretical_series.sort_values()
    #         expiration_theoretical_list = expiration_theoretical_series.index.tolist()
    #         expiration_theoretical_list.append(expiration_theoretical[-1])  # Add back last expiration date
    #         expiration_theoretical_dti = pd.DatetimeIndex(np.asarray(expiration_theoretical_list))
    #     else:
    #         expiration_theoretical_dti = expiration_theoretical
    #     return expiration_theoretical_dti


def csv_2_feather(csv_directory):

    spxw_feather = UpdateSP500Data.TOP_LEVEL_PATH / 'raw_df.feather'
    history = feather.read_dataframe(spxw_feather)
    last_date = pd.DatetimeIndex(history['quote_date'].unique()).sort_values()[-1]

    csv_dates = get_dates(csv_directory, file_type='.csv')
    csv_dates = csv_dates.to_series()

    csv_dates = csv_dates[csv_dates > last_date]
    csv_dates = csv_dates.index
    try:
        file_list = []
        for item in csv_dates:
            file_list.append('UnderlyingOptionsEODCalcs_' + item.strftime('%Y-%m-%d') + '.csv')
        dataframe_list = []
        greek_cols = ['delta_1545',
                      'rho_1545',
                      'vega_1545',
                      'gamma_1545',
                      'theta_1545']
        # for item in os.listdir(csv_directory):
        for item in file_list:
            if item.endswith('.csv'):
                future_df = pd.read_csv(csv_directory / item)
                if pd.DatetimeIndex(future_df['quote_date'].unique()) > last_date:
                    dataframe_list.append(future_df)

        raw_df = pd.concat(dataframe_list, axis=0, ignore_index=True)
        raw_df = raw_df[['quote_date', 'root', 'expiration', 'strike',
                         'option_type', 'open', 'high', 'low', 'close', 'active_underlying_price_1545',
                         'implied_volatility_1545', 'delta_1545', 'gamma_1545', 'theta_1545',
                         'vega_1545', 'rho_1545', 'bid_1545', 'ask_1545']]
        raw_df = raw_df[raw_df['root'] == 'SPXW']
        raw_df.loc[:, ['quote_date', 'expiration']] = raw_df.loc[:, ['quote_date', 'expiration']].apply(
            pd.to_datetime)
        raw_df.loc[:, greek_cols] = raw_df.loc[:, greek_cols].apply(pd.to_numeric, errors='coerce')
        raw_df = pd.concat([history, raw_df], axis=0)
        raw_df = raw_df.sort_values('quote_date').reset_index(drop=True)
        feather.write_dataframe(raw_df, spxw_feather)
        print('Feather updated')
    except ValueError:
        print('Feather file not updated')
        raw_df = history
    # ['underlying_symbol', 'quote_date', 'root', 'expiration', 'strike',
    #  'option_type', 'open', 'high', 'low', 'close', 'trade_volume',
    #  'bid_size_1545', 'bid_1545', 'ask_size_1545', 'ask_1545',
    #  'underlying_bid_1545', 'underlying_ask_1545',
    #  'implied_underlying_price_1545', 'active_underlying_price_1545',
    #  'implied_volatility_1545', 'delta_1545', 'gamma_1545', 'theta_1545',
    #  'vega_1545', 'rho_1545', 'bid_size_eod', 'bid_eod', 'ask_size_eod',
    #  'ask_eod', 'underlying_bid_eod', 'underlying_ask_eod', 'vwap',
    #  'open_interest', 'delivery_code']

    return raw_df



