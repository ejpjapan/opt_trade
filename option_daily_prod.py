#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 08:33:48 2018

@author: macbook2
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from ib_insync import IB, Index, Option
from option_utilities import third_fridays, USSimpleYieldCurve, get_theoretical_strike
from spx_data_update import DividendYieldHistory
from ib_insync.util import isNan
import time

class OptionAsset(ABC):
    def __init__(self, mkt_symbol, vol_symbol, exchange_dict, port=4001, client_id=20):
        """Abstract class for option asset container"""
        exchange_mkt = exchange_dict['exchange_mkt']
        exchange_vol = exchange_dict['exchange_vol']
        exchange_opt = exchange_dict['exchange_opt']
        underlying_index = Index(mkt_symbol, exchange_mkt)
        ib = IB()
        ib.connect('127.0.0.1', port, client_id)
        self.underlying_qc = self.__get_underlying_qc(underlying_index, ib)
        self.sigma_qc = self.get_sigma_qc(vol_symbol, ib, exchange_vol)
        self.chain = self.__get_option_chain(underlying_index, ib, exchange_opt)
        ib.disconnect()
    """"" Abstract option asset container - Each underlying instrument is an instance of the OptionAsset class 
    and each instance is the only argument for the option_market Class. """

    @staticmethod
    def __get_underlying_qc(underlying_index, ib):
        """Retrieve IB qualifying contracts for an index"""
        index_qc = ib.qualifyContracts(underlying_index)
        assert(len(index_qc) == 1)
        return index_qc[0]

    @staticmethod
    def __get_option_chain(underlying_index, ib, exchange):
        """Retrieve IB qualifying options contracts for an index"""
        all_chains = ib.reqSecDefOptParams(underlying_index.symbol, '',
                                           underlying_index.secType,
                                           underlying_index.conId)
        # TO DO Consider moving this to abstract function as different markets will have different
        # conditions around which options to select
        chain = next(c for c in all_chains if c.tradingClass == underlying_index.symbol and c.exchange == exchange)
        return chain

    def get_expirations(self):
        """Retrieve Dataframe of option expirations (last trading day) for option chain in object"""
        expirations = pd.DataFrame(list(self.chain.expirations),
                                   index=pd.DatetimeIndex(self.chain.expirations),
                                   columns=['expirations'])
        timedelta = expirations.index - pd.datetime.today()
        expirations['year_fraction'] = timedelta.days / 365
        # remove negative when latest expiry is today
        expirations = expirations[expirations['year_fraction'] > 0]
        return expirations.sort_index()

    @abstractmethod
    def get_sigma_qc(self, vol_symbol, ib, exchange):
        """Abstract method"""
        # should return empty string if no pre-calculated vol index exists
        pass

    @staticmethod
    @abstractmethod
    def get_dividend_yield():
        """Abstract method - Gets latest dividend yield"""
        # should return empty string if no pre-calculated vol index exists
        pass


class SpxOptionAsset(OptionAsset):
    def __init__(self):
        """"" Asset container for SPX  - S&P 500 Index. """
        mkt_symbol = 'SPX'
        vol_symbol = 'VIX'
        exchange_dict = {'exchange_mkt': 'CBOE', 'exchange_vol': 'CBOE', 'exchange_opt': 'CBOE'}
        super().__init__(mkt_symbol, vol_symbol, exchange_dict)

    def get_sigma_qc(self, vol_symbol, ib, exchange):
        """Returns implied Volatility for market"""
        sigma_index = Index(vol_symbol, exchange)
        sigma_qc = ib.qualifyContracts(sigma_index)
        assert(len(sigma_qc) == 1)
        return sigma_qc[0]

    @staticmethod
    def get_dividend_yield():
        """Gets latest dividend yield"""
        # TO DO: Add check on date of latest dividend yield
        dividend_yield_history = DividendYieldHistory()
        dividend_yield = dividend_yield_history.dy_monthly[dividend_yield_history.dy_monthly.columns[0]][-1] / 100
        return dividend_yield


class RSL2OptionAsset(OptionAsset):
    
    def __init__(self):
        mkt_symbol = 'RUT'
        vol_symbol = 'RVX'
        exchange_dict = {'exchange_mkt': 'RUSSELL', 'exchange_vol': 'CBOE', 'exchange_opt': 'CBOE'}
        super().__init__(mkt_symbol, vol_symbol, exchange_dict)

    def get_sigma_qc(self, vol_symbol, ib, exchange):
        """Returns implied Volatility for market"""
        sigma_index = Index(vol_symbol, exchange)
        sigma_qc = ib.qualifyContracts(sigma_index)
        assert(len(sigma_qc) == 1)
        return sigma_qc[0]

    @staticmethod
    def get_dividend_yield():
        """Gets latest dividend yield"""
        # TO DO: Add check on date of latest dividend yield
        # TO DO: Change to RSL2 dividend yield
        # dividend_yield_history = DividendYieldHistory()
        # dividend_yield = dividend_yield_history.dy_monthly[-1] / 100
        print('Warning: RSL2 Using Fixed Dividend yield')
        dividend_yield = 0.0134

        return dividend_yield

#
# class _emfOptionAsset(OptionAsset):
#    def __init__(self, mkt_symbol='MXEF', vol_symbol='VXEEM', exchange=('CBOE', 'CBOE'), \
#                  currency='USD', multiplier='100', sec_type='IND'):
#       super().__init__(mkt_symbol, vol_symbol, exchange, \
#                  currency, multiplier, sec_type)
#       self.listing_spread = 10
#
#    @staticmethod
#    def get_option_implied_dividend_yld():
#        """Returns latest dividend yield for market"""
#        url = 'http://www.wsj.com/mdc/public/page/2_3021-peyield.html'
#        # Package the request, send the request and catch the response: r
#        raw_html_tbl = pd.read_html(url)
#        dy_df = raw_html_tbl[2]
#        latest_dividend_yield = float(dy_df.iloc[2, 4]) /100
#        return latest_dividend_yield


class TradeChoice:

    def __init__(self, tickers, mkt_prices, account_value, z_score, yield_curve, trade_date):
        self.tickers = tickers
        self.spot = mkt_prices[0]
        self.sigma = mkt_prices[1]
        self.account_value = account_value
        self.z_score = z_score
        last_trade_dates = [item.contract.lastTradeDateOrContractMonth for item in self.tickers]
        unique_last_trade_dates = pd.to_datetime(list(dict.fromkeys(last_trade_dates)))
        self.expirations = unique_last_trade_dates + pd.tseries.offsets.BDay(1)
        self.yield_curve = yield_curve
        self.trade_date = trade_date

    def strike_grid(self):
        strikes = [item.contract.strike for item in self.tickers]
        strike_array = np.array(strikes).astype(int).reshape(len(self.expirations),
                                                             len(strikes) // len(self.expirations))
        df_out = pd.DataFrame(strike_array, index=self.expirations, columns=self.z_score)
        df_out = self._format_index(df_out)
        return df_out

    def premium_grid(self):
        # Debug
        premium_mid = [item.marketPrice() for item in self.tickers]
        premium_mid = np.round(premium_mid, 2)
        premium_mid = premium_mid.reshape(len(self.expirations),
                                          len(premium_mid) // len(self.expirations))
        df_out = pd.DataFrame(premium_mid, index=self.expirations, columns=self.z_score)
        df_out = self._format_index(df_out)
        return df_out

    def prices_grid(self):
        bid, ask = zip(*[(item.bid, item.ask) for item in self.tickers])
        list_val = [np.array(item).reshape((len(self.expirations),
                                            len(item) // len(self.expirations))) for item in [bid, ask]]
        df_lst = [pd.DataFrame(item, index=self.expirations, columns=self.z_score) for item in list_val]
        df_out = df_lst[0].astype(str) + '/' + df_lst[1].astype(str)
        df_out = self._format_index(df_out)
        return df_out

    def pct_otm_grid(self, last_price):
        df_out = self.strike_grid() / last_price - 1
        return df_out

    def option_lots(self, leverage, capital_at_risk):
        risk_free = self.yield_curve.get_zero4_date(self.expirations) / 100
        option_life = np.array([timeDelta.days / 365 for timeDelta in
                                [expiryDate - self.trade_date for expiryDate in self.expirations]])
        strike_discount = np.exp(- risk_free.mul(option_life))
        strike_discount = strike_discount[strike_discount.columns[0]] # convert to series
        notional_capital = self.strike_grid().mul(strike_discount, axis=0) - self.premium_grid()
        contract_lots = [round(capital_at_risk / (notional_capital.copy() / num_leverage * 100), 0)
                         for num_leverage in leverage]
        for counter, df in enumerate(contract_lots):
            df.index.name = 'Lev %i' % leverage[counter]
        contract_lots = [df.apply(pd.to_numeric, downcast='integer') for df in contract_lots]
        return contract_lots

    def margin(self, last_price):
        # 100% of premium + 20% spot price - (spot-strike)
        otm_margin = last_price - self.strike_grid()
        otm_margin[otm_margin < 0] = 0
        single_margin_a = (self.premium_grid() + 0.2 * last_price) - (last_price - self.strike_grid())
        # 100% of premium + 10% * strike
        single_margin_b = self.premium_grid() + 0.1 * self.strike_grid()
        margin = pd.concat([single_margin_a, single_margin_b]).max(level=0)
        margin = margin * int(self.tickers[0].contract.multiplier)
        return margin

    @staticmethod
    def _format_index(df_in):
        df_out = df_in.set_index(df_in.index.strftime('%Y.%m.%d'))
        return df_out

class OptionMarket:
    """IB Interface class that fetches data from IB to pass to trade choice object

        Args:
        param1 (OptionAsset): Option asset that contains description of underlying asset.
        """

    def __init__(self, opt_asset: OptionAsset):
        self.option_asset = opt_asset
        self.trade_date = pd.DatetimeIndex([pd.datetime.today()])
        self.zero_curve = USSimpleYieldCurve()
        self.dividend_yield = self.option_asset.get_dividend_yield()

    # @time_it
    def form_trade_choice(self, z_score, num_expiries, right='P'):
        """Forms option trade choice

            Only public method of OptionMarket class, initiates connection to IB server,
            retrieves account value, prices for underlying instrument, the implied volatility index and
            the relevant option tickers.

            Args:
                z_score (numpy array): Range of Z scores for theoretical option strikes
                num_expiries (int): Number of option expirations
                right (`str`, optional): 'P' or 'C'

            Returns: TradeChoice object

            Raises:
                ."""

        ib = IB()
        ib.connect('127.0.0.1', port=4001, clientId=30)
        liquidation_value = self._get_account_tag(ib, 'NetLiquidationByCurrency')
        # TO DO: this will not work when underlying does not have implied vol index
        # this will happen when we need to calculate an implied vol index
        contracts = [self.option_asset.underlying_qc, self.option_asset.sigma_qc]
        mkt_prices = self._get_market_prices(ib, contracts)

        option_tickers = self._option_tickers(ib, mkt_prices, num_expiries, z_score, right)

        trd_choice = TradeChoice(option_tickers, mkt_prices, liquidation_value, z_score, self.zero_curve,
                                 self.trade_date)

        ib.disconnect()
        return trd_choice

    # @time_it
    def _option_tickers(self, ib, mkt_prices, num_expiries, z_score, right):
        """ Retrieves valid option tickers based on theoretical strikes

        :param ib: Interactive brokers connection
        :param mkt_prices: List of underlying index and vol index prices
        :param num_expiries (int): number of expirations
        :param z_score (numpy array): Range of Z scores for theoretical option strikes
        :param right (str) : Type of option P or C
        :return: Option tickers
        """
        # option_expiry_1 = third_fridays(self.trade_date, num_expiries)

        last_trade_dates_df = self.option_asset.get_expirations()
        # TO DO Expiration is day after last trade date
        # Might have to revisit for PM settled options
        option_expiry = last_trade_dates_df.index[0:num_expiries] + pd.tseries.offsets.BDay(1)
        option_expiry = option_expiry.date
        risk_free = self.zero_curve.get_zero4_date(option_expiry) / 100

        last_price = mkt_prices[0]
        sigma = mkt_prices[1] / 100
        theoretical_strikes = get_theoretical_strike(self.trade_date.date, option_expiry,
                                                     last_price, risk_free.squeeze().values,
                                                     z_score, self.dividend_yield, sigma)

        expiration_date_list = last_trade_dates_df['expirations'].iloc[0:num_expiries].tolist()
        theoretical_strike_list = theoretical_strikes.flatten().tolist()
        expiration_date_list = [item for item in expiration_date_list for i in range(len(z_score))]
        contracts = [self._get_closest_valid_contract(strike, expiration, ib, right) for strike, expiration in
                     zip(theoretical_strike_list, expiration_date_list)]
        contracts_flat = [item for sublist in contracts for item in sublist]

        tickers = ib.reqTickers(*contracts_flat)

        # Alternative to get live tickers
        # for contract in contracts_flat:
        #     ib.reqMktData(contract, '', False, False)
        #
        # tickers = [ib.ticker(contract) for contract in contracts_flat]
        # #debug
        # print('Waiting for tickers')
        # ib.sleep(5)
        # print(tickers)
        # ib.sleep(5)
        # print(tickers)
        # ib.sleep(5)
        # print(tickers)
        # ib.sleep(5)
        # print(tickers)
        return tickers

    @staticmethod
    def _get_account_tag(ib, tag):
        account_tag = [v for v in ib.accountValues() if v.tag == tag and v.currency == 'BASE']
        return account_tag

    @staticmethod
    # @time_it
    def _get_market_prices(ib, contracts):

        # tickers = ib.reqTickers(*contracts)

        # Alternative to get live tickers
        for contract in contracts:
            ib.reqMktData(contract, '', False, False)

        # print('Waiting for tickers')
        ib.sleep(1)
        tickers = [ib.ticker(contract) for contract in contracts]
        # print(tickers)

        mkt_prices = [ticker.last if ticker.marketPrice() == ticker.close else ticker.marketPrice()
                      for ticker in tickers]
        if any([True for item in mkt_prices if isNan(item)]):
            mkt_prices = [ticker.marketPrice() for ticker in tickers]

        return mkt_prices

    def _get_closest_valid_contract(self, theoretical_strike, expiration, ib, right='P'):
        """Return valid contract for expiration closest to theoretical_strike"""
        exchange = self.option_asset.chain.exchange
        symbol = self.option_asset.underlying_qc.symbol
        strikes_sorted = sorted(list(self.option_asset.chain.strikes),
                                key=lambda x: abs(x - theoretical_strike))
        ii = 0
        contract = Option(symbol, expiration, strikes_sorted[ii], right, exchange)
        qualified_contract = ib.qualifyContracts(contract)
        while len(qualified_contract) == 0 or ii > 1000:
            ii = ii + 1
            contract = Option(symbol, expiration, strikes_sorted[ii], right, exchange)
            qualified_contract = ib.qualifyContracts(contract)

        # Assertion to break when infinite loop exits after after ii > 1000
        assert len(qualified_contract) > 0, "No valid contracts found"
        return qualified_contract

    @staticmethod
    def get_closest_valid_twin_contract(qualified_contracts, ib):
        """ Returns call for put (and vice versa) qualified contract
        Will return an error if contract not found"""
        key = lambda x: 'C' if x == 'P' else 'P'
        contracts = [Option(list_elem[0], list_elem[1], list_elem[2], list_elem[3], list_elem[4]) for list_elem \
                     in [[contract.symbol, contract.lastTradeDateOrContractMonth, contract.strike, key(contract.right),
                          contract.exchange] for contract in qualified_contracts]]
        qualified_contract_twins = ib.qualifyContracts(*contracts)

        return qualified_contract_twins

    @staticmethod
    def get_option_implied_dividend_yld(qualified_contracts: list, ib, market_price):
        expiration_str = [contract.lastTradeDateOrContractMonth for contract in qualified_contracts]
        timedelta = pd.DatetimeIndex(expiration_str) - pd.datetime.today()
        year_fraction = timedelta.days / 365

        tickers = ib.reqTickers(*qualified_contracts)
        pv_dividends = [ticker.modelGreeks.pvDividend for ticker in tickers]
        dividend_yield = np.array(pv_dividends) / (market_price * year_fraction)

        return dividend_yield
