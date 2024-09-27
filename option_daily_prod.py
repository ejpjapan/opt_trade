#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 08:33:48 2018

@author: macbook2
"""
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from ib_insync import Index, Option, ContFuture, Future
from option_utilities import time_it, USSimpleYieldCurve, get_theoretical_strike
from spx_data_update import DividendYieldHistory, IbWrapper
from ib_insync.util import isNan
import io
import contextlib
import logging



class OptionAsset(ABC):
    def __init__(self, mkt_symbol, vol_symbol, exchange_dict):
        """Abstract class for option asset container"""
        self.settlement_PM = None
        exchange_mkt = exchange_dict['exchange_mkt']
        exchange_vol = exchange_dict['exchange_vol']
        exchange_opt = exchange_dict['exchange_opt']
        self.trading_class = exchange_dict['trading_class']
        underlying_index = Index(mkt_symbol, exchange_mkt)
        ibw = IbWrapper()
        ib = ibw.ib
        self.underlying_qc = self.__get_underlying_qc(underlying_index, ib)
        self.sigma_qc = self.get_sigma_qc(vol_symbol, ib, exchange_vol)
        self.chain = self.get_option_chain(underlying_index, ib, exchange_opt)

        ib.disconnect()
    """"" Abstract option asset container - Each underlying instrument is an instance of the OptionAsset class 
    and each instance is the only argument for the option_market Class. """

    @staticmethod
    def __get_underlying_qc(underlying_index, ib):
        """Retrieve IB qualifying contracts for an index"""
        index_qc = ib.qualifyContracts(underlying_index)
        assert(len(index_qc) == 1)
        return index_qc[0]

    @property
    def get_expirations(self):
        """Retrieve Dataframe of option expirations (last trading day) for option chain in object"""
        expirations = pd.DataFrame(list(self.chain.expirations),
                                   index=pd.DatetimeIndex(self.chain.expirations),
                                   columns=['expirations'])
        timedelta = expirations.index - datetime.datetime.today()
        expirations['year_fraction'] = timedelta.days / 365
        # remove negative when latest expiry is today
        expirations = expirations[expirations['year_fraction'] > 0]
        return expirations.sort_index()

    @abstractmethod
    def get_option_chain(self, underlying_index, ib, exchange):
        """Abstract method"""
        #
    pass

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
    def __init__(self, trading_class='SPX'):
        """"" Asset container for SPX  - S&P 500 Index. """
        mkt_symbol = 'SPX'
        vol_symbol = 'VIX'
        exchange_dict = {'exchange_mkt': 'CBOE', 'exchange_vol': 'CBOE', 'exchange_opt': 'CBOE',
                         'trading_class': trading_class}  # other choice is SPXW
        super().__init__(mkt_symbol, vol_symbol, exchange_dict)
        if trading_class == 'SPXW':
            self.settlement_PM = True
        else:
            self.settlement_PM = False

    # def get_sigma_qc(self, vol_symbol, ib, exchange):
    #     """Returns implied Volatility for market"""
    #     sigma_index = Index(vol_symbol, exchange)
    #     sigma_qc = ib.qualifyContracts(sigma_index)
    #     assert(len(sigma_qc) == 1)
    #     return sigma_qc[0]

    def get_sigma_qc(self, vol_symbol, ib, exchange):
        """Returns implied Volatility for market - Using continous front month future"""

        # Define the VIX continuous future
        vix_cont_future = ContFuture('VIX', exchange='CFE')
        # print(vix_cont_future)

        # Qualify the contract
        qualified_contracts = ib.qualifyContracts(vix_cont_future)
        # print(qualified_contracts)
        if qualified_contracts:
            vix_contract = qualified_contracts[0]
            ticker = ib.reqMktData(vix_contract)
            ib.sleep(1)
            # print(ticker)
            # Get the latest price
            vix_price = ticker.marketPrice() if ticker.marketPrice() else ticker.close
            print(f"VIX Continuous Future Price: {vix_price}")
        else:
            print("Could not qualify contract for VIX continuous future.")

        # Define the base futures contract for VIX
        base_future = Future('VIX', exchange='CFE')

        # Fetch all available futures contracts for VIX
        futures = ib.reqContractDetails(base_future)
        future_contracts = [cd.contract for cd in futures]

        # Get today's date and the target date one month from today
        today = datetime.datetime.now().date()
        target_date = today + timedelta(days=30)

        # Initialize a list to hold the contract details with expiry dates
        contracts_with_expiry = []

        # Gather expiry dates for each contract
        for contract in future_contracts:
            # Parse the expiry date from the contract
            expiry_date = datetime.datetime.strptime(contract.lastTradeDateOrContractMonth, '%Y%m%d').date()
            contracts_with_expiry.append((contract, expiry_date))

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(contracts_with_expiry, columns=['Contract', 'ExpiryDate'])

        # Calculate the absolute difference in days from the target date
        df['DaysToTarget'] = (df['ExpiryDate'] - target_date).abs().dt.days

        # Select the contract with the smallest difference in days to the target date
        closest_contract = df.loc[df['DaysToTarget'].idxmin()]['Contract']

        # Select the contract with the highest open interest
        # most_open_interest_contract = df.loc[df['OpenInterest'].idxmax()]['Contract']
        ticker = ib.reqMktData(closest_contract)
        ib.sleep(1)
        vix_price = ticker.marketPrice() if ticker.marketPrice() else ticker.close
        print(f"VIX One month out: {vix_price}")
        # print(f'VIX most traded contract month:{ticker}')

        return closest_contract
        # return qualified_contracts[0]

    def get_option_chain(self, underlying_index, ib, exchange):
        """Retrieve IB qualifying options contracts for an index"""
        all_chains = ib.reqSecDefOptParams(underlying_index.symbol, '',
                                           underlying_index.secType,
                                           underlying_index.conId)
        # TO DO Consider moving this to abstract function as different markets will have different
        # conditions around which options to select
        chain = next(c for c in all_chains if c.tradingClass == self.trading_class and c.exchange == exchange)
        return chain

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
    def get_option_chain(underlying_index, ib, exchange):
        """Retrieve IB qualifying options contracts for an index"""
        all_chains = ib.reqSecDefOptParams(underlying_index.symbol, '',
                                           underlying_index.secType,
                                           underlying_index.conId)
        # TO DO Consider moving this to abstract function as different markets will have different
        # conditions around which options to select
        chain = next(c for c in all_chains if c.tradingClass == underlying_index.symbol and c.exchange == exchange)
        return chain

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

    def __init__(self, tickers, mkt_prices, account_value, z_score, yield_curve, trade_date, option_expiry):
        self.tickers = tickers
        self.spot = mkt_prices[0]
        self.sigma = mkt_prices[1]
        self.account_value = account_value
        self.z_score = z_score
        # last_trade_dates = [item.contract.lastTradeDateOrContractMonth for item in self.tickers]
        # unique_last_trade_dates = pd.to_datetime(list(dict.fromkeys(last_trade_dates)))
        self.expirations = option_expiry
        self.yield_curve = yield_curve
        self.trade_date = trade_date

    @property
    def strike_grid(self):
        strikes = [item.contract.strike for item in self.tickers]
        strike_array = np.array(strikes).astype(int).reshape(len(self.expirations),
                                                             len(strikes) // len(self.expirations))
        df_out = pd.DataFrame(strike_array, index=self.expirations, columns=self.z_score)
        df_out = self._format_index(df_out)
        return df_out

    @property
    def premium_grid(self):
        premium_mid = [item.marketPrice() for item in self.tickers]
        premium_mid = np.round(premium_mid, 2)
        premium_mid = premium_mid.reshape(len(self.expirations),
                                          len(premium_mid) // len(self.expirations))
        df_out = pd.DataFrame(premium_mid, index=self.expirations, columns=self.z_score)
        df_out = self._format_index(df_out)
        return df_out

    @property
    def prices_grid(self):
        bid, ask = zip(*[(item.bid, item.ask) for item in self.tickers])
        list_val = [np.array(item).reshape((len(self.expirations),
                                            len(item) // len(self.expirations))) for item in [bid, ask]]
        df_lst = [pd.DataFrame(item, index=self.expirations, columns=self.z_score) for item in list_val]
        df_out = df_lst[0].astype(str) + '/' + df_lst[1].astype(str)
        df_out = self._format_index(df_out)
        return df_out

    def pct_otm_grid(self, last_price):
        df_out = self.strike_grid / last_price - 1
        return df_out

    def option_lots(self, leverage, capital_at_risk):
        risk_free = self.yield_curve.get_zero4_date(self.expirations.date) / 100
        option_life = np.array([timeDelta.days / 365 for timeDelta in
                                [expiryDate - self.trade_date for expiryDate in self.expirations]])
        strike_discount = np.exp(- risk_free.mul(option_life))
        strike_discount = strike_discount.squeeze()  # convert to series
        notional_capital = self.strike_grid.mul(strike_discount, axis=0) - self.premium_grid
        contract_lots = [round(capital_at_risk / (notional_capital.copy() / num_leverage * 100), 0)
                         for num_leverage in leverage]
        for counter, df in enumerate(contract_lots):
            df.index.name = 'Lev %i' % leverage[counter]
        contract_lots = [df.apply(pd.to_numeric, downcast='integer') for df in contract_lots]
        return contract_lots

    def margin(self, last_price):
        # 100% of premium + 20% spot price - (spot-strike)
        # otm_margin = last_price - self.strike_grid
        # otm_margin[otm_margin < 0] = 0
        single_margin_a = (self.premium_grid + 0.2 * last_price) - (last_price - self.strike_grid)
        # 100% of premium + 10% * strike
        single_margin_b = self.premium_grid + 0.1 * self.strike_grid
        margin = pd.concat([single_margin_a, single_margin_b]).max(level=0)
        margin = margin * int(self.tickers[0].contract.multiplier)
        return margin

    @staticmethod
    def _format_index(df_in):
        df_out = df_in.set_index(df_in.index.tz_localize(None).normalize())
        return df_out


class OptionMarket:
    """IB Interface class that fetches data from IB to pass to trade choice object

        Args:
        param1 (OptionAsset): Option asset that contains description of underlying asset.
        """

    def __init__(self, opt_asset: OptionAsset):
        self.option_asset = opt_asset
        self.trade_date = pd.DatetimeIndex([datetime.datetime.today()], tz='US/Eastern')
        self.zero_curve = USSimpleYieldCurve()
        self.dividend_yield = self.option_asset.get_dividend_yield()
        self.option_expiry = None

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

        ibw = IbWrapper()
        ib = ibw.ib
        liquidation_value = self._get_account_tag(ib, 'NetLiquidationByCurrency')
        # TO DO: this will not work when underlying does not have implied vol index
        # this will happen when we need to calculate an implied vol index
        contracts = [self.option_asset.underlying_qc, self.option_asset.sigma_qc]
        mkt_prices = self._get_market_prices(ib, contracts)

        option_tickers = self._option_tickers(ib, mkt_prices, num_expiries, z_score, right)

        trd_choice = TradeChoice(option_tickers, mkt_prices, liquidation_value, z_score, self.zero_curve,
                                 self.trade_date, self.option_expiry)

        ib.disconnect()
        return trd_choice

    def _calculate_option_expiry(self, last_trade_dates_df):
        """Calculate the option expiry based on the settlement type."""

        # Assuming SPXW is identified by a trading class or some attribute of the option asset
        if 'SPXW' in self.option_asset.trading_class:
            # Logic for SPXW, typically PM settled options
            option_expiry = last_trade_dates_df.index.normalize() + pd.DateOffset(hours=16)
        elif self.option_asset.settlement_PM:
            # General PM settled options logic
            option_expiry = last_trade_dates_df.index.normalize() + pd.DateOffset(hours=16)
        else:
            # AM settled options logic
            option_expiry = last_trade_dates_df.index + pd.tseries.offsets.BDay(1)
            option_expiry = option_expiry.normalize() + pd.DateOffset(hours=9) + pd.DateOffset(minutes=45)

        return option_expiry.tz_localize(tz='US/Eastern')

    # @time_it
    def _option_tickers(self, ib, mkt_prices, num_expiries, z_score, right):
        """ Retrieves valid option tickers based on theoretical strikes.

        :param ib: Interactive brokers connection
        :param mkt_prices: List of underlying index and vol index prices
        :param num_expiries (int or list): number of expirations
        :param z_score (numpy array): Range of Z scores for theoretical option strikes
        :param right (str) : Type of option P or C
        :return: Option tickers
        """

        # Validate inputs
        if not mkt_prices or len(mkt_prices) < 2:
            raise ValueError("Market prices should contain at least two values: underlying price and volatility.")
        if not isinstance(z_score, np.ndarray):
            raise TypeError("z_score should be a numpy array.")

        # Handle single int or list for num_expiries
        if isinstance(num_expiries, int):
            num_expiries = range(num_expiries)
        last_trade_dates_df = self.option_asset.get_expirations.iloc[num_expiries]

        # Calculate option expiry
        self.option_expiry = self._calculate_option_expiry(last_trade_dates_df)

        # Calculate the risk-free rate
        risk_free = self.zero_curve.get_zero4_date(self.option_expiry.date) / 100

        # Get the last price and sigma (volatility)
        last_price = mkt_prices[0]
        sigma = mkt_prices[1] / 100

        # Calculate theoretical strikes
        theoretical_strikes = get_theoretical_strike(self.trade_date, self.option_expiry,
                                                     last_price, risk_free.squeeze().values,
                                                     z_score, self.dividend_yield, sigma)

        # Prepare the list of expiration dates
        expiration_date_list = last_trade_dates_df['expirations'].tolist()
        expiration_date_list = [item for item in expiration_date_list for _ in range(len(z_score))]

        # Get the closest valid contract for each theoretical strike and expiration date
        contracts = [self._get_closest_valid_contract(strike, expiration, ib, right)
                     for strike, expiration in zip(theoretical_strikes.flatten(), expiration_date_list)]

        # Pass the list of contracts to reqTickers
        try:
            tickers = ib.reqTickers(*contracts)
            logger.info(f"Successfully retrieved tickers: {tickers}")
        except Exception as e:
            logger.error(f"Error retrieving tickers: {e}")
            raise

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

    # def _get_closest_valid_contract(self, theoretical_strike, expiration, ib, right='P'):
    #     """Return valid contract for expiration closest to theoretical_strike"""
    #     exchange = self.option_asset.chain.exchange
    #     symbol = self.option_asset.underlying_qc.symbol
    #     strikes_sorted = sorted(list(self.option_asset.chain.strikes),
    #                             key=lambda x: abs(x - theoretical_strike))
    #     ii = 0
    #     contract = Option(symbol, expiration, strikes_sorted[ii], right, exchange,
    #                       tradingClass=self.option_asset.trading_class)
    #     qualified_contract = validate_contract(ib, contract)
    #
    #     # Loop until a valid contract is found or ii exceeds 1000
    #     while qualified_contract is None and ii < 1000:
    #         ii += 1
    #         contract = Option(symbol, expiration, strikes_sorted[ii], right, exchange,
    #                           tradingClass=self.option_asset.trading_class)
    #         qualified_contract = validate_contract(ib, contract)
    #
    #     # Assertion to break when infinite loop exits after ii > 1000
    #     assert qualified_contract is not None, "No valid contracts found"
    #     return qualified_contract

    def _get_closest_valid_contract(self, theoretical_strike, expiration, ib, right='P'):
        """Return valid contract for expiration closest to theoretical_strike"""
        exchange = self.option_asset.chain.exchange
        symbol = self.option_asset.underlying_qc.symbol
        strikes_sorted = sorted(self.option_asset.chain.strikes, key=lambda x: abs(x - theoretical_strike))

        for ii, strike in enumerate(strikes_sorted):
            contract = Option(symbol, expiration, strike, right, exchange, tradingClass=self.option_asset.trading_class)
            qualified_contract = validate_contract(ib, contract)

            # Log and return the first valid contract found
            if qualified_contract is not None:
                logger.info(f"Found valid contract at index {ii} with strike {strike}")
                return qualified_contract

            # Early exit: if no valid contract is found in a reasonable range
            if ii > 1000:
                logger.warning(f"Exceeded 1000 iterations without finding a valid contract")
                break

        assert qualified_contract is not None, "No valid contracts found"
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

logging.basicConfig(
    filename='option_market.log',  # Log file name
    level=logging.INFO,            # Log level (INFO, DEBUG, etc.)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
)

logger = logging.getLogger(__name__)

def validate_contract(ib, contract):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            qualified_contracts = ib.qualifyContracts(contract)

        if not qualified_contracts:
            logger.info(f"Invalid contract: {contract}")
            return None
        else:
            logger.info(f"Contract is valid: {qualified_contracts[0]}")
            return qualified_contracts[0]

    except Exception as e:
        logger.error(f"An error occurred during contract validation: {e}")
        return None