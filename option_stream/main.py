from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, DataTable, TableColumn, Div, Slider, NumberFormatter, HTMLTemplateFormatter
from bokeh.layouts import layout, row, column
import pandas as pd
import numpy as np
from ib_insync import Option, Contract, Index, util, ContractDetails, IB, Ticker
from zoneinfo import ZoneInfo  # Available in Python 3.9 and later
from datetime import datetime
from typing import List, Set, Dict, Optional
import time


from stream_utilities import IbWrapper, USSimpleYieldCurve, illiquid_equity

MAX_EXPIRIES = 25
PRICE_UPDATE_MS = 1000
ACCOUNT_UPDATE_MS = 15000


def _best_price(ticker: Optional[Ticker]) -> Optional[float]:
    if ticker is None:
        return None
    price = ticker.marketPrice()
    if pd.isna(price) or price == 0:
        for candidate in (ticker.last, ticker.close):
            if candidate is not None and not pd.isna(candidate):
                price = candidate
                break
    if price is None or pd.isna(price):
        return None
    return float(price)


def _trend_color(previous: Optional[float], current: Optional[float], existing: str = 'black') -> str:
    if previous is None or current is None:
        return existing
    if pd.isna(previous) or pd.isna(current):
        return existing
    if current > previous:
        return 'green'
    if current < previous:
        return 'red'
    return existing


def _format_colored_value(value: Optional[float], color: str, decimals: int) -> str:
    if value is None or pd.isna(value):
        return ''
    return f"<span style='color:{color}'>{value:.{decimals}f}</span>"


def _select_option_expiries(option_params_df: pd.DataFrame, max_expiries: int) -> List[datetime]:
    now = datetime.now(tz=ZoneInfo("America/New_York"))
    expiries: List[datetime] = []
    if 'expirations_timestamps' in option_params_df.columns:
        for exp_list in option_params_df['expirations_timestamps'].dropna().values:
            if isinstance(exp_list, (list, tuple, set)):
                expiries.extend(list(exp_list))
    if not expiries and 'expirations' in option_params_df.columns:
        for exp_list in option_params_df['expirations'].dropna().values:
            if isinstance(exp_list, (list, tuple, set)):
                expiries.extend([datetime.strptime(item, '%Y%m%d').replace(
                    hour=16, minute=0, second=0, tzinfo=ZoneInfo("America/New_York")
                ) for item in exp_list])
    valid = sorted({expiry for expiry in expiries if expiry >= now})
    return valid[:max_expiries]


def convert_to_datestamps(date_lists: List[List[str]]) -> List[List[datetime]]:
    """
    Convert lists of date strings in 'YYYYMMDD' format to lists of datetime objects set to 16:00 EST (America/New_York).

    Args:
        date_lists (List[List[str]]): A list of lists containing date strings in 'YYYYMMDD' format.

    Returns:
        List[List[datetime]]: A list of lists where each date string is converted into a datetime object set to 16:00 EST.

    Example:
        Input: [['20230915', '20231022'], ['20240101']]
        Output: [[datetime(2023, 9, 15, 16, 0, tzinfo=ZoneInfo("America/New_York")),
                  datetime(2023, 10, 22, 16, 0, tzinfo=ZoneInfo("America/New_York"))],
                 [datetime(2024, 1, 1, 16, 0, tzinfo=ZoneInfo("America/New_York"))]]
    """
    # Define the EST time zone using America/New_York (handles EST/EDT)
    est = ZoneInfo("America/New_York")

    # Prepare a list to hold all the converted date lists
    all_datestamps = []

    # Iterate over each list of date strings
    for date_list in date_lists:
        # List to hold converted datetime objects for the current date_list
        datestamps = []

        for date_str in date_list:
            # Parse the date string into a datetime object with default time (00:00)
            date_obj = datetime.strptime(date_str, '%Y%m%d')

            # Set the time to 4:00 PM EST (16:00) and apply the EST timezone
            date_obj = date_obj.replace(hour=16, minute=0, second=0, tzinfo=est)

            # Append the datetime object to the current list
            datestamps.append(date_obj)

        # Append the current list of datetime objects to the final result
        all_datestamps.append(datestamps)

    # Return the list of lists with converted datetime objects
    return all_datestamps


def fetch_option_chain_via_params(ibw, underlying_symbol: str = 'SPX') -> pd.DataFrame:
    """
    Fetches the option chain parameters for a given underlying symbol from Interactive Brokers (IB).

    Args:
        underlying_symbol (str): The ticker symbol of the underlying index for which to fetch the option chain.
                                 Defaults to 'SPX' (S&P 500 index).

    Returns:
        pd.DataFrame: A DataFrame containing the option chain parameters for the specified underlying symbol,
                      filtered to include only SPXW options and the SMART exchange. The expirations are converted
                      to datetime objects in EST time zone.

    Raises:
        ValueError: If no contract details are found for the provided underlying symbol.

    Example:
        Input: 'SPX'
        Output: DataFrame with option chain parameters including expirations converted to EST datetime objects.
    """
    # Define the underlying contract for SPX (Index contract)
    spx_contract = Index(underlying_symbol, 'CBOE', 'USD')

    # Open a connection to IBWrapper only once
   # ibw.ib  # Access the Interactive Brokers (IB) connection

    # Fetch the contract details to get a valid conId (contract ID)
    contract_details: list[ContractDetails] = ibw.ib.reqContractDetails(spx_contract)

    # Raise an error if no valid contract details are found
    if not contract_details:
        raise ValueError(f"Contract details not found for {underlying_symbol}")

    # Extract the conId from the first contract detail (this is required for further queries)
    spx_contract.conId = contract_details[0].contract.conId
    print(f"SPX contract conId: {spx_contract.conId}")

    # Request all option parameters for SPX (using the symbol, security type, and contract ID)
    option_params = ibw.ib.reqSecDefOptParams(spx_contract.symbol, '', spx_contract.secType, spx_contract.conId)

    # Convert the option parameters to a DataFrame for easier manipulation
    option_params_df: pd.DataFrame = util.df(option_params)

    # Filter the DataFrame to include only SPXW options and options listed on the SMART exchange
    filtered_params = option_params_df[
        (option_params_df['tradingClass'].isin(['SPXW'])) & (option_params_df['exchange'] == 'SMART')
        ].copy()

    # Convert expiration dates from strings to datetime objects in EST time zone
    filtered_params['expirations_timestamps'] = convert_to_datestamps(filtered_params.loc[:, 'expirations'].copy())

    # Return the filtered option parameters DataFrame
    return filtered_params


def get_theoretical_strike(
        option_expiry: List[datetime],
        spot_price: List[float],
        risk_free: pd.Series,
        z_score: List[float],
        dividend_yield: float,
        sigma: List[float]
) -> pd.DataFrame:
    """
    Calculate theoretical option strikes with constant delta for given expiries, spot price, risk-free rates,
    volatility, and z-scores.

    Args:
        option_expiry (List[datetime]): List of option expiry dates (in datetime format).
        spot_price (List[float]): A list containing a single float representing the spot price.
        risk_free (pd.Series): Series of risk-free rates (as percentages, e.g., 5 for 5%), with one value for each expiry.
        z_score (List[float]): List of z-scores for which to calculate the theoretical strikes.
        dividend_yield (float): The dividend yield of the underlying security (as a percentage, e.g., 1.3 for 1.3%).
        sigma (List[float]): A list containing a single float representing the implied volatility (as a percentage, e.g., 20 for 20%).

    Returns:
        pd.DataFrame: A DataFrame containing the theoretical strikes for each z-score, with additional columns
                      for option life, time discount, time scale, and strike discount.

    Raises:
        ValueError: If `spot_price` or `sigma` are not lists with exactly one element.

    Example:
        Input:
            option_expiry = [datetime(2024, 12, 31), datetime(2025, 1, 31)]
            spot_price = [4000]
            risk_free = pd.Series([5.0, 5.2])
            z_score = [-1, 0, 1]
            dividend_yield = 1.3
            sigma = [20]
        Output:
            DataFrame with theoretical strikes for each z-score.
    """

    # Ensure sigma and spot_price are lists with exactly one element
    if isinstance(sigma, list) and len(sigma) == 1:
        sigma = sigma[0] / 100  # Convert sigma from percentage to decimal
    else:
        raise ValueError("sigma must be a list with one element.")

    if isinstance(spot_price, list) and len(spot_price) == 1:
        spot_price = spot_price[0]
    else:
        raise ValueError("spot_price must be a list with one element.")

    # Get the current trade date (using EST timezone)
    trade_date = datetime.now(tz=ZoneInfo("America/New_York"))

    # Calculate option life in years for each expiry (time to expiration)
    option_life = [(date - trade_date).total_seconds() / (365 * 24 * 60 * 60) for date in option_expiry]

    # Create a DataFrame to store data, using expiry dates as the index
    df = pd.DataFrame(index=[dt.strftime("%Y%m%d") for dt in option_expiry])
    df['Days to Expiry'] = [(date - trade_date).days for date in option_expiry]
    # Fill the DataFrame with constant values and calculated values
    df['option_life'] = option_life
    df['trade_date'] = trade_date
    df['sigma'] = sigma  # Apply the single sigma value to all rows
    df['dividend_yield'] = dividend_yield
    df['spot_price'] = spot_price  # Apply the single spot price to all rows
    df['risk_free'] = risk_free / 100  # Convert risk-free rates from percentages to decimals (vectorized)

    # Calculate the time discount and time scale
    df['time_discount'] = (df['risk_free'] - df['dividend_yield'] + (df['sigma'] ** 2) / 2) * df['option_life']
    df['time_scale'] = df['sigma'] * np.sqrt(df['option_life'])

    # Calculate the strike discount
    df['strike_discount'] = np.exp(-df['risk_free'].mul(df['option_life']))

    # Vectorized calculation of theoretical strike for each z-score

    df[f'theoretical_strike'] = df['spot_price'] * np.exp(df['time_discount'] + df['time_scale'] * z_score)

    return df


class QualifiedContractsCache:
    def __init__(self):
        # Cache for qualified contracts
        self.cache = {}
        # Cache for unqualified contracts
        self.unqualified_cache = set()

    def get_contract(self, contract_key):
        """
        Retrieves a qualified contract from the cache if it exists.
        """
        return self.cache.get(contract_key)

    def add_contract(self, contract_key, qualified_contract):
        """
        Adds a qualified contract to the cache.
        """
        self.cache[contract_key] = qualified_contract

    def is_unqualified(self, contract_key):
        """
        Checks if the contract is known to be unqualified.
        """
        return contract_key in self.unqualified_cache

    def add_unqualified(self, contract_key):
        """
        Records a contract as unqualified.
        """
        self.unqualified_cache.add(contract_key)



def qualify_all_contracts(
        ib_wrapper: IbWrapper,
        strikes_df: pd.DataFrame,
        available_strikes: List[float],
        cache: QualifiedContractsCache
) -> pd.DataFrame:
    """
    Qualifies option contracts for given strikes and expirations, handling errors and retrying qualification
    with alternative strikes if necessary, while utilizing a cache to avoid re-qualification.
    """
    # Ensure 'expiry_date' is in string format 'YYYYMMDD'
    strikes_df['expiry_date'] = strikes_df['expiry_date'].astype(str)

    # Initialize 'qualified_contracts' column
    strikes_df['qualified_contracts'] = None

    # Initialize a set to keep track of used strikes for each expiry date
    used_strikes_per_expiry: Dict[str, Set[float]] = {expiry: set() for expiry in strikes_df['expiry_date'].unique()}

    # Iterate over each row to process contracts
    for idx, row in strikes_df.iterrows():
        expiry_str = row['expiry_date']
        theoretical_strike = row['theoretical_strike']

        # Start with the closest strike
        closest_strike = min(available_strikes, key=lambda y: abs(theoretical_strike - y))

        # Initialize a list of alternative strikes sorted by proximity
        alternative_strikes = sorted(available_strikes, key=lambda y: abs(theoretical_strike - y))

        # Remove strikes that have already been used for this expiry
        alternative_strikes = [strike for strike in alternative_strikes if
                               strike not in used_strikes_per_expiry[expiry_str]]

        # Attempt to qualify the contract using alternative strikes
        qualified_contract = None
        for strike in alternative_strikes:
            # Create a unique key for the contract
            contract_key = (
                'SPX',
                expiry_str,
                strike,
                'P',  # Assuming Put options
                'SMART',
                'USD',
                'SPXW'
            )

            # Check if the contract is known to be unqualified
            if cache.is_unqualified(contract_key):
                # Skip this contract as we know it cannot be qualified
                used_strikes_per_expiry[expiry_str].add(strike)
                continue

            # Check if the contract is in the cache
            cached_contract = cache.get_contract(contract_key)
            if cached_contract:
                # Use the cached contract
                qualified_contract = cached_contract
                # Mark the strike as used for this expiry date
                used_strikes_per_expiry[expiry_str].add(strike)
                break  # Contract is found and qualified
            else:
                # Create the Option contract
                option = Option(
                    symbol='SPX',
                    lastTradeDateOrContractMonth=expiry_str,
                    strike=strike,
                    right='P',
                    exchange='SMART',
                    currency='USD',
                    tradingClass='SPXW'
                )

                try:
                    # Qualify the contract
                    qualified_contract = ib_wrapper.ib.qualifyContracts(option)[0]
                    # Store in cache
                    cache.add_contract(contract_key, qualified_contract)
                    # Mark the strike as used for this expiry date
                    used_strikes_per_expiry[expiry_str].add(strike)
                    break  # Successfully qualified
                except Exception as e:
                    print(f"Failed to qualify contract {option}: {e}")
                    # Mark the strike as used and add to unqualified cache
                    used_strikes_per_expiry[expiry_str].add(strike)
                    cache.add_unqualified(contract_key)
                    qualified_contract = None
                    continue  # Try the next alternative strike

        if qualified_contract is None:
            print(f"No valid contract found for expiry {expiry_str} and theoretical strike {theoretical_strike}")
            # Handle the case where no valid contract could be qualified
            strikes_df.at[idx, 'qualified_contracts'] = None
            strikes_df.at[idx, 'closest_strike'] = None
        else:
            # Assign the qualified contract and the strike used to the DataFrame
            strikes_df.at[idx, 'qualified_contracts'] = qualified_contract
            strikes_df.at[idx, 'closest_strike'] = qualified_contract.strike

    return strikes_df


def _build_strikes_df(
        ib_wrapper: IbWrapper,
        option_expiries: List[datetime],
        risk_free: pd.Series,
        option_params_df: pd.DataFrame,
        cache: QualifiedContractsCache,
        spot_price: float,
        vix_price: float,
        z_score: float
) -> pd.DataFrame:
    strikes_df = get_theoretical_strike(
        option_expiries, [spot_price], risk_free, [z_score], 0.013, [vix_price]
    )
    strikes_df['expiry_date'] = strikes_df.index
    available_strikes = option_params_df['strikes'].values[0]
    strikes_df = qualify_all_contracts(ib_wrapper, strikes_df, available_strikes, cache)
    strikes_df = strikes_df.reset_index(drop=True)

    for col in ['bid', 'ask', 'last_traded', 'volume', 'market', 'implied_volatility']:
        strikes_df[col] = np.nan
    return strikes_df


def _recompute_derived_fields(df: pd.DataFrame, base_capital: float, leverage: float) -> None:
    df['Mid'] = (df['Bid'] + df['Ask']) / 2
    notional_capital = df['closest_strike'] * df['strike_discount'] - df['Mid']
    with np.errstate(divide='ignore', invalid='ignore'):
        df['Lots'] = np.round(base_capital / (notional_capital / leverage * 100), 0)

    single_margin_a = (df['Mid'] + 0.2 * df['spot_price']) - (df['spot_price'] - df['closest_strike'])
    single_margin_b = df['Mid'] + 0.1 * df['closest_strike']
    margin = pd.concat([single_margin_a, single_margin_b], axis=1).max(axis=1)
    df['Margin'] = margin * 100
    df['Margin'] = df['Margin'] * df['Lots']
    df['Discount'] = df['closest_strike'] / df['spot_price'] - 1


def _build_display_df(strikes_df: pd.DataFrame, leverage: float, base_capital: float) -> pd.DataFrame:
    df = strikes_df.copy()
    df['Bid'] = df['bid']
    df['Ask'] = df['ask']
    df['Implied Volatility'] = df['implied_volatility']
    df['Strike'] = df['closest_strike']
    df['Expiry'] = df['expiry_date'].apply(
        lambda x: datetime.strptime(str(x), '%Y%m%d').strftime('%d %b %Y')
    )

    _recompute_derived_fields(df, base_capital, leverage)

    df['BidColor'] = 'black'
    df['AskColor'] = 'black'
    df['MidColor'] = 'black'
    df['BidDisplay'] = [
        _format_colored_value(value, color, 2) for value, color in zip(df['Bid'], df['BidColor'])
    ]
    df['AskDisplay'] = [
        _format_colored_value(value, color, 2) for value, color in zip(df['Ask'], df['AskColor'])
    ]
    df['MidDisplay'] = [
        _format_colored_value(value, color, 1) for value, color in zip(df['Mid'], df['MidColor'])
    ]

    df = df.drop(
        columns=[
            'option_life',
            'trade_date',
            'time_discount',
            'time_scale',
            'theoretical_strike',
            'last_traded',
            'volume',
            'market',
            'qualified_contracts',
            'bid',
            'ask',
            'implied_volatility',
            'expiry_date',
        ],
        errors='ignore'
    )

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df.fillna({col: 0}, inplace=True)
        else:
            df.fillna({col: '--'}, inplace=True)
    return df


def _subscribe_option_tickers(ib: IB, qualified_contracts: pd.Series) -> Dict[int, Ticker]:
    tickers: Dict[int, Ticker] = {}
    for idx, contract in enumerate(qualified_contracts):
        if contract is None:
            continue
        tickers[idx] = ib.reqMktData(contract, snapshot=False)
    return tickers


def _cancel_market_data(ib: IB, contracts: List[Contract]) -> None:
    for contract in contracts:
        if contract is None:
            continue
        try:
            ib.cancelMktData(contract)
        except Exception:
            continue


def _build_patch(df: pd.DataFrame, columns: List[str]) -> Dict[str, List[tuple]]:
    return {
        col: [(i, value) for i, value in enumerate(df[col].tolist())]
        for col in columns
    }


def get_bid_ask_for_contracts(ib: IB, qualified_contracts: pd.Series) -> pd.DataFrame:
    """
    Fetches bid and ask prices for a list of qualified option contracts from Interactive Brokers (IB).

    Args:
        ib (IB): An instance of the IB connection to interact with the Interactive Brokers API.
        qualified_contracts (pd.Series): A pandas Series of qualified option contracts.

    Returns:
        pd.DataFrame: A DataFrame containing market data including bid, ask, last traded price, volume,
                      and market price (midpoint) for each contract.

    Steps:
        1. Request market data for each qualified contract.
        2. Retry up to 5 times for contracts with invalid bid or ask prices.
        3. Collect bid, ask, last traded price, volume, and market price for each contract and return as a DataFrame.

    Example:
        Input: qualified_contracts as a pandas Series of option contracts.
        Output: DataFrame with bid, ask, last traded, volume, and market price for each contract.
    """

    market_data = []  # List to store the market data for each contract

    # Batch request for market data (snapshot=True for current prices only)
    tickers: Dict[Contract, IB] = {}
    for contract in qualified_contracts:
        # Request market data for each contract (non-snapshot for continuous updates)
        ticker = ib.reqMktData(contract, snapshot=True)
        tickers[contract] = ticker

    # Wait for all the market data to arrive
    ib.sleep(1)  # Sleep to allow the initial data to populate

    for contract, ticker in tickers.items():
        retry_attempts = 5  # Max number of retries for invalid bid/ask data
        attempt = 0
        implied_vol = None
        # Retry until valid bid and ask data is received or max retries is reached
        while (pd.isna(ticker.bid) or pd.isna(ticker.ask)or pd.isna(ticker.modelGreeks)) and attempt < retry_attempts:
            print(f"Invalid bid/ask for {contract}, retrying... (Attempt {attempt + 1})")
            ib.sleep(1)  # Wait for 1 second before retrying
            attempt += 1

        # if ticker.modelGreeks:
        #     implied_vol = ticker.modelGreeks.impliedVol
        #     print(f"Implied Volatility: {implied_vol}")
            # Append the contract and the data after the bid/ask are valid or after max retries
        market_data.append({
            'contract': contract,
            'bid': ticker.bid,
            'ask': ticker.ask,
            'last_traded': ticker.last,  # Last traded price
            'volume': ticker.volume,  # Volume for the day
            'market': ticker.marketPrice(),  # Market price (midpoint of bid/ask)
            'implied_volatility': ticker.modelGreeks.impliedVol if ticker.modelGreeks else None  # Handle None case
        })

    # Convert the market data into a DataFrame
    return pd.DataFrame(market_data, index=qualified_contracts.index)


def get_account_tag(ib, tag):
    account_tag = [v for v in ib.accountValues() if v.tag == tag and v.currency == 'BASE']
    return account_tag


def fetch_data(
        ib_wrapper: IbWrapper,
        option_expiries: List[datetime],
        risk_free: pd.Series,
        option_params_df: pd.DataFrame,
        cache: QualifiedContractsCache,
        leverage: float = 1,
        z_score: float = -1
) -> pd.DataFrame:
    """
    Fetches market data and calculates theoretical strikes for SPX options, including bid/ask prices, volume, and margin calculations.

    Args:
        ib_wrapper (IbWrapper): An instance of the IbWrapper that provides a connection to the IB API.
        option_expiries (List[datetime]): A list of option expiry dates.
        risk_free (pd.Series): Series of risk-free rates corresponding to each expiry.
        option_params_df (pd.DataFrame): DataFrame containing the option parameters, including available strikes.

    Returns:
        pd.DataFrame: A DataFrame with calculated option strikes, bid/ask data, and margin calculations.

    Steps:
        1. Fetch market data for SPX and VIX index contracts.
        2. Calculate theoretical option strikes based on market prices and risk-free rates.
        3. Qualify the option contracts and retrieve bid/ask data.
        4. Calculate margin, notional capital, and leverage for each contract.
        5. Return a cleaned DataFrame with relevant information for display or further analysis.

    Example:
        Input: A list of option expiries, risk-free rates, and option parameter DataFrame.
        Output: DataFrame with bid, ask, margin, and other calculated values.
    """

    # Define the contracts for SPX and VIX
    def get_market_data(ib_wrapper: IbWrapper, contracts: List[Contract]) -> Dict[str, float]:
        """
        Fetches the snapshot market prices for a list of contracts.

        Args:
            ib_wrapper: The instance of the IbWrapper to interact with Interactive Brokers.
            contracts: A list of contracts (e.g., SPX, VIX) for which market data needs to be fetched.

        Returns:
            A dictionary with contract symbols as keys and their corresponding market prices as values.
        """
        tickers = {}

        # Request market data for all contracts in snapshot mode
        for contract in contracts:
            ticker = ib_wrapper.ib.reqMktData(contract, '', snapshot=True)
            tickers[contract.symbol] = ticker

        # Wait for data to be populated and ensure no NaNs
        # ib_wrapper.ib.sleep(1)

        # Extract the market prices and return in a dictionary
        market_prices = {}
        for symbol, ticker in tickers.items():
            while pd.isna(ticker.marketPrice()):
                ib_wrapper.ib.sleep(0.1)
            market_prices[symbol] = ticker.marketPrice()

        return market_prices

    spx_contract = Index('SPX', 'CBOE', 'USD')
    vix_contract = Index('VIX', 'CBOE', 'USD')

    # Qualify each contract separately
    spx_contract = ib_wrapper.ib.qualifyContracts(spx_contract)[0]
    vix_contract = ib_wrapper.ib.qualifyContracts(vix_contract)[0]
    market_prices = get_market_data(ib_wrapper, [spx_contract, vix_contract])


    # Retrieve the market prices for SPX and VIX
    spx_price = market_prices['SPX']
    vix_price = market_prices['VIX']

    # Step 1: Calculate theoretical strikes
    strikes_df = get_theoretical_strike(
        option_expiries, [spx_price], risk_free, [z_score], 0.013, [vix_price]
    )
    available_strikes = option_params_df['strikes'].values[0]
    strikes_df['expiry_date'] = strikes_df.index

    # Step 2: Qualify all option contracts based on theoretical strikes
    strikes_df = qualify_all_contracts(ib_wrapper, strikes_df, available_strikes, cache)

    # Step 3: Get the liquidation value for the account
    liquidation_value = get_account_tag(ib_wrapper.ib, 'NetLiquidationByCurrency')

    # Step 4: Get bid/ask market data for all qualified contracts
    market_data_df = get_bid_ask_for_contracts(ib_wrapper.ib, strikes_df['qualified_contracts'])
    strikes_df['bid'] = market_data_df['bid']
    strikes_df['ask'] = market_data_df['ask']
    strikes_df['last_traded'] = market_data_df['last_traded']
    strikes_df['volume'] = market_data_df['volume']
    strikes_df['market'] = market_data_df['market']
    strikes_df['implied_volatility'] = market_data_df['implied_volatility']

    # Step 5: Calculate mid price and other derived metrics
    strikes_df['mid'] = (strikes_df['bid'] + strikes_df['ask']) / 2
    notional_capital = strikes_df['closest_strike'] * strikes_df['strike_discount'] - strikes_df['mid']

    # Calculate lots for each leverage level and margin
    capital_at_risk = illiquid_equity(discount=0.5) + float(liquidation_value[0].value)
    # for num_leverage in [1, 1.5, 2]:
    #     strikes_df[f'lots_leverage_{num_leverage}'] = round(
    #         capital_at_risk / (notional_capital / num_leverage * 100), 0
    #     )


    strikes_df[f'Lots'] = round(
        capital_at_risk / (notional_capital / leverage * 100), 0
    )

    # Step 6: Calculate margin for each contract
    single_margin_a = (strikes_df['mid'] + 0.2 * strikes_df['spot_price']) - (
            strikes_df['spot_price'] - strikes_df['closest_strike']
    )
    single_margin_b = strikes_df['mid'] + 0.1 * strikes_df['closest_strike']
    margin = pd.concat([single_margin_a, single_margin_b], axis=1).max(axis=1)
    strikes_df['Margin'] = margin * 100
    strikes_df['Margin'] = strikes_df['Margin'] * strikes_df['Lots']

    # Drop unnecessary columns and clean up the DataFrame
    out_df = strikes_df.drop(
        columns=[
            'option_life', 'trade_date', 'time_discount', 'time_scale',
            'strike_discount', 'theoretical_strike', 'last_traded', 'volume',
            'qualified_contracts'
        ]
    ).copy()
    # Convert expiry dates to a more readable format
    out_df['expiry_date'] = out_df['expiry_date'].apply(
        lambda x: datetime.strptime(str(x), '%Y%m%d').strftime('%d %b %Y')
    )
    out_df['Discount'] = out_df['closest_strike'] / out_df['spot_price'] - 1
    out_df.rename(columns={'closest_strike': 'Strike',
                           'expiry_date': 'Expiry',
                           'bid': 'Bid',
                           'ask': 'Ask',
                            'mid': 'Mid',
                           'implied_volatility': 'Implied Volatility',
                           }, inplace=True)

    # Handle missing values in numerical and non-numerical columns
    for col in out_df.columns:
        if pd.api.types.is_numeric_dtype(out_df[col]):
            out_df.fillna({col: 0}, inplace=True)
        else:
            out_df.fillna({col: '--'}, inplace=True)

    # Return the updated DataFrame
    return out_df

class PriceTracker:
    """
    A class to track the current and previous prices for SPX, VIX, or any other asset
    and determine whether the trend is up, down, or unchanged.
    """
    def __init__(self):
        # Store the current and previous prices and trends
        self.previous_prices = {}
        self.trends = {}

    def update_price(self, symbol, current_price):
        """
        Updates the price for a given symbol and determines the trend (up, down, unchanged).
        """
        # Check if we have a previous price stored for this symbol
        if symbol in self.previous_prices:
            previous_price = self.previous_prices[symbol]
            # Determine the trend
            if current_price > previous_price:
                self.trends[symbol] = 'green'
            elif current_price < previous_price:
                self.trends[symbol] = 'red'
            else:
                # If unchanged, keep the previous trend
                self.trends[symbol] = self.trends.get(symbol, 'black')
        else:
            # If no previous price, default to black
            self.trends[symbol] = 'black'

        # Update the previous price with the current one
        self.previous_prices[symbol] = current_price

    def get_trend(self, symbol):
        """
        Returns the trend color for the given symbol.
        """
        return self.trends.get(symbol, 'black')

# Update create_bokeh_app to pass arguments to fetch_data
# Integrate the PriceTracker class into the Bokeh app
def create_bokeh_app():
    """
    Creates a Bokeh app that displays a DataTable with SPX option data and tracks price trends.
    """
    doc = curdoc()
    # nuke previous state on refresh
    doc.clear()
    # Initialize the IBWrapper to connect to Interactive Brokers
    ib_wrapper = IbWrapper()
    # 1) pre-cleanup
    if ib_wrapper.ib.isConnected():
        ib_wrapper.ib.disconnect()


    # Initialize the cache
    cache = QualifiedContractsCache()

    # Initialize PriceTracker to track SPX and VIX trends
    price_tracker = PriceTracker()

    try:
        # Establish connection to Interactive Brokers
        ib_wrapper.connect_to_ib()
        option_params_df = fetch_option_chain_via_params(ib_wrapper)
        option_expiries = _select_option_expiries(option_params_df, MAX_EXPIRIES)

        yld_curve = USSimpleYieldCurve()
        risk_free = yld_curve.get_zero4_date([date.date() for date in option_expiries])
        lev_slider = Slider(start=0.5, end=4, value=1, step=0.5, title="Leverage")
        z_slider = Slider(start=-4, end=4, value=-1, step=1, title="Z-Score")

        spx_contract = Index('SPX', 'CBOE', 'USD')
        vix_contract = Index('VIX', 'CBOE', 'USD')
        spx_contract, vix_contract = ib_wrapper.ib.qualifyContracts(spx_contract, vix_contract)
        spx_ticker = ib_wrapper.ib.reqMktData(spx_contract, '', snapshot=False)
        vix_ticker = ib_wrapper.ib.reqMktData(vix_contract, '', snapshot=False)

        start_time = time.time()
        spx_price = _best_price(spx_ticker)
        vix_price = _best_price(vix_ticker)
        while (spx_price is None or vix_price is None) and time.time() - start_time < 2:
            ib_wrapper.ib.sleep(0.1)
            spx_price = _best_price(spx_ticker)
            vix_price = _best_price(vix_ticker)
        if spx_price is None:
            spx_price = 0.0
        if vix_price is None:
            vix_price = 0.0

        liquidation_value = get_account_tag(ib_wrapper.ib, 'NetLiquidationByCurrency')
        liquidation_amount = float(liquidation_value[0].value) if liquidation_value else 0.0
        base_capital = illiquid_equity(discount=0.5) + liquidation_amount

        strikes_df = _build_strikes_df(
            ib_wrapper,
            option_expiries,
            risk_free,
            option_params_df,
            cache,
            spx_price,
            vix_price,
            z_slider.value
        )
        option_tickers = _subscribe_option_tickers(ib_wrapper.ib, strikes_df['qualified_contracts'])
        out_df = _build_display_df(strikes_df, leverage=lev_slider.value, base_capital=base_capital)

        display_fields = ['Expiry', 'Days to Expiry', 'Strike', 'BidDisplay', 'AskDisplay', 'MidDisplay',
                          'Implied Volatility',
                          'Discount', 'Margin', 'Lots']
        # Mapping of column names to their respective formatters
        formatter_map = {
            "Margin": NumberFormatter(format="$0,0"),  # Currency format for Margin
            "Discount": NumberFormatter(format="0.00%"),  # Percentage format for Discount
            'Implied Volatility': NumberFormatter(format="0.00%"),  # Percentage format for 'Implied Volatility'
        }

        display_df = out_df[display_fields].copy()

        source = ColumnDataSource(display_df)

        spx_div = Div(text=f"<b>SPX Price:</b> {spx_price:.2f}")
        vix_div = Div(text=f"<b>VIX Price:</b> {vix_price:.2f}")
        account_div_1 = Div(text=f"<b>Liquidation Value:</b> ${liquidation_amount:,.0f}")
        account_div_2 = Div(text=f"<b>Capital at Risk:</b> ${base_capital * lev_slider.value:,.0f}")


        # Create the columns for the DataTable, applying formatters where necessary
        columns = [
            TableColumn(field='Expiry', title='Expiry'),
            TableColumn(field='Days to Expiry', title='Days to Expiry'),
            TableColumn(field='Strike', title='Strike'),
            TableColumn(field='BidDisplay', title='Bid', formatter=HTMLTemplateFormatter(template="<%= value %>")),
            TableColumn(field='AskDisplay', title='Ask', formatter=HTMLTemplateFormatter(template="<%= value %>")),
            TableColumn(field='MidDisplay', title='Mid', formatter=HTMLTemplateFormatter(template="<%= value %>")),
            TableColumn(field='Implied Volatility', title='Implied Volatility',
                        formatter=formatter_map.get('Implied Volatility')),
            TableColumn(field='Discount', title='Discount', formatter=formatter_map.get('Discount')),
            TableColumn(field='Margin', title='Margin', formatter=formatter_map.get('Margin')),
            TableColumn(field='Lots', title='Lots'),
        ]

        # Create the DataTable using the data source and defined columns
        data_table = DataTable(
            source=source,
            columns=columns,
            width=1000,
            height=1000,
            index_position=None
        )

        # Create the layout for the Bokeh app
        app_layout = column(row(spx_div, vix_div, account_div_1, account_div_2), row(lev_slider, z_slider), data_table)
        doc.add_root(app_layout)
        state = {
            "data_df": out_df,
            "option_tickers": option_tickers,
            "option_contracts": strikes_df['qualified_contracts'].tolist(),
            "base_capital": base_capital,
            "rebuilding": False
        }

        patch_columns = [
            'BidDisplay', 'AskDisplay', 'MidDisplay', 'Implied Volatility',
            'Discount', 'Margin', 'Lots'
        ]

        def update_account_values():
            liquidation_value = get_account_tag(ib_wrapper.ib, 'NetLiquidationByCurrency')
            liquidation_amount = float(liquidation_value[0].value) if liquidation_value else 0.0
            state['base_capital'] = illiquid_equity(discount=0.5) + liquidation_amount
            account_div_1.text = f"<b>Liquidation Value:</b> ${liquidation_amount:,.0f}"
            account_div_2.text = f"<b>Capital at Risk:</b> ${state['base_capital'] * lev_slider.value:,.0f}"
            if state['data_df'].empty:
                return
            _recompute_derived_fields(state['data_df'], state['base_capital'], lev_slider.value)
            source.patch(_build_patch(state['data_df'], patch_columns))

        def rebuild_contracts(z_score_value: float) -> None:
            state['rebuilding'] = True
            try:
                _cancel_market_data(ib_wrapper.ib, state['option_contracts'])
                spot_price = _best_price(spx_ticker)
                vix_value = _best_price(vix_ticker)
                if spot_price is None and not state['data_df'].empty:
                    spot_price = float(state['data_df']['spot_price'].iloc[0])
                if vix_value is None and not state['data_df'].empty:
                    vix_value = float(state['data_df']['sigma'].iloc[0]) * 100
                if spot_price is None:
                    spot_price = 0.0
                if vix_value is None:
                    vix_value = 0.0

                new_strikes_df = _build_strikes_df(
                    ib_wrapper,
                    option_expiries,
                    risk_free,
                    option_params_df,
                    cache,
                    spot_price,
                    vix_value,
                    z_score_value
                )
                new_option_tickers = _subscribe_option_tickers(ib_wrapper.ib, new_strikes_df['qualified_contracts'])
                new_out_df = _build_display_df(
                    new_strikes_df, leverage=lev_slider.value, base_capital=state['base_capital']
                )
                new_display_df = new_out_df[display_fields].copy()

                state['data_df'] = new_out_df
                state['option_tickers'] = new_option_tickers
                state['option_contracts'] = new_strikes_df['qualified_contracts'].tolist()
                source.data = new_display_df.to_dict(orient='list')
            finally:
                state['rebuilding'] = False

        def update_market():
            if state['rebuilding'] or state['data_df'].empty:
                return

            data_df = state['data_df']
            prev_mid_values = data_df['Mid'].copy()
            spot_price = _best_price(spx_ticker)
            if spot_price is not None:
                data_df['spot_price'] = spot_price
            vix_value = _best_price(vix_ticker)
            if vix_value is not None:
                data_df['sigma'] = vix_value / 100

            for idx, ticker in state['option_tickers'].items():
                if ticker is None:
                    continue
                prev_bid = data_df.at[idx, 'Bid']
                prev_ask = data_df.at[idx, 'Ask']
                if not pd.isna(ticker.bid):
                    new_bid = ticker.bid
                    data_df.at[idx, 'Bid'] = new_bid
                    data_df.at[idx, 'BidColor'] = _trend_color(
                        prev_bid, new_bid, data_df.at[idx, 'BidColor']
                    )
                if not pd.isna(ticker.ask):
                    new_ask = ticker.ask
                    data_df.at[idx, 'Ask'] = new_ask
                    data_df.at[idx, 'AskColor'] = _trend_color(
                        prev_ask, new_ask, data_df.at[idx, 'AskColor']
                    )
                if ticker.modelGreeks and not pd.isna(ticker.modelGreeks.impliedVol):
                    data_df.at[idx, 'Implied Volatility'] = ticker.modelGreeks.impliedVol

            _recompute_derived_fields(data_df, state['base_capital'], lev_slider.value)
            for idx in data_df.index:
                prev_mid = prev_mid_values.at[idx]
                new_mid = data_df.at[idx, 'Mid']
                data_df.at[idx, 'MidColor'] = _trend_color(
                    prev_mid, new_mid, data_df.at[idx, 'MidColor']
                )
            data_df['BidDisplay'] = [
                _format_colored_value(value, color, 2)
                for value, color in zip(data_df['Bid'], data_df['BidColor'])
            ]
            data_df['AskDisplay'] = [
                _format_colored_value(value, color, 2)
                for value, color in zip(data_df['Ask'], data_df['AskColor'])
            ]
            data_df['MidDisplay'] = [
                _format_colored_value(value, color, 1)
                for value, color in zip(data_df['Mid'], data_df['MidColor'])
            ]
            source.patch(_build_patch(data_df, patch_columns))

            current_spx_price = float(data_df['spot_price'].iloc[0])
            current_vix_price = float(data_df['sigma'].iloc[0]) * 100
            price_tracker.update_price('SPX', current_spx_price)
            price_tracker.update_price('VIX', current_vix_price)

            spx_color = price_tracker.get_trend('SPX')
            vix_color = price_tracker.get_trend('VIX')
            spx_div.text = f"<b>SPX Price:</b> <span style='color:{spx_color}'>{current_spx_price:.2f}</span>"
            vix_div.text = f"<b>VIX Price:</b> <span style='color:{vix_color}'>{current_vix_price:.2f}</span>"

        def on_leverage_change(attr, old, new):
            if state['rebuilding'] or state['data_df'].empty:
                return
            _recompute_derived_fields(state['data_df'], state['base_capital'], new)
            source.patch(_build_patch(state['data_df'], patch_columns))
            account_div_2.text = f"<b>Capital at Risk:</b> ${state['base_capital'] * new:,.0f}"

        def on_z_change(attr, old, new):
            rebuild_contracts(new)

        lev_slider.on_change('value', on_leverage_change)
        z_slider.on_change('value', on_z_change)

        update_account_values()
        doc.add_periodic_callback(update_market, PRICE_UPDATE_MS)
        doc.add_periodic_callback(update_account_values, ACCOUNT_UPDATE_MS)

        def _cleanup_session(session_context):
            _cancel_market_data(ib_wrapper.ib, state['option_contracts'])
            _cancel_market_data(ib_wrapper.ib, [spx_contract, vix_contract])
            ib_wrapper.disconnect()

        doc.on_session_destroyed(_cleanup_session)


        # Return the layout for the Bokeh document root
        # return app_layout

    except ConnectionError as e:
        # Handle connection errors by logging the error and returning an empty layout
        print(f"Failed to connect to IB: {e}")
        return column()  # Return an empty layout if IB connection fails


create_bokeh_app()
# Add the Bokeh app layout to the current document root
# curdoc().add_root(create_bokeh_app())
