from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, DataTable, TableColumn, Div, Slider, NumberFormatter
from bokeh.layouts import layout, row, column
import pandas as pd
import numpy as np
from ib_insync import Option, Contract, Index, util, ContractDetails, IB, Ticker
from zoneinfo import ZoneInfo  # Available in Python 3.9 and later
from datetime import datetime
from typing import List, Set, Dict


from stream_utilities import IbWrapper, USSimpleYieldCurve, illiquid_equity


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

        # Retry until valid bid and ask data is received or max retries is reached
        while (pd.isna(ticker.bid) or pd.isna(ticker.ask)) and attempt < retry_attempts:
            print(f"Invalid bid/ask for {contract}, retrying... (Attempt {attempt + 1})")
            ib.sleep(1)  # Wait for 1 second before retrying
            attempt += 1

        # Append the contract and the data after the bid/ask are valid or after max retries
        market_data.append({
            'contract': contract,
            'bid': ticker.bid if ticker.bid >= 1 else None,  # Use None for invalid bids
            'ask': ticker.ask if ticker.ask >= 1 else None,  # Use None for invalid asks
            'last_traded': ticker.last,  # Last traded price
            'volume': ticker.volume,  # Volume for the day
            'market': ticker.marketPrice()  # Market price (midpoint of bid/ask)
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
    # Initialize the IBWrapper to connect to Interactive Brokers
    ib_wrapper = IbWrapper()

    # Initialize the cache
    cache = QualifiedContractsCache()

    # Initialize PriceTracker to track SPX and VIX trends
    price_tracker = PriceTracker()

    try:
        # Establish connection to Interactive Brokers
        ib_wrapper.connect_to_ib()

        # Fetch option chain parameters (including expiration dates and strikes)
        option_params_df = fetch_option_chain_via_params(ib_wrapper)

        # Filter valid option expirations based on the current date
        option_expiries = [
            expiry for expiry in option_params_df['expirations_timestamps'].values[0]
            if expiry >= datetime.now(tz=ZoneInfo("America/New_York"))
        ]

        # Initialize yield curve to fetch risk-free rates for the option expiries
        yld_curve = USSimpleYieldCurve()
        risk_free = yld_curve.get_zero4_date([date.date() for date in option_expiries])
        lev_slider = Slider(start=0.5, end=4, value=1, step=0.5, title="Leverage")
        z_slider = Slider(start=-4, end=4, value=-1, step=1, title="Z-Score")
        # Fetch the full data set with market prices, strikes, and margins
        out_df = fetch_data(
            ib_wrapper, option_expiries, risk_free, option_params_df, cache, leverage=lev_slider.value,
            z_score=z_slider.value
        )

        # Set up the data source for the Bokeh DataTable
        source = ColumnDataSource(out_df)
        previous_source = ColumnDataSource(out_df.copy())

        # Create Divs to display SPX and VIX prices with default values
        spx_div = Div(text=f"<b>SPX Price:</b> {out_df.iloc[0]['spot_price']:.2f}")
        vix_div = Div(text=f"<b>VIX Price:</b> {out_df.iloc[0]['sigma'] * 100:.2f}")
        liquidation_value = get_account_tag(ib_wrapper.ib, 'NetLiquidationByCurrency')
        capital_at_risk = (illiquid_equity(discount=0.5) + float(liquidation_value[0].value)) * lev_slider.value

        account_div_1 = Div(text=f"<b>Liquidation Value:</b> ${float(liquidation_value[0].value):,.0f}")
        account_div_2 = Div(text=f"<b>Capital at Risk:</b> ${capital_at_risk:,.0f}")


        display_cols = ['Expiry', 'Days to Expiry', 'Strike', 'Bid', 'Ask', 'Mid', 'Discount', 'Margin', 'Lots']
        # Mapping of column names to their respective formatters
        formatter_map = {
            "Mid": NumberFormatter(format="0.0"),  # One decimal place for Mid
            "Margin": NumberFormatter(format="$0,0"),  # Currency format for Margin
            "Discount": NumberFormatter(format="0.00%"),  # Percentage format for Discount
        }


        # Create the columns for the DataTable, applying formatters where necessary
        columns = [
            TableColumn(field=col, title=col, formatter=formatter_map.get(col))
            if formatter_map.get(col) else TableColumn(field=col, title=col)
            for col in display_cols
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

        # Periodic callback function to update data every second
        def update():
            # Fetch updated data
            update_df = fetch_data(ib_wrapper, option_expiries, risk_free, option_params_df, cache,
                                   leverage=lev_slider.value,   z_score=z_slider.value
            )

            # Store previous data
            previous_source.data = source.data.copy()

            # Update the data source with the new data
            source.data = update_df.to_dict(orient='list')

            # Grab the current SPX and VIX prices
            current_spx_price = source.data['spot_price'][0]
            current_vix_price = source.data['sigma'][0] * 100

            # Update the price tracker for SPX and VIX
            price_tracker.update_price('SPX', current_spx_price)
            price_tracker.update_price('VIX', current_vix_price)

            # Get the trend colors for SPX and VIX
            spx_color = price_tracker.get_trend('SPX')
            vix_color = price_tracker.get_trend('VIX')

            # Update SPX and VIX prices in the Divs with conditional color formatting
            spx_div.text = f"<b>SPX Price:</b> <span style='color:{spx_color}'>{current_spx_price:.2f}</span>"
            vix_div.text = f"<b>VIX Price:</b> <span style='color:{vix_color}'>{current_vix_price:.2f}</span>"
            liquidation_value = get_account_tag(ib_wrapper.ib, 'NetLiquidationByCurrency')
            capital_at_risk = (illiquid_equity(discount=0.5) + float(liquidation_value[0].value)) * lev_slider.value
            account_div_1.text = f"<b>Liquidation Value:</b> ${float(liquidation_value[0].value):,.0f}"
            account_div_2.text = f"<b>Capital at Risk:</b> ${capital_at_risk:,.0f}"

        # Add a periodic callback to update the data every 1 second (1000 ms)
        curdoc().add_periodic_callback(update, 1000)

        # Return the layout for the Bokeh document root
        return app_layout

    except ConnectionError as e:
        # Handle connection errors by logging the error and returning an empty layout
        print(f"Failed to connect to IB: {e}")
        return column()  # Return an empty layout if IB connection fails



# Add the Bokeh app layout to the current document root
curdoc().add_root(create_bokeh_app())




