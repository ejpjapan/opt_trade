from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from ib_insync import Contract, ContractDetails, IB, Index, Option, Ticker, util
from zoneinfo import ZoneInfo

try:
    from option_stream.config import (
        CURRENCY,
        DEFAULT_DIVIDEND_YIELD,
        INDEX_EXCHANGE,
        OPTION_EXCHANGE,
        OPTION_RIGHT,
        OPTION_TRADING_CLASS,
        UNDERLYING_SYMBOL,
        VIX_SYMBOL,
    )
    from option_stream.stream_utilities import IbWrapper
except ImportError:  # pragma: no cover - supports running from option_stream/
    from config import (  # type: ignore
        CURRENCY,
        DEFAULT_DIVIDEND_YIELD,
        INDEX_EXCHANGE,
        OPTION_EXCHANGE,
        OPTION_RIGHT,
        OPTION_TRADING_CLASS,
        UNDERLYING_SYMBOL,
        VIX_SYMBOL,
    )
    from stream_utilities import IbWrapper  # type: ignore


logger = logging.getLogger(__name__)
APP_TZ = ZoneInfo("America/New_York")


def _singleton_value(value, name: str):
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        if len(value) != 1:
            raise ValueError(f"{name} must be a single value or a one-item sequence.")
        return value[0]
    return value


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


def _trend_color(previous: Optional[float], current: Optional[float], existing: str = "black") -> str:
    if previous is None or current is None:
        return existing
    if pd.isna(previous) or pd.isna(current):
        return existing
    if current > previous:
        return "green"
    if current < previous:
        return "red"
    return existing


def _format_colored_value(value: Optional[float], color: str, decimals: int) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"<span style='color:{color}'>{value:.{decimals}f}</span>"


def _select_option_expiries(option_params_df: pd.DataFrame, max_expiries: int) -> List[datetime]:
    now = datetime.now(tz=APP_TZ)
    expiries: List[datetime] = []

    if "expirations_timestamps" in option_params_df.columns:
        for exp_list in option_params_df["expirations_timestamps"].dropna().values:
            if isinstance(exp_list, (list, tuple, set)):
                expiries.extend(list(exp_list))

    if not expiries and "expirations" in option_params_df.columns:
        for exp_list in option_params_df["expirations"].dropna().values:
            if isinstance(exp_list, (list, tuple, set)):
                expiries.extend(
                    [
                        datetime.strptime(item, "%Y%m%d").replace(
                            hour=16, minute=0, second=0, tzinfo=APP_TZ
                        )
                        for item in exp_list
                    ]
                )

    valid = sorted({expiry for expiry in expiries if expiry >= now})
    return valid[:max_expiries]


def convert_to_datestamps(date_lists: List[List[str]]) -> List[List[datetime]]:
    est = APP_TZ
    all_datestamps: List[List[datetime]] = []

    for date_list in date_lists:
        datestamps: List[datetime] = []
        for date_str in date_list:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            datestamps.append(date_obj.replace(hour=16, minute=0, second=0, tzinfo=est))
        all_datestamps.append(datestamps)

    return all_datestamps


def fetch_option_chain_via_params(ibw: IbWrapper, underlying_symbol: str = UNDERLYING_SYMBOL) -> pd.DataFrame:
    spx_contract = Index(underlying_symbol, INDEX_EXCHANGE, CURRENCY)
    contract_details: list[ContractDetails] = ibw.ib.reqContractDetails(spx_contract)

    if not contract_details:
        raise ValueError(f"Contract details not found for {underlying_symbol}")

    spx_contract.conId = contract_details[0].contract.conId
    logger.info("Contract conId for %s: %s", underlying_symbol, spx_contract.conId)

    option_params = ibw.ib.reqSecDefOptParams(
        spx_contract.symbol, "", spx_contract.secType, spx_contract.conId
    )
    option_params_df: pd.DataFrame = util.df(option_params)

    filtered_params = option_params_df[
        (option_params_df["tradingClass"].isin([OPTION_TRADING_CLASS]))
        & (option_params_df["exchange"] == OPTION_EXCHANGE)
    ].copy()
    filtered_params["expirations_timestamps"] = convert_to_datestamps(
        filtered_params.loc[:, "expirations"].copy()
    )
    return filtered_params


def get_theoretical_strike(
    option_expiry: List[datetime],
    spot_price: List[float],
    risk_free: pd.Series,
    z_score: List[float],
    dividend_yield: float,
    sigma: List[float],
) -> pd.DataFrame:
    sigma = _singleton_value(sigma, "sigma") / 100
    spot_price = _singleton_value(spot_price, "spot_price")
    z_score = _singleton_value(z_score, "z_score")

    trade_date = datetime.now(tz=APP_TZ)
    option_life = [(date - trade_date).total_seconds() / (365 * 24 * 60 * 60) for date in option_expiry]
    risk_free_values = np.asarray(risk_free).reshape(-1)
    if risk_free_values.size == 1 and len(option_expiry) > 1:
        risk_free_values = np.repeat(risk_free_values, len(option_expiry))
    if risk_free_values.size != len(option_expiry):
        raise ValueError("risk_free must provide one rate per expiry or a single rate.")

    df = pd.DataFrame(index=[dt.strftime("%Y%m%d") for dt in option_expiry])
    df["Days to Expiry"] = [(date - trade_date).days for date in option_expiry]
    df["option_life"] = option_life
    df["trade_date"] = trade_date
    df["sigma"] = sigma
    df["dividend_yield"] = dividend_yield
    df["spot_price"] = spot_price
    df["risk_free"] = risk_free_values / 100
    df["time_discount"] = (
        df["risk_free"] - df["dividend_yield"] + (df["sigma"] ** 2) / 2
    ) * df["option_life"]
    df["time_scale"] = df["sigma"] * np.sqrt(df["option_life"])
    df["strike_discount"] = np.exp(-df["risk_free"].mul(df["option_life"]))
    df["theoretical_strike"] = df["spot_price"] * np.exp(
        df["time_discount"] + df["time_scale"] * z_score
    )
    return df


class QualifiedContractsCache:
    def __init__(self):
        self.cache = {}
        self.unqualified_cache = set()

    def get_contract(self, contract_key):
        return self.cache.get(contract_key)

    def add_contract(self, contract_key, qualified_contract):
        self.cache[contract_key] = qualified_contract

    def is_unqualified(self, contract_key):
        return contract_key in self.unqualified_cache

    def add_unqualified(self, contract_key):
        self.unqualified_cache.add(contract_key)


def qualify_all_contracts(
    ib_wrapper: IbWrapper,
    strikes_df: pd.DataFrame,
    available_strikes: List[float],
    cache: QualifiedContractsCache,
) -> pd.DataFrame:
    strikes_df["expiry_date"] = strikes_df["expiry_date"].astype(str)
    strikes_df["qualified_contracts"] = None

    used_strikes_per_expiry: Dict[str, Set[float]] = {
        expiry: set() for expiry in strikes_df["expiry_date"].unique()
    }

    for idx, row in strikes_df.iterrows():
        expiry_str = row["expiry_date"]
        theoretical_strike = row["theoretical_strike"]
        alternative_strikes = sorted(available_strikes, key=lambda y: abs(theoretical_strike - y))
        alternative_strikes = [
            strike for strike in alternative_strikes if strike not in used_strikes_per_expiry[expiry_str]
        ]

        qualified_contract = None
        for strike in alternative_strikes:
            contract_key = (
                UNDERLYING_SYMBOL,
                expiry_str,
                strike,
                OPTION_RIGHT,
                OPTION_EXCHANGE,
                CURRENCY,
                OPTION_TRADING_CLASS,
            )

            if cache.is_unqualified(contract_key):
                used_strikes_per_expiry[expiry_str].add(strike)
                continue

            cached_contract = cache.get_contract(contract_key)
            if cached_contract:
                qualified_contract = cached_contract
                used_strikes_per_expiry[expiry_str].add(strike)
                break

            option = Option(
                symbol=UNDERLYING_SYMBOL,
                lastTradeDateOrContractMonth=expiry_str,
                strike=strike,
                right=OPTION_RIGHT,
                exchange=OPTION_EXCHANGE,
                currency=CURRENCY,
                tradingClass=OPTION_TRADING_CLASS,
            )

            try:
                qualified_contract = ib_wrapper.ib.qualifyContracts(option)[0]
                cache.add_contract(contract_key, qualified_contract)
                used_strikes_per_expiry[expiry_str].add(strike)
                break
            except Exception as exc:
                logger.warning("Failed to qualify %s: %s", option, exc)
                used_strikes_per_expiry[expiry_str].add(strike)
                cache.add_unqualified(contract_key)
                qualified_contract = None

        if qualified_contract is None:
            logger.warning(
                "No valid contract found for expiry %s and theoretical strike %s",
                expiry_str,
                theoretical_strike,
            )
            strikes_df.at[idx, "qualified_contracts"] = None
            strikes_df.at[idx, "closest_strike"] = None
        else:
            strikes_df.at[idx, "qualified_contracts"] = qualified_contract
            strikes_df.at[idx, "closest_strike"] = qualified_contract.strike

    return strikes_df


def _build_strikes_df(
    ib_wrapper: IbWrapper,
    option_expiries: List[datetime],
    risk_free: pd.Series,
    option_params_df: pd.DataFrame,
    cache: QualifiedContractsCache,
    spot_price: float,
    vix_price: float,
    z_score: float,
) -> pd.DataFrame:
    strikes_df = get_theoretical_strike(
        option_expiries, [spot_price], risk_free, [z_score], DEFAULT_DIVIDEND_YIELD, [vix_price]
    )
    strikes_df["expiry_date"] = strikes_df.index
    available_strikes = option_params_df["strikes"].values[0]
    strikes_df = qualify_all_contracts(ib_wrapper, strikes_df, available_strikes, cache)
    strikes_df = strikes_df.reset_index(drop=True)

    for col in ["bid", "ask", "last_traded", "volume", "market", "implied_volatility"]:
        strikes_df[col] = np.nan
    return strikes_df


def _recompute_derived_fields(df: pd.DataFrame, base_capital: float, leverage: float) -> None:
    df["Mid"] = (df["Bid"] + df["Ask"]) / 2
    notional_capital = df["closest_strike"] * df["strike_discount"] - df["Mid"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Lots"] = np.round(base_capital / (notional_capital / leverage * 100), 0)

    single_margin_a = (df["Mid"] + 0.2 * df["spot_price"]) - (df["spot_price"] - df["closest_strike"])
    single_margin_b = df["Mid"] + 0.1 * df["closest_strike"]
    margin = pd.concat([single_margin_a, single_margin_b], axis=1).max(axis=1)
    df["Margin"] = margin * 100
    df["Margin"] = df["Margin"] * df["Lots"]
    df["Discount"] = df["closest_strike"] / df["spot_price"] - 1


def _build_display_df(strikes_df: pd.DataFrame, leverage: float, base_capital: float) -> pd.DataFrame:
    df = strikes_df.copy()
    df["Bid"] = df["bid"]
    df["Ask"] = df["ask"]
    df["Implied Volatility"] = df["implied_volatility"]
    df["Strike"] = df["closest_strike"]
    df["Expiry"] = df["expiry_date"].apply(
        lambda x: datetime.strptime(str(x), "%Y%m%d").strftime("%d %b %Y")
    )

    _recompute_derived_fields(df, base_capital, leverage)

    df["BidColor"] = "black"
    df["AskColor"] = "black"
    df["MidColor"] = "black"
    df["BidDisplay"] = [
        _format_colored_value(value, color, 2) for value, color in zip(df["Bid"], df["BidColor"])
    ]
    df["AskDisplay"] = [
        _format_colored_value(value, color, 2) for value, color in zip(df["Ask"], df["AskColor"])
    ]
    df["MidDisplay"] = [
        _format_colored_value(value, color, 1) for value, color in zip(df["Mid"], df["MidColor"])
    ]

    df = df.drop(
        columns=[
            "option_life",
            "trade_date",
            "time_discount",
            "time_scale",
            "theoretical_strike",
            "last_traded",
            "volume",
            "market",
            "qualified_contracts",
            "bid",
            "ask",
            "implied_volatility",
            "expiry_date",
        ],
        errors="ignore",
    )

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df.fillna({col: 0}, inplace=True)
        else:
            df.fillna({col: "--"}, inplace=True)
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
    return {col: [(i, value) for i, value in enumerate(df[col].tolist())] for col in columns}


def get_bid_ask_for_contracts(ib: IB, qualified_contracts: pd.Series) -> pd.DataFrame:
    market_data = []
    tickers: Dict[Contract, Ticker] = {}

    for contract in qualified_contracts:
        ticker = ib.reqMktData(contract, snapshot=True)
        tickers[contract] = ticker

    ib.sleep(1)

    for contract, ticker in tickers.items():
        retry_attempts = 5
        attempt = 0
        while (pd.isna(ticker.bid) or pd.isna(ticker.ask) or pd.isna(ticker.modelGreeks)) and attempt < retry_attempts:
            logger.warning("Invalid bid/ask for %s, retrying... (Attempt %s)", contract, attempt + 1)
            ib.sleep(1)
            attempt += 1

        market_data.append(
            {
                "contract": contract,
                "bid": ticker.bid,
                "ask": ticker.ask,
                "last_traded": ticker.last,
                "volume": ticker.volume,
                "market": ticker.marketPrice(),
                "implied_volatility": ticker.modelGreeks.impliedVol if ticker.modelGreeks else None,
            }
        )

    return pd.DataFrame(market_data, index=qualified_contracts.index)


def get_account_tag(ib, tag):
    account_tag = [v for v in ib.accountValues() if v.tag == tag and v.currency == "BASE"]
    return account_tag


class PriceTracker:
    """
    Track current and previous prices for a symbol and return a color trend.
    """

    def __init__(self):
        self.previous_prices = {}
        self.trends = {}

    def update_price(self, symbol, current_price):
        if symbol in self.previous_prices:
            previous_price = self.previous_prices[symbol]
            if current_price > previous_price:
                self.trends[symbol] = "green"
            elif current_price < previous_price:
                self.trends[symbol] = "red"
            else:
                self.trends[symbol] = self.trends.get(symbol, "black")
        else:
            self.trends[symbol] = "black"

        self.previous_prices[symbol] = current_price

    def get_trend(self, symbol):
        return self.trends.get(symbol, "black")


__all__ = [
    "_best_price",
    "_trend_color",
    "_format_colored_value",
    "_select_option_expiries",
    "convert_to_datestamps",
    "fetch_option_chain_via_params",
    "get_theoretical_strike",
    "QualifiedContractsCache",
    "qualify_all_contracts",
    "_build_strikes_df",
    "_recompute_derived_fields",
    "_build_display_df",
    "_subscribe_option_tickers",
    "_cancel_market_data",
    "_build_patch",
    "get_bid_ask_for_contracts",
    "get_account_tag",
    "PriceTracker",
]
