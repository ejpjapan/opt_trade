from __future__ import annotations

import os
import plistlib
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PLIST_PATH = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "com~apple~CloudDocs"
    / "localDB"
    / "config.plist"
)

CONFIG_PLIST_PATH = Path(os.getenv("OPT_TRADE_CONFIG_PLIST", str(DEFAULT_CONFIG_PLIST_PATH)))

IB_HOST = os.getenv("OPT_TRADE_IB_HOST", "127.0.0.1")
IB_GATEWAY_PORT = int(os.getenv("OPT_TRADE_IB_GATEWAY_PORT", "4001"))
IB_TWS_PORT = int(os.getenv("OPT_TRADE_IB_TWS_PORT", "7496"))
IB_CONNECT_TIMEOUT = float(os.getenv("OPT_TRADE_IB_CONNECT_TIMEOUT", "10"))

MAX_EXPIRIES = int(os.getenv("OPT_TRADE_MAX_EXPIRIES", "25"))
PRICE_UPDATE_MS = int(os.getenv("OPT_TRADE_PRICE_UPDATE_MS", "1000"))
ACCOUNT_UPDATE_MS = int(os.getenv("OPT_TRADE_ACCOUNT_UPDATE_MS", "15000"))

UNDERLYING_SYMBOL = os.getenv("OPT_TRADE_UNDERLYING_SYMBOL", "SPX")
VIX_SYMBOL = os.getenv("OPT_TRADE_VIX_SYMBOL", "VIX")
INDEX_EXCHANGE = os.getenv("OPT_TRADE_INDEX_EXCHANGE", "CBOE")
CURRENCY = os.getenv("OPT_TRADE_CURRENCY", "USD")

OPTION_EXCHANGE = os.getenv("OPT_TRADE_OPTION_EXCHANGE", "SMART")
OPTION_TRADING_CLASS = os.getenv("OPT_TRADE_OPTION_TRADING_CLASS", "SPXW")
OPTION_RIGHT = os.getenv("OPT_TRADE_OPTION_RIGHT", "P")

DEFAULT_DIVIDEND_YIELD = float(os.getenv("OPT_TRADE_DIVIDEND_YIELD", "0.013"))
ILLIQUID_EQUITY_DISCOUNT = float(os.getenv("OPT_TRADE_ILLIQUID_EQUITY_DISCOUNT", "0.5"))


@lru_cache(maxsize=1)
def load_app_config() -> dict[str, Any]:
    if not CONFIG_PLIST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing config plist at {CONFIG_PLIST_PATH}. "
            "Set OPT_TRADE_CONFIG_PLIST if the file lives elsewhere."
        )

    with CONFIG_PLIST_PATH.open("rb") as handle:
        return plistlib.load(handle)


def config_key(dict_key: str):
    config = load_app_config()
    try:
        return config[dict_key]
    except KeyError as exc:
        available = ", ".join(sorted(config))
        raise KeyError(
            f"Missing key {dict_key!r} in {CONFIG_PLIST_PATH}. Available keys: {available}"
        ) from exc
