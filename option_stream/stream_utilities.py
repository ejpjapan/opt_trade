from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta

import nest_asyncio
import pandas as pd
import pandas_datareader.data as web
from dateutil.relativedelta import relativedelta
from ib_insync import IB

try:
    from option_stream.config import (
        IB_CONNECT_TIMEOUT,
        IB_GATEWAY_PORT,
        IB_HOST,
        IB_TWS_PORT,
        config_key,
    )
except ImportError:  # pragma: no cover - supports running from option_stream/
    from config import (  # type: ignore
        IB_CONNECT_TIMEOUT,
        IB_GATEWAY_PORT,
        IB_HOST,
        IB_TWS_PORT,
        config_key,
    )


logger = logging.getLogger(__name__)

class IbWrapper:
    def __init__(self, base_client_id: int = 1000):
        """
        Lightweight wrapper around ib_insync.IB
        - Always picks a fresh clientId to avoid clashes
        - Applies nest_asyncio so .connect() can be called in a running event loop
        """
        nest_asyncio.apply()
        self.ib = IB()
        self.ib.errorEvent += self.on_error

        # Generate a semi-unique clientId each time:
        # base_client_id + a random offset (0–999)
        self.client_id = base_client_id + random.randint(0, 999)
        logger.debug(f"[IbWrapper] initialized with clientId={self.client_id}")

    def on_error(self, reqId, errorCode, errorString, contract):
        logger.warning(f"[IB Error] reqId={reqId} code={errorCode} msg={errorString}")

    def connect_to_ib(self,
                      host: str = IB_HOST,
                      gateway_port: int = IB_GATEWAY_PORT,
                      tws_port: int = IB_TWS_PORT,
                      timeout: float = IB_CONNECT_TIMEOUT):
        """
        Try IB Gateway first, then fall back to TWS.
        Always disconnect any existing connection first.
        """
        # 1) Tear down any existing IB session
        if self.ib.isConnected():
            logger.info("[IbWrapper] existing IB connection found; disconnecting.")
            self.ib.disconnect()
            # Give it a moment to close cleanly
            time.sleep(0.2)

        # 2) Attempt Gateway → TWS with retry/back-off
        for port, name in ((gateway_port, "Gateway"), (tws_port, "TWS")):
            try:
                logger.info(f"[IbWrapper] connecting to IB {name} on port {port} (clientId={self.client_id})")
                self.ib.connect(host, port, clientId=self.client_id, timeout=timeout)
                logger.info(f"[IbWrapper] SUCCESS: connected to {name} on port {port}")
                return
            except ConnectionRefusedError as cre:
                logger.warning(f"[IbWrapper] {name} refused: {cre!r}")
            except Exception as e:
                # catch “clientId in use” and timeouts
                logger.warning(f"[IbWrapper] {name} connect error: {e!r}")

            # small back-off before next attempt
            time.sleep(0.5)

        # 3) If we get here, both failed
        raise ConnectionError(f"Could not connect to IB Gateway ({gateway_port}) or TWS ({tws_port}).\n"
                              "• Make sure the API port is open in TWS/IBG\n"
                              "• Check your host/port settings")

    def disconnect(self):
        """Cleanly tear down the IB session if live."""
        if self.ib.isConnected():
            logger.info(f"[IbWrapper] disconnecting clientId={self.client_id}")
            self.ib.disconnect()
        else:
            logger.debug(f"[IbWrapper] no active connection to disconnect")


class USSimpleYieldCurve:
    """Simple US Zero coupon yield curve for today up to one year"""
    # Simple Zero yield curve built from TBill discount yields and effective fund rate
    # This is a simplified approximation for a full term structure model
    # Consider improving by building fully specified yield curve model using
    # Quantlib
    def __init__(self):
        end = datetime.now()
        start = end - timedelta(days=10)
        fred_api_key = config_key('fred_api_key')
        columns = ['DFF', 'DTB4WK', 'DTB3', 'DTB6', 'DTB1YR', 'DGS2']
        try:
            zero_rates = web.DataReader(columns, 'fred', start, end, api_key=fred_api_key).dropna(axis=0)
            if zero_rates.empty:
                raise ValueError('No zero rate data returned')
            zero_yld_date = zero_rates.index[-1]
        except Exception:
            zero_yld_date = pd.Timestamp(end - timedelta(days=1))
            zero_rates = pd.DataFrame(data=[[3.62] * len(columns)],
                                      index=[zero_yld_date],
                                      columns=columns)
        new_index = [zero_yld_date + relativedelta(days=1),
                     zero_yld_date + relativedelta(weeks=4),
                     zero_yld_date + relativedelta(months=3),
                     zero_yld_date + relativedelta(months=6),
                     zero_yld_date + relativedelta(years=1),
                     zero_yld_date + relativedelta(years=2)]
        dt_time_index = pd.DatetimeIndex(new_index, tz='America/New_York')
        zero_curve = pd.DataFrame(data=zero_rates.iloc[-1].values, index=pd.DatetimeIndex(dt_time_index.date),
                                  columns=[end])
        self.zero_curve = zero_curve.resample('D').interpolate(method='polynomial', order=2)

    def get_zero4_date(self, input_date):
        """Retrieve zero yield maturity for input_date"""
        return self.zero_curve.loc[input_date]


def illiquid_equity(discount=0.5):
    return sum(config_key('illiquid_equity').values()) * discount
