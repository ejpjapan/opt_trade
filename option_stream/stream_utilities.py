# from future.backports.datetime import datetime
from ib_insync import IB
import nest_asyncio
from dateutil.relativedelta import relativedelta
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import plistlib
from pathlib import Path
import time
import random
import logging
# from ib_insync import IB
# import nest_asyncio

# class IbWrapper:
#     def __init__(self, client_id=30):
#         """Wrapper function for Interactive Broker API connection."""
#         self.ib = IB()
#         self.ib.errorEvent += self.on_error  # Attach the error handler
#         nest_asyncio.apply()
#         self.client_id = client_id  # Store client_id for reuse
#
#     def connect_to_ib(self):
#         """Attempt to connect to IB Gateway or TWS, suppressing connection errors."""
#         if self.ib.isConnected():
#             print("An existing IB connection is found. Disconnecting before reconnecting.")
#             self.ib.disconnect()
#
#         try:
#             # Attempt to connect to IB Gateway
#             self.ib.connect('127.0.0.1', port=4001, clientId=self.client_id, timeout=10)
#             print("Connected to IB Gateway on port 4001")
#         except ConnectionRefusedError:
#             print("IB Gateway connection failed. Attempting to connect to TWS...")
#             try:
#                 # Attempt to connect to TWS as a fallback
#                 self.ib.connect('127.0.0.1', port=7496, clientId=self.client_id, timeout=10)
#                 print("Connected to TWS on port 7496")
#             except ConnectionRefusedError:
#                 raise ConnectionError("TWS connection also failed. Please ensure the API port is open and try again.")
#
#     def disconnect_from_ib(self):
#         """Disconnect from IB Gateway or TWS."""
#         if self.ib.isConnected():
#             self.ib.disconnect()
#             print("Disconnected from IB.")
#
#     @staticmethod
#     def on_error(req_id, error_code, error_string, contract):
#         """Custom error handling method for the IB API."""
#         if error_code == 200:
#             # Suppress or log the specific Error 200 - No security definition found
#             pass  # Suppressing the output completely
#         elif error_code in [2104, 2106, 2158]:
#             # These are not errors, just information about data farm connections
#             pass  # Suppressing the output completely
#         else:
#             print(f"Error {error_code}, reqId {req_id}: {error_string}, contract: {contract}")
#
#     def __enter__(self):
#         """Enter the runtime context related to this object and connect to IB."""
#         self.connect_to_ib()  # Connect to IB
#         return self  # Return the instance so that `self.ib` can be accessed
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         """Exit the runtime context and disconnect from IB."""
#         self.disconnect_from_ib()  # Disconnect from IB



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
                      host: str = "127.0.0.1",
                      gateway_port: int = 4001,
                      tws_port: int = 7496,
                      timeout: float = 10):
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
        zero_rates = web.DataReader(['DFF', 'DTB4WK', 'DTB3', 'DTB6', 'DTB1YR', 'DGS2'],
                                    'fred', start, end, api_key=fred_api_key)
        zero_rates = zero_rates.dropna(axis=0)
        zero_yld_date = zero_rates.index[-1]
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


def config_key(dict_key: str):
    file_name = Path.home() / 'Library' / 'Mobile Documents' / 'com~apple~CloudDocs' / 'localDB' / 'config.plist'
    assert (file_name.is_file())
    f = open(str(file_name), 'rb')
    pl = plistlib.load(f)
    return pl[dict_key]
