# from future.backports.datetime import datetime
from ib_insync import IB
import nest_asyncio
from dateutil.relativedelta import relativedelta
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import plistlib
from pathlib import Path

class IbWrapper:
    def __init__(self, client_id=30):
        """Wrapper function for Interactive Broker API connection."""
        self.ib = IB()
        self.ib.errorEvent += self.on_error  # Attach the error handler
        nest_asyncio.apply()
        self.client_id = client_id  # Store client_id for reuse

    def connect_to_ib(self):
        """Attempt to connect to IB Gateway or TWS, suppressing connection errors."""
        if self.ib.isConnected():
            print("An existing IB connection is found. Disconnecting before reconnecting.")
            self.ib.disconnect()

        try:
            # Attempt to connect to IB Gateway
            self.ib.connect('127.0.0.1', port=4001, clientId=self.client_id, timeout=10)
            print("Connected to IB Gateway on port 4001")
        except ConnectionRefusedError:
            print("IB Gateway connection failed. Attempting to connect to TWS...")
            try:
                # Attempt to connect to TWS as a fallback
                self.ib.connect('127.0.0.1', port=7496, clientId=self.client_id, timeout=10)
                print("Connected to TWS on port 7496")
            except ConnectionRefusedError:
                raise ConnectionError("TWS connection also failed. Please ensure the API port is open and try again.")

    def disconnect_from_ib(self):
        """Disconnect from IB Gateway or TWS."""
        if self.ib.isConnected():
            self.ib.disconnect()
            print("Disconnected from IB.")

    @staticmethod
    def on_error(req_id, error_code, error_string, contract):
        """Custom error handling method for the IB API."""
        if error_code == 200:
            # Suppress or log the specific Error 200 - No security definition found
            pass  # Suppressing the output completely
        elif error_code in [2104, 2106, 2158]:
            # These are not errors, just information about data farm connections
            pass  # Suppressing the output completely
        else:
            print(f"Error {error_code}, reqId {req_id}: {error_string}, contract: {contract}")

    def __enter__(self):
        """Enter the runtime context related to this object and connect to IB."""
        self.connect_to_ib()  # Connect to IB
        return self  # Return the instance so that `self.ib` can be accessed

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and disconnect from IB."""
        self.disconnect_from_ib()  # Disconnect from IB


class USSimpleYieldCurve:
    """Simple US Zero coupon yield curve for today up to one year"""
    # Simple Zero yield curve built from TBill discount yields and effective fund rate
    # This is a simplified approximation for a full term structure model
    # Consider improving by building fully specified yield curve model using
    # Quantlib
    def __init__(self):
        end = datetime.now()
        start = end - timedelta(days=10)
        zero_rates = web.DataReader(['DFF', 'DTB4WK', 'DTB3', 'DTB6', 'DTB1YR', 'DGS2'], 'fred', start, end)
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