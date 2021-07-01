import os
import re
import zipfile
# from ftplib import FTP
import pysftp
from pathlib import Path
from time import time
# import feather
import pandas as pd
import numpy as np
import quandl
from scipy.io import loadmat
from pyfolio.timeseries import cum_returns
from urllib.request import urlretrieve
import plistlib
import nest_asyncio

from option_utilities import USZeroYieldCurve, write_feather, read_feather, matlab2datetime, get_asset
from ib_insync import IB, util, Index
from twilio.rest import Client


class SMSMessage:
    account_sid = 'AC51119e549b9cee8945cc432d27dfa7f8'
    twilio_sms_number = '+13342343055'

    def __init__(self, sms_text='This message is empty'):
        client = Client(self.account_sid, config_key('twilio_token'))
        message = client.messages \
            .create(
                body=sms_text,
                from_=self.twilio_sms_number,
                to=config_key('cell_number')
                    )
        print(message.sid)


class UpdateSP500Data:
    DATA_BASE_PATH = Path.home() / 'Library' / 'Mobile Documents' / 'com~apple~CloudDocs' / 'localDB'
    TOP_LEVEL_PATH = DATA_BASE_PATH / 'cboeRawVolData'

    def __init__(self):
        # Check basic file structure exists, if not create it
        path_list = [self.TOP_LEVEL_PATH]
        path_list.extend([self.TOP_LEVEL_PATH / sub_directory
                          for sub_directory in ['zip', 'csv', 'feather']])
        for directory in path_list:
            if not os.path.isdir(directory):
                print('Warning: {0} does not exist - creating it'.format(str(directory)))
                os.mkdir(directory)

        self.GetRawOptionData = GetRawCBOEOptionData(self.TOP_LEVEL_PATH)
        self.GetRawOptionData.update_data_files(self.TOP_LEVEL_PATH / 'test')

        self.ImpliedVol = ImpliedVolatilityHistory()
        self.DividendYieldHistory = DividendYieldHistory()
        self.usZeroYldCurve = USZeroYieldCurve(update_data=True)
        self.ClosingPriceHistory = ClosingPriceHistory(self.TOP_LEVEL_PATH / 'feather')
        self.save_data()

    def save_data(self):
        self.ImpliedVol.save_vix_df(self.TOP_LEVEL_PATH)
        self.DividendYieldHistory.save_dividend_yield_df(self.TOP_LEVEL_PATH)
        self.ClosingPriceHistory.save_daily_close(self.TOP_LEVEL_PATH)


class GetRawCBOEOptionData:
    OPTION_TYPES = ['P', 'C']
    # Need to update this string each year for subscription renewal
    if pd.datetime.today().date() > pd.to_datetime('20-Mar-2021').date():
        print('Warning - Update subscription string for SPX from CBOE Datashop')
        exit(0)
    SUBSCRIPTION_STR = 'subscriptions/order_000012838/item_000016265/'

    # SUBSCRIPTION_STR = '/subscriptions/order_000008352/item_000011077/'
    # SUBSCRIPTION_STR = 'order_000008421/item_000011148/'

    SYMBOL_DEFINITION_FILE = 'OptionSymbolConversionHistory.xlsx'

    def __init__(self, top_level_directory):

        self.top_level_directory = top_level_directory
        # Specific to SPX - Get option symbol string
        root_symbols_file = self.top_level_directory / self.SYMBOL_DEFINITION_FILE
        assert (root_symbols_file.is_file())
        root_symbols_df = pd.read_excel(root_symbols_file, sheet_name='spxSymbols', skiprows=[0],
                                        usecols=[0], index_col=None, names=['root_symbols'])
        self.root_symbols_str = root_symbols_df['root_symbols'].dropna().str.strip().values

    @staticmethod
    def open_sftp():
        user_dict = data_shop_login()
        "Open ftp connection to CBOE datashop"
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        sftp = pysftp.Connection('sftp.datashop.livevol.com',
                                 username=user_dict['user'],
                                 password=user_dict['password'],
                                 cnopts=cnopts)
        # ftp = FTP(host='ftp.datashop.livevol.com',
        #           user=user_dict['user'],
        #           passwd=user_dict['password'])
        return sftp

    @staticmethod
    def unzip_file(in_directory, out_directory):
        """Unzip files to csv """
        for item in os.listdir(in_directory):  # loop through items in dir
            if item.endswith('.zip'):
                file_name = in_directory / item  # get full path of files
                zip_ref = zipfile.ZipFile(file_name)  # create zipfile object
                try:
                    zip_ref.extractall(out_directory)  # extract file to dir
                except zipfile.BadZipFile as err:
                    print("Zipfile error: {0} for {1}".format(err, item))
                zip_ref.close()  # close file

    def __get_zip_files(self, output_directory, order_string):
        """Download zip files from order_string to output_directory"""
        sftp = self.open_sftp()
        sftp.get_d(order_string, output_directory, preserve_mtime=True)
        sftp_file_list = sftp.listdir(order_string)
        # ftp.cwd(order_string)
        # ftp_file_list = ftp.nlst()
        for file in sftp_file_list:
            if file.endswith('.zip'):
                print("Downloading..." + file)
        sftp.close()

    def get_subscription_files(self, output_directory: Path):
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)
        assert(output_directory.is_dir())
        self.__get_zip_files(output_directory, self.SUBSCRIPTION_STR)

    def update_data_files(self, temporary_file_directory):
        """ Download zip files from CBOE, unzip to csv, process and turn into feather
        TODO: Should be in separate simulation data update & fetch class that creates/updates database
         :rtype: Bool"""
        feather_directory = self.top_level_directory / 'feather'
        assert(feather_directory.is_dir())
        assert temporary_file_directory.is_dir(), '{} directory does not exist'.format(temporary_file_directory)
        latest_business_date = pd.to_datetime('today') - pd.tseries.offsets.BDay(1)
        opt_dates_all = get_dates(feather_directory)
        if opt_dates_all[-1].date() != latest_business_date.date():
            start_time = time()
            print('Downloading Option data from CBOE')
            self.get_subscription_files(temporary_file_directory)
            self.unzip_file(temporary_file_directory, temporary_file_directory)
            self.csv2feather(temporary_file_directory, feather_directory)
            end_time = time()
            files_updated = True
            print('Option files updated in: ' + str(round(end_time - start_time)) + ' seconds')
        else:
            files_updated = False
            print('Option files not updated')
        return files_updated

    def csv2feather(self, in_directory, out_directory, archive_files=True):
        """Open raw csv files, remove weekly options and all options not in
        root_symbols_file build dataframe and convert to feather
        archive zip and csv files"""
        zip_archive_directory = self.top_level_directory / 'zip'
        csv_archive_directory = self.top_level_directory / 'csv'
    # Check/create output directory
        if not os.path.isdir(out_directory):
            os.mkdir(out_directory)
        # list of all files in directory (includes .DS_store hidden file)
        regex_pattern = '|'.join(self.root_symbols_str)
        for item in os.listdir(in_directory):  # loop through items in dir
            if item.endswith('.csv'):
                option_df = pd.read_csv(in_directory / item)
                # Convert quote_date and expiration to datetime format
                option_df[['quote_date', 'expiration']] = option_df[['quote_date', 'expiration']].apply(pd.to_datetime)
                # Convert option type to upper cap
                option_df['option_type'] = option_df['option_type'].apply(str.upper)
                # Remove SPXW because its the only root that contains SPX
                option_df = option_df[~option_df['root'].str.contains('SPXW')]
                # Create new column of days2Expiry
                option_df = option_df[option_df['root'].str.contains(regex_pattern)]
                for option_type in self.OPTION_TYPES:
                    df2save = option_df[option_df['option_type'] == option_type]
                    file_name = os.path.splitext(item)[0] + '_' + option_type + '.feather'
                    #
                    # feather.write_dataframe(df2save, str(out_directory / file_name))
                    df2save.to_feather(str(out_directory / file_name))
        if archive_files:
            # This makes sure we keep the archive - we will be missing zip and csv
            for item in os.listdir(in_directory):
                if item.endswith('.csv'):
                    os.rename(in_directory / item, str(csv_archive_directory / item))
                elif item.endswith('.zip'):
                    os.rename(in_directory / item, str(zip_archive_directory / item))
                else:
                    os.remove(in_directory / item)


class ImpliedVolatilityHistory:

    def __init__(self):
        vix = get_vix()
        self.implied_vol_index = vix.rename('vix_index')

    def save_vix_df(self, out_directory: Path, file_name='vix_index'):
        write_feather(self.implied_vol_index.to_frame(), str(out_directory / file_name))


class DividendYieldHistory:

    def __init__(self):
        dy_monthly = get_sp5_dividend_yield()
        self.dy_monthly = dy_monthly.rename(columns={"Value": "Yield Value"})

    def save_dividend_yield_df(self, out_directory: Path, file_name='sp500_dividend_yld'):
        # dividend_yield_df = self.dy_monthly.to_frame()
        write_feather(self.dy_monthly, str(out_directory / file_name))


class ClosingPriceHistory:

    def __init__(self, feather_directory):
        self.option_data_dates = get_dates(feather_directory)
        self.daily_close = get_daily_close(self.option_data_dates, str(feather_directory) + '/')

    def save_daily_close(self, output_directory):
        write_feather(self.daily_close, str(output_directory / 'sp500_close'))


class VixTSM:
    def __init__(self, expiry_type=0):
        """ Class to retrieve tsm vix futures data and create return and index series"""
        try:
            raw_tsm = loadmat('/Volumes/ExtraStorage/base/db/fut/vix.mat')
        except FileNotFoundError:
            raw_tsm = loadmat(str(UpdateSP500Data.DATA_BASE_PATH / 'mat' / 'vix.mat'))
        python_dates = matlab2datetime(raw_tsm['t'].squeeze())
        column_names = [item[0] for item in raw_tsm['h'][:, 0]]
        raw_x_data = np.round(raw_tsm['x'], 4)
        self.raw_tsm_df = pd.DataFrame(data=raw_x_data, index=python_dates, columns=column_names)
        self.raw_tsm_df = self.raw_tsm_df.iloc[:-1, :]  # remove last row
        self.start_date = self.raw_tsm_df.index[0]
        self.expiry_type = expiry_type  # expiry_type is either string or positive integer
        self.rolled_return, self.rolled_expiries, self.days_2_exp, self.rolled_future = self._rolled_future_return()

    def _rolled_future_return(self):
        """Returns arithmetic return from long position in vix future"""
        expiry_dates = pd.to_datetime(self.raw_tsm_df['exp1'].astype(int), format='%Y%m%d')
        returns = self._expiry_returns
        days_2_exp = self._expiration_days_2_expiry
        if self.expiry_type == 'eom':
            eom_dates = returns.index[returns.reset_index().groupby(returns.index.to_period('M'))['index'].idxmax()]
            last_month_end = eom_dates[-1] + pd.offsets.MonthEnd(0)
            eom_dates = eom_dates[:-1]
            eom_dates = eom_dates.insert(-1, last_month_end)
            roll_dates = eom_dates.sort_values()
        else:
            # TODO: add checks to make sure roll_dates are subset of return index dates
            expiry_dates_unique = pd.to_datetime(self.raw_tsm_df['exp1'].unique().astype(int), format='%Y%m%d')
            roll_dates = expiry_dates_unique - pd.offsets.BDay(self.expiry_type)

        expiry_for_roll = []
        for dts in expiry_dates:
            idx = roll_dates.get_loc(dts, method='ffill')
            expiry_for_roll.append(roll_dates[idx])
        day_diff = expiry_dates.index - pd.DatetimeIndex(expiry_for_roll)
        front_month_bool = day_diff.days < 0
        back_month_bool = ~front_month_bool

        rolled_return = pd.concat([returns['close2'][back_month_bool], returns['close1'][front_month_bool]],
                                  axis=0).sort_index()
        rolled_return[0] = np.nan  # replace first empty observation with NaN

        rolled_expiries = pd.concat([self.raw_tsm_df['exp2'][back_month_bool],
                                     self.raw_tsm_df['exp1'][front_month_bool]], axis=0).sort_index()

        days_2_exp = pd.concat([days_2_exp['exp2'][back_month_bool],
                                days_2_exp['exp1'][front_month_bool]], axis=0).sort_index()

        rolled_future = pd.concat([self.raw_tsm_df['close2'][back_month_bool],
                                   self.raw_tsm_df['close1'][front_month_bool]], axis=0).sort_index()

        return rolled_return, rolled_expiries, days_2_exp, rolled_future

    @property
    def _expiry_returns(self):
        """ Returns future arithmetic return if contracts are held to expiry"""
        close_cols = [col for col in self.raw_tsm_df.columns if 'close' in col]
        close = self.raw_tsm_df[close_cols].copy()
        roll_rows = self.raw_tsm_df['exp1'].diff() > 0  # Day after expiry
        returns = close.pct_change()
        # Cross the columns on the day after expiry
        column_shift_ret = close.divide(close.shift(periods=-1, axis='columns').shift(periods=1, axis='rows')) - 1
        returns[roll_rows] = column_shift_ret[roll_rows]
        return returns

    @property
    def _expiration_days_2_expiry(self):
        """Returns number of days to expiry for each contract month"""
        # TODO: This is an approximation that assumes there is only one day between expiration date and last day of
        #  contract
        # exp_cols = [col for col in self.raw_tsm_df.columns if 'exp' in col]
        # expiries = self.raw_tsm_df[exp_cols].fillna(0).astype(int).apply(pd.to_datetime,
        #                                                                  format='%Y%m%d',
        #                                                                  errors='coerce')
        # expiry_dates = expiries.subtract(self.raw_tsm_df.index, axis=0)
        # # Dates in TSM are last trading day so add one day for expiration
        # expiry_dates = expiry_dates + pd.Timedelta(days=1)
        # expiry_days = pd.concat([expiry_dates[cols].dt.days for cols in expiry_dates.columns], axis=1)

        exp_cols = [col for col in self.raw_tsm_df.columns if 'exp' in col]
        expiries = self.raw_tsm_df[exp_cols].fillna(0).astype(int).apply(pd.to_datetime,
                                                                         format='%Y%m%d',
                                                                         errors='coerce')
        # Dates in TSM are last trading day so add one day for expiration
        expiry_list = [expiries[cols].add(pd.Timedelta(days=1)) for cols in expiries.columns]
        num_bus_days = [np.busday_count(item.index.values.astype('<M8[D]'), item.values.astype('<M8[D]')) for item in
                    expiry_list[:-1]]
        num_bus_days = pd.DataFrame(index=expiries.index, data=np.transpose(num_bus_days), columns=expiries.columns[:-1])
        return num_bus_days

    @property
    def rolled_idx(self):
        """Returns cumulative return index from long position in vix future"""
        start_idx = 100
        cumulative_returns = cum_returns(self.rolled_return, start_idx)
        # Add back start of index
        cumulative_returns[self.start_date] = start_idx
        idx = cumulative_returns.reindex(cumulative_returns.index.sort_values())
        return idx.rename('long_vix')

    @property
    def rolled_idx_short(self):
        """Returns cumulative return index from short position in vix future"""
        idx = 1 / self.rolled_idx
        idx = idx / idx[0] * 100
        return idx.rename('short_vix')

    @property
    def rolled_return_short(self):
        """Returns arithmetic return from short position in vix future"""
        return self.rolled_idx_short.pct_change()


class SP500Index:
    def __init__(self, **kwargs):
        return_index_list = get_asset({'^SP500TR': 'S&P 500TR', '^GSPC': 'S&P 500'}, **kwargs)
        return_index, price_index = [item for item in return_index_list]
        self.return_index = return_index[return_index.columns[0]]
        self.price_index = price_index[price_index.columns[0]]

    @property
    def excess_return_index(self):
        zeros = USZeroYieldCurve(update_data=True)
        cash_index = zeros.cash_index[self.return_index.index]
        excess_return_index = self.return_index / cash_index
        return excess_return_index / excess_return_index[0] * 100


class CBOEIndex:
    def __init__(self):
        db_directory = UpdateSP500Data.DATA_BASE_PATH / 'xl'
        cboe_dict = {'cboe_vvix': 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vvixtimeseries.csv',
                     'cboe_skew': 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/skewdailyprices.csv'}

        [urlretrieve(value, db_directory / str(key + '.csv')) for key, value in cboe_dict.items()]

        vvix, skew = [pd.read_csv(value, str(db_directory / str(key + '.csv')),
                                  skiprows=1,
                                  delimiter=',') for key, value in cboe_dict.items()]

        vvix['Date'], skew['Date'] = [df['Date'].apply(pd.to_datetime) for df in [vvix, skew]]

        vvix, skew = [df.set_index('Date') for df in [vvix, skew]]
        vvix, skew = [item[item.columns[0]] for item in [vvix, skew]]
        self.vvix, self.skew = [item.ffill() for item in [vvix, skew]]


def get_daily_close(in_dates: pd.DatetimeIndex, in_dir: str):
    """Retrieve closing price for S&P 500"""
    daily_close = np.empty(len(in_dates))
    for i, item in enumerate(in_dates):
        # dtf = feather.read_dataframe(in_dir + 'UnderlyingOptionsEODCalcs_' +
        #                              item.strftime(format='%Y-%m-%d') + '_P' + '.feather')

        dtf = pd.read_feather(in_dir + 'UnderlyingOptionsEODCalcs_' +
                                      item.strftime(format='%Y-%m-%d') + '_P' + '.feather')
        daily_close[i] = dtf['underlying_bid_eod'][0]
    daily_close = pd.DataFrame(data=daily_close, index=in_dates, columns=['sp500_close'])
    return daily_close


def get_dates(feather_directory, file_type='.feather'):
    """ Fetch dates from feather file names
    :rtype: pd.DatetimeIndex
    """
    regex_pattern = r'\d{4}-\d{2}-\d{2}'  # this will fail if month>12 or days>31
    opt_dates_list = []
    for item in os.listdir(feather_directory):  # loop through items in dir
        if item.endswith(file_type):
            date_string = re.search(regex_pattern, item)
            if date_string:
                opt_dates_list.append(date_string.group())
    opt_dates_list = list(set(opt_dates_list))
    opt_dates_all = pd.DatetimeIndex([pd.to_datetime(date_item, yearfirst=True,
                                                     format='%Y-%m-%d')
                                      for date_item in opt_dates_list])
    opt_dates_all = opt_dates_all.sort_values()
    return opt_dates_all


def get_vix():
    """Fetch vix from Interactive Brokers and append to history'''
    :return: Dataframe
    """
    ibw = IbWrapper()
    ib = ibw.ib
    vix = Index('VIX')
    cds = ib.reqContractDetails(vix)

    # contracts = [cd.contract for cd in cds]
    bars = ib.reqHistoricalData(cds[0].contract,
                                endDateTime='',
                                durationStr='1 Y',
                                barSizeSetting='1 day',
                                whatToShow='TRADES',
                                useRTH=True,
                                formatDate=1)
    ib.disconnect()
    vix = util.df(bars)
    vix = vix.set_index('date')
    vix.index = pd.to_datetime(vix.index)
    vix = vix[['open', 'high', 'low', 'close']]

    vix_history = read_feather(str(UpdateSP500Data.TOP_LEVEL_PATH / 'vix_history'))

    full_hist = vix.combine_first(vix_history)
    write_feather(full_hist, str(UpdateSP500Data.TOP_LEVEL_PATH / 'vix_history'))
    return full_hist['close']


def get_sp5_dividend_yield():
    """Fetch dividend yield from Quandl'''
    :return: Dataframe
    """
    quandl.ApiConfig.api_key = quandle_api()
    # try:
    spx_dividend_yld = quandl.get('MULTPL/SP500_DIV_YIELD_MONTH', collapse='monthly')
    spx_dividend_yld = spx_dividend_yld.resample('MS').bfill()
    # except:
    #     print('Quandl failed - Scraping dividend yield from Mutlp.com')
    # else:
    #     print('Quandl failed - Scraping dividend yield from Mutlp.com')
    #     spx_dividend_yld = scrape_sp5_div_yield()
    return spx_dividend_yld


def scrape_sp5_div_yield():
    """Scrape S&P 500 dividend yield from www.multpl.com
    :rtype: pd.Dataframe
    """
    url = 'http://www.multpl.com/s-p-500-dividend-yield/table?f=m'
    # Package the request, send the request and catch the response: r
    raw_html_tbl = pd.read_html(url)
    dy_df = raw_html_tbl[0]
    # Clear dataframe
    dy_df.columns = dy_df.iloc[0]
    dy_df = dy_df.drop([0])
    dy_df[dy_df.columns[0]] = pd.to_datetime(dy_df.loc[:, dy_df.columns[0]],
                                             format='%b %d, %Y')
    dy_df = dy_df.set_index(dy_df.columns[0])
    dy_df = dy_df[dy_df.columns[0]]
    spx_dividend_yld = pd.to_numeric(dy_df.str.replace('%', '').str.replace('estimate', '').str.strip())
    spx_dividend_yld = spx_dividend_yld.reindex(spx_dividend_yld.index[::-1])
    spx_dividend_yld = spx_dividend_yld.resample('MS').bfill()
    return spx_dividend_yld


def quandle_api():
    return config_key('Quandl')


def data_shop_login():
    return config_key('cbeoDataShop_dict')


def illiquid_equity(discount=0.5):
    return sum(config_key('illiquid_equity').values()) * discount


def config_key(dict_key: str):
    file_name = UpdateSP500Data.DATA_BASE_PATH / 'config.plist'
    assert (file_name.is_file())
    f = open(str(file_name), 'rb')
    pl = plistlib.load(f)
    return pl[dict_key]


def feather_clean(in_directory):
    """ Utility function to clean feather files"""
    # in_directory = UpdateSP500Data.TOP_LEVEL_PATH / 'feather'
    Path.is_dir(in_directory)
    all_files = os.listdir(in_directory)
    for item in all_files:
        if item.endswith('.feather'):
            # Remove options with strikes at 5$
            option_df = feather.read_dataframe(in_directory / item)
            idx = option_df['strike'] == 5
            option_df = option_df.drop(option_df.index[idx])
            # # Remove Quarterly options
            # idx2 = option_df['root'] == 'SPXQ'
            # option_df = option_df.drop(option_df.index[idx2])
            # # Remove Monthly options
            # idx2 = option_df['root'] == 'SPXM'
            # option_df = option_df.drop(option_df.index[idx2])
            # feather.write_dataframe(option_df, str(in_directory / item))
            option_df.to_feather(str(in_directory / item))


class IbWrapper:
    def __init__(self, client_id=30):
        """Wrapper function for Interactive Broker API connection"""
        self.ib = IB()
        nest_asyncio.apply()
        try:
            # IB Gateway
            self.ib.connect('127.0.0.1', port=4001, clientId=client_id)
        except ConnectionRefusedError:
            # TWS
            try:
                self.ib.connect('127.0.0.1', port=7496, clientId=client_id)
            # TWS Paper
            except ConnectionRefusedError:
                print('Warning- Connected to Paper Portfolio - Account Values are hypothetical')
                self.ib.connect('127.0.0.1', port=7497, clientId=client_id)


def main():
    try:
        raw_file_updater = GetRawCBOEOptionData(UpdateSP500Data.TOP_LEVEL_PATH)
        raw_file_updater.update_data_files(UpdateSP500Data.TOP_LEVEL_PATH / 'test')
        _ = SMSMessage('Option files downloaded')
    except Exception:
        _ = SMSMessage('CBOE Data download failed')
    try:
        USZeroYieldCurve(update_data=True)
    except Exception:
        _ = SMSMessage('Yield Curve download failed')


if __name__ == '__main__':
    main()
