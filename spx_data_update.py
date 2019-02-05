import os
import plistlib
import re
import zipfile
from ftplib import FTP
from pathlib import Path
from time import time
import feather
import pandas as pd
import numpy as np
# import pandas_datareader.data as web
import quandl
from option_utilities import USZeroYieldCurve, write_feather, read_feather
from ib_insync import IB, util, Index


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
    SUBSCRIPTION_STR = '/subscriptions/order_000004576/item_000006417/'
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
    def __open_ftp():
        user_dict = data_shop_login()
        "Open ftp connection to CBOE datashop"
        ftp = FTP(host='ftp.datashop.livevol.com',
                  user=user_dict['user'],
                  passwd=user_dict['password'])
        return ftp

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
        """Download zip all zip files from order_string to output_directory"""
        ftp = self.__open_ftp()
        ftp.cwd(order_string)
        ftp_file_list = ftp.nlst()
        for file in ftp_file_list:
            if file.endswith('.zip'):
                print("Downloading..." + file)
                ftp.retrbinary("RETR " + file, open(output_directory / file, 'wb').write)
        ftp.close()

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
                # Think about improving this with regex
                option_df = option_df[~option_df['root'].str.contains('SPXW')]
                # Create new column of days2Expiry
        #       option_df['days2Expiry'] = option_df['expiration'] - option_df['quote_date']
        #       timedelata int64 not stored in feather
        #       option_df = option_df[option_df['root'] == rootSymbols]
                option_df = option_df[option_df['root'].str.contains(regex_pattern)]
                for option_type in self.OPTION_TYPES:
                    df2save = option_df[option_df['option_type'] == option_type]
                    file_name = os.path.splitext(item)[0] + '_' + option_type + '.feather'
                    feather.write_dataframe(df2save, str(out_directory / file_name))
        if archive_files:
            # This makes sure we keep the archive - we will be missing zip and csv
            # files from March 24th 2018 to March October 10 2018 - will need to re-purchase
            # if we want to run analysis on weekly options
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


def get_daily_close(in_dates: pd.DatetimeIndex, in_dir: str):
    "Retrieve closing price for S&P 500"
    daily_close = np.empty(len(in_dates))
    for i, item in enumerate(in_dates):
        dtf = feather.read_dataframe(in_dir + 'UnderlyingOptionsEODCalcs_' +
                                     item.strftime(format='%Y-%m-%d') + '_P' + '.feather')
        daily_close[i] = dtf['underlying_bid_eod'][0]
    daily_close = pd.DataFrame(data=daily_close, index=in_dates, columns=['sp500_close'])
    return daily_close


def get_dates(feather_directory):
    """ Fetch dates from feather file names
    :rtype: pd.DatetimeIndex
    """
    regex_pattern = r'\d{4}-\d{2}-\d{2}'  # this will fail if month>12 or days>31
    opt_dates_list = []
    for item in os.listdir(feather_directory):  # loop through items in dir
        if item.endswith('.feather'):
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
    ib = IB()
    ib.connect('127.0.0.1', port=4001, clientId=30)
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


# def data_shop_login():
#     """Get CBOE Datashop password and user name from config plist"""
#     file_name = UpdateSP500Data.DATA_BASE_PATH / 'config.plist'
#     assert(file_name.is_file())
#     f = open(str(file_name), 'rb')
#     pl = plistlib.load(f)
#     return pl['cbeoDataShop_dict']


# def quandle_api():
#     """Get Quandl API key"""
#     file_name = UpdateSP500Data.DATA_BASE_PATH / 'config.plist'
#     assert(file_name.is_file())
#     f = open(str(file_name), 'rb')
#     pl = plistlib.load(f)
#     return pl['Quandl']


def quandle_api():
    return config_ket('Quandl')


def data_shop_login():
    return config_ket('cbeoDataShop_dict')


def config_ket(dict_key: str):
    file_name = UpdateSP500Data.DATA_BASE_PATH / 'config.plist'
    assert (file_name.is_file())
    f = open(str(file_name), 'rb')
    pl = plistlib.load(f)
    return pl[dict_key]


def feather_clean(in_directory):
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
            feather.write_dataframe(option_df, str(in_directory / item))


def main():
    _ = UpdateSP500Data()


if __name__ == '__main__':
    main()
