import pandas as pd
from pathlib import Path
import plistlib
import pysftp
import zipfile
import os
import warnings
from time import time
import re
import requests
from io import StringIO
from datetime import datetime




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


def data_shop_login():
    return config_key('cbeoDataShop_dict')


def illiquid_equity(discount=0.5):
    return sum(config_key('illiquid_equity').values()) * discount


def config_key(dict_key: str):
    file_name = GetRawCBOEOptionData.DATA_BASE_PATH / 'config.plist'
    assert (file_name.is_file())
    f = open(str(file_name), 'rb')
    pl = plistlib.load(f)
    return pl[dict_key]



class GetRawCBOEOptionData:
    """Class for handling raw option data downloads and processing from the CBOE DataShop."""
    DATA_BASE_PATH = Path.home() / 'Library' / 'Mobile Documents' / 'com~apple~CloudDocs' / 'localDB'
    TOP_LEVEL_PATH = DATA_BASE_PATH / 'cboeRawVolData'
    OPTION_TYPES = ['P', 'C']  # Option types: P for Put, C for Call
    SUBSCRIPTION_STR = 'subscriptions/order_000012838/item_000016265/'  # Subscription string for data access
    SYMBOL_DEFINITION_FILE = 'OptionSymbolConversionHistory.xlsx'  # Excel file containing symbol conversion history

    def __init__(self, top_level_directory: Path):
        """
        Initialize with the directory path for storing the data.

        :param top_level_directory: Path object pointing to the top-level directory.
        """
        self.top_level_directory = top_level_directory
        # Load root symbol strings from the Excel file
        self.root_symbols_str = self._load_symbol_definitions

    @property
    def _load_symbol_definitions(self) -> list:
        """
        Load option symbol string definitions from an Excel file.

        :return: List of root symbol strings.
        """
        # Path to the symbol definition file
        root_symbols_file: Path = self.top_level_directory / self.SYMBOL_DEFINITION_FILE

        # Check if the file exists
        assert root_symbols_file.is_file(), f"{root_symbols_file} does not exist."

        # Load root symbols from the 'spxSymbols' sheet in the Excel file
        root_symbols_df: pd.DataFrame = pd.read_excel(root_symbols_file, sheet_name='spxSymbols', skiprows=[0],
                                        usecols=[0], names=['root_symbols'])

        # Strip whitespace and return as a list of strings
        return root_symbols_df['root_symbols'].dropna().str.strip().values.tolist()

    @staticmethod
    def open_sftp():
        """
        Open an SFTP connection to CBOE DataShop using stored credentials.

        :return: pysftp.Connection object for SFTP communication.
        :raises ConnectionError: If unable to establish a connection.
        """
        user_dict = data_shop_login()  # Retrieve login credentials
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None  # Disable host key verification for this connection

        # Suppress warning related to host key verification
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

        try:
            # Establish SFTP connection
            sftp = pysftp.Connection('sftp.datashop.livevol.com',
                                     username=user_dict['user'],
                                     password=user_dict['password'],
                                     cnopts=cnopts)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SFTP: {e}")

        return sftp

    @staticmethod
    def unzip_files(in_directory: Path, out_directory: Path):
        """
        Unzip all .zip files from the input directory into the output directory.

        :param in_directory: Path object for the input directory containing zip files.
        :param out_directory: Path object for the output directory to extract files to.
        """
        # Loop through all files in the input directory
        for item in os.listdir(in_directory):
            if item.endswith('.zip'):
                file_path = in_directory / item
                try:
                    # Extract the contents of the zip file
                    with zipfile.ZipFile(file_path) as zip_ref:
                        zip_ref.extractall(out_directory)
                except zipfile.BadZipFile as err:
                    print(f"Error extracting {item}: {err}")

    def __get_zip_files(self, output_directory: Path, order_string: str):
        """
        Download zip files from the SFTP server into the specified directory.

        :param output_directory: Path object where downloaded zip files will be stored.
        :param order_string: String for the SFTP folder containing the files.
        """
        # Establish SFTP connection
        sftp = self.open_sftp()

        # Download the directory from the server to the local directory
        sftp.get_d(order_string, output_directory, preserve_mtime=True)

        # List the files on the SFTP server
        sftp_file_list = sftp.listdir(order_string)

        # Print the names of the downloaded files
        for file in sftp_file_list:
            if file.endswith('.zip'):
                print(f"Downloading... {file}")

        sftp.close()  # Close the SFTP connection

    def get_subscription_files(self, output_directory: Path):
        """
        Download the subscription files from the CBOE DataShop SFTP server.

        :param output_directory: Path object where the downloaded files will be saved.
        """
        # Ensure the output directory exists, if not, create it
        if not output_directory.is_dir():
            output_directory.mkdir(parents=True)

        # Download the zip files
        self.__get_zip_files(output_directory, self.SUBSCRIPTION_STR)

    def update_data_files(self, temporary_file_directory: Path) -> bool:
        """
        Download, unzip, process, and update option data if not already up-to-date.

        :param temporary_file_directory: Path object for temporary storage of raw files.
        :return: True if data files were updated, False otherwise.
        """
        feather_directory = self.top_level_directory / 'feather'  # Directory for processed data
        assert feather_directory.is_dir(), f"{feather_directory} does not exist."
        assert temporary_file_directory.is_dir(), f"{temporary_file_directory} does not exist."

        # Get the most recent business day
        latest_business_date = pd.to_datetime('today') - pd.tseries.offsets.BDay(1)

        # Retrieve the list of available option dates from the existing data
        opt_dates_all = get_dates(feather_directory)

        # Check if the data is up-to-date
        if opt_dates_all[-1].date() != latest_business_date.date():
            print('Downloading Option data from CBOE...')
            start_time = time()

            # Download and process the data files
            self.get_subscription_files(temporary_file_directory)
            self.unzip_files(temporary_file_directory, temporary_file_directory)
            self.csv_to_feather(temporary_file_directory, feather_directory)

            end_time = time()
            print(f"Option files updated in {round(end_time - start_time)} seconds.")
            return True
        else:
            print('Option files are up-to-date.')
            return False

    def csv_to_feather(self, in_directory: Path, out_directory: Path, archive_files=True):
        """
        Convert CSV files to Feather format and optionally archive the original files.

        :param in_directory: Path object containing CSV files.
        :param out_directory: Path object where Feather files will be stored.
        :param archive_files: Boolean flag indicating whether to archive the original files.
        """
        zip_archive_directory = self.top_level_directory / 'zip'  # Directory for zip file archives
        csv_archive_directory = self.top_level_directory / 'csv'  # Directory for csv file archives

        # Ensure the output directory exists
        if not out_directory.is_dir():
            out_directory.mkdir(parents=True)

        # Compile a regex pattern for filtering option symbols
        regex_pattern = '|'.join(self.root_symbols_str)

        # Process each CSV file in the input directory
        for item in os.listdir(in_directory):
            if item.endswith('.csv'):
                file_path = in_directory / item
                option_df = pd.read_csv(file_path)

                # Convert quote_date and expiration to datetime format
                option_df[['quote_date', 'expiration']] = option_df[['quote_date', 'expiration']].apply(pd.to_datetime)

                # Ensure option_type is uppercase
                option_df['option_type'] = option_df['option_type'].str.upper()

                # Remove rows with SPXW root symbol and filter by root_symbols
                option_df = option_df[~option_df['root'].str.contains('SPXW')]
                option_df = option_df[option_df['root'].str.contains(regex_pattern)]

                # Save data by option type (P for Put, C for Call) in Feather format
                for option_type in self.OPTION_TYPES:
                    df_filtered = option_df[option_df['option_type'] == option_type]
                    file_name = f"{os.path.splitext(item)[0]}_{option_type}.feather"
                    df_filtered.reset_index().to_feather(out_directory / file_name)

        # Archive the original zip and csv files if required
        if archive_files:
            self._archive_files(in_directory, csv_archive_directory, zip_archive_directory)

    @staticmethod
    def _archive_files(in_directory: Path, csv_archive_directory: Path, zip_archive_directory: Path):
        """
        Archive CSV and ZIP files by moving them to the archive directories.

        :param in_directory: Path object for the directory containing files to be archived.
        :param csv_archive_directory: Path object where CSV files will be moved.
        :param zip_archive_directory: Path object where ZIP files will be moved.
        """
        # Move files to their respective archive directories
        for item in os.listdir(in_directory):
            file_path = in_directory / item
            if item.endswith('.csv'):
                file_path.rename(csv_archive_directory / item)
            elif item.endswith('.zip'):
                file_path.rename(zip_archive_directory / item)
            else:
                file_path.unlink()  # Remove other non-relevant files



class DividendYieldScraper:
    def __init__(self, url='https://www.multpl.com/s-p-500-dividend-yield/table/by-month'):
        self.url = url
        self.dy_df = self._fetch_and_clean_data()

    def _fetch_and_clean_data(self):
        """Fetches and cleans the dividend yield data from the given URL."""
        # Package the request, send the request and catch the response
        response = requests.get(self.url)

        # Set the correct encoding
        response.encoding = 'utf-8'

        # Wrap the HTML content in a StringIO object
        raw_html_tbl = pd.read_html(StringIO(response.text))

        # Access the first table
        dy_df = raw_html_tbl[0]

        # Convert the 'Date' column to datetime format
        dy_df['Date'] = pd.to_datetime(dy_df['Date'], format='%b %d, %Y')

        # Set 'Date' as the index
        dy_df.set_index('Date', drop=True, inplace=True)

        # Clean the 'Value' column by removing '†' and '%' symbols
        dy_df['Value'] = dy_df['Value'].str.replace('†', '').str.replace('%', '').str.strip()

        # Convert the cleaned 'Value' column to numeric
        dy_df['Value'] = pd.to_numeric(dy_df['Value'])

        # Filter out future end-of-month dates
        today = datetime.today()
        dy_df = dy_df[~((dy_df.index.is_month_end) & (dy_df.index > today))]

        return dy_df

    @property
    def latest_yield(self):
        """Returns the latest available dividend yield."""
        return self.dy_df['Value'].iloc[0]

    @property
    def full_history(self):
        """Returns the full history of dividend yields."""
        return self.dy_df


def main():
    # try:
    raw_file_updater = GetRawCBOEOptionData(GetRawCBOEOptionData.TOP_LEVEL_PATH)
    raw_file_updater.update_data_files(GetRawCBOEOptionData.TOP_LEVEL_PATH / 'test')


if __name__ == '__main__':
    main()
