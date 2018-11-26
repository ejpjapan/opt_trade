from lxml import etree
from datetime import datetime
import pandas as pd


class XmlConverter(object):
    """
    Converts XML to pandas dataframe.

    Example Usage:

    converter = XmlConverter()
    converter.parse()
    df = converter.build_dataframe()
    """

    def __init__(self, input_path='feds200628.xls', first_header='SVENY01', last_header='TAU2'):
        """
        Constructs a new XMLConverter instance.

        Inputs:
        - input_path: Path to input xml file.
        - first_header: String containing first header.
        - last_header: String containing last header.
        """

        self._first_header = first_header
        self._last_header = last_header
        self._header_list = []
        self._table_reached = False
        self._input_path = input_path
        self._parser = etree.XMLParser(target=self)
        self._in_headers = False
        self._in_data_tag = False
        self._in_actual_data = False
        self._data = {}
        self._is_possibly_empty = False

    def parse(self):
        """
        Parses xml file to generate dictionary containing data.
        """
        etree.parse(self._input_path, self._parser)

    def start(self, tag, attrib):
        """
        Callback function for XMLParser event of start of a new tag.

        Inputs:
        - tag: String containing the tag name.
        - attrib: String containing the attributes for this tag.
        """
        if tag.split('}', 1)[1] == 'Data':
            self._in_data_tag = True

        if self._in_actual_data:
            self._is_possibly_empty = True
        
    def end(self, tag):
        """
        Callback function for XMLParser event of end of a tag
        
        Inputs:
        - tag: String containing the tag name.
        """
        if self._is_possibly_empty:
            self._is_possibly_empty = False                    
            self._data[self._curr_date_idx].append('0')

        self._in_data_tag = False

    def data(self, data):
        """
        Callback function for XMLParser event of data of a tag

        Inputs:
        - data: String containing the text data for this tag.
        """
        if self._in_data_tag == True:
            if self._in_headers:
                if data == self._last_header:
                    self._in_headers = False
                    self._header_list.append(data)
                    self._in_actual_data = True

                    return

                self._header_list.append(data)

                return

            if data == self._first_header:
                self._in_headers = True
                self._header_list.append(data)

                return
            
            if self._in_actual_data:
                self._is_possibly_empty = False                    

                try:
                    datetime.strptime(data, '%Y-%m-%d')
                    self._curr_date_idx = data
                    self._data[self._curr_date_idx] = []

                except ValueError:
                    self._data[self._curr_date_idx].append(data)

    def close(self):
        """
        Callback function for XMLParser event of close.
        """
        pass

    def build_dataframe(self):
        """
        Builds a pandas dataframe.
        """
        df = pd.DataFrame.from_dict(self._data, orient='index')
        df.columns = self._header_list
        df = df.set_index(pd.to_datetime(df.index))
        df = df.astype(float)

        return df