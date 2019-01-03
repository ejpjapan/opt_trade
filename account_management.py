import pandas as pd
from pathlib import Path
import lxml.objectify as lxml_objectify
import numpy as np

# etree = ET.parse() #create an ElementTree object
file_name = 'daily_nav copy.xml'


tree = lxml_objectify.parse(open(Path.home() / 'Downloads' / file_name, 'rb'))
root = tree.getroot()
stmt = root.FlexStatements.FlexStatement

# Withdrawals and deposits
node = stmt.CashTransactions
cash_trans = pd.DataFrame([dict(zip(c.keys(), c.values())) for
                         c in node.getchildren()])
cash_trans.amount = cash_trans.amount.astype(np.float64)
cash_trans.dateTime = pd.to_datetime(cash_trans.dateTime)
cash_trans = cash_trans.groupby('dateTime').sum()

# Transfers
node = stmt.Transfers
transfers = pd.DataFrame([dict(zip(c.keys(), c.values())) for
                         c in node.getchildren()])

