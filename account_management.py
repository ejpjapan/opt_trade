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
transfers.reportDate = pd.to_datetime(transfers.reportDate)
transfers = transfers[['assetCategory', 'reportDate', 'positionAmountInBase',
                       'cashTransfer', 'direction']]
transfers['total'] = transfers.positionAmountInBase.astype(np.float) \
                     + transfers.cashTransfer.astype(np.float)
transfers = transfers.groupby('reportDate').sum()

# Transfers total
all_transfers = transfers.add(cash_trans, fill_value=0).fillna(0).sum(axis=1)


# Change in NAV
node = stmt.EquitySummaryInBase
daily_nav = pd.DataFrame([dict(zip(c.keys(), c.values())) for
                         c in node.getchildren()])
daily_nav.reportDate = pd.to_datetime(daily_nav.reportDate)
daily_nav = daily_nav.set_index(daily_nav.reportDate)
daily_nav['total'] = daily_nav['total'].astype(np.float)
all_transfers = all_transfers.reindex(daily_nav.index, fill_value=0)

daily_nav['AdjTotal'] = daily_nav['total'].shift(1)
daily_nav['AdjTotal'] = daily_nav['AdjTotal'] + all_transfers
daily_nav['TotalReturn'] = daily_nav['total'] / daily_nav['AdjTotal'] - 1

