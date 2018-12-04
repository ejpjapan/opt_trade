from spx_data_update import UpdateSP500Data
import pandas_datareader.data as web
from implied_to_realized import VolatilityYZ, VolatilitySD
import pandas as pd
import pyfolio as pf
import statsmodels.formula.api as sm
import quandl
import numpy as np

[sp500, vix] = [web.get_data_yahoo(item, 'JAN-01-90') for item in ['^GSPC', '^VIX']]


sp_monthly_ret  = sp500['Adj Close'].resample('BM').bfill().dropna().pct_change().dropna()
# sp_monthly_ret  = np.log(sp500['Adj Close'].resample('BM').bfill().dropna() / sp500['Adj Close'].resample('BM').bfill().dropna().shift(1))
sp_monthly_ret = sp_monthly_ret.rename('sp5_ret')


vrp_data = pd.read_csv(UpdateSP500Data.DATA_BASE_PATH / 'xl' / 'vol_risk_premium.csv',
                       usecols=['VRP', 'EVRP', 'IV','RV','ERV'])

vrp_data = vrp_data.set_index(pd.date_range('31-jan-1990', '31-dec-2017',freq='BM'))
cape = quandl.get('MULTPL/SHILLER_PE_RATIO_MONTH', collapse='monthly')
regression_data = pd.concat([sp_monthly_ret, vrp_data.shift(1) / 100, cape.apply(np.log).shift(1).resample('BM').bfill()], axis=1)
regression_data = regression_data.dropna(axis=0, how='any')


annual_returns_dict = {}
for col_name in ['sp5_ret']:
    annual_returns_dict[col_name] = pf.timeseries.annual_return(regression_data[col_name], period='monthly')

regression_string = 'sp5_ret ~ VRP + RV + Value'
#
results = sm.ols(formula=regression_string, data=regression_data).fit()

results.summary()



# vol_yz= VolatilityYZ(22, 10, sp500)
# vol_sd= VolatilitySD(22, 10, sp500)
#
# vol_rf = (vix['Close']/100 - vol_sd.compute())
# vol_rf.plot()
# vol_rf_yz = (vix['Close']/100 - vol_yz.compute())
# vol_rf_yz.mean()



from option_simulation import OptionSimulation, OptionTrades
from time import time
before = time()
optsim = OptionSimulation(update_simulation_data=False)

dtfs = optsim.trade_sim(-1, 1, trade_type='EOM', option_type='P')

variable_leverage = pd.Series(np.linspace(1,2,len(dtfs[2])), index=dtfs[2])
opt_trade = OptionTrades(dtfs, 2)

opt_sim_index = pf.timeseries.cum_returns(opt_trade.returns[1], starting_value=100)

opt_sim_index = opt_sim_index.resample('BM').bfill().dropna().pct_change().dropna()
opt_sim_index = opt_sim_index.rename('options')


regression_data_2 = pd.concat([regression_data, opt_sim_index], axis=1)
regression_data_2 = regression_data_2.dropna(axis=0, how='any')


regression_string = 'options ~ ERV'
#
results = sm.ols(formula=regression_string, data=regression_data_2).fit()

results.summary()

# print('elapsed: ', after - before )
#



# optsim.trade_sim(1, 1, trade_type='THU3', option_type='C')
# rets_c = optsim.sell_option(2)
#
#
# total = rets_p[1] + rets_c[1]
# pf.plot_monthly_returns_heatmap(total)
# pf.timeseries.annual_return(total)
# pf.timeseries.annual_return(rets_c[1])

# from spx_data_update import UpdateSP500Data, get_dates
# from option_utilities import read_feather
# import pandas as pd
# import feather
#
#
# file_names ={'spot': 'sp500_close', 'sigma': 'vix_index', 'dividend_yield': 'sp500_dividend_yld'}
# input_path = UpdateSP500Data.TOP_LEVEL_PATH
#
# file_strings = [str(input_path / file_name) for file_name in file_names.values()]
#
#
#
# seconds_since_upate = time() - os.path.getmtime(fed_zero_feather)
pf.timeseries.perf_stats()
pf.utils.get_symbol_from_yahoo