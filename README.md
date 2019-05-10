# Volatility Trading and Historical Simulation Engine

Project contains classes and functions used to build historical simulations and daily trading estimates for two equity index volatility based investment strategies: 

  - Equity index options: Algorithm is drawn from JUREK, J. W. and STAFFORD, E. (2015),  [The Cost of Capital for Alternative Investments](https://www.hbs.edu/faculty/Publication%20Files/Cost%20of%20Capital%20for%20Alternative%20Investments_57a4f444-65fa-4f0c-b51a-116408f1dab9.pdf) The Journal of Finance
  
  - VIX futures: Algorithm is drawn from Cheng, I-H. (2018), [The VIX Premium](https://ssrn.com/abstract=2495414). Review of Financial Studies, Forthcoming. 


### Prerequisites

Requires active session of Interactive Brokers TWS or IB Gateway 

Requires directory structure defined in UpdateSP500Data class

### Example

Equity index option simulation example
```python
z_score_strike = -1
option_life_in_months = 2
from option_simulation import OptionSimulation
opt_sim = OptionSimulation(update_simulation_data=False)
sim_output = [opt_sim.trade_sim(z_score_strike, option_life_in_months, trade_day_type=day_type) \
                    for day_type in ['EOM', (0,22)]
```
Equity index option [notebooks](https://github.com/ejpjapan/jupyter_nb/tree/master/spx/)

VIX futures [notebooks](https://github.com/ejpjapan/jupyter_nb/tree/master/vixp/)

## Authors

* **Edmund Bellord** - [ejpjapan](https://github.com/ejpjapan/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/ejpjapan/opt_trade/blob/master/LICENSE) file for details
