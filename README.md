
# Historical simulation engine is deprecated as I rebuild the daily production application

The stream directory contains all the code needed to display a bokeh application displaying a live list of SPXW options to trade according algorithm described below


# Volatility Trading and Historical Simulation Engine

Project contains classes and functions used to build historical simulations and daily trading estimates for two equity index volatility based investment strategies: 

  - Equity index options: Algorithm is drawn from JUREK, J. W. and STAFFORD, E. (2015),  [The Cost of Capital for Alternative Investments](https://www.hbs.edu/faculty/Publication%20Files/Cost%20of%20Capital%20for%20Alternative%20Investments_57a4f444-65fa-4f0c-b51a-116408f1dab9.pdf) The Journal of Finance
  
  - VIX futures: Algorithm is drawn from Cheng, I-H. (2018), [The VIX Premium](https://ssrn.com/abstract=2495414). Review of Financial Studies, Forthcoming. 


### Prerequisites

Requires active session of Interactive Brokers TWS or IB Gateway 

Requires directory structure defined in UpdateSP500Data class

### Run the live option stream with Docker

Install and start Docker Desktop, and wait until its engine is running. Then start and log in to Trader Workstation (TWS) or IB Gateway on your Mac. Enable API socket connections in TWS. From the repository root, run:

```sh
docker compose up --build
```

Open http://localhost:5006/main in your browser. Compose mounts the app's existing `~/Library/Mobile Documents/com~apple~CloudDocs/localDB/config.plist` read-only; it does not copy the file into the image. If your plist is elsewhere, set `OPT_TRADE_CONFIG_PLIST_PATH` to its absolute path before running Compose.

The defaults target live TWS on port 7496 or live IB Gateway on port 4001. The app tries Gateway first. For a paper session, use:

```sh
OPT_TRADE_TWS_PORT=7497 OPT_TRADE_IB_GATEWAY_PORT=4002 docker compose up --build
```

The container reaches TWS through Docker Desktop's [`host.docker.internal`](https://docs.docker.com/desktop/features/networking/networking-how-tos/). If TWS rejects the connection, check its API socket port and follow [IBKR's remote API connection settings](https://www.interactivebrokers.com/docs/tws-api/doc/connectivity/remote-tws-api-connections-with-trader-workstation): turn off "Allow connection from localhost only" and add the IP shown in the TWS connection prompt to Trusted IPs. The app only reads data, so TWS's Read-Only API setting can stay enabled. Keep TWS running and logged in; Docker packages only this Bokeh app. Live SPX, VIX, and SPXW quotes still require the corresponding IBKR market data permissions.

Docker keeps the Conda packages from `environment.yml` together with the app and makes it easy to run on another machine with the same setup. The first build downloads those packages and can be large; source or dependency changes require another build. Stop the app with Ctrl+C or `docker compose down`.

If Compose reports that `~/.docker/run/docker.sock` does not exist, Docker Desktop's engine has not started. Open Docker Desktop or run `docker desktop start`, wait for it to finish, then retry `docker compose up --build`. Run `docker info` to check that the engine responds.

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
