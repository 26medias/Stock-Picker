import pandas as pd
import skopt
import matplotlib.pyplot as plt
import pandas as pd
import os
import neptune

from Sim import Sim
from Analysis import Analysis


if not os.path.exists('data'):
	os.makedirs('data')


neptune.init('leap-forward/sandbox')
neptune.create_experiment('sim-opt-2', upload_source_files=['*.py'])


def train_evaluate(search_params):
	print('------------')
	print(search_params)
	
	sim    = Sim(neptune=neptune, period='1y', timedelay=100, window=100, timestep=1, budget=5000, stockPicks=5, avoidDowntrends=True, sellAllOnCrash=False, **search_params)
	stats  = sim.run()

	analysis = Analysis(neptune=neptune, stats=stats, positions=sim.portfolio.holdings, prices=sim.downloader.prices)
	analysis.chart()
	output, advanced_stats = analysis.positionStats()
	
	print(output)
	
	#neptune.log_artifact('data/output_1y.pkl')
	sharpe = analysis.sharpe()
	stats = sim.portfolio.summary()
	
	neptune.log_metric('sharpe', sharpe)
	neptune.log_metric('start_value', 5000)
	neptune.log_metric('end_value', stats['total_value'])
	
	return sharpe



SEARCH_PARAMS = {'sl': -0.04, 'tp': 3.0, 'ts': 0.05, 'ts_threshold': 0.05}

SPACE = [
	skopt.space.Real(-0.2, -0.01, name='sl'),
	skopt.space.Real(0.005, 3.0, name='tp'),
	skopt.space.Real(0.005, 0.5, name='ts'),
	skopt.space.Real(0.005, 0.25, name='ts_threshold')
]


@skopt.utils.use_named_args(SPACE)
def objective(**params):
	return -1*train_evaluate(params)


results = skopt.forest_minimize(objective, SPACE, n_calls=30, n_random_starts=10)

print(results)