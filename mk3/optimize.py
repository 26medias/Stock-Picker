import pandas as pd
import skopt
import matplotlib.pyplot as plt
import pandas as pd
import os
import neptune
import json
import math

from Sim import Sim
from Analysis import Analysis


if not os.path.exists('data'):
	os.makedirs('data')


hyperparameters_init={"sl": -0.06, "tp": 2, "ts": -0.02, "v_100d": 0.24531455566424112, "v_dfh": 0.05274405649047917, "v_rfl": 0.01703439651254952, "w_100d": 0.19102857670958306, "w_dfh": 0.5799832209142695, "w_sharpe": 0.655623531391156}


neptune.init('leap-forward/sandbox')
neptune.create_experiment('mk4_2y', upload_source_files=['*.py'])


def train_evaluate(search_params):
	
	hyperparameters = {}
	pick_kwargs = {}
	for k in list(search_params.keys()):
		if k in ['w_dfh','w_sharpe','w_100d','v_100d','v_dfh','v_rfl']:
			pick_kwargs[k] = search_params[k]
		else:
			hyperparameters[k] = search_params[k]
	
	hyperparameters['pick_kwargs'] = pick_kwargs
	print('------------')
	print(json.dumps(hyperparameters, indent=2, sort_keys=True))
	
	sim    = Sim(neptune=neptune, period='2y', timedelay=100, window=100, timestep=1, budget=5000, stockPicks=5, avoidDowntrends=True, sellAllOnCrash=False, **hyperparameters)
	stats  = sim.run()

	analysis = Analysis(neptune=neptune, stats=stats, positions=sim.portfolio.holdings, prices=sim.downloader.prices)
	#analysis.chart()
	output, advanced_stats, obj_stats = analysis.positionStats()
	
	for k in list(obj_stats.keys()):
		neptune.log_metric(k, obj_stats[k])
	
	print(output)
	
	#neptune.log_artifact('data/output_1y.pkl')
	sharpe = analysis.sharpe()
	stats = sim.portfolio.summary()
	
	if math.isnan(sharpe) or math.isinf(sharpe) or sharpe <= -2 or sharpe >= 5:
		sharpe = -5
	
	#neptune.log_metric('sharpe', sharpe)
	#neptune.log_metric('start_value', 5000)
	#neptune.log_metric('end_value', stats['total_value'])
	
	report = {
		'hyperparameters': hyperparameters,
		'sharpe': sharpe,
		'end_value': stats['total_value'],
		'gains': (stats['total_value']-5000.0)/5000.0
	}
	
	
	neptune.log_text('report', json.dumps(report, indent=2, sort_keys=True))
	
	return sharpe

	
SPACE = [
	skopt.space.Real(-0.2, -0.01, name='sl'),
	skopt.space.Real(0.005, 3.0, name='tp'),
	skopt.space.Real(-0.2, -0.005, name='ts'),
	skopt.space.Real(0.1, 0.5, name='v_100d'),
	skopt.space.Real(0.005, 0.25, name='v_dfh'),
	skopt.space.Real(0.005, 0.25, name='v_rfl'),
	skopt.space.Real(0.1, 0.8, name='w_100d'),
	skopt.space.Real(0.1, 0.8, name='w_dfh'),
	skopt.space.Real(0.1, 0.8, name='w_sharpe'),
]


@skopt.utils.use_named_args(SPACE)
def objective(**params):
	return -1*train_evaluate(params)

results = skopt.forest_minimize(objective, SPACE, n_calls=500, n_random_starts=5, n_jobs=3, x0=list(hyperparameters_init.values()))

print(results)