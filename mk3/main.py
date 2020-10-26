import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

"""
Watcher:
	- Open the portfolio
	- Load the latest data
	- Get TS, TP, SL list
	- Get daily picks
	- Get order list

Executor:
	- Open the portfolio
	- Record sells & buys
	- Save portfolio
"""

from Watcher import Watcher
from Watcher import Executor



if not os.path.exists('data'):
	os.makedirs('data')

pick_kwargs = { "v_100d": 0.24531455566424112, "v_dfh": 0.05274405649047917, "v_rfl": 0.01703439651254952, "w_100d": 0.19102857670958306, "w_dfh": 0.5799832209142695, "w_sharpe": 0.655623531391156 }
hyperparameters = {"sl": -0.06, "tp": 2, "ts": -0.02, 'pick_kwargs':pick_kwargs}

def Main():
	if len(sys.argv)<=1:
		print("Argument required: create, watch, buy, sell, stats")
	else:
		if sys.argv[1] == 'create':
			if len(sys.argv) < 3:
				print("Invalid arguments. `python main.py create 5000`")
			else:
				watcher = Watcher(filename='data/portfolio', hyperparameters=hyperparameters)
				watcher.create(balance=float(sys.argv[2]))
		
		elif sys.argv[1] == 'watch':
			watcher = Watcher(filename='data/portfolio', hyperparameters=hyperparameters)
			watcher.start()
		
		elif sys.argv[1] == 'buy':
			# main.py buy 20 AMD 81.26
			if len(sys.argv) < 5:
				print("Invalid arguments. `python main.py buy 20 AMD 81.26`")
			else:
				executor = Executor(filename='data/portfolio', hyperparameters=hyperparameters)
				print(executor.buy(symbol=sys.argv[3], count=int(sys.argv[2]), value=float(sys.argv[4])))
		
		elif sys.argv[1] == 'sell':
			# main.py sell 20 AMD 81.26
			if len(sys.argv) < 5:
				print("Invalid arguments. `python main.py sell 20 AMD 81.26`")
			else:
				executor = Executor(filename='data/portfolio', hyperparameters=hyperparameters)
				print(executor.sell(symbol=sys.argv[3], count=int(sys.argv[2]), value=float(sys.argv[4])))
		
		elif sys.argv[1] == 'stats':
			watcher = Watcher(filename='data/portfolio', hyperparameters=hyperparameters)
			watcher.stats()
		
		elif sys.argv[1] == 'sim':
			from Sim import Sim
			from Analysis import Analysis
			
			sim    = Sim(period='1y', timedelay=100, window=100, timestep=1, budget=5000, stockPicks=5, avoidDowntrends=True, sellAllOnCrash=False, **hyperparameters)
			stats  = sim.run()

			analysis = Analysis(stats=stats, positions=sim.portfolio.holdings, prices=sim.downloader.prices)
			analysis.chart('data/best_optimized_3y.png')
			output, advanced_stats = analysis.positionStats()
			print(output)


			g = sim.portfolio.holdings.copy().groupby('label').sum()
			g['profits_pct'] = (g['current_price']-g['purchase_price'])/g['purchase_price']*100
			print(g)
		
		elif sys.argv[1] == 'test':
			from Sim import Sim
			from Analysis import Analysis
			
			sim    = Sim(period='1y', timedelay=100, window=100, timestep=1, budget=5000, stockPicks=5, avoidDowntrends=True, sellAllOnCrash=False, **hyperparameters)
			#stats  = sim.run()
			for n in range(0,15):
				sim.tick()
			
			analysis = Analysis(stats=stats, positions=sim.portfolio.holdings, prices=sim.downloader.prices)
			analysis.chart('data/best_optimized_3y.png')
			output, advanced_stats = analysis.positionStats()
			print(output)


			g = sim.portfolio.holdings.copy().groupby('label').sum()
			g['profits_pct'] = (g['current_price']-g['purchase_price'])/g['purchase_price']*100
			print(g)
		else:
			print("Unknown command")
		
Main()