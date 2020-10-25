import sys
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

def Main():
	if len(sys.argv)<=1:
		print("Argument required: create, watch, buy, sell, stats")
	else:
		if sys.argv[1] == 'create':
			if len(sys.argv) < 3:
				print("Invalid arguments. `python main.py create 5000`")
			else:
				watcher = Watcher(filename='data/portfolio')
				watcher.create(balance=float(sys.argv[2]))
		elif sys.argv[1] == 'watch':
			watcher = Watcher(filename='data/portfolio')
			watcher.start()
		elif sys.argv[1] == 'buy':
			# main.py buy 20 AMD 81.26
			if len(sys.argv) < 5:
				print("Invalid arguments. `python main.py buy 20 AMD 81.26`")
			else:
				executor = Executor(filename='data/portfolio')
				print(executor.buy(symbol=sys.argv[3], count=int(sys.argv[2]), value=float(sys.argv[4])))
		elif sys.argv[1] == 'sell':
			# main.py sell 20 AMD 81.26
			if len(sys.argv) < 5:
				print("Invalid arguments. `python main.py sell 20 AMD 81.26`")
			else:
				executor = Executor(filename='data/portfolio')
				print(executor.sell(symbol=sys.argv[3], count=int(sys.argv[2]), value=float(sys.argv[4])))
		elif sys.argv[1] == 'stats':
			watcher = Watcher(filename='data/portfolio')
			watcher.stats()
		else:
			print("Unknown command")
		
Main()