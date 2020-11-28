import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


from Portfolio import Portfolio
from Downloader import Downloader
from TradingZones import TradingZones
from Picker import Picker
from Order import Order

"""
Watcher:
	- Open the portfolio
	- Load the latest data
	- Get TS, TP, SL list
	- Get daily picks
	- Get order list
"""

class Watcher:
	def __init__(self, filename, period='1yr', stockPicks=10, avoidDowntrends=True, sellAllOnCrash=True, hyperparameters={}):
		self.filename	= filename
		self.period 	= period
		self.stockPicks = stockPicks
		self.hyperparameters = hyperparameters
		self.sl         = hyperparameters['sl']
		self.tp         = hyperparameters['tp']
		self.ts         = hyperparameters['ts']
		self.avoidDowntrends = avoidDowntrends
		self.sellAllOnCrash = sellAllOnCrash
	
	def create(self, balance):
		# Create the portolio
		self.portfolio  = Portfolio(cash=balance, sl=self.sl, tp=self.tp)
		self.portfolio.save(self.filename)
	
	def start(self):
		# Open the portolio
		self.portfolio  = Portfolio(cash=0, sl=self.sl, tp=self.tp)
		self.portfolio.load(self.filename);
		# Download the stock data
		self.downloader = Downloader(cache=False)
		self.downloader.download(period=self.period)
		
		date = self.downloader.prices.index[-1]
		prices = self.downloader.prices
		
		print("#", date)

		# Refresh the portfolio with the latest prices
		self.portfolio.refresh(prices=prices, date=date)
		
		# Get the portfolio Summary
		portfolio_summary = self.portfolio.summary()

		# Get the opened positions
		opened = self.portfolio.holdings[self.portfolio.holdings['status']=='open']

		# Get the current trend
		zt 	 = TradingZones(prices=prices)
		zones = zt.get()
		
		output = pd.DataFrame()
		
		# Activate the trailing stops
		targets   = self.portfolio.getPositionsBelowTS(self.ts)
		if len(targets) > 0:
			sell_orders = self.portfolio.getSellOrdersFromSubset(subset=targets)
			print("\n----- TS -----\n",sell_orders)
			#self.portfolio.sell(sell_orders, prices, date, label='TS')

		# Activate the stop loss
		if self.sl < 0:
			poor_perfs	= opened[opened['profits_pct']<=self.sl]
			if len(poor_perfs) > 0:
				sell_list 	= poor_perfs['symbol'].unique()
				sell_orders = self.portfolio.getSellAllOrders(symbols=sell_list)
				print("\n----- SL -----\n",sell_orders)
				#self.portfolio.sell(sell_orders, prices, date, label='SL')
		
		# Sell all on drop
		if self.sellAllOnCrash==True and zones.iloc[-1]['signal']=='DOWN' and len(opened)>0:
			print("Sell all")
			print("\n----- Sell All -----\n")
			#self.portfolio.sellAll(prices, date, label='Market Drop')
		
		# Get the ones that reached take-profit
		if self.tp > 0:
			good_perfs	= opened[opened['profits_pct']>=self.tp]
			if len(good_perfs) > 0:
				sell_list 	= good_perfs['symbol'].unique()
				sell_orders = self.portfolio.getSellAllOrders(symbols=sell_list)
				print("\n----- TP -----\n",sell_orders)
				#self.portfolio.sell(sell_orders, prices, date, label='TP')

		# Avoid trading in a general market downtrend
		if self.avoidDowntrends==False or (self.avoidDowntrends==True and zones.iloc[-1]['signal']=='UP'):
			# Pick the stocks
			picker = Picker(prices=prices)
			picker.pick(count=self.stockPicks)
			#print(">", picker.selected)
			# Get the order size
			portfolio_summary = self.portfolio.summary()
			order 	= Order(prices=prices, picks=picker.selected)
			order.get(budget=portfolio_summary['cash'])
			#print("!", portfolio_summary['cash'], order.orders)
			print("\n----- BUY -----\n",order.orders)
			# Update our portfolio
			#self.portfolio.update(orders=order.orders, prices=prices, date=date)
		else:
			# No new positions in a downtrend!
			# Update our portfolio
			self.portfolio.update(orders=pd.DataFrame(), prices=prices, date=date)
	
	def stats(self):
		# Open the portolio
		self.portfolio  = Portfolio(cash=0, sl=self.sl, tp=self.tp)
		self.portfolio.load(self.filename);
		
		symbols = list(self.portfolio.holdings['symbol'].unique())
		
		if len(symbols)==0:
			print("Nothing to update")
			return False
		elif len(symbols)==1:
			symbols = symbols + ['MSFT','TSLA'] # otherwise single symbol = no multi-index = breaking
		
		# Download the stock data
		self.downloader = Downloader(cache=False)
		self.downloader.download(period=self.period, symbols=symbols)
		
		date = self.downloader.prices.index[-1]
		prices = self.downloader.prices
		
		print("#", date)

		# Refresh the portfolio with the latest prices
		self.portfolio.refresh(prices=prices, date=date)
		
		# Save the portfolio with the latest prices
		self.portfolio.save(self.filename)
		
		print(self.portfolio.summary())
		print('')
		
		g = self.portfolio.holdings[self.portfolio.holdings['status']=='open'].copy().groupby('symbol').sum()
		g['profits_pct']	= (g['current_price']-g['purchase_price'])/g['purchase_price']*100
		g['price']			= prices.loc[date]
		print(g)
		print(list(g.index))
		print(prices.loc[date])
		
	def check(self, symbol='TSLA', period='6mo'):
		# Download the stock data
		self.downloader = Downloader(cache=False)
		if symbol not in self.downloader.symbols:
			self.downloader.add(symbol)
			print('Symbol added to the watch list')
		self.downloader.download(period=period, symbols=[symbol])
		print(self.downloader.prices[symbol].tail(50))





class Executor:
	def __init__(self, filename, hyperparameters={}):
		self.hyperparameters = hyperparameters
		self.filename	= filename
		self.portfolio  = Portfolio()
		self.portfolio.load(self.filename)
		self.downloader = Downloader(cache=False)
		self.downloader.download(period='3mo')
	
	def buy(self, symbol, count, value):
		print("Buying", count, "shares of", symbol, "at", value)
		date = self.downloader.prices.index[-1]
		prices = self.downloader.prices
		orders = pd.DataFrame()
		orders = orders.append({
			"price":		value,
			"last_price":	value,
			"order_size": 	count,
			"symbol": 		symbol
		}, ignore_index=True)
		orders = orders.set_index('symbol')
		# Execute the orders
		self.portfolio.update(orders=orders, prices=prices, date=date)
		# Refresh the data
		self.portfolio.refresh(prices=prices, date=date)
		# Save the portfolio
		self.portfolio.save(self.filename)
		return self.portfolio.summary()
	
	def sell(self, symbol, count, value):
		print("Selling", count, "shares of", symbol, "at", value)
		date = self.downloader.prices.index[-1]
		prices = self.downloader.prices
		orders = pd.DataFrame()
		orders = orders.append({
			"current_price":value,
			"count": 		count,
			"total_value":	value*count,
			"symbol": 		symbol
		}, ignore_index=True)
		orders = orders.set_index('symbol')
		# Execute the orders
		self.portfolio.update(orders=orders, prices=prices, date=date)
		self.portfolio.sell(orders, prices, date, label='TS')
		# Refresh the data
		self.portfolio.refresh(prices=prices, date=date)
		# Save the portfolio
		self.portfolio.save(self.filename)
		return self.portfolio.summary()