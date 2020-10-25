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

class Sim:
  def __init__(self, period='2y', timedelay=100, window=100, timestep=5, budget=5000, stockPicks=10, sl=-0.1, tp=0.25, ts=0.2, ts_threshold=0.05, avoidDowntrends=True, sellAllOnCrash=True):
    self.period     = period
    self.timedelay  = timedelay
    self.timestep   = timestep
    self.budget     = budget
    self.stockPicks = stockPicks
    self.sl         = sl
    self.tp         = tp
    self.ts         = ts
    self.ts_threshold = ts_threshold
    self.avoidDowntrends = avoidDowntrends
    self.sellAllOnCrash = sellAllOnCrash
    self.portfolio  = Portfolio(cash=budget, sl=self.sl, tp=self.tp)
    self.downloader = Downloader()
    self.downloader.download(period=self.period)
    self.current_index = timedelay
    self.is_notebook   = self.isNotebook()
  
  def isNotebook(self):
	  try:
	    cfg = get_ipython().config 
	    return True
	  except NameError:
	    return False
  
  
  def progress(self, count, total=100, status=''):
    if self.is_notebook == True:
	    return HTML("""
	        <progress
	            value='{value}'
	            max='{max}',
	            style='width: 50%'
	        >
	            {value}
	        </progress>
	    """.format(value=count, max=total))
    else:
	    bar_len = 60
	    filled_len = int(round(bar_len * count / float(total)))

	    percents = round(100.0 * count / float(total), 1)
	    bar = '=' * filled_len + '-' * (bar_len - filled_len)

	    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
	    sys.stdout.flush()
  
  def run(self):
    #out = display(self.progress(0, 100), display_id=True)
    steps = (len(self.downloader.prices)-self.timedelay)
    stepN = 0
    self.stats = pd.DataFrame(columns=['invested','portfolio_value','Positions','closed','opened','cash','total_value','portfolio_resistances','portfolio_supports','drawdown','drawdown_pct'])
    while self.current_index+self.timestep < len(self.downloader.prices):
      output = self.tick()
      output['date'] = self.downloader.prices.index[self.current_index]
      self.stats = self.stats.append(output, ignore_index=True)
      stepN = stepN + 1
      pct = math.ceil((stepN/steps)*100)
      if self.is_notebook == True:
      	out.update(self.progress(pct, 100))
      else:
      	self.progress(pct, 100)
      #print(str(round(pct))+"%")
    self.portfolio.sellAll(prices=self.downloader.prices[:self.current_index], date=self.downloader.prices.index[self.current_index], label='End Sim')
    self.stats = self.stats.set_index('date')
    return self.stats

  # Run every night after the market close
  def tick(self):
    #print('\n\n----------- tick -----------')
    # Get the latest prices & date as of today
    date = self.downloader.prices.index[self.current_index]
    prices = self.downloader.prices[:self.current_index+1] # Include the current price
    
    #print("#", date)

    # Refresh the portfolio with the latest prices
    self.portfolio.refresh(prices=prices, date=date)
    
    # Get the portfolio Summary
    portfolio_summary = self.portfolio.summary()

    # Get the opened positions
    opened = self.portfolio.holdings[self.portfolio.holdings['status']=='open']

    # Get the current trend
    zt    = TradingZones(prices=prices)
    zones = zt.get()
    
    # Activate the trailing stops
    targets   = self.portfolio.getPositionsBelowTS(self.ts)
    targets   = targets[targets['high_pct']>=self.ts_threshold]
    if len(targets) > 0:
      sell_orders = self.portfolio.getSellOrdersFromSubset(subset=targets)
      #print("\n-----------\n",targets,"\n\n",sell_orders)
      self.portfolio.sell(sell_orders, prices, date, label='TS')

    # Activate the stop loss
    if self.sl < 0:
      poor_perfs  = opened[opened['profits_pct']<=self.sl]
      if len(poor_perfs) > 0:
        sell_list   = poor_perfs['symbol'].unique()
        sell_orders = self.portfolio.getSellAllOrders(symbols=sell_list)
        self.portfolio.sell(sell_orders, prices, date, label='SL')
    
    # Sell all on drop
    if self.sellAllOnCrash==True and zones.iloc[-1]['signal']=='DOWN' and len(opened)>0:
      print("Sell all")
      self.portfolio.sellAll(prices, date, label='Market Drop')
    
    # Get the ones that reached take-profit
    if self.tp > 0:
      good_perfs  = opened[opened['profits_pct']>=self.tp]
      if len(good_perfs) > 0:
        sell_list   = good_perfs['symbol'].unique()
        sell_orders = self.portfolio.getSellAllOrders(symbols=sell_list)
        self.portfolio.sell(sell_orders, prices, date, label='TP')

    # Avoid trading in a general market downtrend
    if self.avoidDowntrends==False or (self.avoidDowntrends==True and zones.iloc[-1]['signal']=='UP'):
      # Pick the stocks
      picker = Picker(prices=prices)
      picker.pick(count=self.stockPicks)
      #print(">", picker.selected)
      # Get the order size
      portfolio_summary = self.portfolio.summary()
      order   = Order(prices=prices, picks=picker.selected)
      order.get(budget=portfolio_summary['cash'])
      #print("!", portfolio_summary['cash'], order.orders)
      # Update our portfolio
      self.portfolio.update(orders=order.orders, prices=prices, date=self.downloader.prices.index[self.current_index])
    else:
      # No new positions in a downtrend!
      # Update our portfolio
      self.portfolio.update(orders=pd.DataFrame(), prices=prices, date=self.downloader.prices.index[self.current_index])
    
    # Update the index
    if self.current_index + self.timestep < len(self.downloader.prices):
      self.current_index = self.current_index + self.timestep
    portfolio_summary = self.portfolio.summary()
    return portfolio_summary


