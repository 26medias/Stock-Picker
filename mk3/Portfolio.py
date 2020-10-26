import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import json


class Portfolio:
  def __init__(self, cash=5000, sl=-0.1, tp=0.25):
    self.cash     = cash
    self.sl       = sl
    self.tp       = tp
    self.holdings = pd.DataFrame(columns=['symbol','purchase_date','sell_date','status','purchase_price','current_price','profits','profits_pct', 'high', 'high_pct','label'])

  def load(self, filename):
    self.holdings = pd.read_pickle(filename+'.pkl')
    with open(filename+'.json') as json_file:
      data = json.load(json_file)
      self.cash = data['cash']
  
  def save(self, filename):
    self.holdings.to_pickle(filename+'.pkl')
    with open(filename+'.json', 'w') as outfile:
      data = {}
      data['cash'] = self.cash
      json.dump(data, outfile)
  
  # Refresh the positions with fresh price data
  # Delete the positions that aren't performing well
  def refresh(self, prices, date):
    # Update our positions
    opened = self.holdings[self.holdings['status']=='open']
    symbols = opened['symbol'].unique()
    for symbol in symbols:
      subset    = opened[opened['symbol']==symbol]
      subsetIds = subset.index
      # Update the holdings
      self.holdings.loc[subsetIds, 'current_price'] = prices.loc[date, symbol]
      self.holdings.loc[subsetIds, 'profits']       = self.holdings.loc[subsetIds, 'current_price'] - self.holdings.loc[subsetIds, 'purchase_price']
      self.holdings.loc[subsetIds, 'profits_pct']   = (self.holdings.loc[subsetIds, 'current_price'] - self.holdings.loc[subsetIds, 'purchase_price'])/self.holdings.loc[subsetIds, 'purchase_price']
      symbolPrices = prices[symbol]
      # Update the high since the positions were purchased
      for _date in list(self.holdings.loc[subsetIds, 'purchase_date'].unique()):
        high = symbolPrices[_date:].max()
        sub_subset = subset[subset['purchase_date']==_date]
        self.holdings.loc[sub_subset.index, 'high'] = high
        self.holdings.loc[sub_subset.index, 'high_pct']   = (self.holdings.loc[subsetIds, 'high'] - self.holdings.loc[subsetIds, 'purchase_price'])/self.holdings.loc[subsetIds, 'purchase_price']
    
    #print('-------------------\n',self.getPositionsBelowTS())

  # Get the list of open positions that have reached a specific trailing stop
  def getPositionsBelowTS(self, ts=-0.2):
    opened = self.holdings[self.holdings['status']=='open'].copy()
    #opened['dfh'] = (opened['high']-opened['current_price'])/(opened['high']-opened['purchase_price'])
    opened['dfh'] = (opened['current_price']-opened['high'])/opened['high']
    opened['dfh'] = opened['dfh'].replace(np.inf, 0)
    return opened[(opened['dfh'] <= ts) & (opened['profits'] >= 0)].copy()
  
  # Update the holdings based on purchase orders
  def update(self, orders, prices, date):
    if len(orders)>0:
      #buy_orders, sell_orders = self.computeOrders(orders)
      buy_orders = self.getBuyOrders(orders)
      #print('')
      #print('-- Buy orders --')
      #print(buy_orders)
      self.buy(buy_orders, prices, date)
  
  # Sell All
  def sellAll(self, prices, date, symbols=None, label=''):
    sell_orders = self.getSellAllOrders(symbols=symbols)
    return self.sell(sell_orders, prices, date, label)
  
  def buy(self, buy_orders, prices, date):
    #print("Buy Value", buy_orders.sum()['total_value'])
    for symbol, row in buy_orders.iterrows():
      for i in range(0, int(row['count'])):
        # Pay for the order
        if self.cash >= buy_orders.at[symbol, 'current_price']:
          self.cash = self.cash - buy_orders.at[symbol, 'current_price']
          #print("Buying 1x "+symbol+" for $"+str(buy_orders.at['AMD', 'current_price'])+" | Cash: $"+str(self.cash))
          self.holdings = self.holdings.append({
              "symbol":         symbol,
              "purchase_date":  date,
              "sell_date":      None,
              "purchase_price": buy_orders.at[symbol, 'current_price'],
              "current_price":  buy_orders.at[symbol, 'current_price'],
              "status":         'open'
          }, ignore_index=True)
        else:
          print("# Not enough cash to purchase "+symbol+" at ", buy_orders.at['AMD', 'current_price'])
  
  def buy2(self, buy_orders, prices, date):
    #print("Buy Value", buy_orders.sum()['total_value'])
    for symbol, row in buy_orders.iterrows():
      for i in range(0, int(row['count'])):
        # Pay for the order
        if self.cash >= prices[symbol].iloc[-1]:
          self.cash = self.cash - prices[symbol].iloc[-1]
          #print("Buying 1x "+symbol+" for $"+str(prices[symbol].iloc[-1])+" | Cash: $"+str(self.cash))
          self.holdings = self.holdings.append({
              "symbol":         symbol,
              "purchase_date":  date,
              "sell_date":      None,
              "purchase_price": prices[symbol].iloc[-1],
              "current_price":  prices[symbol].iloc[-1],
              "status":         'open'
          }, ignore_index=True)
        else:
          print("# Not enough cash to purchase "+symbol+" at ", prices[symbol].iloc[-1])
  
  def sell(self, sell_orders, prices, date, label=''):
    for symbol, row in sell_orders.iterrows():
      subsetIds = self.holdings[self.holdings['symbol']==symbol]
      subsetIds = subsetIds[subsetIds['status']=='open']
      subsetIds = subsetIds.head(int(row['count'])).index
      # Update the holdings
      self.holdings.loc[subsetIds, 'current_price'] = sell_orders.at[symbol, 'current_price']
      self.holdings.loc[subsetIds, 'status']        = 'closed'
      self.holdings.loc[subsetIds, 'profits']       = self.holdings.loc[subsetIds, 'current_price'] - self.holdings.loc[subsetIds, 'purchase_price']
      self.holdings.loc[subsetIds, 'profits_pct']   = (self.holdings.loc[subsetIds, 'current_price'] - self.holdings.loc[subsetIds, 'purchase_price'])/self.holdings.loc[subsetIds, 'purchase_price']
      self.holdings.loc[subsetIds, 'sell_date']     = date
      self.holdings.loc[subsetIds, 'label']         = label
      # Calculate the profits & how much we're getting back out
      subset    = self.holdings[self.holdings.index.isin(subsetIds)]
      self.cash = self.cash + subset['current_price'].sum()
      #print("Selling "+str(len(subsetIds))+"x "+symbol+" for $"+str(subset['current_price'].sum())+" with a $"+str(subset['profits'].sum())+" gain | Cash: $"+str(self.cash))
  
  def getSellAllOrders(self, symbols=None):
    opened = self.holdings[self.holdings['status']=='open']
    if symbols is not None:
      opened = opened[opened['symbol'].isin(symbols)]
    opened = opened.groupby(['symbol'])
    holding_stats = opened.sum()
    if len(holding_stats) > 0:
      holding_stats['count'] = opened.count()['purchase_date']
      sell_orders = holding_stats.copy()
      sell_orders = sell_orders.drop(columns=['purchase_price','profits'])
      sell_orders['current_price'] = sell_orders['current_price']/sell_orders['count']
      sell_orders['total_value'] = sell_orders['current_price']*sell_orders['count']
      return sell_orders
    return pd.DataFrame()
  
  
  def getSellOrdersFromSubset(self, subset):
    opened = subset.groupby(['symbol'])
    holding_stats = opened.sum()
    if len(holding_stats) > 0:
      holding_stats['count'] = opened.count()['purchase_date']
      sell_orders = holding_stats.copy()
      sell_orders = sell_orders.drop(columns=['purchase_price','profits','profits_pct','high','dfh'])
      sell_orders['current_price'] = sell_orders['current_price']/sell_orders['count']
      sell_orders['total_value'] = sell_orders['current_price']*sell_orders['count']
      return sell_orders
    return pd.DataFrame()
  
  def getBuyOrders(self, orders):
    buy_orders = pd.DataFrame()
    buy_orders['count'] = orders['order_size']
    buy_orders['current_price'] = orders['last_price']
    buy_orders['total_value'] = buy_orders['current_price']*buy_orders['count']
    buy_orders = buy_orders[buy_orders['count']>0]
    return buy_orders

  def holdingStats(self):
    opened = self.holdings[self.holdings['status']=='open'].copy()
    closed = self.holdings[self.holdings['status']=='closed'].copy()
    openGroup = opened.groupby(['symbol'])
    stats_open = openGroup.sum()
    stats_open['count'] = openGroup.count()['purchase_date']

    closedGroup = closed.groupby(['symbol'])
    stats_closed = closedGroup.sum()
    stats_closed['count'] = closedGroup.count()['purchase_date']
    print("-- Opened --")
    print(stats_open)
    print("-- Closed --")
    print(stats_closed)
  
  def getOrderValue(self, order):
    if len(order):
      data = order.copy();
      data['val'] = data['count']*data['current_price']
      print(data)
      sum = data.sum()['val']
      print("Sum: ", sum)
    else:
      print("None")
  
  def summary(self):
    output = {
        "invested":       self.holdings[self.holdings['status']=='open'][['purchase_price']].sum()[0],
        "portfolio_value":self.holdings[self.holdings['status']=='open'][['current_price']].sum()[0],
        "Positions":      len(self.holdings),
        "closed":         len(self.holdings[self.holdings['status']=='closed']),
        "opened":         len(self.holdings[self.holdings['status']=='open']),
        "cash":           self.cash,
        "total_value":    self.holdings[self.holdings['status']=='open'][['current_price']].sum()[0]+self.cash,
        "closed_profits": self.holdings[self.holdings['status']=='closed'][['profits']].sum()[0],
        "open_profits":   self.holdings[self.holdings['status']=='open'][['profits']].sum()[0]
    }
    output['drawdown']      = output['portfolio_value']-output['invested']
    if output['invested'] > 0:
    	output['drawdown_pct']  = (output['portfolio_value']-output['invested'])/output['invested']
    else:
    	output['drawdown_pct']  = 0
    return output

