import sys
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import hashlib

from os import path
import json
import threading
import time
import math
import random


pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)


class Downloader:
  def __init__(self, figsize=(30,10)):
    self.figsize  = figsize
    self.getStocklist()

  def getStocklist(self):
    self.stocklist = pd.read_csv('constituents.csv')
    self.symbols   = list(self.stocklist[['Symbol']].values.flatten())
    self.sectors   = self.stocklist['Sector'].unique()
  
  def stats(self):
    return self.stocklist.groupby('Sector').aggregate(['count'])
  
  def download(self, sector=None, symbols=None, period='1y'):
    print("Download: ", sector, symbols)
    if sector is not None:
      self.sector = sector
      rows      = self.stocklist[self.stocklist['Sector']==sector]
      symbols   = list(rows[['Symbol']].values.flatten())
      filename  = sector+'.pkl'
      self.symbols = self.getSymbolsBySector(self.sector)
    else:
      if symbols is not None:
        self.symbols = symbols
      filename  = 'data_'+period+'_'+(hashlib.sha224((''.join(self.symbols)).encode('utf-8')).hexdigest())+".json"
    
    if path.exists(filename):
      print("Using cached data")
      self.data = pd.read_pickle(filename)
    else:
      print("Downloading the historical prices")
      self.data = yf.download(self.symbols, period=period, threads=True)
      self.data.to_pickle(filename)
    #self.data = self.data.dropna()
    self.data = self.data[~self.data.index.duplicated(keep='last')]
    self.build()

    return self.data
  
  def getSymbolsBySector(self, sector):
    sectorRows    = self.stocklist[self.stocklist['Sector']==sector]
    return list(sectorRows[['Symbol']].values.flatten())
  
  def build(self):
    self.prices = self.data.loc[:,('Adj Close', slice(None))]
    if math.isnan(self.prices.iloc[0].max()):
      self.data = self.data.iloc[1:]
      self.prices = self.data.loc[:,('Adj Close', slice(None))]
    self.prices.columns = self.prices.columns.droplevel(0)
    self.changes  = (self.prices-self.prices.shift())/self.prices.shift()
    self.returns  = (self.prices[:]-self.prices[:].loc[list(self.prices.index)[0]])/self.prices[:].loc[list(self.prices.index)[0]]
    self.sector_mean = self.returns[self.symbols].T.describe().T[['mean']]
    return self.stocklist.groupby('Sector').aggregate(['count'])









class TradingZones:
  def __init__(self, prices):
    self.prices = prices
    self.returns  = (self.prices[:]-self.prices[:].loc[list(self.prices.index)[0]])/self.prices[:].loc[list(self.prices.index)[0]]
  
  def get(self):
    self.stats = pd.DataFrame()
    self.stats['mean'] = self.returns.copy().T.mean().T
    #self.stats['mean'] = self.returns.copy().diff().T.mean().T
    self.stats['3d'] = self.stats['mean']-self.stats['mean'].shift(3)
    self.stats['7d'] = self.stats['mean']-self.stats['mean'].shift(7)
    self.stats['zone'] = (self.stats['7d']+self.stats['3d'])/2
    
    self.stats['resistances']  = self.stats[(self.stats['mean'].shift(1) < self.stats['mean']) & (self.stats['mean'].shift(-1) < self.stats['mean'])]['mean']
    self.stats['supports']     = self.stats[(self.stats['mean'].shift(1) > self.stats['mean']) & (self.stats['mean'].shift(-1) > self.stats['mean'])]['mean']
    self.stats = self.stats.ffill(axis = 0) 
    self.stats['band_up']      = self.stats['resistances'].rolling(10).max()
    self.stats['band_down']    = self.stats['supports'].rolling(10).min()
    self.stats['band_mean']    = (self.stats['band_up']+self.stats['band_down'])/2
    self.stats.at[self.stats[self.stats['mean'] > self.stats['band_mean']].index, 'signal'] = 'UP'
    self.stats.at[self.stats[self.stats['mean'] < self.stats['band_mean']].index, 'signal'] = 'DOWN'

    return self.stats
  
  def chart3(self):
    zones = self.stats
    fig, axs = plt.subplots(3, figsize=(30,10))
    fig.suptitle('Analysis')
    axs[0].plot(zones['mean'])
    axs[0].plot(zones['mean'].rolling(5).mean())
    axs[0].plot(zones['mean'].shift(5).rolling(5).mean())
    osc = zones['mean'].rolling(5).mean()-zones['mean'].shift(5).rolling(5).mean()
    
    axs[0].fill_between(zones.index, 0, zones['mean'], where=zones['zone'] >= 0, facecolor='green', interpolate=True, alpha=0.2)
    axs[0].fill_between(zones.index, 0, zones['mean'], where=zones['zone'] <= 0, facecolor='red', interpolate=True, alpha=0.2)

    axs[1].plot(zones['3d'])
    axs[1].plot(zones['7d'])
    axs[1].plot(zones['zone'])
    axs[1].fill_between(zones.index, 0, zones['zone'], where=zones['zone'] > 0, facecolor='green', interpolate=True, alpha=0.2)
    axs[1].fill_between(zones.index, 0, zones['zone'], where=zones['zone'] < 0, facecolor='red', interpolate=True, alpha=0.2)
    axs[2].fill_between(osc.index, 0, osc, where=osc > 0, facecolor='green', interpolate=True, alpha=0.2)
    axs[2].fill_between(osc.index, 0, osc, where=osc < 0, facecolor='red', interpolate=True, alpha=0.2)
    axs[2].plot(osc)
    plt.show()
  
  def chart2(self):
    zones = self.stats
    zones['osc'] = zones['mean'].rolling(5).mean()-zones['mean'].shift(5).rolling(5).mean()

    fig, axs = plt.subplots(3, figsize=(30,10))
    fig.suptitle('Analysis')
    axs[0].plot(zones['mean'])
    
    axs[0].fill_between(zones.index, 0, zones['mean'], where=zones['zone'] >= 0, facecolor='green', interpolate=True, alpha=0.2)
    axs[0].fill_between(zones.index, 0, zones['mean'], where=zones['zone'] <= 0, facecolor='red', interpolate=True, alpha=0.2)

    axs[1].plot(zones['3d'])
    axs[1].plot(zones['7d'])
    axs[1].plot(zones['zone'])
    axs[1].fill_between(zones.index, 0, zones['zone'], where=zones['zone'] > 0, facecolor='green', interpolate=True, alpha=0.2)
    axs[1].fill_between(zones.index, 0, zones['zone'], where=zones['zone'] < 0, facecolor='red', interpolate=True, alpha=0.2)
    
    axs[2].fill_between(zones['osc'].index, 0, zones['osc'], where=zones['osc'] > 0, facecolor='green', interpolate=True, alpha=0.2)
    axs[2].fill_between(zones['osc'].index, 0, zones['osc'], where=zones['osc'] < 0, facecolor='red', interpolate=True, alpha=0.2)
    axs[2].plot(zones['osc'])
    plt.show()
    zones = self.stats
    fig, axs = plt.subplots(3, figsize=(30,10))
    fig.suptitle('Analysis')
    axs[0].plot(zones['mean'])
    axs[0].plot(zones['mean'].rolling(5).mean())
    axs[0].plot(zones['mean'].shift(5).rolling(5).mean())
    osc = zones['mean'].rolling(5).mean()-zones['mean'].shift(5).rolling(5).mean()
    
    axs[0].fill_between(zones.index, 0, zones['mean'], where=zones['zone'] >= 0, facecolor='green', interpolate=True, alpha=0.2)
    axs[0].fill_between(zones.index, 0, zones['mean'], where=zones['zone'] <= 0, facecolor='red', interpolate=True, alpha=0.2)

    axs[1].plot(zones['3d'])
    axs[1].plot(zones['7d'])
    axs[1].plot(zones['zone'])
    axs[1].fill_between(zones.index, 0, zones['zone'], where=zones['zone'] > 0, facecolor='green', interpolate=True, alpha=0.2)
    axs[1].fill_between(zones.index, 0, zones['zone'], where=zones['zone'] < 0, facecolor='red', interpolate=True, alpha=0.2)
    axs[2].fill_between(osc.index, 0, osc, where=osc > 0, facecolor='green', interpolate=True, alpha=0.2)
    axs[2].fill_between(osc.index, 0, osc, where=osc < 0, facecolor='red', interpolate=True, alpha=0.2)
    axs[2].plot(osc)
    plt.show()
  
  def chart(self):
    zones = self.stats.copy()
    zones.at[zones[zones['signal']=='UP'].index, 'sig_up'] = 0
    zones.at[zones[zones['signal']=='DOWN'].index, 'sig_down'] = 0

    fig, axs = plt.subplots(2, figsize=(30,10))
    fig.suptitle('Analysis')
    axs[0].plot(zones['mean'])
    
    axs[0].fill_between(zones.index, 0, zones['mean'], where=zones['mean'] >= 0, facecolor='green', interpolate=True, alpha=0.2)
    axs[0].fill_between(zones.index, 0, zones['mean'], where=zones['mean'] <= 0, facecolor='red', interpolate=True, alpha=0.2)

    axs[1].plot(zones['mean'])
    #axs[1].plot(zones['resistances'], 'r--', alpha=0.2)
    #axs[1].plot(zones['supports'], 'g--', alpha=0.2)

    axs[1].plot(zones['band_up'], 'b--', alpha=0.2)
    axs[1].plot(zones['band_down'], 'b--', alpha=0.2)
    axs[1].plot(zones['band_mean'], 'r--', alpha=0.9)

    axs[1].plot(zones['sig_up'], 'g-', alpha=0.9)
    axs[1].plot(zones['sig_down'], 'r-', alpha=0.9)

    plt.show()
    #return zones.tail(50)










class Picker:
  def __init__(self, prices):
    self.prices = prices
  
  def noiseless(self, data, steps=20):
    out = data.copy();
    firstValues = out.iloc[0]
    lastValues = out.iloc[-1]
    for x in range(0, steps):
      out = (out.shift(-1)+out.shift(1))/2
      out.iloc[0] = firstValues
      out.iloc[-1] = lastValues
    return out
  
  def pick(self, count=5):
    self.returns  = (self.prices[:]-self.prices[:].loc[list(self.prices.index)[0]])/self.prices[:].loc[list(self.prices.index)[0]]
    noiseless = self.noiseless(self.returns, steps=30).diff()
    noiseless = self.noiseless(noiseless, steps=10) # Smooth it a bit
    stats = pd.DataFrame()
    stats['min'] = self.prices.min()
    stats['max'] = self.prices.max()
    stats['first_price'] = self.prices.iloc[0]
    stats['last_price'] = self.prices.iloc[-1]
    stats['return'] = (stats['last_price']-stats['first_price'])/stats['first_price']
    stats['3d'] = (stats['last_price']-self.prices.iloc[-3])/stats['first_price']
    stats['7d'] = (stats['last_price']-self.prices.iloc[-7])/stats['first_price']
    stats['20d'] = (stats['last_price']-self.prices.iloc[-20])/stats['first_price']
    stats['50d'] = (stats['last_price']-self.prices.iloc[-50])/stats['first_price']
    stats['100d'] = (stats['last_price']-self.prices.iloc[-100])/stats['first_price']
    stats['sharpe'] = (252**0.5) * self.returns.diff().mean() / self.returns.diff().std()
    stats['trend'] = noiseless.iloc[-1]

    rolling_period = 20

    dfh = ((self.prices.rolling(rolling_period).max()-self.prices)/self.prices.rolling(rolling_period).max()).tail(1).T
    dfh.columns = ['dfh']
    rfl = ((self.prices-self.prices.rolling(rolling_period).min())/self.prices).tail(1).T
    rfl.columns = ['rfl']

    stats = pd.concat([stats, dfh, rfl], axis=1)
    #print(self.prices.diff())
    #stats['7d_trend'] = self.noiseless(self.returns.iloc[-7:].diff(), steps=30).iloc[-1]
    #stats['20d_trend'] = self.noiseless(self.returns.iloc[-20:].diff(), steps=30).iloc[-1]

    stats["w"] = (stats['dfh'] * 0.6) + (stats['sharpe'] * 0.2) + (stats['100d'] * 0.2)

    stats.sort_values(['w'], ascending=[0], inplace=True)
    selected = stats.copy();
    selected = selected[selected['3d'] > 0]
    #selected = selected[selected['7d'] > 0]
    selected = selected[selected['100d'] > 0.3]
    #selected = selected[selected['sharpe'] > 1]
    selected = selected[(selected['dfh'] >= 0.05) & (selected['rfl'] >= 0.01)]
    #selected = selected[selected['7d_trend'] > 0]
    selected = selected.head(count)
    self.selected = selected
    return selected
  
  def pick_classic(self, count=5):
    self.returns  = (self.prices[:]-self.prices[:].loc[list(self.prices.index)[0]])/self.prices[:].loc[list(self.prices.index)[0]]
    noiseless = self.noiseless(self.returns, steps=30).diff()
    noiseless = self.noiseless(noiseless, steps=10) # Smooth it a bit
    stats = pd.DataFrame()
    stats['min'] = self.prices.min()
    stats['max'] = self.prices.max()
    stats['first_price'] = self.prices.iloc[0]
    stats['last_price'] = self.prices.iloc[-1]
    stats['return'] = (stats['last_price']-stats['first_price'])/stats['first_price']
    stats['3d'] = (stats['last_price']-self.prices.iloc[-3])/stats['first_price']
    stats['7d'] = (stats['last_price']-self.prices.iloc[-7])/stats['first_price']
    stats['20d'] = (stats['last_price']-self.prices.iloc[-20])/stats['first_price']
    stats['50d'] = (stats['last_price']-self.prices.iloc[-50])/stats['first_price']
    stats['100d'] = (stats['last_price']-self.prices.iloc[-100])/stats['first_price']
    stats['sharpe'] = (252**0.5) * self.returns.diff().mean() / self.returns.diff().std()
    stats['trend'] = noiseless.iloc[-1]
    #print(self.prices.diff())
    #stats['7d_trend'] = self.noiseless(self.returns.iloc[-7:].diff(), steps=30).iloc[-1]
    #stats['20d_trend'] = self.noiseless(self.returns.iloc[-20:].diff(), steps=30).iloc[-1]

    stats.sort_values(['sharpe'], ascending=[0], inplace=True)
    selected = stats.copy();
    selected = selected[selected['3d'] < 0]
    selected = selected[selected['7d'] > 0]
    selected = selected[selected['20d'] > 0]
    selected = selected[selected['sharpe'] > 1]
    #selected = selected[selected['7d_trend'] > 0]
    selected = selected.head(count)
    self.selected = selected
    return selected
  
  def pick_best(self, count=5):
    self.returns  = (self.prices[:]-self.prices[:].loc[list(self.prices.index)[0]])/self.prices[:].loc[list(self.prices.index)[0]]
    noiseless = self.noiseless(self.returns, steps=30).diff()
    noiseless = self.noiseless(noiseless, steps=10) # Smooth it a bit
    stats = pd.DataFrame()
    stats['min'] = self.prices.min()
    stats['max'] = self.prices.max()
    stats['first_price'] = self.prices.iloc[0]
    stats['last_price'] = self.prices.iloc[-1]
    stats['return'] = (stats['last_price']-stats['first_price'])/stats['first_price']
    stats['3d'] = (stats['last_price']-self.prices.iloc[-3])/stats['first_price']
    stats['7d'] = (stats['last_price']-self.prices.iloc[-7])/stats['first_price']
    stats['20d'] = (stats['last_price']-self.prices.iloc[-20])/stats['first_price']
    stats['50d'] = (stats['last_price']-self.prices.iloc[-50])/stats['first_price']
    stats['100d'] = (stats['last_price']-self.prices.iloc[-100])/stats['first_price']
    stats['sharpe'] = (252**0.5) * self.returns.diff().mean() / self.returns.diff().std()
    stats['trend'] = noiseless.iloc[-1]
    #print(self.prices.diff())
    #stats['7d_trend'] = self.noiseless(self.returns.iloc[-7:].diff(), steps=30).iloc[-1]
    #stats['20d_trend'] = self.noiseless(self.returns.iloc[-20:].diff(), steps=30).iloc[-1]

    stats.sort_values(['sharpe'], ascending=[0], inplace=True)
    selected = stats.copy();
    selected = selected[selected['3d'] > 0]
    selected = selected[selected['7d'] > 0]
    selected = selected[selected['20d'] > 0]
    selected = selected[selected['sharpe'] > 1]
    selected = selected[selected['trend'] > 0]
    #selected = selected[selected['7d_trend'] > 0]
    selected = selected.head(count)
    self.selected = selected
    return selected






class Order:
  def __init__(self, prices, picks):
    self.picks = picks
    self.prices = prices[list(picks.index)]
  
  def get(self, budget=5000):
    weights = {}
    pickWeights = self.picks[['sharpe']].copy()
    pickWeights['weight']      = pickWeights[['sharpe']]/pickWeights[['sharpe']].sum()
    for symbol in list(self.picks.index):
      weights[symbol] = pickWeights.at[symbol, 'weight']#1/len(list(self.picks.index))
    weights = self.toDataFrame(weights)
    weights['curr_weight'] = 0
    weights['delta_w'] = weights['weight']
    weights['order_size'] = 0
    done = False
    c = 0
    balance = budget
    while (done is False or c<=50):
      balance, weights, done = self.update(balance, weights)
      c = c + 1

    orders, leftovers   = weights, balance
    orders = pd.concat([orders, self.picks], axis=1)
    self.orders = orders
    self.leftovers = leftovers
    return orders, leftovers
  
  def update(self, budget, weights):
    weights.sort_values(['delta_w'], ascending=[0], inplace=True)
    _weights = weights[weights['price'] <= budget]
    done = True
    if len(_weights)>0:
      line = _weights.iloc[0]
      budget = budget - line['price']
      weights.at[line.name, 'order_size'] = weights.at[line.name, 'order_size'] + 1
      done = False
      weights['curr_weight']  = weights[['order_size']]/weights[['order_size']].sum()
      weights['delta_w']      = weights['weight']-weights['curr_weight']
    return budget, weights, done

  # Convert a weight dict to a dataframe with latest prices
  def toDataFrame(self, portfolio):
    dfo = {'symbol': [], 'price': [], 'weight': []}
    dfo['symbol'] = list(portfolio.keys())
    last_index  = self.prices.index[-1]
    last_prices = self.prices.loc[last_index]
    dfo['price'] = list(last_prices)
    dfo['weight'] = list(portfolio.values())
    prices = pd.DataFrame.from_dict(dfo)
    prices = prices.set_index('symbol')
    return prices









class Portfolio:
  def __init__(self, cash=5000, sl=-0.1, tp=0.25):
    self.cash     = cash
    self.sl       = sl
    self.tp       = tp
    self.holdings = pd.DataFrame(columns=['symbol','purchase_date','sell_date','status','purchase_price','current_price','profits','profits_pct', 'high', 'high_pct','label'])

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
  def getPositionsBelowTS(self, ts=0.2):
    opened = self.holdings[self.holdings['status']=='open'].copy()
    opened['dfh'] = (opened['high']-opened['current_price'])/(opened['high']-opened['purchase_price'])
    opened['dfh']  = opened['dfh'].replace(np.inf, 0)
    return opened[(opened['dfh'] >= ts) & (opened['profits_pct'] > 0)].copy()
  
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
      self.holdings.loc[subsetIds, 'current_price'] = prices[symbol].iloc[-1]
      self.holdings.loc[subsetIds, 'status']        = 'closed'
      self.holdings.loc[subsetIds, 'profits']       = self.holdings.loc[subsetIds, 'current_price'] - self.holdings.loc[subsetIds, 'purchase_price']
      self.holdings.loc[subsetIds, 'profits_pct']   = (self.holdings.loc[subsetIds, 'current_price'] - self.holdings.loc[subsetIds, 'purchase_price'])/self.holdings.loc[subsetIds, 'purchase_price']
      self.holdings.loc[subsetIds, 'sell_date']     = date
      self.holdings.loc[subsetIds, 'label']         = label
      # Calculate the profits & how much we're getting back out
      subset    = self.holdings[self.holdings.index.isin(subsetIds)]
      self.cash = self.cash + subset['current_price'].sum()
      #print("Selling "+str(len(subsetIds))+"x "+symbol+" for $"+str(subset['current_price'].sum())+" with a $"+str(subset['profits'].sum())+" gain | Cash: $"+str(self.cash))
  
  def computeOrders(self, orders):
    opened = self.holdings[self.holdings['status']=='open'].groupby(['symbol'])
    holding_stats = opened.sum()
    holding_stats['count'] = opened.count()['purchase_date']
    
    # Sell the positions that aren't part of the new portfolio
    sell_list = [i for i in list(holding_stats.index) if i not in list(orders.index)]
    sell_orders = pd.DataFrame()
    if len(sell_list) > 0:
      # Sell those positions entirely
      sell_orders = holding_stats[holding_stats.index.isin(sell_list)].copy()
      sell_orders = sell_orders.drop(columns=['purchase_price','profits'])
      sell_orders['current_price'] = sell_orders['current_price']/sell_orders['count']
      sell_orders['total_value'] = sell_orders['current_price']*sell_orders['count']
    # Find the positions to update
    update_list = [i for i in list(holding_stats.index) if i in list(orders.index)]
    update_order = orders[orders.index.isin(update_list)]
    diff = update_order[['order_size']].copy()
    diff['prev_order_size'] = holding_stats['count']
    diff['diff'] = diff['order_size']-diff['prev_order_size']
    diff['current_price'] = orders[orders.index.isin(diff.index)]['last_price']
    #print('-- diff --')
    #print(diff)
    buy_orders = pd.DataFrame()
    buy_orders['count'] = orders['order_size']
    buy_orders['current_price'] = orders['last_price']
    buy_orders['total_value'] = buy_orders['current_price']*buy_orders['count']
    for symbol, row in diff.iterrows():
      if row['diff'] > 0:
        # Buy
        if symbol in list(buy_orders.index):
          buy_orders.loc[symbol, 'count'] = buy_orders.loc[symbol, 'count'] + row['diff']
        else:
          buy_orders.loc[symbol, 'count'] = row['diff']
      elif row['diff'] < 0:
        # Sell
        if symbol in list(sell_orders.index):
          sell_orders.loc[symbol, 'count'] = sell_orders.loc[symbol, 'count'] + abs(row['diff'])
        else:
          sell_orders.loc[symbol, 'count'] = abs(row['diff'])
        sell_orders.loc[symbol, 'current_price']  = row['current_price']
        sell_orders.loc[symbol, 'total_value']    = row['current_price']*abs(row['diff'])
        buy_orders.loc[symbol, 'count'] = 0
    buy_orders = buy_orders[buy_orders['count'] > 0]
    return buy_orders, sell_orders
  
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
  
  def progress(self, value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 50%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))
  
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
      print(str(round(pct))+"%")
      #out.update(self.progress(pct, 100))
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









class Analysis:
  def __init__(self, stats, positions, prices):
    self.stats      = stats
    self.positions  = positions
    self.prices     = prices
  
  def chart2(self):
    zt = TradingZones(prices=self.prices)
    zones = zt.get()
    self.stats = pd.concat([self.stats, zones[zones.index.isin(self.stats.index)]], axis=1)
    self.stats[['sign_up','sig_down']] = None
    self.stats.at[self.stats[self.stats['signal']=='UP'].index, 'sig_up'] = 0
    self.stats.at[self.stats[self.stats['signal']=='DOWN'].index, 'sig_down'] = 0

    #self.stats[['invested','portfolio_value','total_value']].plot(figsize=(30,5))
    self.stats['profits_pct'] = (self.stats['total_value']-self.stats.iloc[0]['total_value'])/self.stats.iloc[0]['total_value']
    self.stats['profits'] = self.stats['total_value']-self.stats.iloc[0]['total_value']
    self.stats['drawdown'] = self.stats['portfolio_value']-self.stats['invested']
    fig, axs = plt.subplots(4, figsize=(30,10))
    fig.suptitle('Analysis')
    for col in ['invested','portfolio_value','total_value']:
      axs[0].plot(self.stats[col], '--', label=col)
      if col == 'total_value':
        axs[0].fill_between(self.stats.index, self.stats.iloc[0]['total_value'], self.stats['total_value'], where=self.stats['total_value'] > self.stats.iloc[0]['total_value'], facecolor='green', interpolate=True, alpha=0.2)
        axs[0].fill_between(self.stats.index, self.stats.iloc[0]['total_value'], self.stats['total_value'], where=self.stats['total_value'] < self.stats.iloc[0]['total_value'], facecolor='red', interpolate=True, alpha=0.2)
    
    # Profits (returns)
    axs[1].fill_between(self.stats.index, 0, self.stats['profits_pct'], where=self.stats['profits_pct'] > 0, facecolor='green', interpolate=True, alpha=0.2)
    axs[1].fill_between(self.stats.index, 0, self.stats['profits_pct'], where=self.stats['profits_pct'] < 0, facecolor='red', interpolate=True, alpha=0.2)
    axs[1].plot(self.stats['profits_pct'], label='Profits (Returns)')
    axs[1].plot(self.stats['mean'], label='Market Mean')
    axs[1].plot(self.stats['sig_up'], 'g--')
    axs[1].plot(self.stats['sig_down'], 'r--')

    # Drawdown (portfolio value - invested value)
    axs[2].plot(self.stats['drawdown'], label='Drawdown')
    axs[2].fill_between(self.stats.index, 0, self.stats['drawdown'], where=self.stats['drawdown'] > 0, facecolor='green', interpolate=True, alpha=0.2)
    axs[2].fill_between(self.stats.index, 0, self.stats['drawdown'], where=self.stats['drawdown'] < 0, facecolor='red', interpolate=True, alpha=0.2)

    # Profits (USD) + Positions
    sellGroup = self.positions.copy()
    sellGroup = sellGroup.set_index('sell_date')
    sellGroup = sellGroup.groupby(sellGroup.index)['profits'].sum()
    profits = pd.DataFrame(index=stats.index, data=np.zeros(len(stats.index)))
    profits['profits'] = sellGroup
    axs[3].plot(profits['profits'][profits['profits']>0], 'gv')
    axs[3].plot(profits['profits'][profits['profits']<0], 'rv')
    axs[3].fill_between(self.stats.index, 0, self.stats['profits'], where=self.stats['profits'] > 0, facecolor='green', interpolate=True, alpha=0.2)
    axs[3].fill_between(self.stats.index, 0, self.stats['profits'], where=self.stats['profits'] < 0, facecolor='red', interpolate=True, alpha=0.2)
    axs[3].plot(self.stats['profits'], label='Profits (Returns)')
    axs[3].plot(self.stats['sig_up'], 'g--')
    axs[3].plot(self.stats['sig_down'], 'r--')


    axs[0].legend(shadow=True, fancybox=True)
    axs[1].legend(shadow=True, fancybox=True)
    axs[2].legend(shadow=True, fancybox=True)
    plt.show()
    return self.stats
  
  def chart(self):
    zt = TradingZones(prices=self.prices)
    zones = zt.get()
    #print(zones)
    self.stats = pd.concat([self.stats, zones[zones.index.isin(self.stats.index)]], axis=1)
    self.stats[['sign_up','sig_down']] = None
    self.stats.at[self.stats[self.stats['signal']=='UP'].index, 'sig_up'] = 0
    self.stats.at[self.stats[self.stats['signal']=='DOWN'].index, 'sig_down'] = 0

    fig, axs = plt.subplots(6, figsize=(30,20))
    fig.suptitle('Analysis')

    # Cash, etc...
    axs[0].plot(self.stats['sig_down']+self.stats.iloc[0]['total_value'], 'r--')
    for col in ['invested','cash','total_value']:
      axs[0].plot(self.stats[col], ':', label=col)
      if col == 'total_value':
        axs[0].fill_between(self.stats.index, self.stats.iloc[0]['total_value'], self.stats['total_value'], where=self.stats['total_value'] > self.stats.iloc[0]['total_value'], facecolor='green', interpolate=True, alpha=0.2)
        axs[0].fill_between(self.stats.index, self.stats.iloc[0]['total_value'], self.stats['total_value'], where=self.stats['total_value'] < self.stats.iloc[0]['total_value'], facecolor='red', interpolate=True, alpha=0.2)
    
    # Profits (returns)
    axs[1].plot(self.stats['closed_profits'], ':', label='Closed profits')
    axs[1].plot(self.stats['closed_profits']+self.stats['open_profits'], ':', label='Open profits')
    axs[1].plot(self.stats['sig_up']+self.stats['closed_profits']+self.stats['open_profits']+100, 'g-', alpha=0.2)
    axs[1].plot(self.stats['sig_down']+self.stats['closed_profits']+self.stats['open_profits']+100, 'r-', alpha=0.2)

    axs[1].fill_between(self.stats.index, self.stats['closed_profits'], self.stats['closed_profits']+self.stats['open_profits'], where=self.stats['closed_profits'] <= self.stats['closed_profits']+self.stats['open_profits'], facecolor='green', interpolate=True, alpha=0.4)
    axs[1].fill_between(self.stats.index, self.stats['closed_profits'], self.stats['closed_profits']+self.stats['open_profits'], where=self.stats['closed_profits'] >= self.stats['closed_profits']+self.stats['open_profits'], facecolor='red', interpolate=True, alpha=0.4)

    axs[1].fill_between(self.stats.index, 0, self.stats['closed_profits']+self.stats['open_profits'], where=self.stats['closed_profits']+self.stats['open_profits']>=0, facecolor='green', interpolate=True, alpha=0.05)
    axs[1].fill_between(self.stats.index, 0, self.stats['closed_profits']+self.stats['open_profits'], where=self.stats['closed_profits']+self.stats['open_profits']<=0, facecolor='red', interpolate=True, alpha=0.05)

    # Chart the gains on sells
    sellGroup = self.positions.copy()
    sellGroup = sellGroup.set_index('sell_date')
    sellGroup = sellGroup.groupby(sellGroup.index)['profits'].sum()
    profits = pd.DataFrame(index=stats.index, data=np.zeros(len(stats.index)))
    profits['profits'] = sellGroup
    axs[1].plot(self.stats['closed_profits']+profits['profits'][profits['profits']>0], 'g+', label='Gains')
    axs[1].plot(self.stats['closed_profits']+profits['profits'][profits['profits']<0], 'r+', label='Losses')

    buyGroup = self.positions.copy()
    buyGroup = buyGroup.set_index('purchase_date')
    buyGroup = buyGroup.groupby(buyGroup.index)['profits'].sum()
    #print(buyGroup)
    profits2 = pd.DataFrame(index=stats.index, data=np.zeros(len(stats.index)))
    profits2['profits'] = buyGroup
    axs[1].plot(profits2['profits'][profits2['profits']>0], 'g.', label='Entries', alpha=0.2)
    axs[1].plot(profits2['profits'][profits2['profits']<0], 'r.', label='Entries', alpha=0.2)
    
    self.stats['open_profits_pct'] = self.stats['open_profits']/self.stats['portfolio_value']
    self.stats['open_profits_pct'] = self.stats['open_profits_pct'].fillna(0)
    axs[2].plot(self.stats['open_profits_pct'], 'g:', label='Open profits')
    axs[2].fill_between(self.stats.index, 0, self.stats['open_profits_pct'], where=self.stats['open_profits_pct']>=0, facecolor='green', interpolate=True, alpha=0.2)
    axs[2].fill_between(self.stats.index, 0, self.stats['open_profits_pct'], where=self.stats['open_profits_pct']<=0, facecolor='red', interpolate=True, alpha=0.2)

    axs[3].plot(self.stats['total_value'], 'g:', label='Total Value')
    axs[3].fill_between(self.stats.index, self.stats.iloc[0]['total_value'], self.stats['total_value'], where=self.stats['total_value'] > self.stats.iloc[0]['total_value'], facecolor='green', interpolate=True, alpha=0.2)
    axs[3].fill_between(self.stats.index, self.stats.iloc[0]['total_value'], self.stats['total_value'], where=self.stats['total_value'] < self.stats.iloc[0]['total_value'], facecolor='red', interpolate=True, alpha=0.2)
    
    axs[4].plot(self.stats.index, zones[zones.index.isin(self.stats.index)][['mean']], 'b:', label='Market Average', alpha=0.2)
    axs[4].plot(self.stats['sig_up']+zones[zones.index.isin(self.stats.index)][['mean']]+100, 'g-', alpha=0.2)
    axs[4].plot(self.stats['sig_down']+zones[zones.index.isin(self.stats.index)][['mean']]+100, 'r-', alpha=0.2)
    
    axs[5].plot(self.stats['closed_profits'], 'r-', alpha=0.5)

    axs[0].legend(shadow=True, fancybox=True)
    axs[1].legend(shadow=True, fancybox=True)
    axs[2].legend(shadow=True, fancybox=True)
    axs[3].legend(shadow=True, fancybox=True)
    axs[4].legend(shadow=True, fancybox=True)
    axs[5].legend(shadow=True, fancybox=True)
    plt.savefig('foo.png')
    return self.stats

  def chartPies(self, figsize=(5,5)):
    holdings = self.positions.copy()
    holdings = holdings.groupby('symbol')

    holdings.count().plot.pie(y='purchase_date', figsize=figsize, title="By Shares", legend=False)
    plt.show()

    sums = holdings.sum()

    sums.plot.pie(y='purchase_price', figsize=figsize, title="By Investment", legend=False)
    plt.show()

    in_profit = sums[sums['profits']>0]
    in_profit.plot.pie(y='profits', figsize=figsize, title="Biggest Winners (value)", legend=False)
    plt.show()

    in_loss = sums[sums['profits']<0].copy()
    in_loss['profits'] = abs(in_loss['profits'])
    in_loss.plot.pie(y='profits', figsize=figsize, title="Biggest Losers (value)", legend=False)
    plt.show()

    in_profit.plot.pie(y='profits_pct', figsize=figsize, title="Biggest Winners (Gains)", legend=False)
    plt.show()

    in_loss['profits_pct'] = abs(in_loss['profits_pct'])
    in_loss.plot.pie(y='profits_pct', figsize=figsize, title="Biggest Losers (Loss)", legend=False)
    plt.show()
  
  def sharpe(self):
    total_value = self.stats[['total_value']]
    returns  = (total_value[:]-total_value[:].loc[list(total_value.index)[0]])/total_value[:].loc[list(total_value.index)[0]]
    return ((252**0.5) * returns.diff().mean() / returns.diff().std())['total_value']
  
  def positionStats(self):
    positives = self.positions[self.positions['profits']>0]
    positives = dict(positives.describe()['profits'])
    negatives = self.positions[self.positions['profits']<0]
    negatives = dict(negatives.describe()['profits'])
    _obj = {
        "startValue": self.stats.iloc[0]['total_value'],
        "endValue":   self.stats.iloc[-1]['total_value'],
        "gains":      (self.stats.iloc[-1]['total_value']-self.stats.iloc[0]['total_value'])/self.stats.iloc[0]['total_value']*100,
        "sharpe":     self.sharpe(),
        "winRate":    positives['count']/(positives['count']+negatives['count'])*100,
        "profitRatio":positives['max']/abs(negatives['min']),
        "maxValue":   self.stats['total_value'].max(),
        "minValue":   self.stats['total_value'].min(),
        "maxGain":    (self.stats['total_value'].max()-self.stats.iloc[0]['total_value'])/self.stats.iloc[0]['total_value']*100,
        "minGain":   (self.stats['total_value'].min()-self.stats.iloc[0]['total_value'])/self.stats.iloc[0]['total_value']*100,
        "dropFromHigh":   (self.stats['total_value'].max()-self.stats.iloc[-1]['total_value'])/self.stats['total_value'].max(),
        "riseFromLow":   (self.stats.iloc[-1]['total_value']-self.stats['total_value'].min())/self.stats['total_value'].min(),
        "profits":      self.stats.iloc[-1]['total_value']-self.stats.iloc[0]['total_value']
    }
    output = pd.DataFrame(columns=['startValue','endValue','profits','gains','minGain','maxGain','minValue','maxValue','dropFromHigh','riseFromLow','profitRatio','winRate','sharpe'])
    output = output.append(_obj, ignore_index=True)
    stats = {
        "positives":  positives,
        "negatives":  negatives,
        "value":      dict(self.stats.describe()['total_value']),
        "profits":    dict(self.positions.describe()['profits_pct']),
    }
    return output, stats








sim    = Sim(period='1y', timedelay=100, window=100, timestep=1, budget=10000, stockPicks=5, sl=-0.04, tp=3.0, ts=0.05, ts_threshold=0.05, avoidDowntrends=True, sellAllOnCrash=False)
stats  = sim.run()



analysis = Analysis(stats=stats, positions=sim.portfolio.holdings, prices=sim.downloader.prices)
analysis.chart()
output, advanced_stats = analysis.positionStats()
print(output)


g = sim.portfolio.holdings.copy().groupby('label').sum()
g['profits_pct'] = (g['current_price']-g['purchase_price'])/g['purchase_price']*100
print(g)