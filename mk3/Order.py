import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math



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

