import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math



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
  
  def pick(self, count=5, kwargs={'w_dfh': 0.6, 'w_sharpe': 0.2, 'w_100d': 0.2, 'v_100d': 0.3, 'v_dfh': 0.05, 'v_rfl': 0.01}):
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

    stats["w"] = (stats['dfh'] * kwargs['w_dfh']) + (stats['sharpe'] * kwargs['w_sharpe']) + (stats['100d'] * kwargs['w_100d'])

    stats.sort_values(['w'], ascending=[0], inplace=True)
    selected = stats.copy();
    selected = selected[selected['3d'] > 0]
    #selected = selected[selected['7d'] > 0]
    selected = selected[selected['100d'] > kwargs['v_100d']]
    #selected = selected[selected['sharpe'] > 1]
    selected = selected[(selected['dfh'] >= kwargs['v_dfh']) & (selected['rfl'] >= kwargs['v_rfl'])]
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


