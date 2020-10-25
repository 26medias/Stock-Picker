import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math



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
