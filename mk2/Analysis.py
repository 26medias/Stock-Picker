import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

from TradingZones import TradingZones

class Analysis:
  def __init__(self, stats, positions, prices, neptune=None):
    self.neptune	= neptune # neptune.ai logging
    self.stats      = stats
    self.positions  = positions
    self.prices     = prices
  
  def chart(self, filename=None):
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
    profits = pd.DataFrame(index=self.stats.index, data=np.zeros(len(self.stats.index)))
    profits['profits'] = sellGroup
    axs[1].plot(self.stats['closed_profits']+profits['profits'][profits['profits']>0], 'g+', label='Gains')
    axs[1].plot(self.stats['closed_profits']+profits['profits'][profits['profits']<0], 'r+', label='Losses')

    buyGroup = self.positions.copy()
    buyGroup = buyGroup.set_index('purchase_date')
    buyGroup = buyGroup.groupby(buyGroup.index)['profits'].sum()
    #print(buyGroup)
    profits2 = pd.DataFrame(index=self.stats.index, data=np.zeros(len(self.stats.index)))
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
    
    if self.neptune is not None:
      self.neptune.log_image('Output', fig)
    elif filename is not None:
      plt.savefig(filename)
    else:
      plt.show()
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

