import sys
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
import math
from os import path


class Downloader:
  def __init__(self, cache=True, figsize=(30,10)):
    self.figsize  = figsize
    self.cache    = cache
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
      filename  = 'data/'+sector+'.pkl'
      self.symbols = self.getSymbolsBySector(self.sector)
    else:
      if symbols is not None:
        self.symbols = symbols
      filename  = 'data/'+'data_'+period+'_'+(hashlib.sha224((''.join(self.symbols)).encode('utf-8')).hexdigest())+".pkl"
    
    if path.exists(filename) and self.cache==True:
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
    if len(self.symbols)==1:
      self.data.columns = pd.MultiIndex.from_product([[self.symbols[0]], self.data.columns])
    self.prices = self.data.loc[:,pd.IndexSlice[:,'Adj Close']]
    if math.isnan(self.prices.iloc[0].max()):
      self.data = self.data.iloc[1:]
      self.prices = self.data.loc[:,pd.IndexSlice[:,'Adj Close']]
    self.prices.columns = self.prices.columns.droplevel(1)
    self.changes  = (self.prices-self.prices.shift())/self.prices.shift()
    self.returns  = (self.prices[:]-self.prices[:].loc[list(self.prices.index)[0]])/self.prices[:].loc[list(self.prices.index)[0]]
    self.sector_mean = self.returns[self.symbols].T.describe().T[['mean']]
    self.sector_mean_val = self.prices[self.symbols].T.describe().T[['mean']]
    return self.stocklist.groupby('Sector').aggregate(['count'])
  
  def add(self, symbol):
    if symbol in self.symbols:
      return False
    else:
      info = yf.Ticker(symbol)
      self.stocklist = self.stocklist.append({'Symbol':symbol,'Name':info.info['shortName'],'Sector':info.info['sector']}, ignore_index=True)
      self.stocklist.to_csv('constituents.csv', index=False)
