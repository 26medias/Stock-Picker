import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

from Sim import Sim
from Analysis import Analysis

sim    = Sim(period='1y', timedelay=100, window=100, timestep=1, budget=10000, stockPicks=5, sl=-0.04, tp=3.0, ts=0.05, ts_threshold=0.05, avoidDowntrends=True, sellAllOnCrash=False)
stats  = sim.run()

analysis = Analysis(stats=stats, positions=sim.portfolio.holdings, prices=sim.downloader.prices)
analysis.chart('data/output_1y.png')
output, advanced_stats = analysis.positionStats()
print(output)


g = sim.portfolio.holdings.copy().groupby('label').sum()
g['profits_pct'] = (g['current_price']-g['purchase_price'])/g['purchase_price']*100
print(g)