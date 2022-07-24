# %%
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from pandas_datareader import data
import pandas_datareader.data as web
# %% [markdown]
# import by web.DataReader

# %%
DAX = web.DataReader(name = '^GDAXI', data_source = 'yahoo', start = '2015-01-01')
DAX.info()
DAX.tail()
# %% [markdown]
# import by yahoo finance

# %%
import yfinance as yf
TSMC = yf.download('2330.TW', start = '2015-01-01')
TSMC.tail()

# %% [markdown]
# data visualizing and strat
# %%
# calculating return
TSMC['return'] = 0.0
for i in range(1, len(TSMC)):
    TSMC['return'][i] = (TSMC['Close'][i] - TSMC['Close'][i-1]) / TSMC['Close'][i-1]

TSMC['return'].tail()
figure = plt.figure(figsize = (24, 8))
plot1 = figure.add_subplot(1, 2, 1)
plt2 = figure.add_subplot(1, 2, 2)
plt.plot(TSMC['return'])
plt.grid('True')
_ = plot1.plot(TSMC['Close'])
plt.grid('True')

# another method of subplot
TSMC[['Close', 'return']].plot(subplots = True, color = 'blue', figsize = (12, 8))
# %%
# calculating Simple Moving Average line

TSMC['5SMA'] = TSMC['Close'].rolling(5).mean()
TSMC['13SMA'] = TSMC['Close'].rolling(13).mean()
TSMC['21SMA'] = TSMC['Close'].rolling(21).mean()

TSMC[['Close', '5SMA', '13SMA', '21SMA']].tail()
TSMC[['Close', '5SMA', '13SMA', '21SMA']].plot(figsize = (12, 8))


