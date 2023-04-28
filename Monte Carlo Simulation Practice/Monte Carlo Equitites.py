import math

from scipy import stats
import numpy as np
import pandas as pd
import requests
from pandas_datareader import data as wb
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf


yf.pdr_override()

class Stock():
    def __init__(self, stock, start_date, end_date):
        self.stock = stock
        stock_data = pd.DataFrame()
        stock_data = yf.download(stock, start = start_date, end = end_date)
        self.closing_price = stock_data['Close']
        
    def PltPrice(self): 
        self.closing_price.plot(figsize = (10,6));
        plt.title(self.stock+' Price');
        plt.show();
    
    def PltLogReturns(self):
        log_returns = np.log(1+self.closing_price.pct_change())
        plt.title(self.stock+' Log Returns');
        log_returns.plot(figsize = (10,6));
        plt.show();
        
    def Monte(self):
        log_returns = np.log(1+self.closing_price.pct_change())
        #drift
        u = log_returns.mean()
        var = log_returns.var()
        drift = u-(.5*var)
       
        stdev = log_returns.std()
        norm.ppf(.95)
        x = np.random.rand(10,2)
        Z = norm.ppf(x)
        t_intervals = 365
        iterations = 300
        daily_returns = np.exp(np.array(drift)+np.array(stdev)*norm.ppf(np.random.rand(t_intervals,iterations)))
       
        s0 = self.closing_price.iloc[-1]
        price_list = np.zeros_like(daily_returns)
        price_list[0] = s0
        
        for t in range (1,t_intervals):
            price_list[t] = price_list[t-1]*daily_returns[t]
            
        plt.figure(figsize = (10,6));
        plt.plot(price_list);
        plt.title(self.stock+' Stock Predictions');
        plt.show();
                
        print('Expected price: ', round(np.mean(price_list),2))
        print('Quantile (5%) ',np.percentile((price_list),5))
        print('Quantile (95%) ',np.percentile((price_list),95))  
        
ORCL = Stock('ORCL', '2012-01-01', '2022-04-27')
MELI = Stock('MELI', '2012-01-01', '2022-04-27')

ORCL.PltPrice()
ORCL.PltLogReturns()
ORCL.Monte()

MELI.PltPrice()
MELI.PltLogReturns()
MELI.Monte()