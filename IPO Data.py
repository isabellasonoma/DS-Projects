import pathlib
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
plt.style.use('ggplot')
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import math
import requests
import warnings
warnings.filterwarnings("ignore")


import yfinance as yf

yf.pdr_override()



class IPOClean():
    def data_clean():
        obs = pd.read_csv(r'C:\Users\Isabe\Oxygen Exposure of Southern California Megafauna\HK IPO 2022.csv')
        obs.columns = obs.iloc[0]
        obs = obs.drop(obs.columns[[0]], axis=1)
        obs = obs.drop([0])
        obs.set_index('Stock Code')
        obs[['Funds Raised (HK$)',
                                                                                        ' IPO Subscription Price (HK$)',
                                                                                            'Funds Raised_international',
                                                                                '  IPO Subscription Price_international ',
                                                                                                                'FX rate',
                                                                                                'Authorized Share Capital',
                                                                                                'Total (without option)',
                                                                                        'Global Offering (without option)',
                                                                'Number of offer shares under the capitalization Issue',
                                                                        'Number of offer shares under Capitalization Rest',
                                                                                                            'Sale Shares',
                                                                                                            'New shares ',
                                                                                                        'Placing Shares',
                                                                                                    'Public Offer shares',
                                                                                                    'Maximum Offer Price',
                                                                                                    'Minimum Offer Price',
                                                                            'total assets in year-3 (3 years before IPO)',
                                                                                                'total assets in year-2',
                                                                                                'total assets in year-1',
                                                                                                'total equity in year-3',
                                                                                                'total equity in year-2',
                                                                                                'total equity in year-1',
                                                                                                'total liability in year-3',
                                                                                                'total liability in year-2',
                                                                                                'total liability in year-1',
                                                                                                    'Net sales in year-3',
                                                                                                    'Net sales in year-2',
                                                                                                    'Net sales in year-1',
                                                                                            'Profit before tax in year-3',
                                                                                            'Profit before tax in year-2',
                                                                                            'Profit before tax in year-1',
                                                                                            'Profit for the year in year-3',
                                                                                            'Profit for the year in year-2',
                                                                                            'Profit for the year in year-1',
                                                                                                        'Gross spread ',
                                                                                                'Over-allotment Option (%)',
                                                                                            'Subscription Ratio (times)'
        ]] = obs[['Funds Raised (HK$)',
                                                                                          ' IPO Subscription Price (HK$)',
                                                                                            'Funds Raised_international',
                                                                                '  IPO Subscription Price_international ',
                                                                                                                'FX rate',
                                                                                                'Authorized Share Capital',
                                                                                                 'Total (without option)',
                                                                                        'Global Offering (without option)',
                                                                 'Number of offer shares under the capitalization Issue',
                                                                        'Number of offer shares under Capitalization Rest',
                                                                                                            'Sale Shares',
                                                                                                            'New shares ',
                                                                                                        'Placing Shares',
                                                                                                    'Public Offer shares',
                                                                                                    'Maximum Offer Price',
                                                                                                    'Minimum Offer Price',
                                                                            'total assets in year-3 (3 years before IPO)',
                                                                                                   'total assets in year-2',
                                                                                                    'total assets in year-1',
                                                                                                    'total equity in year-3',
                                                                                                    'total equity in year-2',
                                                                                                    'total equity in year-1',
                                                                                                'total liability in year-3',
                                                                                                'total liability in year-2',
                                                                                                'total liability in year-1',
                                                                                                     'Net sales in year-3',
                                                                                                        'Net sales in year-2',
                                                                                                        'Net sales in year-1',
                                                                                                'Profit before tax in year-3',
                                                                                            'Profit before tax in year-2',
                                                                                            'Profit before tax in year-1',
                                                                                            'Profit for the year in year-3',
                                                                                            'Profit for the year in year-2',
                                                                                            'Profit for the year in year-1',
                                                                                                        'Gross spread ',
                                                                                                'Over-allotment Option (%)',
                                                                                            'Subscription Ratio (times)'    
        ]].apply(pd.to_numeric)
       
        obs['Date of Prospectus (dd/mm/yy)'] = pd.to_datetime(obs['Date of Prospectus (dd/mm/yy)'])
        obs['Date of Listing (dd/mm/yy)'] = pd.to_datetime(obs['Date of Listing (dd/mm/yy)'])   

        return self.xrcobs

class Explore():
    def __innit__(self):
        self.obs = IPOClean.data_clean()

    def SectorGroup(self):
        plot = self.obs.groupby(['Sector']).agg({'Funds Raised (HK$)': ['sum']}).plot.barh(legend = True)
        plot.show()

    def Accountants(self):
        print(self.obs['Reporting Accountants'].value_counts())

    
class Future():  
    def __innit__(self,obs):
        self.obs = IPOClean.data_clean(obs)  

    def Mo3Data(self):
        stock_codes = self.obs['Stock Code'].tolist()
        stock_codes[7] = '0'+stock_codes[7]
        i=0
        while i<42:
            stock_codes[i] = stock_codes[i]+'.HK'
            i= i+1
        
        symbols = stock_codes
        start = '2022-01-03'
        end = '2023-03-08'

        # Read data 
        self.dataset = yf.download(symbols,start,end)['Adj Close']

    def PltStocks(self):
        plt.rcParams["figure.figsize"] = (18,15)
        ax = self.dataset.plot()
        ax.show()

    def PltChange(self):
        pct_change = self.dataset.pct_change()
        plt.rcParams["figure.figsize"] = (22,13)
        percent_change = pct_change.plot(title = 'Percent Change')
        percent_change.show()

        log_change = np.log(self.dataset / self.dataset.shift())
        plt.rcParams["figure.figsize"] = (22,13)
        log_change = log_change.plot(title = 'Log Change')
        percent_change.show()


IPO = Explore()
IPO.SectorGroup()


