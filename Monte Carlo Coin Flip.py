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


import warnings
warnings.filterwarnings("ignore")


class Coin:
    def coin_flip():
            return np.random.randint(0,2)
    
    def Monte_Carlo(n):  
        results = 0
        list1=[]

        for i in range(n):
            flip_result = Coin.coin_flip()
            results = results + flip_result

            #calc probabilities vals
            prob_value = results/(i+1)

            #append the prob values to the list
            list1.append(prob_value)       
        
            plt.axhline(y=.5,color='r',linestyle='-')
            plt.title('Coin Flip Probability');
            plt.xlabel('iteration');
            plt.ylabel('probability');
            plt.plot(list1);
        return results/n


Coin.Monte_Carlo(1000)
plt.show();