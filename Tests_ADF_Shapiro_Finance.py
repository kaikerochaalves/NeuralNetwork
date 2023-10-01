# -*- coding: utf-8 -*-
"""
Created on Sep 2023

@author: Kaike Alves
"""

# Import the series
import yfinance as yf

# Shapiro-Wilk test for normality
from scipy.stats import shapiro

# Augmented Dickey-Fuller test for stationarity
from statsmodels.tsa.stattools import adfuller

#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------
    
Serie = "S&P 500"
    
horizon = 1
ncols = 4
    
# Importing the data
SP500 = yf.Ticker('^GSPC')
SP500 = SP500.history(start = "2020-01-01", end = "2022-01-01", interval='1d')

# Prepare the data
columns = SP500.columns
Data = SP500[columns[3]]

# Defining the atributes and the target value
X = Data.values

#-----------------------------------------------------------------------------
# Implement tests
#-----------------------------------------------------------------------------

# Perform Shapiro-Wilk test
shapiro_test = shapiro(X)
print(f'\nShapiro test centre: {shapiro_test}')

# Perform augmented Dickey-Fuller test
adf = adfuller(X)
print(f'\nADF test: {adf}')


#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------
    
Serie = "NASDAQ"
    
horizon = 1
ncols = 4
    
# Importing the data
NASDAQ = yf.Ticker('^IXIC')
NASDAQ = NASDAQ.history(start = "2020-01-01", end = "2022-01-01", interval='1d')

# Prepare the data
columns = NASDAQ.columns
Data = NASDAQ[columns[3]]

# Defining the atributes and the target value
X = Data.values

#-----------------------------------------------------------------------------
# Implement tests
#-----------------------------------------------------------------------------

# Perform Shapiro-Wilk test
shapiro_test = shapiro(X)
print(f'\nShapiro test centre: {shapiro_test}')

# Perform augmented Dickey-Fuller test
adf = adfuller(X)
print(f'\nADF test: {adf}')

#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "TAIEX"
    
horizon = 1
ncols = 4
    
# Importing the data
TAIEX = yf.Ticker('^TWII')
TAIEX = TAIEX.history(start = "2020-01-01", end = "2022-01-01", interval='1d')

# Prepare the data
columns = TAIEX.columns
Data = TAIEX[columns[3]]

# Defining the atributes and the target value
X = Data.values

#-----------------------------------------------------------------------------
# Implement tests
#-----------------------------------------------------------------------------

# Perform Shapiro-Wilk test
shapiro_test = shapiro(X)
print(f'\nShapiro test centre: {shapiro_test}')

# Perform augmented Dickey-Fuller test
adf = adfuller(X)
print(f'\nADF test: {adf}')