# -*- coding: utf-8 -*-
"""
Created on Sep 2023

@author: Kaike Alves
"""

# Import libraries
import pandas as pd

# Shapiro-Wilk test for normality
from scipy.stats import shapiro

# Augmented Dickey-Fuller test for stationarity
from statsmodels.tsa.stattools import adfuller


#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "Alice_91_Site_1A_Trina_Power"

horizon = 1
    
# Importing the data
Data = pd.read_excel(f'Datasets/{Serie}.xlsx')

# Defining the atributes and the target value
y = Data[Data.columns[13]].values

#-----------------------------------------------------------------------------
# Implement tests
#-----------------------------------------------------------------------------

# Perform Shapiro-Wilk test
shapiro_test = shapiro(y)
print(f'\nShapiro test centre: {shapiro_test}')

# Perform augmented Dickey-Fuller test
adf = adfuller(y)
print(f'\nADF test: {adf}')


#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "Alice_59_Site_38_QCELLS_Power"

horizon = 1
    
# Importing the data
Data = pd.read_excel(f'Datasets/{Serie}.xlsx')

# Defining the atributes and the target value
y = Data[Data.columns[13]].values

#-----------------------------------------------------------------------------
# Implement tests
#-----------------------------------------------------------------------------

# Perform Shapiro-Wilk test
shapiro_test = shapiro(y)
print(f'\nShapiro test centre: {shapiro_test}')

# Perform augmented Dickey-Fuller test
adf = adfuller(y)
print(f'\nADF test: {adf}')


#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "Yulara_5_Site_1_Power"

horizon = 1
    
# Importing the data
Data = pd.read_excel(f'Datasets/{Serie}.xlsx')

# Defining the atributes and the target value
y = Data[Data.columns[16]].values

#-----------------------------------------------------------------------------
# Implement tests
#-----------------------------------------------------------------------------

# Perform Shapiro-Wilk test
shapiro_test = shapiro(y)
print(f'\nShapiro test centre: {shapiro_test}')

# Perform augmented Dickey-Fuller test
adf = adfuller(y)
print(f'\nADF test: {adf}')


#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "Yulara_8_Site_5_Power"

horizon = 1
    
# Importing the data
Data = pd.read_excel(f'Datasets/{Serie}.xlsx')

# Defining the atributes and the target value
y = Data[Data.columns[16]].values

#-----------------------------------------------------------------------------
# Implement tests
#-----------------------------------------------------------------------------

# Perform Shapiro-Wilk test
shapiro_test = shapiro(y)
print(f'\nShapiro test centre: {shapiro_test}')

# Perform augmented Dickey-Fuller test
adf = adfuller(y)
print(f'\nADF test: {adf}')