# -*- coding: utf-8 -*-
"""
Created on Sep 2023

@author: Kaike Alves
"""

# Import libraries
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import statistics as st
import matplotlib.pyplot as plt

# Neural Network
from tensorflow import keras
from keras.utils.vis_utils import plot_model

# Feature scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Import the series
import yfinance as yf

#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "NASDAQ"

horizon = 1
    
# Importing the data
NASDAQ = yf.Ticker('^IXIC')
NASDAQ = NASDAQ.history(start = "2020-01-01", end = "2022-01-01", interval='1d')

# Prepare the data
columns = NASDAQ.columns
Data = NASDAQ[columns[:4]]

# Add the target column value
NextClose = Data.iloc[horizon:,-1].values
Data = Data.drop(Data.index[-horizon:])
Data['NextClose'] = NextClose

# Convert to array
X = Data[Data.columns[:-1]].values
y = Data[Data.columns[-1]].values

# Spliting the data into train and test
n = Data.shape[0]
training_size = round(n*0.8)
X_train, X_test = X[:training_size,:], X[training_size:,:]
y_train, y_test = y[:training_size], y[training_size:]

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# Executing the simulations for the NASDAQ time series
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Load the best network and run predictions
# -----------------------------------------------------------------------------

# Load the model
model = keras.models.load_model(f'GridSearchResults/{Serie}.h5')

# Implement the prediction method
y_pred = model.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Serie}.eps', format='eps', dpi=1200)
plt.show()

# Print the summary of the model
print(model.summary())

# Plot the model architeture
# You must install pydot (`pip install pydot`) and install graphviz (https://graphviz.gitlab.io/download/).
plot_model(model, to_file=f'ModelArchiteture/{Serie}.png', show_shapes=True, show_layer_names=True)

