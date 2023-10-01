# -*- coding: utf-8 -*-
"""
Created on Sep 2023

@author: Kaike Alves
"""

# Import libraries
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import statistics as st
import matplotlib.pyplot as plt

# Neural Network
from tensorflow import keras
from keras.utils.vis_utils import plot_model
# Optimize Network
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasRegressor

# Feature scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Import the series
import yfinance as yf

#-----------------------------------------------------------------------------
# Import the time series
#-----------------------------------------------------------------------------

Serie = "TAIEX"

horizon = 1
    
# Importing the data
TAIEX = yf.Ticker('^TWII')
TAIEX = TAIEX.history(start = "2020-01-01", end = "2022-01-01", interval='1d')

# Prepare the data
columns = TAIEX.columns
Data = TAIEX[columns[:4]]

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
# Executing the Random Search for the TAIEX time series
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Optimize hyper-parameters
# -----------------------------------------------------------------------------

# Define the function to create models for the optimization method
def build_model(n_hidden=1, n_neurons=30, activation = "relu", learning_rate=3e-3, input_shape=[4]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation=activation))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# Wrapper around eras model built
keras_reg = KerasRegressor(model=build_model, n_hidden=1, n_neurons=30, activation = "relu", learning_rate=3e-3, input_shape=[4])

# Random search options
param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1,100),
    # Some activation functions
    #"deserialize", "get", "mish", "serialize", "sigmoid", 
    # "elu", "exponential", "gelu", "hard_sigmoid", "linear", "relu", "selu", "sigmoid", "softplus", "softsign", "swish", "tanh"
    "activation": ["elu", "exponential", "gelu", "hard_sigmoid", "linear", "relu", "selu", "sigmoid", "softplus", "softsign", "swish", "tanh"],
    "learning_rate": reciprocal(1e-5,0.5),
    "input_shape": [X_train.shape[1]]}

# Call the Random Search function
rnd_search_cv = RandomizedSearchCV(estimator=keras_reg, param_distributions=param_distribs, n_iter=100, cv = 3)
# Start the optimization
rnd_search_cv.fit(X_train, y_train, epochs = 100, validation_data=(X_test, y_test), callbacks=[keras.callbacks.EarlyStopping(patience=10)])

# Print the best model parameters
print(f'\nBest parameters:\n {rnd_search_cv.best_params_}')

# Print the best model score
print(f'\nBest score:\n {rnd_search_cv.best_score_}\n\n')

# Implement the prediction method
y_pred = rnd_search_cv.predict(X_test)

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


# -----------------------------------------------------------------------------
# Run and save the network for the best hyper-parameters
# -----------------------------------------------------------------------------

# Define the neural network
model = build_model(**rnd_search_cv.best_params_)

# Checkpoint functions to recover the best model
checkpoint_cb = keras.callbacks.ModelCheckpoint(f'GridSearchResults/{Serie}.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs = 100, validation_data=(X_test, y_test), callbacks=[checkpoint_cb, early_stopping_cb])

# Compute the mse error
#mse_test = model.evaluate(X_test, y_test)

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

