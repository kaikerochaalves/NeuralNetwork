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

# Including to the path another fold
import sys

# Including to the path another fold
sys.path.append(r'Functions')
# Import the serie generator
from LorenzAttractorGenerator import Lorenz

#-----------------------------------------------------------------------------
# Generate the time series
#-----------------------------------------------------------------------------

Serie = "Lorenz"

# Input parameters
x0 = 0.
y0 = 1.
z0 = 1.05
sigma = 10
beta = 2.667
rho=28
num_steps = 10000

# Creating the Lorenz Time Series
x, y, z = Lorenz(x0 = x0, y0 = y0, z0 = z0, sigma = sigma, beta = beta, rho = rho, num_steps = num_steps)

# Ploting the graphic
plt.rc('font', size=10)
plt.rc('axes', titlesize=15)
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x, y, z, lw = 0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

def Create_Leg(data, ncols, leg, leg_output = None):
    X = np.array(data[leg*(ncols-1):].reshape(-1,1))
    for i in range(ncols-2,-1,-1):
        X = np.append(X, data[leg*i:leg*i+X.shape[0]].reshape(-1,1), axis = 1)
    X_new = np.array(X[:,-1].reshape(-1,1))
    for col in range(ncols-2,-1,-1):
        X_new = np.append(X_new, X[:,col].reshape(-1,1), axis=1)
    if leg_output == None:
        return X_new
    else:
        y = np.array(data[leg*(ncols-1)+leg_output:].reshape(-1,1))
        return X_new[:y.shape[0],:], y

# Defining the atributes and the target value
X = np.concatenate([x[:-1].reshape(-1,1), y[:-1].reshape(-1,1), z[:-1].reshape(-1,1)], axis = 1)
y = x[1:].reshape(-1,1)

# Spliting the data into train and test
X_train, X_test = X[:8000,:], X[8000:,:]
y_train, y_test = y[:8000,:], y[8000:,:]

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.show()

# -----------------------------------------------------------------------------
# Executing the Random Search for the Lorenz time series
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

