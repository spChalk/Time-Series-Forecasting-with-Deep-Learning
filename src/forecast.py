"""
Software Development for Algorithmic Problems
Assignment 3 - Supervised Learning with Neural Networks
---
Team no.59
Maraziaris Charalampos - 1115 2018 00105
Chalkias Spyridon - 1115 2018 00209
"""

import datetime
import os
import random
from math import sqrt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import (LSTM, BatchNormalization, Bidirectional, Conv1D,
                          Dense, Dropout, Flatten, Input, MaxPooling1D,
                          RepeatVector, Reshape, UpSampling1D)
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from preprocess import (reproducibleResults, get_dataset, stack_columns,
                        plot_loss_vs_epochs_diagram, visualise_prediction,
                        split_datasets)

import argparse

'''
    Recurrent Neural Network with stacked LSTM layers. 
'''
class LSTM_Model(tf.keras.Model):
    '''
        Input: 
        - lstm_dims: An iterable representing the number of stacked LSTM layers with 
                        their respective number of units. e.g: [5, 7, 9]
        - dropout: Dropout ratio for every LSTM layer 
    '''
    def __init__(self, lstm_dims, dropout=.2):
        super(LSTM_Model, self).__init__()
        
        self.multi_lstm_model = tf.keras.Sequential()
        for i in range(0, len(lstm_dims) - 1):
            self.multi_lstm_model.add(LSTM(units=lstm_dims[i], return_sequences=True, dropout=dropout))
        self.multi_lstm_model.add(LSTM(units=lstm_dims[-1], dropout=dropout))
        self.multi_lstm_model.add(Dense(units = 1))

    '''
        Loads data into the model.
    '''
    def load_data(self, training_set_scaled, n_time_steps=60, lag=1):
        
        self.n_time_steps = n_time_steps
        self.lag = lag

        # Create a data structure with the given number of lag and time-steps
        self.X_train = []
        self.y_train = []
        for i in range(n_time_steps, len(training_set_scaled)):
            self.X_train.append(training_set_scaled[i - n_time_steps : i : lag, 0])
            self.y_train.append(training_set_scaled[i, 0])
        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))

    '''
        Defines early stopping, compiles and runs the model on the loaded data
    '''
    def fit(self, batch_size=32, max_epochs=100):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=2,
                                                    mode='min')
        
        self.multi_lstm_model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

        return self.multi_lstm_model.fit(self.X_train, self.y_train, epochs = max_epochs, batch_size = batch_size, 
                                                validation_split=0.1, callbacks=[early_stopping])

    def predict(self, label, inputs, test_set):
        X_test = []
        for i in range(self.n_time_steps, len(test_set) + self.n_time_steps):
            X_test.append(inputs[i - self.n_time_steps : i : self.lag, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_stock_price = self.multi_lstm_model.predict(X_test)

        return predicted_stock_price

    def save(self, label):
        self.multi_lstm_model.save(label)

'''
    Creates a model with the given set of hyperparmeters, 
    scales the data, trains on a single stock price and makes
    predictions on that stock.
'''
def A__train_model_for_a_stock(df, hpars):

    # Gather hyperparameters
    label = hpars['label']
    dropout = hpars['dropout']
    lstm_dims = hpars['lstm_dims']
    n_time_steps = hpars['n_time_steps']
    batch_size = hpars['batch_size']
    max_epochs = hpars['max_epochs']
    lag = hpars['lag']

    # Split and scale the data
    training_set, test_set = split_datasets(df, label)
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    test_inputs = df[label][len(df[label]) - len(test_set) -n_time_steps:].values.reshape(-1,1)
    test_inputs = sc.transform(test_inputs)

    # Create a model and fit the data
    model = LSTM_Model(lstm_dims, dropout)
    model.load_data(training_set_scaled, n_time_steps, lag)
    history = model.fit(batch_size, max_epochs)
    plot_loss_vs_epochs_diagram(history.history['loss'], history.history['val_loss'], label)

    # Make prediction and visualise it
    predicted_stock_price = model.predict(label, test_inputs, test_set)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)    

    visualise_prediction(test_set, predicted_stock_price, label)

    model.save(f'./saved_models/forecast_model_trained_on_{label.upper()}.h5')

'''
    Creates a model with the given set of hyperparmeters, 
    scales the data, trains on the first N stock prices and makes
    predictions on them.
'''
def A__train_model_for_N_stocks(df, hpars, N):

    dropout = hpars['dropout']
    lstm_dims = hpars['lstm_dims']
    n_time_steps = hpars['n_time_steps']
    batch_size = hpars['batch_size']
    max_epochs = hpars['max_epochs']
    lag = hpars['lag']

    df = df.iloc[: , :N]

    training_set, test_set = train_test_split(df, test_size=.3, shuffle=False)
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(stack_columns(training_set).values)

    model = LSTM_Model(lstm_dims, dropout)
    model.load_data(training_set_scaled, n_time_steps, lag)
    history = model.fit(batch_size, max_epochs)
    plot_loss_vs_epochs_diagram(history.history['loss'], history.history['val_loss'], f'the first {N} stocks')

    for label in list(test_set.columns):
        test_inputs = df[label][len(df[label]) - len(test_set[label]) - n_time_steps:].values.reshape(-1,1)
        test_inputs = sc.transform(test_inputs)

        predicted_stock_price = model.predict(label, test_inputs, test_set[label].to_numpy().reshape(-1, 1))
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)    

        visualise_prediction(test_set[label].values, predicted_stock_price, label)

    model.save(f'./saved_models/forecast_model_trained_on_all_stocks.h5')

def A__test_model_on_a_stock(df, loaded_model, hpars, label):

    label = label.lower()
    n_time_steps = hpars['n_time_steps']
    lag = hpars['lag']
    loaded_model.summary()

    # Split and scale the data
    training_set, test_set = split_datasets(df, label)
    sc = MinMaxScaler(feature_range = (0, 1))
    sc.fit_transform(training_set)

    test_inputs = df[label][len(df[label]) - len(test_set) -n_time_steps:].values.reshape(-1,1)
    test_inputs = sc.transform(test_inputs)

    # Make prediction and visualise 
    X_test = []
    for i in range(n_time_steps, len(test_set) + n_time_steps):
        X_test.append(test_inputs[i - n_time_steps : i : lag, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = loaded_model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)    

    visualise_prediction(test_set, predicted_stock_price, label)    

def A__test_model_on_N_stocks(df, loaded_model, hpars, N):
    
    n_time_steps = hpars['n_time_steps']
    lag = hpars['lag']
    loaded_model.summary()

    df = df.iloc[: , :N]

    training_set, test_set = train_test_split(df, test_size=.3, shuffle=False)
    sc = MinMaxScaler(feature_range = (0, 1))
    sc.fit_transform(stack_columns(training_set).values)

    for label in list(test_set.columns):
        test_inputs = df[label][len(df[label]) - len(test_set[label]) - n_time_steps:].values.reshape(-1,1)
        test_inputs = sc.transform(test_inputs)

        # Make prediction and visualise 
        X_test = []
        for i in range(n_time_steps, len(test_set[label].to_numpy().reshape(-1, 1)) + n_time_steps):
            X_test.append(test_inputs[i - n_time_steps : i : lag, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_stock_price = loaded_model.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)    

        visualise_prediction(test_set[label].values, predicted_stock_price, label)

'''
- `lstm_dims`: Number of stacked LSTM layers and their corresponding number of units. e.g: [50, 50] stands for 2 stacked LSTM layers each one made up of 50 units.
- `label`: 
    - In case `train_model_for_a_stock` is run, label corresponds to the requested stock to train on.
    - In case `train_model_for_N_stocks` is run, it is being ignored.
- `dropout`: Dropout probability on LSTM layers.
- `n_time_steps`: The number of steps the model will look in order to predict a label. e.g [x1, x2, x3] -> [y1] stands for `n_time_steps=3`
- `lag`: The parsing step in order to predict a label. e.g [1,2,3,4,5] => [6] corresponds to `lag=1`, while [1,3,5] => [6] corresponds to `lag=2` etc.
- `batch_size`: Batch size for training
- `max_epochs`: Number of maximum epochs to train the model.
'''
if __name__ == "__main__":

    '''Usage: python3 forecast.py –d <dataset> -n <number of time series selected>'''

    seed = 420
    reproducibleResults(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-n', '--number_of_time_series_selected')
    args = parser.parse_args()

    df = get_dataset(args.dataset)
    
    if args.number_of_time_series_selected:
        num_of_timeseries = int(args.number_of_time_series_selected)
    
        hyperparameters = {'lstm_dims':[100, 100], 'dropout':0.25, 'n_time_steps':60, 'lag':2, 'batch_size':128, 'max_epochs':100}
        #A__train_model_for_N_stocks(df, hyperparameters, num_of_timeseries)

        loaded_model = tf.keras.models.load_model('./saved_models/forecast_model_trained_on_all_stocks.h5')
        A__test_model_on_N_stocks(df, loaded_model, hyperparameters, num_of_timeseries)
    else:
        hyperparameters = {'lstm_dims':[500, 500, 500, 500, 500], 'dropout':0.25, 'label':'aapl', 'n_time_steps':60, 'lag':2, 'batch_size':128, 'max_epochs':100}
        #A__train_model_for_a_stock(df, hyperparameters)

        loaded_model = tf.keras.models.load_model('./saved_models/forecast_model_trained_on_AAPL.h5')
        A__test_model_on_a_stock(df, loaded_model, hyperparameters, 'AAPL')

### EOF ###