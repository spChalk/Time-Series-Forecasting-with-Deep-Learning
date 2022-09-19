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
                        split_datasets, detect_and_plot_anomalies)

import argparse

'''
    Autoencoder with stacked LSTM layers. 
'''
class LSTM_Autoencoder(tf.keras.Model):
    def __init__(self):
        super(LSTM_Autoencoder, self).__init__()
    '''
        Builds a model with the given parameters.
        Input: 
        - lstm_dims: An iterable representing the first half of stacked LSTM layers with 
                        their respective number of units. e.g: [10, 5, 2]
            NOTE: Due to the nature of the autoencoder, the given layers are mirrored 
                    in the model. For example, the above input is translated into [10, 5, 2, 2, 5, 10]  
        
        - dropout: Dropout ratio for every LSTM layer 
    '''
    def build_model(self, lstm_dims, dropout=.2):    
            
        self.multi_lstm_model = tf.keras.Sequential()
        for i in range(0, len(lstm_dims) - 1):
            self.multi_lstm_model.add(LSTM(units=lstm_dims[i], return_sequences=True, dropout=dropout))
        self.multi_lstm_model.add(LSTM(units=lstm_dims[-1], dropout=dropout))

        self.multi_lstm_model.add(tf.keras.layers.RepeatVector(n=self.X_train.shape[1]))
        
        for i in range(1, len(lstm_dims) + 1):
            self.multi_lstm_model.add(LSTM(units=lstm_dims[len(lstm_dims) - i], return_sequences=True, dropout=dropout))

        self.multi_lstm_model.add(
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=self.X_train.shape[2])
            )
        )

    def load_data(self, training_set_scaled, n_time_steps=60, lag=1):
        
        self.n_time_steps = n_time_steps
        self.lag = lag

        self.X_train,  self.y_train = [], []
        for i in range(len(training_set_scaled) - n_time_steps):
            v = training_set_scaled[i:(i + n_time_steps):lag]
            self.X_train.append(v)        
            self.y_train.append(training_set_scaled[i + n_time_steps])
        self.X_train,  self.y_train = np.array(self.X_train), np.array(self.y_train)   

    def fit(self, batch_size=32, max_epochs=100):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=2,
                                                    mode='min')

        self.multi_lstm_model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

        return self.multi_lstm_model.fit(self.X_train, self.y_train, epochs = max_epochs, batch_size = batch_size, 
                                                validation_split=0.1, callbacks=[early_stopping], shuffle=False)

    def predict(self, label, test_set_scaled, test_set):

        self.X_test,  self.y_test = [], []
        for i in range(len(test_set_scaled) - self.n_time_steps):
            v = test_set_scaled[i:(i + self.n_time_steps):self.lag]
            self.X_test.append(v)        
            self.y_test.append(test_set_scaled[i + self.n_time_steps])
        self.X_test, self.y_test = np.array(self.X_test), np.array(self.y_test)

        # Calculate the Mean Absolute Error (MAE) in order to spot anomalies
        train_pred = self.multi_lstm_model.predict(self.X_train)
        train_mae_loss = np.mean(np.abs(train_pred - self.X_train), axis=1)

        test_pred = self.multi_lstm_model.predict(self.X_test)
        test_mae_loss = np.mean(np.abs(test_pred - self.X_test), axis=1)

        return train_mae_loss, test_mae_loss

    def save(self, label):
        self.multi_lstm_model.save(label)

def B__train_model_for_a_stock(df, hpars):

    label = hpars['label']
    dropout = hpars['dropout']
    lstm_dims = hpars['lstm_dims']
    n_time_steps = hpars['n_time_steps']
    batch_size = hpars['batch_size']
    max_epochs = hpars['max_epochs']
    lag = hpars['lag']
    threshold = hpars['anomaly_threshold']

    training_set, test_set = split_datasets(df, label)
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    test_set_scaled = sc.transform(test_set)

    model = LSTM_Autoencoder()
    model.load_data(training_set_scaled, n_time_steps, lag)
    model.build_model(lstm_dims, dropout)
    history = model.fit(batch_size, max_epochs)
    plot_loss_vs_epochs_diagram(history.history['loss'], history.history['val_loss'], label)

    train_mae_loss, test_mae_loss = model.predict(label, test_set_scaled, test_set)
    
    plt.figure(figsize = [12, 7])
    sns.histplot(train_mae_loss, bins=50, kde=True, stat="density", linewidth=1)
    plt.title(f'Anomaly Detector for {label}')
    plt.show()

    # If the anomaly threshold is not given by the user, then it is 
    # being automatically computed by taking the max of the training set's MAE
    if hpars['anomaly_threshold'] is None:
        threshold = float(max(train_mae_loss))

    detect_and_plot_anomalies(threshold, test_set, test_mae_loss, label, n_time_steps)
    visualise_prediction(test_set, sc.inverse_transform(test_mae_loss), label)

    model.save(f'./saved_models/detect_model_trained_on_{label.upper()}.h5')

def B__train_model_for_N_stocks(df, hpars, N):

    dropout = hpars['dropout']
    lstm_dims = hpars['lstm_dims']
    n_time_steps = hpars['n_time_steps']
    batch_size = hpars['batch_size']
    max_epochs = hpars['max_epochs']
    lag = hpars['lag']
    threshold = hpars['anomaly_threshold']

    df = df.iloc[: , :N]

    training_set, test_set = train_test_split(df, test_size=.2, shuffle=False)
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(stack_columns(training_set).values)

    if not hpars['anomaly_threshold'] is None:
        threshold = hpars['anomaly_threshold']

    model = LSTM_Autoencoder()
    model.load_data(training_set_scaled, n_time_steps, lag)
    model.build_model(lstm_dims, dropout)
    history = model.fit(batch_size, max_epochs)
    plot_loss_vs_epochs_diagram(history.history['loss'], history.history['val_loss'], f'the first {N} stocks')

    for k, label in enumerate(list(test_set.columns)):
        test_inputs = sc.transform(test_set[label].to_numpy().reshape(-1, 1))

        train_mae_loss, test_mae_loss = model.predict(label, test_inputs, test_set[label].to_numpy().reshape(-1, 1))
    
        if k == 0:
            plt.figure(figsize = [12, 7])
            sns.histplot(train_mae_loss, bins=50, kde=True, stat="density", linewidth=1)
            plt.title(f'Anomaly Detector for the training set')
            plt.show()

        if hpars['anomaly_threshold'] is None:
            threshold = float(max(train_mae_loss))

        detect_and_plot_anomalies(threshold, test_set[label].to_numpy().reshape(-1, 1), test_mae_loss, label, n_time_steps)
        visualise_prediction(test_set[label].values, sc.inverse_transform(test_mae_loss), label) 

    model.save(f'./saved_models/detect_model_trained_on_all_stocks.h5')

def B__test_model_on_a_stock(df, loaded_model, hpars, label, mae):

    label = label.lower()
    n_time_steps = hpars['n_time_steps']
    lag = hpars['lag']
    loaded_model.summary()

    training_set, test_set = split_datasets(df, label)
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    test_set_scaled = sc.transform(test_set)

    X_train = []
    for i in range(len(training_set_scaled) - n_time_steps):
        v = training_set_scaled[i:(i + n_time_steps):lag]
        X_train.append(v)        
    X_train = np.array(X_train)

    X_test = []
    for i in range(len(test_set_scaled) - n_time_steps):
        v = test_set_scaled[i:(i + n_time_steps):lag]
        X_test.append(v)        
    X_test = np.array(X_test)

    # Calculate the Mean Absolute Error (MAE) in order to spot anomalies
    train_pred = loaded_model.predict(X_train)
    train_mae_loss = np.mean(np.abs(train_pred - X_train), axis=1)

    test_pred = loaded_model.predict(X_test)
    test_mae_loss = np.mean(np.abs(test_pred - X_test), axis=1)

    plt.figure(figsize = [12, 7])
    sns.histplot(train_mae_loss, bins=50, kde=True, stat="density", linewidth=1)
    plt.title(f'Anomaly Detector for {label}')
    plt.show()
    
    if mae is None:
        if hpars['anomaly_threshold'] is None:
            mae = float(max(train_mae_loss))
        else:
            mae = hpars['anomaly_threshold']

    detect_and_plot_anomalies(mae, test_set, test_mae_loss, label, n_time_steps)
    visualise_prediction(test_set, sc.inverse_transform(test_mae_loss), label)  

def B__test_model_on_N_stocks(df, loaded_model, hpars, N, mae):
    
    n_time_steps = hpars['n_time_steps']
    lag = hpars['lag']
    loaded_model.summary()

    df = df.iloc[: , :N]

    training_set, test_set = train_test_split(df, test_size=.3, shuffle=False)
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(stack_columns(training_set).values)
    
    X_train = []
    for i in range(len(training_set_scaled) - n_time_steps):
        v = training_set_scaled[i:(i + n_time_steps):lag]
        X_train.append(v)        
    X_train = np.array(X_train) 

    for k, label in enumerate(list(test_set.columns)):
        test_set_scaled = sc.transform(test_set[label].to_numpy().reshape(-1, 1))
        X_test = []
        for i in range(len(test_set_scaled) - n_time_steps):
            v = test_set_scaled[i:(i + n_time_steps):lag]
            X_test.append(v)        
        X_test = np.array(X_test)

        # Calculate the Mean Absolute Error (MAE) in order to spot anomalies
        train_pred = loaded_model.predict(X_train)
        train_mae_loss = np.mean(np.abs(train_pred - X_train), axis=1)

        test_pred = loaded_model.predict(X_test)
        test_mae_loss = np.mean(np.abs(test_pred - X_test), axis=1)

        if k == 0:
            plt.figure(figsize = [12, 7])
            sns.histplot(train_mae_loss, bins=50, kde=True, stat="density", linewidth=1)
            plt.title(f'Anomaly Detector for the training set')
            plt.show()
    
        if mae is None:
            if hpars['anomaly_threshold'] is None:
                mae = float(max(train_mae_loss))
            else:
                mae = hpars['anomaly_threshold']

        detect_and_plot_anomalies(mae, test_set[label].to_numpy().reshape(-1, 1), test_mae_loss, label, n_time_steps)
        visualise_prediction(test_set[label].values, sc.inverse_transform(test_mae_loss), label) 

if __name__ == "__main__":

    '''Usage: python3 detect.py –d <dataset> -n <number of time series selected> -mae <error value as double>'''

    seed = 420
    reproducibleResults(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-n', '--number_of_time_series_selected')
    parser.add_argument('-mae', '--minimum_abs_error')
    args = parser.parse_args()

    df = get_dataset(args.dataset)

    mae_threshold = float(args.minimum_abs_error) if args.minimum_abs_error else None
    
    if args.number_of_time_series_selected:
        num_of_timeseries = int(args.number_of_time_series_selected)
        
        hyperparameters = {'lstm_dims':[50, 10], 'dropout':0.2, 'n_time_steps':30, 'lag':1, 'batch_size':64, 'max_epochs':100, 'anomaly_threshold':0.5}
        #B__train_model_for_N_stocks(df, hyperparameters, num_of_timeseries)

        loaded_model = tf.keras.models.load_model('./saved_models/detect_model_trained_on_all_stocks.h5')
        B__test_model_on_N_stocks(df, loaded_model, hyperparameters, num_of_timeseries, mae_threshold)
    else:
        hyperparameters = {'lstm_dims':[50, 10], 'label':'aapl', 'dropout':0.2, 'n_time_steps':30, 'lag':1, 'batch_size':64, 'max_epochs':100, 'anomaly_threshold':0.5}
        #B__train_model_for_a_stock(df, hyperparameters)

        loaded_model = tf.keras.models.load_model('./saved_models/detect_model_trained_on_AAPL.h5')
        B__test_model_on_a_stock(df, loaded_model, hyperparameters, 'AAPL', mae_threshold)

### EOF ###