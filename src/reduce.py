"""
Time Series Forecasting with Deep Learning
---
Chalkias Spyridon - Maraziaris Charalampos
"""

import argparse
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
                        visualise_predictions_seperately, prime_factors,
                        split_datasets)

import argparse

class Conv_Autoencoder(tf.keras.Model):
    '''
    Input:
        - conv_filters: Iterable representing the convolution layers and their corresponding filters. e.g [32, 10]
        - kernels: Iterable representing the kernels of the convolution layers. e.g [3, 3]
        - window_length: The window size that the input data is being divided into
        - latent_dim: Dimension of the "bottleneck" layer

    Note considering the window size:
    - As with every parameter, window size is fully defined by the user.
    - The model computes the prime factors of the window size.
    - If the number of the prime factors is less or equal than the number of the convolution layers requested,
        then the convolution layers along with their kernels are being trimmed, in order to match the 
        length of the prime factors.
    - In the other case, the prime factors are being trimmed in order to match the convolution filters and the pooling/upsampling layers
        reduce/enhance the input's dimension based on the prime factors' values. 
    '''
    def __init__(self, conv_filters, kernels, window_length, latent_dim):
        super(Conv_Autoencoder, self).__init__()
        
        pf = prime_factors(window_length)
        if len(pf) <= len(conv_filters):
            conv_filters = conv_filters[:len(pf)]
        else:
            pf = pf[:len(conv_filters)]

        # Define input
        input_window = Input(shape=(window_length, 1))

        # Stack convolution and pooling layers
        cx = Conv1D(conv_filters[0], kernels[0], activation="relu", padding="same")(input_window)
        x = MaxPooling1D(pf[0], padding="same")(cx)
        for i in range(1, len(conv_filters)):
            cx = Conv1D(conv_filters[i], kernels[i], activation="relu", padding="same")(x)
            x = MaxPooling1D(pf[i], padding="same")(cx)
        # Flatten output and define the bottleneck layer
        x = Flatten()(x)
        x = Dense(latent_dim, activation="relu")(x)
        # Define encoder
        self.encoder = Model(input_window, x)

        curr_dim = window_length
        for factor in pf:
            curr_dim = curr_dim / factor
        # Build the decoder 
        x = Dense(int(curr_dim) * conv_filters[-1], activation="relu")(x)
        x = Reshape(( int(curr_dim) , conv_filters[-1]))(x)

        for i in range(1, len(conv_filters) + 1):
            cx = Conv1D(conv_filters[len(conv_filters) - i], kernels[len(kernels) - i], activation="relu", padding="same")(x)
            x = UpSampling1D(pf[len(pf) - i])(cx)

        decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) 
        
        self.autoencoder = Model(input_window, decoded)
        self.autoencoder.summary()
        
    def fit(self, x_train, x_test, max_epochs=100, batch_size=32):

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=2,
                                                    mode='min')
        
        self.autoencoder.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])
        
        history = self.autoencoder.fit(x_train, x_train,
                        epochs=max_epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, x_test), 
                        callbacks=[early_stopping])
        return history

    def predict(self, x_test):
        return self.encoder.predict(x_test)

    def save(self, label):
        self.encoder.save(label)

def C__train_model_for_N_stocks(df, hpars, N):

    window_size = hpars['window_size']
    if hpars['window_size'] is None:
        window_size = 10
    latent_dim = hpars['latent_dim']
    if hpars['latent_dim'] is None:
        latent_dim = 3
    batch_size = hpars['batch_size']
    max_epochs = hpars['max_epochs']
    conv_filters = hpars['conv_filters']
    kernels = hpars['kernels']

    df = df.iloc[: , :N]

    x_train_nonscaled, x_test_nonscaled = train_test_split(df, test_size=.3, shuffle=False)

    train_nonscaled, test_nonscaled = stack_columns(x_train_nonscaled).values, stack_columns(x_test_nonscaled).values

    train_nonscaled = train_nonscaled.flatten()
    test_nonscaled = test_nonscaled.flatten()

    x_train = np.array([train_nonscaled[x : x + window_size] for x in range(0, len(train_nonscaled) - window_size, window_size)])
    x_test = np.array([test_nonscaled[x : x + window_size] for x in range(0, len(test_nonscaled) - window_size, window_size)])

    scaler = MinMaxScaler(feature_range = (0, 1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    model = Conv_Autoencoder(conv_filters, kernels, window_size, latent_dim)
    h = model.fit(x_train, x_test, max_epochs, batch_size)
    plot_loss_vs_epochs_diagram(h.history['loss'], h.history['val_loss'], f'the first {N} stocks')

    predictions = {}
    for i, label in enumerate(list(x_test_nonscaled.columns)): 
        t = np.array([x_test_nonscaled[label].values[x : x + window_size] for x in range(0, len(x_test_nonscaled[label].values) - window_size, window_size)])
        t = scaler.transform(t)
        t =  t.reshape(t.shape[0], t.shape[1], 1)
        predicted_stock_price = model.predict(t)
        predicted_stock_price = np.array([-np.mean(i) for i in predicted_stock_price])
        predictions[label] = predicted_stock_price.flatten()
        visualise_predictions_seperately(x_test_nonscaled[label].values, predicted_stock_price.flatten(), label)

    model.save(f'./saved_models/reduce_model_trained_on_all_stocks.h5')
    return predictions

def C__test_model_on_N_stocks(df, loaded_model, hpars, N):
    
    window_size = hpars['window_size']
    
    df = df.iloc[: , :N]

    x_train_nonscaled, x_test_nonscaled = train_test_split(df, test_size=.3, shuffle=False)

    train_nonscaled, test_nonscaled = stack_columns(x_train_nonscaled).values, stack_columns(x_test_nonscaled).values

    train_nonscaled = train_nonscaled.flatten()
    test_nonscaled = test_nonscaled.flatten()

    x_train = np.array([train_nonscaled[x : x + window_size] for x in range(0, len(train_nonscaled) - window_size, window_size)])
    x_test = np.array([test_nonscaled[x : x + window_size] for x in range(0, len(test_nonscaled) - window_size, window_size)])

    scaler = MinMaxScaler(feature_range = (0, 1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    predictions = {}
    for i, label in enumerate(list(x_test_nonscaled.columns)): 
        t = np.array([x_test_nonscaled[label].values[x : x + window_size] for x in range(0, len(x_test_nonscaled[label].values) - window_size, window_size)])
        t = scaler.transform(t)
        t =  t.reshape(t.shape[0], t.shape[1], 1)
        predicted_stock_price = loaded_model.predict(t)
        predicted_stock_price = np.array([-np.mean(i) for i in predicted_stock_price])
        predictions[label] = predicted_stock_price.flatten()
        visualise_predictions_seperately(x_test_nonscaled[label].values, predicted_stock_price.flatten(), label)

    return predictions

if __name__ == "__main__":

    '''Usage: python3 reduce.py –d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file>'''

    seed = 420
    reproducibleResults(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-q', '--queryset')
    parser.add_argument('-od', '--output_dataset_file')
    parser.add_argument('-oq', '--output_query_file')
    args = parser.parse_args()

    input_df = get_dataset(args.dataset)
    query_df = get_dataset(args.queryset)
    df = pd.concat((input_df, query_df), axis=1)

    hyperparameters = {'conv_filters':[10, 10, 5], 'kernels':[3, 3, 3], 'batch_size':64, 'max_epochs':75, 'window_size':30, 'latent_dim':3}
    predictions = C__train_model_for_N_stocks(df, hyperparameters, len(df.columns))
    
    #loaded_model = tf.keras.models.load_model('./saved_models/reduce_model_trained_on_all_stocks.h5')
    #predictions = C__test_model_on_N_stocks(df, loaded_model, hyperparameters, len(df.columns))

    def write_data_to_out_file(out_filename, df, predictions):
        with open(out_filename, 'w', encoding = 'utf-8') as out:
            for label in df.columns:
                out.write(f'{label}\t')
                for price in predictions[label]:
                    out.write(f'{price}' + '\t')
                out.write('\n')

    write_data_to_out_file(args.output_dataset_file, input_df, predictions)
    write_data_to_out_file(args.output_query_file, query_df, predictions)

### EOF ###
