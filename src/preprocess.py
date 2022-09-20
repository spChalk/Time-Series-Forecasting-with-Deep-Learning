"""
Time Series Forecasting with Deep Learning
---
Chalkias Spyridon - Maraziaris Charalampos
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

# Generate reproducible results by giving a specific seed.
def reproducibleResults(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def get_dataset(filename):
    # Read the dataset and instantiate the corresponding variables
    training_set = pd.read_csv(filename, dialect="excel-tab", index_col=0, header=None)

    df = training_set.transpose()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    
    return df

# Stacks all the columns of a dataset and returns a new dataset with the stacked columns as a single column.
def stack_columns(dataframe):
    df = dataframe.copy()
    concatd_df = pd.concat([df, df.T.stack().reset_index(name='data')['data']], axis=1)
    return pd.DataFrame(concatd_df['data'])

def plot_loss_vs_epochs_diagram(train_loss, val_loss, label):
    plt.figure(figsize=[10, 5])
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title(f"Model's loss during the training on {label}")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    plt.show()

# Compares a stock price over the time with a predicted one
def visualise_prediction(test_set, predicted_stock_price, label):
    plt.figure(figsize=[15, 7])
    plt.plot(test_set, color = 'red', label = 'Real Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
    plt.title(f'{label} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def visualise_predictions_seperately(test_set, predicted_stock_price, label):
    plt.figure(figsize=[12, 5])
    plt.plot(test_set, color = 'blue', label = 'Real Stock Price')
    plt.title(f'{label} Real Stock Price')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    plt.figure(figsize=[12, 5])
    plt.plot(predicted_stock_price, color = 'red', label = 'Encoded Stock Price')
    plt.title(f'{label} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Splits a dataset's column to a training set and a test set by a factor of 3
def split_datasets(data, label):
    return data[label].iloc[: 2*len(data)//3 ].to_numpy().reshape(-1, 1), \
            data[label].iloc[ 2*len(data)//3 :].to_numpy().reshape(-1, 1)

# Plot the anomalies on a graph using a given thresohold
def detect_and_plot_anomalies(threshold, test_set, test_mae_loss, label, n_time_steps):
    test_score_df = pd.DataFrame(index=[i for i in range(n_time_steps, len(test_set))])
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['price'] = test_set.flatten()[n_time_steps:]

    plt.figure(figsize=[12, 7])
    plt.plot(test_score_df.loss, label='Loss')
    plt.plot(test_score_df.threshold, label='Threshold')
    plt.title(f'Anomaly Threshold for {label}')
    plt.xticks(rotation=25)
    plt.legend()
    plt.show()

    anomalies = test_score_df[test_score_df.anomaly == True]

    plt.figure(figsize=[20, 10])
    plt.plot(
        [i for i in range(n_time_steps, len(test_set))], 
        test_set[n_time_steps:], 
        label='Stock Price'
        );

    sns.scatterplot(
        x=anomalies.index,
        y=anomalies.price.to_numpy().reshape(-1, 1).flatten(),
        color=sns.color_palette()[3],
        s=52,
        label='Anomaly'
    )
    
    plt.title(f'Anomalies in {label}')
    plt.xticks(rotation=25)
    plt.legend()
    plt.show()

# Compute and return the prime factors of a given number n
def prime_factors(n):
    if n == 1:
        return [1]
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
