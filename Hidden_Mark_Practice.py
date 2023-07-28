#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practice for Hidden Markov Models 

@author: treywood
"""

#### Libraries ####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
import nasdaqdatalink
from sklearn.model_selection import train_test_split
from meteostat import Point, Daily
from datetime import datetime


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### Import Gold Price Data ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Set base directory to pull data #
base_dir = "https://github.com/natsunoyuki/Data_Science/blob/master/gold/gold/gold_price_usd.csv?raw=True"


# Import data #
gold_data = pd.read_csv(base_dir)


# Convert date from str to datetime object #
gold_data['datetime'] = pd.to_datetime(gold_data['datetime'])


# Get gold price difference #
gold_data['gold_price_diff'] = gold_data['gold_price_usd'].diff()
gold_data = gold_data.dropna(axis = 0)


# Plot daily gold prices and difference #
plt.figure(figsize = (15, 10))
plt.subplot(2, 1, 1)
plt.plot(gold_data['datetime'], gold_data['gold_price_usd'])
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(gold_data['datetime'], gold_data['gold_price_diff'])
plt.xlabel('Date')
plt.ylabel("Gold Price Change (USD)")
plt.grid(True)
plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### Set HMM fro Gold Price ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Daily prices as observed values #
X = gold_data[['gold_price_diff']]


# Build HMM and fit to gold price change #
model = hmm.GaussianHMM(n_components = 3, covariance_type = 'diag',
                        n_iter = 20000, random_state = 1234)
model.fit(X)


# Predict hidden states #
Z = model.predict(X)
states = pd.unique(Z)


# Mapping of dictionary of numeric states into labels #
state_labels = {
    0: 'Medium',
    1: 'Low',
    2: 'High'}


# Convert states into labels #
states_str = [state_labels[state] for state in states]


# Plot observations with states
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
for i in states:
    want = (Z == i)
    x = gold_data["datetime"].iloc[want]
    y = gold_data["gold_price_usd"].iloc[want]
    plt.plot(x, y, '.', label=state_labels[i])  # Use the state_labels dictionary to get the label for each state
plt.legend(fontsize=16)
plt.grid(True)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Gold Price (USD)", fontsize=16)
plt.subplot(2, 1, 2)
for i in states:
    want = (Z == i)
    x = gold_data["datetime"].iloc[want]
    y = gold_data["gold_price_diff"].iloc[want]
    plt.plot(x, y, '.', label=state_labels[i])  # Use the state_labels dictionary to get the label for each state
plt.legend(fontsize=16)
plt.grid(True)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Gold Price Difference (USD)", fontsize=16)

plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### Import Oil Price Data ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Set NASDAQ API Key #
nasdaqdatalink.ApiConfig.api_key = "XssrY3keCyssQs9FmqPr"


# Pull Oil Data #
oil_data = nasdaqdatalink.get('NSE/OIL', start_date = '1990-01-01').reset_index()


# Make Date into datetime #
oil_data['Date'] = pd.to_datetime(oil_data['Date'])


# Get oil price difference #
oil_data['oil_price_diff'] = oil_data['Close'].diff()
oil_data = oil_data.dropna(axis = 0)


# Plot daily oil prices and difference #
plt.figure(figsize = (15, 10))
plt.subplot(2, 1, 1)
plt.plot(oil_data['Date'], oil_data['Close'])
plt.xlabel('Date')
plt.ylabel('Oil Price (USD)')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(oil_data['Date'], oil_data['oil_price_diff'])
plt.xlabel('Date')
plt.ylabel("Oil Price Change (USD)")
plt.grid(True)
plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### Set HMM for Oil Price ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Daily prices as observed values #
X = oil_data[['oil_price_diff']]


# Split into training and val sets #
X_train, X_val = train_test_split(X, test_size = 0.25, random_state = 1234)


# Make hyperparameter grid #
n_components_range = [2, 10]
covariance_types = ['spherical', 'tied', 'diag', 'full']
n_iter = [1000, 5000, 10000, 20000, 30000]
best_score = float("-inf")
best_model = None

# Grid Search #
for n_components in n_components_range:
    for covariance_type in covariance_types:
        for iters in n_iter:
            model = hmm.GaussianHMM(n_components = n_components,
                                    covariance_type = covariance_type, 
                                    n_iter = iters, random_state = 1234)
            model.fit(X_train)
            ll_val = model.score(X_val)
            if ll_val > best_score:
                best_score = ll_val
                best_model = model 
    """
    Since this is an unsupervised method, I attempted to minimze the 
    log-likelihood score of the model to find the best fit. I allowed
    the number of components and covariance type to change. 
    """


# Print best model #
print("Best model:", best_model)


# Predict hidden states #
Z = best_model.predict(X)
states = pd.unique(Z)


# Set dictionary for labels #
state_labels = {
    0: 'Low',
    1: 'High'}


# Plot observations with states
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
for i in states:
    want = (Z == i)
    x = oil_data["Date"].iloc[want]
    y = oil_data["Close"].iloc[want]
    plt.plot(x, y, '.', label=state_labels[i])  # Use the state_labels dictionary to get the label for each state
plt.legend(fontsize=16)
plt.grid(True)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Oil Price (USD)", fontsize=16)
plt.subplot(2, 1, 2)
for i in states:
    want = (Z == i)
    x = oil_data["Date"].iloc[want]
    y = oil_data["oil_price_diff"].iloc[want]
    plt.plot(x, y, '.', label=state_labels[i])  # Use the state_labels dictionary to get the label for each state
plt.legend(fontsize=16)
plt.grid(True)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Oil Price Difference (USD)", fontsize=16)

plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### Import Lexington, KY Weather Data ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# Set time period #
start = datetime(2000, 1, 1)
end = datetime(2022, 12, 31)


# Create Point for Lexington, KY #
location = Point(38.0406, -84.5037)


# Get daily data for 2018 #
data = Daily(location, start, end)
ky_data = data.fetch()
ky_data = ky_data.reset_index()


# Make time into datetime #
ky_data['Date'] = pd.to_datetime(ky_data['time'])
ky_data = ky_data.drop('time', axis = 1)


# Remove rows where 'tavg' is NaN
ky_data = ky_data.dropna(subset=['tavg'])


# Convert tavg into Fahrenheit #
ky_data['temp'] = (ky_data['tavg'] * 9/5) + 32


# Plot temperature over time #
plt.figure(figsize = (15, 10))
plt.plot(ky_data['Date'], ky_data['temp'])
plt.xlabel('Date')
plt.ylabel('Temperature (F)')
plt.grid(True)
plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### Set HMM for KY Temp ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Daily prices as observed values #
X = ky_data[['temp']]


# Build HMM and fit to gold price change #
model = hmm.GaussianHMM(n_components = 4, covariance_type = 'tied',
                        n_iter = 200000, random_state = 1234)
model.fit(X)
print(model.score(X))


# Predict hidden states #
Z = model.predict(X)
states = pd.unique(Z)


# Set dictionary for labels #
state_labels = {
    0: 'Summer',
    1: 'Winter',
    2: 'Spring',
    3: 'Fall'}


# Plot observations with states
plt.figure(figsize=(15, 10))
for i in states:
    want = (Z == i)
    x = ky_data["Date"].iloc[want]
    y = ky_data["temp"].iloc[want]
    plt.plot(x, y, '.', label=state_labels[i])  # Use the state_labels dictionary to get the label for each state
plt.legend(fontsize=16)
plt.grid(True)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Temperature (F)", fontsize=16)

plt.show()



