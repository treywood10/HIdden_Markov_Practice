#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practice for Hidden Markov Models 

@author: treywood
"""

#### Libraries ####
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
import nasdaqdatalink
from sklearn.model_selection import train_test_split
from meteostat import Point, Daily
from datetime import datetime


# Seed seeds #
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)


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
plt.xlabel('Date', fontsize = 16)
plt.ylabel('Gold Price (USD)', fontsize = 16)
plt.grid(True)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.subplot(2, 1, 2)
plt.plot(gold_data['datetime'], gold_data['gold_price_diff'])
plt.xlabel('Date', fontsize = 16)
plt.ylabel("Gold Price Change (USD)", fontsize = 16)
plt.grid(True)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### Set HMM for Gold Price ####
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
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
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
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Gold Price Difference (USD)", fontsize=16)

plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### Import Coffee Price Data ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Set NASDAQ API Key #
nasdaqdatalink.ApiConfig.api_key = "XssrY3keCyssQs9FmqPr"


# Pull Coffee Data #
cof_data = nasdaqdatalink.get('ODA/PCOFFOTM_USD', start_date = '1990-01-01',
                              end_date = '2022-12-31').reset_index()

# Make Date into datetime #
cof_data['Date'] = pd.to_datetime(cof_data['Date'])


# Get coffee price difference #
cof_data['coffee_price_diff'] = cof_data['Value'].diff()
cof_data = cof_data.dropna(subset = ['coffee_price_diff'])


# Plot monthly coffee prices and difference #
plt.figure(figsize = (15, 10))
plt.subplot(2, 1, 1)
plt.plot(cof_data['Date'], cof_data['Value'])
plt.xlabel('Date', fontsize = 16)
plt.ylabel('Monthly Coffee Price (USD)', fontsize = 16)
plt.grid(True)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.subplot(2, 1, 2)
plt.plot(cof_data['Date'], cof_data['coffee_price_diff'])
plt.xlabel('Date', fontsize = 16)
plt.ylabel("Montly Coffee Price Change (USD)", fontsize = 16)
plt.grid(True)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#### Set HMM for Coffee Price ####
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Daily prices as observed values #
X = cof_data[['coffee_price_diff']]


# Make hyperparameter grid #
n_components_range = [2, 8]
covariance_types = ['spherical', 'tied', 'diag', 'full']
n_iter = [40000, 45000, 50000, 55000, 60000]
num_seeds = 10
best_scores = []
best_models = []
best_seeds= []


# Grid Search with multiple random seeds#
for seed in range(num_seeds):
    
    # Split into training and val sets #
    X_train, X_val = train_test_split(X, test_size = 0.25, random_state = seed)
    X_train = X_train.dropna()
    X_val = X_val.dropna()
    
    # Initialize variables for the best model and score for this seed  #
    best_score_seed = float('-inf')
    best_model_seed = None
    
    # Grid Search for this seed #
    for n_components in n_components_range:
        for covariance_type in covariance_types:
            for iters in n_iter:
                model = hmm.GaussianHMM(n_components=n_components,
                                        covariance_type=covariance_type,
                                        n_iter=iters, random_state=seed)
                model.fit(X_train)
                ll_val = model.score(X_val)
                if ll_val > best_score_seed:
                    best_score_seed = ll_val
                    best_model_seed = model
                    
    # Store the results for this seed #
    best_models.append(best_model_seed)
    best_scores.append(best_score_seed)
    best_seeds.append(seed)
    """
    Since this is an unsupervised method, I attempted to minimze the 
    log-likelihood score of the model to find the best fit. I allowed
    the number of components, covariance type, iterations, and seeds 
    to change. 
    """

# Find the index of the best model among all seeds #
best_index = np.argmax(best_scores)


# Print the best overall model and its associated seed #
print("Best model:", best_models[best_index])
print("Best log-likelihood score:", best_scores[best_index])
print("Seed used for the best model:", best_seeds[best_index])


# Pull best model #
best_model = best_models[best_index]


# Predict hidden states #
Z = best_model.predict(X)
states = pd.unique(Z)


# Set dictionary for labels #
state_labels = {
    0: 'Low',
    1: 'High'}


# Plot observations with states #
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
for i in states:
    want = (Z == i)
    x = cof_data["Date"].iloc[want]
    y = cof_data["Value"].iloc[want]
    plt.plot(x, y, '.', label=state_labels[i])  # Use the state_labels dictionary to get the label for each state
plt.legend(fontsize=16)
plt.grid(True)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Monthly Coffee Price (USD)", fontsize=16)
plt.subplot(2, 1, 2)
for i in states:
    want = (Z == i)
    x = cof_data["Date"].iloc[want]
    y = cof_data["coffee_price_diff"].iloc[want]
    plt.plot(x, y, '.', label=state_labels[i])  # Use the state_labels dictionary to get the label for each state
plt.legend(fontsize=16)
plt.grid(True)
plt.xlabel("Date", fontsize=16)
plt.ylabel("Monthly Coffee Price Difference (USD)", fontsize=16)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
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
plt.xlabel('Date', fontsize = 16)
plt.ylabel('Temperature (F)', fontsize = 16)
plt.grid(True)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
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
    0: 'Hot',
    1: 'Cold',
    2: 'Warm',
    3: 'Cool'}


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
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.show()




