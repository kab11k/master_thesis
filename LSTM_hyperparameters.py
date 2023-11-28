import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, GRU
import itertools
from hampel import hampel
from keras.layers import Bidirectional
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization
import random
import LSTM_functions
drive.mount('/content/drive')

##################################################################################################################
#dataset:
df = pd.read_csv("/content/drive/MyDrive/master_thesis/df_clean.csv")
#remove south africa:
df=df[df['area_name']!='South Africa'].copy()

#Data Transformation:

#hampel:
#Identify outliers and replace with rolling mean:

cols_used = ['domestic_fleet_landings', 'effort','CPUE',
             'foreign_landings', 'foreign_Reported_landings','foreign_Unreported_landings']

#Identify outliers and replace with 5-year rolling average:
df_hampel = pd.DataFrame()

# Loop through each country
for country in df['area_name'].unique():
    country_data = df[df['area_name'] == country].copy().reset_index(drop=True)  # Subset data for the current country and reset index

    # Loop through each time series column
    for column in cols_used:
        # Apply Hampel function to replace outliers
        data_series = country_data[column].squeeze().reset_index(drop=True) #hampel function requires that data be in pandas series form
        original_values = data_series.copy()
        hampel_series = hampel(data_series, window_size=5, n_sigma=3.0)
        country_data[column] = hampel_series.filtered_data #covert series back to dataframe column

    # Update the dataframe with the modified data for the current country
    df_hampel = pd.concat([df_hampel, country_data], ignore_index=True)

#Min/max scaler:
#apply the min max scaler by country and data column (with hampel)

cols_used = ['domestic_fleet_landings', 'effort', 'number_boats','CPUE',
             'foreign_landings', 'foreign_Reported_landings','foreign_Unreported_landings'] #columns to transform

df_scaled = pd.DataFrame() #initiate new dataframe

scaler = MinMaxScaler(feature_range=(0, 1)) #initiate scaler

for country, group in df_hampel.groupby('area_name'): #apply by country
    group_copy = group.copy()
    for column in cols_used: #apply by column
      group_copy[column] = scaler.fit_transform(group[[column]])
    df_scaled = pd.concat([df_scaled, group_copy], ignore_index=True)

df_scaled.reset_index(drop=True, inplace=True)

#encode country column:
encoder = LabelEncoder()
df_scaled['country']= encoder.fit_transform(df_scaled['area_name'])

#Masking
df_mask = df_scaled.copy()
df_mask = df_mask.fillna(-1) #masking value will be -1 so setting NAs to -1

#baseline MAE (MAE values resulting from baseline model, for comparison):
MAE_1_2005_2010=20071.994239899
MAE_5_2005_2010=41751.8767145466
MAE_11_2005_2010=58121.5629626857

MAE_1_2013_2019=25772.3083759616
MAE_5_2013_2019=57168.9294177568
MAE_11_2013_2019=80186.8393054145

##################################################################################################################
#Set seeds:
seed=0
def set_seed(seed_value=seed):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)

set_seed(seed)

##################################################################################################################

#hyperparameter tuning:
data=df_mask

forecast_steps=5

train_year_end=1998
val_year_end=2004
test_year_end=2010
verbose=0
activation_Dense='sigmoid'
patience=10
dep_var='domestic_fleet_landings'
pred_vars=['domestic_fleet_landings','country'] #change to test for different models
filename='base_5' #set filename for model being tested
save_path='/content/drive/MyDrive/master_thesis/results/LSTM/hyperparam/h_5'
model_save = f'{save_path}/models/{filename}.h5' #path to save models

data_start=1951

config = [[3,5,10], [16, 32, 50, 64], [2000], [1, 4, 8, 16, 32], ['relu'], [.1], [2,3,4], [0], [0], [0], [1], [0]] #lags, neurons, epochs, batch size, LSTM activation, dropout, number of layers, use GRU (0-no, 1-yes), use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm

hist = lstm_hyperparam(config, forecast_steps, activation_Dense, verbose, patience, model_save, save_path, filename)

#choose best model
hist = pd.DataFrame(hist)
#hist = hist.sort_values(by=[5], ascending=True)
hist.to_csv(f'{save_path}/{filename}.csv', index=False)
