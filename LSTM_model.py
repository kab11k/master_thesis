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
#Period 1 (forecast years 2005-2010):

train_year_end=1998
val_year_end=2004
forecast_start=2005
forecast_stop=2010
data_start=1951
dep_var='domestic_fleet_landings'
activation_LSTM='relu'
activation_Dense='sigmoid'
data=df_mask
patience=10
save_path='/content/drive/MyDrive/master_thesis/results/LSTM/2005_2010/'

performance_metrics = pd.DataFrame()

input_configs = [

                 (['domestic_fleet_landings','country'], 1, 3, 32, 2000, 32, .1, 3, 0, 0, 0, 1, 0, 'base_1', 0),
                 (['domestic_fleet_landings','chlorophyll','country'], 1, 3, 16, 2000, 32, .1, 2, 1, 1, 1, 0, 0, 'chl_1', 0),
                 (['domestic_fleet_landings','CPUE','effort','country'], 1, 5, 50, 2000, 16, .2, 2, 1, 1, 0, 1, 0, 'cpue_effort_1', 0),
                 (['domestic_fleet_landings','CPUE','effort','foreign_landings','country'], 1, 3, 16, 2000, 32, .1, 3, 0, 0, 0, 0, 0, 'cpue_effort_foreign_1', 0),
                 (['domestic_fleet_landings','CPUE','effort','foreign_Reported_landings','foreign_Unreported_landings','country'], 1, 3, 16, 2000, 4, .2, 2, 0, 1, 0, 0, 0, 'cpue_effort_foreign_repvunrep_1', 0),
                 (['domestic_fleet_landings','CPUE','effort','sst','country'], 1, 3, 16, 2000, 1, .1, 2, 1, 0, 1, 0, 0, 'cpue_effort_sst_1', 0),
                 (['domestic_fleet_landings','CPUE','effort','sst','foreign_landings','country'], 1, 3, 32, 2000, 8, .1, 3, 0, 1, 1, 0, 1, 'cpue_effort_sst_foreign_1', 0),
                 (['domestic_fleet_landings','CPUE','effort','sst','foreign_Reported_landings','foreign_Unreported_landings','country'], 1, 3, 64, 2000, 1, .1, 3, 1, 0, 0, 1, 0, 'cpue_effort_sst_foreign_repvunrep_1', 0),
                 (['domestic_fleet_landings','foreign_landings','country'], 1, 3, 50, 2000, 32, .1, 2, 1, 1, 0, 0, 0, 'foreign_1', 0),
                 (['domestic_fleet_landings','foreign_Reported_landings','foreign_Unreported_landings','country'], 1, 3, 64, 2000, 16, .1, 2, 0, 0, 0, 1, 0, 'foreign_repvunrep_1', 0),
                 (['domestic_fleet_landings','avg_governance_score','country'], 1, 3, 64, 2000, 8, .1, 2, 0, 1, 1, 1, 0, 'gov_1', 0),
                 (['domestic_fleet_landings','sst','country'], 1, 3, 64, 2000, 4, .1, 2, 0, 0, 1, 1, 0, 'sst_1', 0),
                 (['domestic_fleet_landings','sst','foreign_landings','country'], 1, 3, 64, 2000, 16, .1, 4, 1, 1, 1, 1, 1, 'sst_foreign_1', 0),
                 (['domestic_fleet_landings','sst','foreign_Reported_landings','foreign_Unreported_landings','country'], 1, 5, 50, 2000, 8, .1, 2, 0, 0, 1, 1, 1, 'sst_foreign_1', 0),
  
                 (['domestic_fleet_landings','country'], 11, 3, 16, 2000, 4, .1, 2, 0, 0, 0, 1, 0, 'base_11', 0),
                 (['domestic_fleet_landings','chlorophyll','country'], 11, 3, 50, 2000, 8, .1, 3, 1, 1, 1, 0, 0, 'chl_11', 0),
                 (['domestic_fleet_landings','CPUE','effort','country'], 11, 3, 64, 2000, 1, .1, 3, 1, 1, 0, 1, 0, 'cpue_effort_11', 0),
                 (['domestic_fleet_landings','CPUE','effort','foreign_landings','country'], 11, 3, 32, 2000, 16, .1, 4, 0, 0, 0, 0, 0, 'cpue_effort_foreign_11', 0),
                 (['domestic_fleet_landings','CPUE','effort','foreign_Reported_landings','foreign_Unreported_landings','country'], 11, 3, 16, 2000, 32, .1, 2, 0, 1, 0, 0, 0, 'cpue_effort_foreign_repvunrep_11', 0),
                 (['domestic_fleet_landings','CPUE','effort','sst','country'], 11, 3, 64, 2000, 32, .1, 3, 1, 0, 1, 0, 0, 'cpue_effort_sst_11', 0),
                 (['domestic_fleet_landings','CPUE','effort','sst','foreign_landings','country'], 11, 5, 32, 2000, 8, .2, 2, 0, 1, 1, 0, 1, 'cpue_effort_sst_foreign_11', 0),
                 (['domestic_fleet_landings','CPUE','effort','sst','foreign_Reported_landings','foreign_Unreported_landings','country'], 11, 3, 32, 2000, 8, .1, 3, 1, 0, 0, 1, 0, 'cpue_effort_sst_foreign_repvunrep_11', 0),
                 (['domestic_fleet_landings','foreign_landings','country'], 11, 3, 16, 2000, 32, .2, 2, 1, 1, 0, 0, 0, 'foreign_11', 0),
                 (['domestic_fleet_landings','foreign_Reported_landings','foreign_Unreported_landings','country'], 11, 3, 32, 2000, 16, .1, 3, 0, 0, 0, 1, 0, 'foreign_repvunrep_11', 0),
                 (['domestic_fleet_landings','avg_governance_score','country'], 11, 3, 16, 2000, 1, .2, 4, 0, 1, 1, 1, 0, 'gov_11', 0),
                 (['domestic_fleet_landings','sst','country'], 11, 3, 16, 2000, 32, .1, 4, 0, 0, 1, 1, 0, 'sst_11', 0),
                 (['domestic_fleet_landings','sst','foreign_landings','country'], 11, 5, 32, 2000, 16, .1, 2, 1, 1, 1, 1, 1, 'sst_foreign_11', 0),
                 (['domestic_fleet_landings','sst','foreign_Reported_landings','foreign_Unreported_landings','country'], 11, 3, 64, 2000, 8, .1, 3, 0, 0, 1, 1, 1, 'sst_foreign_11', 0),

 ]

for config in input_configs:
  pred_vars, forecast_steps, lags, neurons, epochs, batch_size, dropout, num_layers, use_GRU, use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm, filename, verbose = config
  model_save = f'{save_path}/models/{filename}.h5' #path to save models

  preds = []
  set_seed(seed)
  for i in range(0, forecast_stop-forecast_start+1): #Makes iterative 1-yr forecasts. Range is number of years to forecast
    X, y, X_test, y_test, val_x, val_y = data_sequences(data, lags, forecast_steps, data_start, train_year_end+i, val_year_end+i, forecast_start+i, verbose)
    pred, train_accuracy, test_accuracy = lstm_build(X, y, X_test, y_test, val_x, val_y, forecast_steps, lags, neurons, epochs, batch_size, activation_LSTM, activation_Dense, dropout, num_layers, use_GRU, use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm, verbose, patience, model_save)
    preds.append(pred[:, -1])
  preds = np.array(preds)


  performance = evaluate_model(data, lags, preds, pred_vars, forecast_start, forecast_stop, forecast_steps, data_start, neurons, epochs, batch_size, dropout, num_layers, use_GRU, use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm)

  performance_metrics = pd.concat([performance_metrics, performance], ignore_index=True)

  performance_metrics.to_csv(f'{save_path}/metrics_LSTM.csv', index=False)

##################################################################################################################
#Period 2 (forecast years 2013-2019):

train_year_end=2005
val_year_end=2012
forecast_start=2013
forecast_stop=2019
data_start=1951
dep_var='domestic_fleet_landings'
activation_LSTM='relu'
activation_Dense='sigmoid'
data=df_mask
patience=10
save_path='/content/drive/MyDrive/master_thesis/results/LSTM/2013_2019/'

performance_metrics = pd.DataFrame()

input_configs = [

                 (['domestic_fleet_landings','country'], 1, 3, 32, 2000, 32, .1, 3, 0, 0, 0, 1, 0, 'base_1', 0),
                 (['domestic_fleet_landings','chlorophyll','country'], 1, 3, 16, 2000, 32, .1, 2, 1, 1, 1, 0, 0, 'chl_1', 0),
                 (['domestic_fleet_landings','CPUE','effort','country'], 1, 5, 50, 2000, 16, .2, 2, 1, 1, 0, 1, 0, 'cpue_effort_1', 0),
                 (['domestic_fleet_landings','CPUE','effort','foreign_landings','country'], 1, 3, 16, 2000, 32, .1, 3, 0, 0, 0, 0, 0, 'cpue_effort_foreign_1', 0),
                 (['domestic_fleet_landings','CPUE','effort','foreign_Reported_landings','foreign_Unreported_landings','country'], 1, 3, 16, 2000, 4, .2, 2, 0, 1, 0, 0, 0, 'cpue_effort_foreign_repvunrep_1', 0),
                 (['domestic_fleet_landings','CPUE','effort','sst','country'], 1, 3, 16, 2000, 1, .1, 2, 1, 0, 1, 0, 0, 'cpue_effort_sst_1', 0),
                 (['domestic_fleet_landings','CPUE','effort','sst','foreign_landings','country'], 1, 3, 32, 2000, 8, .1, 3, 0, 1, 1, 0, 1, 'cpue_effort_sst_foreign_1', 0),
                 (['domestic_fleet_landings','CPUE','effort','sst','foreign_Reported_landings','foreign_Unreported_landings','country'], 1, 3, 64, 2000, 1, .1, 3, 1, 0, 0, 1, 0, 'cpue_effort_sst_foreign_repvunrep_1', 0),
                 (['domestic_fleet_landings','foreign_landings','country'], 1, 3, 50, 2000, 32, .1, 2, 1, 1, 0, 0, 0, 'foreign_1', 0),
                 (['domestic_fleet_landings','foreign_Reported_landings','foreign_Unreported_landings','country'], 1, 3, 64, 2000, 16, .1, 2, 0, 0, 0, 1, 0, 'foreign_repvunrep_1', 0),
                 (['domestic_fleet_landings','avg_governance_score','country'], 1, 3, 64, 2000, 8, .1, 2, 0, 1, 1, 1, 0, 'gov_1', 0),
                 (['domestic_fleet_landings','sst','country'], 1, 3, 64, 2000, 4, .1, 2, 0, 0, 1, 1, 0, 'sst_1', 0),
                 (['domestic_fleet_landings','sst','foreign_landings','country'], 1, 3, 64, 2000, 16, .1, 4, 1, 1, 1, 1, 1, 'sst_foreign_1', 0),
                 (['domestic_fleet_landings','sst','foreign_Reported_landings','foreign_Unreported_landings','country'], 1, 5, 50, 2000, 8, .1, 2, 0, 0, 1, 1, 1, 'sst_foreign_1', 0),
  
                 (['domestic_fleet_landings','country'], 11, 3, 16, 2000, 4, .1, 2, 0, 0, 0, 1, 0, 'base_11', 0),
                 (['domestic_fleet_landings','chlorophyll','country'], 11, 3, 50, 2000, 8, .1, 3, 1, 1, 1, 0, 0, 'chl_11', 0),
                 (['domestic_fleet_landings','CPUE','effort','country'], 11, 3, 64, 2000, 1, .1, 3, 1, 1, 0, 1, 0, 'cpue_effort_11', 0),
                 (['domestic_fleet_landings','CPUE','effort','foreign_landings','country'], 11, 3, 32, 2000, 16, .1, 4, 0, 0, 0, 0, 0, 'cpue_effort_foreign_11', 0),
                 (['domestic_fleet_landings','CPUE','effort','foreign_Reported_landings','foreign_Unreported_landings','country'], 11, 3, 16, 2000, 32, .1, 2, 0, 1, 0, 0, 0, 'cpue_effort_foreign_repvunrep_11', 0),
                 (['domestic_fleet_landings','CPUE','effort','sst','country'], 11, 3, 64, 2000, 32, .1, 3, 1, 0, 1, 0, 0, 'cpue_effort_sst_11', 0),
                 (['domestic_fleet_landings','CPUE','effort','sst','foreign_landings','country'], 11, 5, 32, 2000, 8, .2, 2, 0, 1, 1, 0, 1, 'cpue_effort_sst_foreign_11', 0),
                 (['domestic_fleet_landings','CPUE','effort','sst','foreign_Reported_landings','foreign_Unreported_landings','country'], 11, 3, 32, 2000, 8, .1, 3, 1, 0, 0, 1, 0, 'cpue_effort_sst_foreign_repvunrep_11', 0),
                 (['domestic_fleet_landings','foreign_landings','country'], 11, 3, 16, 2000, 32, .2, 2, 1, 1, 0, 0, 0, 'foreign_11', 0),
                 (['domestic_fleet_landings','foreign_Reported_landings','foreign_Unreported_landings','country'], 11, 3, 32, 2000, 16, .1, 3, 0, 0, 0, 1, 0, 'foreign_repvunrep_11', 0),
                 (['domestic_fleet_landings','avg_governance_score','country'], 11, 3, 16, 2000, 1, .2, 4, 0, 1, 1, 1, 0, 'gov_11', 0),
                 (['domestic_fleet_landings','sst','country'], 11, 3, 16, 2000, 32, .1, 4, 0, 0, 1, 1, 0, 'sst_11', 0),
                 (['domestic_fleet_landings','sst','foreign_landings','country'], 11, 5, 32, 2000, 16, .1, 2, 1, 1, 1, 1, 1, 'sst_foreign_11', 0),
                 (['domestic_fleet_landings','sst','foreign_Reported_landings','foreign_Unreported_landings','country'], 11, 3, 64, 2000, 8, .1, 3, 0, 0, 1, 1, 1, 'sst_foreign_11', 0),

 ]

for config in input_configs:
  pred_vars, forecast_steps, lags, neurons, epochs, batch_size, dropout, num_layers, use_GRU, use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm, filename, verbose = config
  model_save = f'{save_path}/models/{filename}.h5' #path to save models

  preds = []
  set_seed(seed)
  for i in range(0, forecast_stop-forecast_start+1): #Makes iterative 1-yr forecasts. Range is number of years to forecast
    X, y, X_test, y_test, val_x, val_y = data_sequences(data, lags, forecast_steps, data_start, train_year_end+i, val_year_end+i, forecast_start+i, verbose)
    pred, train_accuracy, test_accuracy = lstm_build(X, y, X_test, y_test, val_x, val_y, forecast_steps, lags, neurons, epochs, batch_size, activation_LSTM, activation_Dense, dropout, num_layers, use_GRU, use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm, verbose, patience, model_save)
    preds.append(pred[:, -1])
  preds = np.array(preds)


  performance = evaluate_model(data, lags, preds, pred_vars, forecast_start, forecast_stop, forecast_steps, data_start, neurons, epochs, batch_size, dropout, num_layers, use_GRU, use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm)

  performance_metrics = pd.concat([performance_metrics, performance], ignore_index=True)

  performance_metrics.to_csv(f'{save_path}/metrics_LSTM.csv', index=False)
