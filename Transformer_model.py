import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from gluonts.itertools import Map
from datasets import Dataset, Features, Value, Sequence
from functools import lru_cache
from functools import partial
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from gluonts.time_feature import get_lags_for_frequency
from gluonts.time_feature import time_features_from_frequency_str
from typing import Iterable
import torch
from gluonts.itertools import Cyclic, Cached
from gluonts.dataset.loader import as_stacked_batches
from gluonts.transform.sampler import InstanceSampler
from typing import Optional
from transformers import PretrainedConfig
from gluonts.time_feature import (
    time_features_from_frequency_str,
    TimeFeature,
    get_lags_for_frequency,
)
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)

from accelerate import Accelerator
from torch.optim import AdamW
from evaluate import load
from gluonts.time_feature import get_seasonality
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from hampel import hampel
from itertools import product
from gluonts.dataset.common import DataEntry
from gluonts.transform import MapTransformation
from typing import Type
import random
import Transformer_functions
drive.mount('/content/drive')

#load evaluation metrics
mse_metric = load("evaluate-metric/mse")
mae_metric = load("evaluate-metric/mae")
r2_metric = load("evaluate-metric/r_squared")

##################################################################################################################
#load data:
df = pd.read_csv("/content/drive/MyDrive/master_thesis/df_clean.csv")
#remove south africa:
df=df[df['area_name']!='South Africa'].copy()

#baseline MAE:
MAE_1_2005_2010=20071.994239899
MAE_5_2005_2010=41751.8767145466
MAE_11_2005_2010=58121.5629626857

MAE_1_2013_2019=25772.3083759616
MAE_5_2013_2019=57168.9294177568
MAE_11_2013_2019=80186.8393054145

##################################################################################################################
#Set seeds:
seed=0
random.seed(seed)
np.random.seed(seed)

def set_seed(seed_value=seed):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

##################################################################################################################
#Data Transformation:

#encode country column:
encoder = LabelEncoder()
df['country']= encoder.fit_transform(df['area_name'])
df['country'] = df['country'].astype(str) #must be type string for transformer

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
        hampel_series = hampel(data_series)
        country_data[column] = hampel_series.filtered_data #covert series back to dataframe column

    # Update the dataframe with the modified data for the current country
    df_hampel = pd.concat([df_hampel, country_data], ignore_index=True)

#Min/max scaler:
#apply min max scaler by country and data column:
cols_used = ['domestic_fleet_landings', 'effort', 'number_boats','CPUE',
             'foreign_landings', 'foreign_Reported_landings','foreign_Unreported_landings',
             'sst', 'chlorophyll'] #columns to transform

df_scaled = pd.DataFrame() #initiate new dataframe

scaler = MinMaxScaler(feature_range=(0, 1)) #initiate scaler

for country, group in df_hampel.groupby('area_name'): #apply by country
    group_copy = group.copy()
    for column in cols_used: #apply by column
      group_copy[column] = scaler.fit_transform(group[[column]])
    df_scaled = pd.concat([df_scaled, group_copy], ignore_index=True)

#years with 0 are NA in data. Need to replace with 0s to differentiate between actual missing data.
cols = ['foreign_landings', 'foreign_Reported_landings', 'foreign_Unreported_landings']
df_scaled[cols] = df_scaled[cols].fillna(0)

#replacing NAs in exogenous variables with average. Transformer model data must not contain NAs in exogenous variables.
columns_to_fill = ['effort', 'CPUE', 'sst', 'chlorophyll', 'avg_governance_score']
for column in columns_to_fill:
    df_scaled[column] = df_scaled.groupby('area_name')[column].transform(lambda x: x.fillna(x.mean()))

##################################################################################################################
#Set parameters:
freq = "A"
dep_var='domestic_fleet_landings'

context_length = 20
lags_sequence = get_lags_for_frequency(freq)
time_features = time_features_from_frequency_str(freq)
embedding_dimension=[2]
encoder_layers=2
decoder_layers=2
d_model=16

learning_rate=.0001
weight_decay=1e-1
betas=(0.9, 0.95)
verbose=0
patience=10

##################################################################################################################
#Period 1 (forecast years 2005-2010):
performance_metrics = pd.DataFrame()
df_scaled=df_scaled[df_scaled['year']>1950].copy()

forecast_start=2005
forecast_stop=2010

filename='metrics_step1'

input_configs = [
    ('base_1',1,[],16,.1,16,80),
    ('chl_1',1,['chlorophyll'],64,.1,8,80),
    ('cpue_effort_1',1,['CPUE','effort'],32,.2,8,60),
    ('cpue_effort_foreign_1',1,['CPUE','effort','foreign_landings'],32,.2,16,100),
    ('cpue_effort_foreign_repvunrep_1',1,['CPUE','effort','foreign_Reported_landings','foreign_Unreported_landings'],16,.1,8,80),
    ('cpue_effort_sst_1',1,['CPUE','effort','sst'],32,.2,16,100),
    ('cpue_effort_sst_foreign_1',1,['CPUE','effort','sst','foreign_landings'],16,.1,8,60),
    ('cpue_effort_sst_foreign_repvunrep_1',1,['CPUE','effort','sst','foreign_Reported_landings','foreign_Unreported_landings'],64,.2,16,100),
    ('foreign_1',1,['foreign_landings'],64,.1,8,60),
    ('foreign_repvunrep_1',1,['foreign_Reported_landings','foreign_Unreported_landings'],32,.1,8,60),
    ('gov_1',1,['avg_governance_score'],16,.1,8,80),
    ('sst_1',1,['sst'],32,.2,16,80),
    ('sst_foreign_1',1,['sst','foreign_landings'],16,.1,32,80),
    ('sst_foreign_repvunrep_1',11,['sst','foreign_Reported_landings','foreign_Unreported_landings'],64,.1,16,100),

    ('base_11',11,[],16,.1,16,80),
    ('chl_11',11,['chlorophyll'],64,.1,8,80),
    ('cpue_effort_11',11,['CPUE','effort'],32,.2,8,60),
    ('cpue_effort_foreign_11',11,['CPUE','effort','foreign_landings'],32,.2,16,100),
    ('cpue_effort_foreign_repvunrep_11',11,['CPUE','effort','foreign_Reported_landings','foreign_Unreported_landings'],16,.1,8,80),
    ('cpue_effort_sst_11',11,['CPUE','effort','sst'],32,.2,16,100),
    ('cpue_effort_sst_foreign_11',11,['CPUE','effort','sst','foreign_landings'],16,.1,8,60),
    ('cpue_effort_sst_foreign_repvunrep_11',11,['CPUE','effort','sst','foreign_Reported_landings','foreign_Unreported_landings'],64,.2,16,100),
    ('foreign_11',11,['foreign_landings'],64,.1,8,60),
    ('foreign_repvunrep_11',11,['foreign_Reported_landings','foreign_Unreported_landings'],32,.1,8,60),
    ('gov_11',11,['avg_governance_score'],16,.1,8,80),
    ('sst_11',11,['sst'],32,.2,16,80),
    ('sst_foreign_11',11,['sst','foreign_landings'],16,.1,32,80),
    ('sst_foreign_repvunrep_11',11,['sst','foreign_Reported_landings','foreign_Unreported_landings'],64,.1,16,100),
    ('gov_foreign_11',11,['avg_governance_score','foreign_landings'],16,.1,8,80),

    ]

for config in input_configs:
  modelname, prediction_length, pred_var, batch_size, dropout, num_batches_per_epoch, epochs = config
  save_path=f'/content/drive/MyDrive/master_thesis/results/Transformer/2005_2010/models/best_model_{modelname}.pth' #save model path
  num_dynamic_real_features=len(pred_var)
  inverted_forecast_median = [] #initialize list for forecasts
  inverted_forecast_mean = []
  inverted_forecast_std = []

  for i in range(forecast_start,forecast_stop+1):
    start=stop=i

    dataset_train, dataset_test, dataset_val, dataset_test_original = create_GluonTS_dataset (df_scaled, start, stop, prediction_length, dep_var, pred_var)

    config = TimeSeriesTransformerConfig(
      prediction_length=prediction_length,
      context_length=context_length, #input sequence length
      lags_sequence=lags_sequence, #sequence of lags to be used as features
      num_time_features=len(time_features) + 1, #1 time feature "age" which indactes the "age" of each data point in the time series
      num_static_categorical_features=1, #country time series ID
      cardinality=[len(dataset_train)], #specifies the number of categories or distinct items in the static categorical feature
      embedding_dimension=embedding_dimension, #hyperparameter to test
      dropout=dropout,
      num_dynamic_real_features=num_dynamic_real_features, #exogenous variables

      #other items to potentially add:
      #distribution_output= default="student_t". Other options: "normal" and "negative_binomial"
      #activation_function: default: "gelu". Can also use "relu"

      encoder_layers=encoder_layers,
      decoder_layers=decoder_layers,
      d_model=d_model,
      )

    forecasts = transformer_build(config, freq, dataset_train, dataset_test, dataset_val, batch_size, num_batches_per_epoch, learning_rate, betas, weight_decay, epochs, patience, verbose, filename, save_path)
    preds_median, preds_mean, preds_std = invert_forecasts(forecasts, dataset_test_original)

    #keep only lst forecast:
    preds_median = preds_median[:, -1][:, np.newaxis]
    preds_mean = preds_mean[:, -1][:, np.newaxis]
    preds_std = preds_std[:, -1][:, np.newaxis]

    if len(inverted_forecast_median) == 0:
      inverted_forecast_median = preds_median
      inverted_forecast_mean = preds_mean
      inverted_forecast_std = preds_std
    else:
      inverted_forecast_median = np.hstack((inverted_forecast_median, preds_median))
      inverted_forecast_mean = np.hstack((inverted_forecast_mean, preds_mean))
      inverted_forecast_std = np.hstack((inverted_forecast_std, preds_std))

  performance = evaluate_model(dataset_test_original, inverted_forecast_median, forecast_start, forecast_stop, pred_var)
  performance_metrics = pd.concat([performance_metrics, performance], ignore_index=True)
  performance_metrics.to_csv(f'/content/drive/MyDrive/master_thesis/results/Transformer/2005_2010/{filename}.csv', index=False)

##################################################################################################################
#Period 2 (forecast years 2013-2019):

performance_metrics = pd.DataFrame()
df_scaled=df_scaled[df_scaled['year']>1950].copy()

forecast_start=2013
forecast_stop=2019
filename='metrics_step1'

input_configs = [
    ('base_1',1,[],16,.1,16,80),
    ('chl_1',1,['chlorophyll'],64,.1,8,80),
    ('cpue_effort_1',1,['CPUE','effort'],32,.2,8,60),
    ('cpue_effort_foreign_1',1,['CPUE','effort','foreign_landings'],32,.2,16,100),
    ('cpue_effort_foreign_repvunrep_1',1,['CPUE','effort','foreign_Reported_landings','foreign_Unreported_landings'],16,.1,8,80),
    ('cpue_effort_sst_1',1,['CPUE','effort','sst'],32,.2,16,100),
    ('cpue_effort_sst_foreign_1',1,['CPUE','effort','sst','foreign_landings'],16,.1,8,60),
    ('cpue_effort_sst_foreign_repvunrep_1',1,['CPUE','effort','sst','foreign_Reported_landings','foreign_Unreported_landings'],64,.2,16,100),
    ('foreign_1',1,['foreign_landings'],64,.1,8,60),
    ('foreign_repvunrep_1',1,['foreign_Reported_landings','foreign_Unreported_landings'],32,.1,8,60),
    ('gov_1',1,['avg_governance_score'],16,.1,8,80),
    ('sst_1',1,['sst'],32,.2,16,80),
    ('sst_foreign_1',1,['sst','foreign_landings'],16,.1,32,80),
    ('sst_foreign_repvunrep_1',1,['sst','foreign_Reported_landings','foreign_Unreported_landings'],64,.1,16,100),

    ('base_11',11,[],16,.1,16,80),
    ('chl_11',11,['chlorophyll'],64,.1,8,80),
    ('cpue_effort_11',11,['CPUE','effort'],32,.2,8,60),
    ('cpue_effort_foreign_11',11,['CPUE','effort','foreign_landings'],32,.2,16,100),
    ('cpue_effort_foreign_repvunrep_11',11,['CPUE','effort','foreign_Reported_landings','foreign_Unreported_landings'],16,.1,8,80),
    ('cpue_effort_sst_11',11,['CPUE','effort','sst'],32,.2,16,100),
    ('cpue_effort_sst_foreign_11',11,['CPUE','effort','sst','foreign_landings'],16,.1,8,60),
    ('cpue_effort_sst_foreign_repvunrep_11',11,['CPUE','effort','sst','foreign_Reported_landings','foreign_Unreported_landings'],64,.2,16,100),
    ('foreign_11',11,['foreign_landings'],64,.1,8,60),
    ('foreign_repvunrep_11',11,['foreign_Reported_landings','foreign_Unreported_landings'],32,.1,8,60),
    ('gov_11',11,['avg_governance_score'],16,.1,8,80),
    ('sst_11',11,['sst'],32,.2,16,80),
    ('sst_foreign_11',11,['sst','foreign_landings'],16,.1,32,80),
    ('sst_foreign_repvunrep_11',11,['sst','foreign_Reported_landings','foreign_Unreported_landings'],64,.1,16,100),
    ('gov_foreign_11',11,['avg_governance_score','foreign_landings'],16,.1,8,80),

    ]

for config in input_configs:
  modelname, prediction_length, pred_var, batch_size, dropout, num_batches_per_epoch, epochs = config
  save_path=f'/content/drive/MyDrive/master_thesis/results/Transformer/2013_2019/models/best_model_{modelname}.pth' #save model path
  num_dynamic_real_features=len(pred_var)
  inverted_forecast_median = [] #initialize list for forecasts
  inverted_forecast_mean = []
  inverted_forecast_std = []

  for i in range(forecast_start,forecast_stop+1):
    start=stop=i

    dataset_train, dataset_test, dataset_val, dataset_test_original = create_GluonTS_dataset (df_scaled, start, stop, prediction_length, dep_var, pred_var)

    config = TimeSeriesTransformerConfig(
      prediction_length=prediction_length,
      context_length=context_length, #input sequence length
      lags_sequence=lags_sequence, #sequence of lags to be used as features
      num_time_features=len(time_features) + 1, #1 time feature "age" which indactes the "age" of each data point in the time series
      num_static_categorical_features=1, #country time series ID
      cardinality=[len(dataset_train)], #specifies the number of categories or distinct items in the static categorical feature
      embedding_dimension=embedding_dimension, #hyperparameter to test
      dropout=dropout,
      num_dynamic_real_features=num_dynamic_real_features, #exogenous variables

      #other items to potentially add:
      #distribution_output= default="student_t". Other options: "normal" and "negative_binomial"
      #activation_function: default: "gelu". Can also use "relu"

      encoder_layers=encoder_layers,
      decoder_layers=decoder_layers,
      d_model=d_model,
      )

    forecasts = transformer_build(config, freq, dataset_train, dataset_test, dataset_val, batch_size, num_batches_per_epoch, learning_rate, betas, weight_decay, epochs, patience, verbose, filename, save_path)
    preds_median, preds_mean, preds_std = invert_forecasts(forecasts, dataset_test_original)

    #keep only lst forecast:
    preds_median = preds_median[:, -1][:, np.newaxis]
    preds_mean = preds_mean[:, -1][:, np.newaxis]
    preds_std = preds_std[:, -1][:, np.newaxis]

    if len(inverted_forecast_median) == 0:
      inverted_forecast_median = preds_median
      inverted_forecast_mean = preds_mean
      inverted_forecast_std = preds_std
    else:
      inverted_forecast_median = np.hstack((inverted_forecast_median, preds_median))
      inverted_forecast_mean = np.hstack((inverted_forecast_mean, preds_mean))
      inverted_forecast_std = np.hstack((inverted_forecast_std, preds_std))

  performance = evaluate_model(dataset_test_original, inverted_forecast_median, forecast_start, forecast_stop, pred_var)
  performance_metrics = pd.concat([performance_metrics, performance], ignore_index=True)
  performance_metrics.to_csv(f'/content/drive/MyDrive/master_thesis/results/Transformer/2013_2019/{filename}.csv', index=False)

##################################################################################################################
#Final Forecast (years 2020-2030):

performance_metrics = pd.DataFrame()
df_scaled=df_scaled[df_scaled['year']>1950].copy()

forecast_start=2020
forecast_stop=2030
prediction_length=11
pred_var=['avg_governance_score','foreign_landings']
batch_size=16
dropout=.1
num_batches_per_epoch=8
epochs=80
modelname='final'


save_path=f'/content/drive/MyDrive/master_thesis/results/Transformer/2020_2030/models/best_model_{modelname}.pth' #save model path
num_dynamic_real_features=len(pred_var)
inverted_forecast_median = [] #initialize list for forecasts
inverted_forecast_mean = []
inverted_forecast_std = []

dataset_train, dataset_test, dataset_val, dataset_test_original = create_GluonTS_dataset (df_scaled, start, stop, prediction_length, dep_var, pred_var)

config = TimeSeriesTransformerConfig(
      prediction_length=prediction_length,
      context_length=context_length, #input sequence length
      lags_sequence=lags_sequence, #sequence of lags to be used as features
      num_time_features=len(time_features) + 1, #1 time feature "age" which indactes the "age" of each data point in the time series
      num_static_categorical_features=1, #country time series ID
      cardinality=[len(dataset_train)], #specifies the number of categories or distinct items in the static categorical feature
      embedding_dimension=embedding_dimension, #hyperparameter to test
      dropout=dropout,
      num_dynamic_real_features=num_dynamic_real_features, #exogenous variables

      #other items to potentially add:
      #distribution_output= default="student_t". Other options: "normal" and "negative_binomial"
      #activation_function: default: "gelu". Can also use "relu"

      encoder_layers=encoder_layers,
      decoder_layers=decoder_layers,
      d_model=d_model,
      )

forecasts = transformer_build(config, freq, dataset_train, dataset_test, dataset_val, batch_size, num_batches_per_epoch, learning_rate, betas, weight_decay, epochs, patience, verbose, filename, save_path)
preds_median, preds_mean, preds_std = invert_forecasts(forecasts, dataset_test_original)


if len(inverted_forecast_median) == 0:
      inverted_forecast_median = preds_median
      inverted_forecast_mean = preds_mean
      inverted_forecast_std = preds_std
else:
      inverted_forecast_median = np.hstack((inverted_forecast_median, preds_median))
      inverted_forecast_mean = np.hstack((inverted_forecast_mean, preds_mean))
      inverted_forecast_std = np.hstack((inverted_forecast_std, preds_std))

inverted_forecast_median_df = pd.DataFrame(inverted_forecast_median)
inverted_forecast_mean_df = pd.DataFrame(inverted_forecast_mean)
inverted_forecast_std_df = pd.DataFrame(inverted_forecast_std)

inverted_forecast_median_df.to_csv(f'/content/drive/MyDrive/master_thesis/results/Transformer/2020_2030/forecasts_median_{modelname}.csv', index=False)
inverted_forecast_mean_df.to_csv(f'/content/drive/MyDrive/master_thesis/results/Transformer/2020_2030/forecasts_mean_{modelname}.csv', index=False)
inverted_forecast_std_df.to_csv(f'/content/drive/MyDrive/master_thesis/results/Transformer/2020_2030/forecasts_std_{modelname}.csv', index=False)
