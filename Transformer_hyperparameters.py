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

# Hyperparameters grid
param_grid = {
    'learning_rate': [1e-4], #1e-6, 1e-4, 1e-2
    'batch_size': [16, 32, 64], #8, 16, 32, 64, 128, 256
    'num_batches_per_epoch': [8, 16, 32],
    'context_length': [20], #20, 30, 40
    'encoder_layers': [2],#2, 4, 6
    'decoder_layers': [2],#2, 4, 6
    'd_model': [16], #16, 32
    'dropout': [.1, .2],#.1, .2
    'weight_decay': [1e-1], #[0, 1e-1, 1e-2]
    'betas': [(0.9, 0.95)], #, (0.9, 0.98)
    'epochs': [60, 80, 100], #20, 40, 60
}

freq = "A"
dep_var='domestic_fleet_landings'
lags_sequence = get_lags_for_frequency(freq)
time_features = time_features_from_frequency_str(freq)
embedding_dimension=[2]
verbose=0
patience=10

forecast_start=2013
forecast_stop=2019
prediction_length=5
pred_var=['foreign_landings'] #update here to run for specific model type. If running for a model without exogenous variables, pred_var=[].
filename='foreign_5' #update filename

#number of combinations:
all_names = sorted(param_grid)
combinations = list(product(*(param_grid[name] for name in all_names)))
len(combinations)

#Run hyperparameter search:
save_path=f'/content/drive/MyDrive/master_thesis/results/Transformer/hyperparam/h_5/models/best_model_{filename}.pth' #save model path
hyperparams_df = grid_search(param_grid)
