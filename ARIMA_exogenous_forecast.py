import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pmdarima as pm
from pmdarima import model_selection
import seaborn as sns
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from numpy import cumsum, log, polyfit, sqrt, std, subtract
import matplotlib.transforms as mtransforms
import matplotlib.cm as cm
from scipy.stats import spearmanr, pearsonr
from scipy.stats import probplot, moment
import statsmodels.tsa.api as smt
from statsmodels.tsa.api import VAR, VARMAX
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.statespace.varmax import VARMAXResults
from statsmodels.stats.stattools import durbin_watson
from dieboldmariano import dm_test
import pickle
from hampel import hampel
import ARIMA_functions
drive.mount('/content/drive')

##################################################################################################################
#import cleaned dataset:
df = pd.read_csv("/content/drive/MyDrive/master_thesis/df_clean.csv")
#exclude South Africa:
df=df[df['area_name']!='South Africa'].copy()

#Data Transformation:

#years with 0 are NA in data. Need to replace with 0s to differentiate between actual missing data.
cols = ['foreign_landings', 'foreign_Reported_landings', 'foreign_Unreported_landings']
df[cols] = df[cols].fillna(0)

#hampel filtering:
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

        # Print the replaced values
        #replaced_values = original_values[original_values != hampel_series]
        #for replaced_value in replaced_values:
        #    print(f"Replaced outlier: {replaced_value} in column {column} for country {country}")

    # Update the dataframe with the modified data for the current country
    df_hampel = pd.concat([df_hampel, country_data], ignore_index=True)
  
#Percent change transformation:
#Transform data to year over year percent change:
cols_used = ['domestic_fleet_landings', 'effort','CPUE',
             'foreign_landings', 'foreign_Reported_landings','foreign_Unreported_landings',
             'sst', 'chlorophyll', 'avg_governance_score']

df_perc = df_hampel.sort_values(['area_name', 'year'])
grouped = df_perc.groupby('area_name')# group by country

replacement_value = 0.000001 #need to replace 0 landings with a small, insignificant value so that percent changes can be calculated without losing records where landings were truely 0

for column in cols_used:
  df_perc[column] = df_perc[column].replace(0, replacement_value)
  df_perc[column + '_change'] = grouped[column].pct_change().replace([np.inf, -np.inf], np.nan)

df_perc.index = pd.DatetimeIndex(pd.to_datetime(df_perc['year'], format='%Y')) #set year as index

##################################################################################################################
#Function to produce final ARIMA final forecasts (i.e, non-iterative):

#Fit auto arima and perform iterative forecasts:
def arima_final_forecast(data, dep_var, forecast_start, forecast_stop, steps, exog_var=None):

  def fit_arima(data, column, exog_var=None, p=0, q=0, max_p=10, max_q=10, stepwise=False, seasonal=False, maxiter=10000, information_criterion='aic', trace=False, error_action='ignore', suppress_warnings=True):
    arima_model = pm.auto_arima(data[column], X=exog_var, start_p=p, start_q=q, max_p=max_p, max_q=max_q, stepwise=stepwise, seasonal=seasonal, maxiter=maxiter, information_criterion=information_criterion, trace=trace, error_action=error_action, suppress_warnings=suppress_warnings)
    return arima_model

  def predict_arima(model, var, exog_var=None, steps=steps):
    prediction, conf_int = model.predict(n_periods=steps, X=exog_var, return_conf_int=True)
    #prediction = prediction[-1:]
    #conf_int = conf_int[-1:]
    pred_df = pd.DataFrame({f"{var}": prediction, f"{var}_lower": conf_int[:, 0], f"{var}_upper": conf_int[:, 1]})
    return pred_df

  forecast = pd.DataFrame() #dataframe to save forecasts
  arima_models = {} #dictionary to save all arima model results

  for country in data['area_name'].unique():
    country_data = data[data['area_name'] == country]
    country_data_test = country_data[country_data["year"]>=forecast_start]

    country_forecast = []


    country_data_train = country_data[country_data["year"]<forecast_start]
    if exog_var==None:
      model = fit_arima(country_data_train, f"{dep_var}_change")
      pred_df = predict_arima(model, f"{dep_var}_change", steps=steps)
      country_forecast.append(pred_df)
    else:
      model = fit_arima(country_data_train, f"{dep_var}_change", exog_var=pd.DataFrame(country_data_train[exog_var]))
      pred_df = predict_arima(model, f"{dep_var}_change", exog_var=pd.DataFrame(exog_forecast[(exog_forecast["area_name"]==country)&(exog_forecast["year"]>=forecast_start)&(exog_forecast["year"]<=forecast_stop)][exog_var]), steps=steps)
      country_forecast.append(pred_df)

    model_key = f"{country}"
    arima_models[model_key] = model # Save the current arima model to the dictionary

    country_forecast_df = pd.concat(country_forecast, axis=0) #create dataframe of forecasts for the entire horizon


    #Convert to original units:
    country_forecast_df[f"{dep_var}_pred"] = country_data[country_data["year"]==(forecast_start-1)][dep_var].values[0] * (1 + country_forecast_df[f"{dep_var}_change"]).cumprod()
    country_forecast_df[f"{dep_var}_lower"] = country_data[country_data["year"]==(forecast_start-1)][dep_var].values[0]  * (1 + country_forecast_df[f"{dep_var}_change_lower"]).cumprod()
    country_forecast_df[f"{dep_var}_upper"] = country_data[country_data["year"]==(forecast_start-1)][dep_var].values[0]  * (1 + country_forecast_df[f"{dep_var}_change_upper"]).cumprod()

    country_forecast_df['area_name'] = country #add country as a column and assign current country
    forecast = pd.concat([forecast, country_forecast_df], axis=0) #combine all countries' forecasts

  return forecast, arima_models
  
##################################################################################################################
#Calculate base ARIMA performance for exogenous variables:

performance_metrics = pd.DataFrame()

save_dir = '/content/drive/MyDrive/master_thesis/results/ARIMA/exogenous_forecasting/testing_models/'

data_start=1951

input_configs = [

    ('exog_cpue', 'CPUE', 11, 2005, 2010, ['domestic_fleet_landings_change','effort_change'], ['area_name','year','CPUE','CPUE_change','domestic_fleet_landings','effort','domestic_fleet_landings_change','effort_change']),
    ('exog_effort', 'effort', 11, 2005, 2010, None, ['area_name','year','CPUE','CPUE_change','domestic_fleet_landings','effort','domestic_fleet_landings_change','effort_change']),
    ('exog_sst', 'sst', 11, 2013, 2019, None, ['area_name','year','sst','sst_change']),
    ('exog_chl', 'chlorophyll', 11, 2013, 2019, None, ['area_name','year','chlorophyll','chlorophyll_change']),
    ('exog_gov', 'avg_governance_score', 11, 2013, 2019, None, ['area_name','year','avg_governance_score','avg_governance_score_change']),
    ('exog_foreign', 'foreign_landings', 11, 2013, 2019, None, ['area_name','year','foreign_landings','foreign_landings_change']),
    ('exog_foreign_rep', 'foreign_Reported_landings', 11, 2013, 2019, None, ['area_name','year','foreign_Reported_landings','foreign_Reported_landings_change']),
    ('exog_foreign_unrep', 'foreign_Unreported_landings', 11, 2013, 2019, None, ['area_name','year','foreign_Unreported_landings','foreign_Unreported_landings_change'])

]

for config in input_configs:
    filename, dep_var, steps, forecast_start, forecast_stop, exog_var, to_keep = config

    data=df_perc
    data=data[(data['year']>=data_start)&(data['year']<=forecast_stop)]
    data=data[to_keep].dropna()

    landings_forecast, landings_model = arima_training(data, dep_var, forecast_start, forecast_stop, steps, exog_var) #model fit and forecasting

    if not os.path.exists(save_dir): #save ARIMA models
      os.makedirs(save_dir)
    for country, model in landings_model.items(): #loop through and save each country's model
      file_path = os.path.join(save_dir, f'arima_model_{filename}_{country}.pkl')

      with open(file_path, 'wb') as file:
        pickle.dump(model, file)

#calculate model metrics:
    actuals = data[(data['year'] >= forecast_start)&(data['year'] <= forecast_stop)] #select actual values
    if not exog_var:
      pred_var = ','.join([f'{dep_var}_change'])
      landings_metrics = calculate_performance_metrics(landings_forecast, actuals, pred_var, steps)
    else:
      pred_var = ','.join([f'{dep_var}_change'] + exog_var)
      landings_metrics = calculate_performance_metrics(landings_forecast, actuals, pred_var, steps)

    performance_metrics = pd.concat([performance_metrics, landings_metrics], ignore_index=True)
    performance_metrics.to_csv(f'/content/drive/MyDrive/master_thesis/results/ARIMA/exogenous_forecasting/metrics_{filename}.csv', index=False)
    
##################################################################################################################
#forecast:
performance_metrics = pd.DataFrame()

save_dir = '/content/drive/MyDrive/master_thesis/results/ARIMA/exogenous_forecasting/forecast_models/'

data_start=1951


input_configs = [

    ('cpue', 'CPUE', 20, 2011, 2030, None, ['area_name','year','CPUE','CPUE_change','domestic_fleet_landings','effort','domestic_fleet_landings_change','effort_change']),
    ('effort', 'effort', 20, 2011, 2030, None, ['area_name','year','effort','effort_change']),
]

for config in input_configs:
    filename, dep_var, steps, forecast_start, forecast_stop, exog_var, to_keep = config

    data=df_perc
    data=data[(data['year']>=data_start)&(data['year']<=forecast_stop)]
    data=data[to_keep].dropna()

    forecast, landings_model = arima_final_forecast(data, dep_var, forecast_start, forecast_stop, steps, exog_var) #model fit and forecasting

    if not os.path.exists(save_dir): #save ARIMA models
      os.makedirs(save_dir)
    for country, model in landings_model.items(): #loop through and save each country's model
      file_path = os.path.join(save_dir, f'arima_model_{filename}_{country}.pkl')

      with open(file_path, 'wb') as file:
        pickle.dump(model, file)


    forecast.to_csv(f'/content/drive/MyDrive/master_thesis/results/ARIMA/exogenous_forecasting/forecasts_{filename}.csv', index=False)

