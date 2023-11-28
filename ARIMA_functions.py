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
drive.mount('/content/drive')

##################################################################################################################
#Fit auto arima and perform iterative forecasts:
def arima_training(data, dep_var, forecast_start, forecast_stop, steps, exog_var=None):

  def fit_arima(data, column, exog_var=None, p=0, q=0, max_p=10, max_q=10, stepwise=False, seasonal=False, maxiter=10000, information_criterion='aic', trace=False, error_action='ignore', suppress_warnings=True):
    arima_model = pm.auto_arima(data[column], X=exog_var, start_p=p, start_q=q, max_p=max_p, max_q=max_q, stepwise=stepwise, seasonal=seasonal, maxiter=maxiter, information_criterion=information_criterion, trace=trace, error_action=error_action, suppress_warnings=suppress_warnings)
    return arima_model

  def predict_arima(model, var, exog_var=None, steps=steps):
    prediction, conf_int = model.predict(n_periods=steps, X=exog_var, return_conf_int=True)
    prediction = prediction[-1:]
    conf_int = conf_int[-1:]
    pred_df = pd.DataFrame({f"{var}": prediction, f"{var}_lower": conf_int[:, 0], f"{var}_upper": conf_int[:, 1]})
    return pred_df

  forecast = pd.DataFrame() #dataframe to save forecasts
  arima_models = {} #dictionary to save all arima model results

  for country in data['area_name'].unique():
    country_data = data[data['area_name'] == country]
    country_data_test = country_data[country_data["year"]>=forecast_start]

    country_forecast = []

    for i in range (forecast_start-steps,forecast_stop-steps+1,1): #iteratively make forecasts over the horizon period
      country_data_train = country_data[country_data["year"]<=i]
      if exog_var==None:
        model = fit_arima(country_data_train, f"{dep_var}_change")
        pred_df = predict_arima(model, f"{dep_var}_change", steps=steps)
        country_forecast.append(pred_df)

      else:
        model = fit_arima(country_data_train, f"{dep_var}_change", exog_var=pd.DataFrame(country_data_train[exog_var]))
        pred_df = predict_arima(model, f"{dep_var}_change", exog_var=pd.DataFrame(country_data[(country_data["year"]>i)&(country_data["year"]<=i+steps)][exog_var]), steps=steps)
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
#Calculate model performance metrics:
def calculate_performance_metrics(predicted, actual, pred_var, steps):
  performance_metrics = pd.DataFrame()
  for country in predicted['area_name'].unique():
    mse = mean_squared_error(actual[actual['area_name']==country][dep_var], predicted[predicted['area_name']==country][f'{dep_var}_pred'])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual[actual['area_name']==country][dep_var], predicted[predicted['area_name']==country][f'{dep_var}_pred'])
    mean_actual = actual[actual['area_name']==country][dep_var].mean()
    normalized_mae = mae / mean_actual
    r2 = r2_score(actual[actual['area_name']==country][dep_var], predicted[predicted['area_name']==country][f'{dep_var}_pred'])

    metrics_current = {"Country": country,"Predictor variable": pred_var, "Time step": steps,"Prediction_start": forecast_start,"Prediction_end": forecast_stop,"MSE": mse,"RMSE": rmse,"MAE": mae,"Normalized MAE": normalized_mae,"R^2": r2}
    metrics_current = pd.DataFrame([metrics_current])
    performance_metrics = pd.concat([performance_metrics, metrics_current], ignore_index=True)

  #average metrics:
  filtered = performance_metrics[(performance_metrics['Time step'] == steps)&(performance_metrics['Predictor variable'] == pred_var)&(performance_metrics['Prediction_start'] == forecast_start)&(performance_metrics['Prediction_end'] == forecast_stop)]
  avg_mse = filtered['MSE'].mean()
  avg_rmse = np.sqrt(avg_mse)
  avg_mae = filtered['MAE'].mean()
  actual_values_all = actual[dep_var].values
  avg_normalized_mae = avg_mae / np.mean(actual_values_all)
  if steps==1:
    if forecast_start==2005:
      relative_mae = avg_mae / MAE_1_2005_2010
    if forecast_start==2013:
      relative_mae = avg_mae / MAE_1_2013_2019
  if steps==5:
    if forecast_start==2005:
      relative_mae = avg_mae / MAE_5_2005_2010
    if forecast_start==2013:
      relative_mae = avg_mae / MAE_5_2013_2019
  if steps==11:
    if forecast_start==2005:
      relative_mae = avg_mae / MAE_11_2005_2010
    if forecast_start==2013:
      relative_mae = avg_mae / MAE_11_2013_2019
  avg_r2 = filtered["R^2"].mean()

  average_metrics = {"Country": 'Average - All countries',"Predictor variable": pred_var,"Time step": steps,"Prediction_start": forecast_start,"Prediction_end": forecast_stop,"MSE": avg_mse,"RMSE": avg_rmse,"MAE": avg_mae,"Normalized MAE": avg_normalized_mae,"R^2": avg_r2, "MAE (relative to baseline model)": relative_mae}
  average_metrics = pd.DataFrame([average_metrics])
  performance_metrics = pd.concat([performance_metrics, average_metrics], ignore_index=True)

  return performance_metrics

##################################################################################################################
#create dataframe of model results:
def arima_model_results(model):
    data = {
        'Country': [],
        'Model': [],
        'Variable': [],
        'Coefficient': [],
        'P-value': []}

    for country, model in model.items(): #loop through each country's model
        summary = model.summary()
        model_order = model.order
        variables = summary.tables[1]
        for row in variables.data[1:]:
            variable = row[0].strip()
            coefficient = float(row[1].strip())
            p_value = float(row[4].strip())

            data['Country'].append(country)
            data['Model'].append(model_order)
            data['Variable'].append(variable)
            data['Coefficient'].append(coefficient)
            data['P-value'].append(p_value)

    result_df = pd.DataFrame(data)
    return result_df
