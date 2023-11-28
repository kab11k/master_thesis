import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
drive.mount('/content/drive')

##################################################################################################################
df = pd.read_csv("/content/drive/MyDrive/master_thesis/df_clean.csv")
df=df[df['area_name']!='South Africa'].copy() #exclude South Africa

#persistence model:

def persistence_model(x): #simple persistence model
  return x

performance_metrics = pd.DataFrame()

input_configs = [
    (df, 2005, 2010, 1),
    (df, 2005, 2010, 11),
    (df, 2013, 2019, 1),
    (df, 2013, 2019, 11),
]

for config in input_configs:
    data, forecast_start, forecast_stop, steps = config

    for country in data['area_name'].unique():
      country_data = data[data['area_name']==country].copy()

      country_data['lagged'] = country_data.groupby('area_name')['domestic_fleet_landings'].shift(steps) #create lagged value by country
      dataframe = pd.concat([country_data['lagged'] , country_data['domestic_fleet_landings']],axis=1) #join landings and lags
      dataframe.index = pd.DatetimeIndex(pd.to_datetime(country_data['year'], format='%Y')) #set year as index

      train, test = dataframe[dataframe.index.year < forecast_start].copy().values, dataframe[(dataframe.index.year >= forecast_start) & (dataframe.index.year <= forecast_stop)].copy().values #test/train split
      train_X, train_y = train[:,0], train[:,1]
      test_X, test_y = test[:,0], test[:,1]

      predictions = list() #generate predictions
      for x in test_X:
        yhat = persistence_model(x)
        predictions.append(yhat)

      #metric calculations:
      mse = mean_squared_error(test_y, predictions)
      rmse = np.sqrt(mse)
      mae = mean_absolute_error(test_y, predictions)
      mean_actual = test_y.mean()
      normalized_mae = mae / mean_actual
      r2 = r2_score(test_y, predictions)

      metrics_current = {"Country": country,"Steps": steps,"Prediction_start": forecast_start,"Prediction_end": forecast_stop,"MSE": mse,"RMSE": rmse,"MAE": mae,"Normalized MAE": normalized_mae,"R^2": r2}
      metrics_current = pd.DataFrame([metrics_current])
      performance_metrics = pd.concat([performance_metrics, metrics_current], ignore_index=True)

    #average results over all countries:
    filtered = performance_metrics[(performance_metrics['Steps'] == steps)&(performance_metrics['Prediction_start'] == forecast_start)&(performance_metrics['Prediction_end'] == forecast_stop)]
    avg_mse = filtered['MSE'].mean()
    avg_rmse = np.sqrt(avg_mse)
    avg_mae = filtered['MAE'].mean()
    actual_values_all = data[(data["year"] >= forecast_start) & (data["year"] <= forecast_stop)]['domestic_fleet_landings'].values
    avg_normalized_mae = avg_mae / np.mean(actual_values_all)
    #avg_normalized_mae = filtered["Normalized MAE"].mean()
    avg_r2 = filtered["R^2"].mean()

    average_metrics = {"Country": 'Average - All countries',"Steps": steps,"Prediction_start": forecast_start,"Prediction_end": forecast_stop,"MSE": avg_mse,"RMSE": avg_rmse,"MAE": avg_mae,"Normalized MAE": avg_normalized_mae,"R^2": avg_r2}
    average_metrics = pd.DataFrame([average_metrics])
    performance_metrics = pd.concat([performance_metrics, average_metrics], ignore_index=True)

performance_metrics_avg = performance_metrics[performance_metrics['Country']=="Average - All countries"] #results on average 
performance_metrics_bycountry = performance_metrics[performance_metrics['Country']!="Average - All countries"] #results by country

performance_metrics_avg.to_csv(f'/content/drive/MyDrive/master_thesis/results/baseline/performance_metrics_avg.csv', index=False)
performance_metrics_bycountry.to_csv(f'/content/drive/MyDrive/master_thesis/results/baseline/performance_metrics_bycountry.csv', index=False)

