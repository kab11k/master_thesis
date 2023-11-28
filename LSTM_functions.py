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
drive.mount('/content/drive')

#baseline MAE (MAE values resulting from baseline model, for comparison):
MAE_1_2005_2010=20071.994239899
MAE_5_2005_2010=41751.8767145466
MAE_11_2005_2010=58121.5629626857

MAE_1_2013_2019=25772.3083759616
MAE_5_2013_2019=57168.9294177568
MAE_11_2013_2019=80186.8393054145

##################################################################################################################

def data_sequences (data, lags, forecast_steps, data_start, train_year_end, val_year_end, test_year_end, verbose):
#def data_sequences (data, lags, forecast_steps, forecast_year, forecast_stop, num_val_years, verbose):

  '''''
        This function builds the training, testing, and validation sequences to feed into a LSTM prediction model. It creates input sequences of lagged predictor variables and
        output sequences of the dependent variable depending on the number of forecasting steps specified.
  '''''

  all_X, all_y, all_X_test, all_y_test, all_val_x, all_val_y = [], [], [], [], [], []# initialize lists to store sequences for each country

  for country in data['area_name'].unique():
    country_data = data[data['area_name'] == country] #filter on current country
    #split out training, testing, and validation data:
    train_seq = np.array(country_data[(country_data["year"] >= data_start) & (country_data["year"] <= train_year_end)][pred_vars + [dep_var]])
    val_seq = np.array(country_data[(country_data["year"] > train_year_end-lags-forecast_steps+1) & (country_data["year"] <= val_year_end)][pred_vars + [dep_var]])
    test_seq = np.array(country_data[(country_data["year"] > val_year_end-lags-forecast_steps+1) & (country_data["year"] <= test_year_end)][pred_vars + [dep_var]])

    def split_sequence(seq):
      X, y = [], []
      for i in range(len(seq)):
        input_end = i + lags
        output_end = input_end + forecast_steps
        if output_end > len(seq):
          break
        data_x, data_y = seq[i:input_end, :-1], seq[input_end:output_end, -1]
        X.append(data_x)
        y.append(data_y)
      return np.array(X), np.array(y)

    X, y = split_sequence(train_seq)
    X_test, y_test = split_sequence(test_seq)
    val_x, val_y = split_sequence(val_seq)

    if verbose == 1: #print statements to check sequences
      print("----------------------------------------------------------------------------------")
      print(f"Country: {country}")
      print("Train sequence:")
      print(X.shape, y.shape)
      for i in range(len(X)):
        print(X[i], y[i])
      print("Validation sequence:")
      print(val_x.shape, val_y.shape)
      for i in range(len(val_x)):
        print(val_x[i], val_y[i])
      print("Test sequence:")
      print(X_test.shape, y_test.shape)
      for i in range(len(X_test)):
        print(X_test[i], y_test[i])
      print("----------------------------------------------------------------------------------")

    all_X.append(X)
    all_y.append(y)
    all_X_test.append(X_test)
    all_y_test.append(y_test)
    all_val_x.append(val_x)
    all_val_y.append(val_y)

  # combine input and output sequences for all countries:
  X = np.concatenate(all_X)
  y = np.concatenate(all_y)
  X_test = np.concatenate(all_X_test)
  y_test = np.concatenate(all_y_test)
  val_x = np.concatenate(all_val_x)
  val_y = np.concatenate(all_val_y)

  return X, y, X_test, y_test, val_x, val_y


def evaluate_model(data, lags, preds, pred_vars, forecast_start, forecast_stop, forecast_steps, data_start, neurons, epochs, batch_size, dropout, num_layers, use_GRU, use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm, save_path, filename):

  '''''
        This function analyses the results of the LSTM model compared to actuals. Calculates the MSE, RMSE, MAE and compares the MAE to the baseline model's results. Returns results as a csv file.
  '''''

  pred_vars_str = ', '.join(pred_vars)
  performance_metrics = pd.DataFrame()

  #by country results:
  for i in data['country'].unique():
    country = data[data['country'] == i]['area_name'].unique()[0]
    country_preds = [row[i] for row in preds] #select predictions for current country
    country_preds = np.array(country_preds)

    #reverse the min max scaler:
    scaler.fit(df[df['area_name']==country][dep_var].values.reshape(-1, 1))
    inverse_predictions = scaler.inverse_transform(country_preds.reshape(-1, 1))
    inverse_predictions = inverse_predictions.reshape(-1)
    predicted_values=inverse_predictions.copy()

    actual_values = df[(df["area_name"]  == country) & (df["year"] >= forecast_start) & (df["year"] <= forecast_stop)][dep_var].values


    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, predicted_values)
    normalized_mae = mae / np.mean(actual_values)
    r2 = r2_score(actual_values, predicted_values)

    metrics_current = {"Country": country, "Predictor Variable": pred_vars_str, "Lags": lags,
                       "Time step": forecast_steps, "Data Start":  data_start, "Prediction_start": forecast_start, "Prediction_end": forecast_stop,
                       "Neurons": neurons, "Epochs": epochs, "Batch size": batch_size, "Dropout": dropout, "Number of layers": num_layers, "GRU": use_GRU,
                       "Bidirect": use_Bidirectional, "Grad clip": use_grad_clip, "lr scheduler": use_lrscheduler, "Batch norm": use_batchnorm,
                       "MSE": mse, "RMSE": rmse, "MAE": mae, "Normalized MAE": normalized_mae,"R^2": r2}
    metrics_current= pd.DataFrame([metrics_current])
    performance_metrics = pd.concat([performance_metrics, metrics_current], ignore_index=True)


  #average results over all countries:
  filtered = performance_metrics[(performance_metrics['Predictor Variable'] == pred_vars_str)&(performance_metrics['Lags'] == lags)&(performance_metrics['Time step'] == forecast_steps)&
          (performance_metrics['Data Start'] == data_start)&(performance_metrics['Prediction_start'] == forecast_start)&(performance_metrics['Prediction_end'] == forecast_stop)&
          (performance_metrics['Neurons'] == neurons)&(performance_metrics['Epochs'] == epochs)&(performance_metrics['Batch size'] == batch_size)&(performance_metrics['Dropout'] == dropout)&
          (performance_metrics['Number of layers'] == num_layers)&(performance_metrics['GRU'] == use_GRU)&(performance_metrics['Bidirect'] == use_Bidirectional)&
          (performance_metrics['Grad clip'] == use_grad_clip)&(performance_metrics['lr scheduler'] == use_lrscheduler)&(performance_metrics['Batch norm'] == use_batchnorm)]

  avg_mse = filtered['MSE'].mean()
  avg_rmse = np.sqrt(avg_mse)
  avg_mae = filtered['MAE'].mean()
  actual_values_all = df[(df["year"] >= forecast_start) & (df["year"] <= forecast_stop)][dep_var].values
  avg_normalized_mae = avg_mae / np.mean(actual_values_all)
  avg_r2 = filtered["R^2"].mean()
  if forecast_steps==1:
    if forecast_start==2005:
      relative_mae = avg_mae / MAE_1_2005_2010
    if forecast_start==2013:
      relative_mae = avg_mae / MAE_1_2013_2019
  if forecast_steps==5:
    if forecast_start==2005:
      relative_mae = avg_mae / MAE_5_2005_2010
    if forecast_start==2013:
      relative_mae = avg_mae / MAE_5_2013_2019
  if forecast_steps==11:
    if forecast_start==2005:
      relative_mae = avg_mae / MAE_11_2005_2010
    if forecast_start==2013:
      relative_mae = avg_mae / MAE_11_2013_2019

  average_metrics = {"Country": 'Average - All countries',"Predictor Variable": pred_vars_str,"Lags": lags,
                     "Time step": forecast_steps,"Data Start":  data_start,"Prediction_start": forecast_start,"Prediction_end": forecast_stop,
                     "Neurons": neurons, "Epochs": epochs, "Batch size": batch_size, "Dropout": dropout, "Number of layers": num_layers, "GRU": use_GRU,
                     "Bidirect": use_Bidirectional, "Grad clip": use_grad_clip, "lr scheduler": use_lrscheduler, "Batch norm": use_batchnorm,
                     "MSE": avg_mse,"RMSE": avg_rmse,"MAE": avg_mae,"Normalized MAE": avg_normalized_mae,"R^2": avg_r2,"MAE (relative to baseline model)": relative_mae}

  average_metrics = pd.DataFrame([average_metrics])
  performance_metrics = pd.concat([performance_metrics, average_metrics], ignore_index=True)

  return performance_metrics



def lstm_build (X, y, X_test, y_test, val_x, val_y, forecast_steps, lags, neurons, epochs, batch_size, activation_LSTM, activation_Dense, dropout, num_layers, use_GRU, use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm, verbose, patience, model_save):

  '''''
        This function builds the LSTM model under various archietctures and hyperparameters as specified in the variables passed when calling the function. Function returns forecasts as well as training and testing loss.
  '''''

  model = Sequential(Masking(mask_value=-1))

  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience, verbose=verbose, min_lr=1e-8)

  optimizer = tf.keras.optimizers.Adam(clipvalue=1.0) if use_grad_clip==1 else 'adam'

  if use_Bidirectional==1:
    model.add(Bidirectional(LSTM(units=neurons, return_sequences=True, activation=activation_LSTM), input_shape=(lags, X.shape[2])))
    if use_batchnorm==1:
      model.add(BatchNormalization())
    model.add(Dropout(dropout))

    #Add additional recurrent layers:
    for _ in range(3, num_layers + 1):
      if use_GRU==1:
        model.add(Bidirectional(GRU(units=neurons, return_sequences=True)))
        if use_batchnorm==1:
          model.add(BatchNormalization())
      else:
        model.add(Bidirectional(LSTM(units=neurons, return_sequences=True, activation=activation_LSTM)))
        if use_batchnorm==1:
          model.add(BatchNormalization())
      model.add(Dropout(dropout))

  else:
    model.add(LSTM(units=neurons, return_sequences=True, activation=activation_LSTM, input_shape=(lags, X.shape[2])))
    if use_batchnorm==1:
      model.add(BatchNormalization())
    model.add(Dropout(dropout))

    #Add additional recurrent layers:
    for _ in range(3, num_layers + 1):
      if use_GRU==1:
        model.add(GRU(units=neurons, return_sequences=True))
        if use_batchnorm==1:
          model.add(BatchNormalization())
      else:
        model.add(LSTM(units=neurons, return_sequences=True, activation=activation_LSTM))
        if use_batchnorm==1:
          model.add(BatchNormalization())
      model.add(Dropout(dropout))

  model.add(LSTM(units=neurons, return_sequences=False))
  if use_batchnorm==1:
    model.add(BatchNormalization())
  #model.add(Dropout(dropout))
  model.add(Dense(units=forecast_steps, activation=activation_Dense))

  model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])


  callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=patience),
              ModelCheckpoint(model_save, monitor='val_loss', mode='min', verbose=verbose, save_best_only=True)]

  if use_lrscheduler:
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience, verbose=verbose, min_lr=1e-8))

  model.fit(X, y, validation_data=(val_x, val_y), batch_size=batch_size, callbacks=callbacks, epochs=epochs, verbose=0)

  #save training and testing accuracy:
  train_accuracy = model.evaluate(X, y, verbose=0)
  test_accuracy = model.evaluate(X_test, y_test, verbose=0)

  #make predictions
  pred = model.predict(X_test, verbose=0)

  return pred, train_accuracy, test_accuracy


def lstm_hyperparam (config, forecast_steps, activation_Dense, verbose, patience, model_save, save_path, filename):

  '''''
        This function performs a grid search over hyperparameters and architectural options generating the training and testing accuracy of different model configuraitons for comparison.
  '''''

  lags, neurons, epochs, batch_size, activation_LSTM, dropout, num_layers, use_GRU, use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm = config
  possible_combinations = list(itertools.product(lags, neurons, epochs, batch_size, activation_LSTM, dropout, num_layers, use_GRU, use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm))

  hist = []

  for i in range(0, len(possible_combinations)):

    lags, neurons, epochs, batch_size, activation_LSTM, dropout, num_layers, use_GRU, use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm = possible_combinations[i]

    X, y, X_test, y_test, val_x, val_y = data_sequences(data, lags, forecast_steps, data_start, train_year_end, val_year_end, test_year_end, verbose)

    pred, train_accuracy, test_accuracy = lstm_build(X, y, X_test, y_test, val_x, val_y, forecast_steps, lags, neurons, epochs, batch_size, activation_LSTM, activation_Dense, dropout, num_layers, use_GRU, use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm, verbose, patience, model_save)

    hist.append(list((lags, neurons, epochs, batch_size, activation_LSTM, dropout, num_layers, use_GRU, use_Bidirectional, use_grad_clip, use_lrscheduler, use_batchnorm, train_accuracy, test_accuracy)))

    #save temp results:
    hist_temp = pd.DataFrame(hist)
    hist_temp.to_csv(f'{save_path}/{filename}.csv', index=False)


  return hist
