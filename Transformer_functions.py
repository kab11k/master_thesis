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
drive.mount('/content/drive')

#baseline MAE:
MAE_1_2005_2010=20071.994239899
MAE_5_2005_2010=41751.8767145466
MAE_11_2005_2010=58121.5629626857

MAE_1_2013_2019=25772.3083759616
MAE_5_2013_2019=57168.9294177568
MAE_11_2013_2019=80186.8393054145

##################################################################################################################
class ProcessStartField():
    ts_id = 0

    def __call__(self, data):
        '''''
          Function maps pandas dataset 'start' field to a timestamp rather than pd.Period
        '''''

        data["start"] = data["start"].to_timestamp()
        data["feat_static_cat"] = [self.ts_id]
        self.ts_id += 1

        return data


def target_transformation_length(
    target: np.ndarray, pred_length: int, is_train: bool
) -> int:
    return target.shape[-1] + (0 if is_train else pred_length)

class ExtendDynamicFeature(MapTransformation):
    '''''
    If `is_train=True` the dynamic real feature(s) has the same length as the `target` field.
    If `is_train=False` the dynamic real feature(s) has length len(target) + pred_length
   '''''
    def __init__(
        self,
        feature_field: str,
        target_field: str,
        pred_length: int,
        dtype: Type = np.float32,
    ) -> None:
        self.feature_field = feature_field
        self.target_field = target_field
        self.pred_length = pred_length
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )
        if len(data[self.feature_field][0]) == length:
            return data

        extended_feature = np.concatenate([
            data[self.feature_field],
            np.tile(data[self.feature_field][:, -1:], (1, self.pred_length))
        ], axis=1)

        data[self.feature_field] = extended_feature[:, :length].astype(self.dtype)

        return data


def create_transformation (freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    return Chain(
        [RemoveFields(field_names=remove_field_names)]
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_DYNAMIC_REAL,
                    expected_ndim=2,
                )
            ]
            if config.num_dynamic_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=1,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
        ]
        + (
            [
                ExtendDynamicFeature(
                    target_field=FieldName.TARGET,
                    feature_field=FieldName.FEAT_DYNAMIC_REAL,
                    pred_length=config.prediction_length
                )
            ]
            if config.num_dynamic_real_features > 0
            else []
        )
        + [
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )


def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )


def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of context length + lags + prediction length randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(
        stream, is_train=True
    )

    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )



def create_test_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    #creates a Test Instance splitter which will sample the very last context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "test")

    testing_instances = instance_sampler.apply(transformed_data, is_train=False)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )



def create_val_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    instance_splitter = create_instance_splitter(config, "validation")

    stream = Cyclic(transformed_data).stream()
    validation_instances = instance_splitter.apply(stream, is_train=True)

    return as_stacked_batches(
        validation_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )


def transformer_build (config, freq, dataset_train, dataset_test, dataset_val, batch_size, num_batches_per_epoch, learning_rate, betas, weight_decay, epochs, patience, verbose, filename, save_path):
  '''''
    Fits a transformer model and produces forecasts
  '''''

  min_val_loss = float('inf') #initialize loss
  epochs_without_improvement = 0 #patience counter

  set_seed(23) #set seed to stabalize results

  model = TimeSeriesTransformerForPrediction(config) #define model

  train_dataloader = create_train_dataloader(config=config,freq=freq,data=dataset_train,batch_size=batch_size,num_batches_per_epoch=num_batches_per_epoch) #load train data
  test_dataloader = create_test_dataloader(config=config,freq=freq,data=dataset_test,batch_size=batch_size) #load test data
  val_dataloader = create_val_dataloader(config=config,freq=freq,data=dataset_val,batch_size=batch_size,num_batches_per_epoch=num_batches_per_epoch) #load validation data

  #Train the model:
  accelerator = Accelerator()
  device = accelerator.device

  model.to(device)
  optimizer = AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True) #Initialize lr scheduler

  model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

  for epoch in range(epochs):
    model.train()

    for idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device, dtype=torch.float32),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device, dtype=torch.float32),
            future_values=batch["future_values"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
            future_observed_mask=batch["future_observed_mask"].to(device),
        )
        loss = outputs.loss

        # Backpropagation
        accelerator.backward(loss)
        optimizer.step()

        if verbose==1:
          print(f"Epoch {epoch}, Batch {idx}, Loss: {loss.item()}")


    #Evaluate validation loss:
    model.eval()
    val_losses = []
    with torch.no_grad():
      for batch in val_dataloader:
        outputs = model(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device, dtype=torch.float32),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device, dtype=torch.float32),
            future_values=batch["future_values"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
            future_observed_mask=batch["future_observed_mask"].to(device),
        )

        val_loss=outputs.loss
        val_losses.append(val_loss.item())
    val_loss = np.mean(val_losses)

    if verbose==1:
      print(f"Epoch {epoch+1}, Validation loss: {val_loss:.4f}")


    # Early stopping and model checkpointing
    if val_loss < min_val_loss:
      min_val_loss = val_loss
      torch.save(model.state_dict(), save_path) # Save the best model
      epochs_without_improvement = 0
    else:
      epochs_without_improvement += 1
      if epochs_without_improvement == patience:
        print(f"Early stopping after {patience} epochs without improvement!")
        break
    scheduler.step(val_loss)


  #Inference:
  # Load best model for inference
  model.load_state_dict(torch.load(save_path))
  model.eval()

  forecasts = []

  for batch in test_dataloader:
    outputs = model.generate(
        static_categorical_features=batch["static_categorical_features"].to(device)
        if config.num_static_categorical_features > 0
        else None,
        static_real_features=batch["static_real_features"].to(device)
        if config.num_static_real_features > 0
        else None,
        past_time_features=batch["past_time_features"].to(device, dtype=torch.float32),
        past_values=batch["past_values"].to(device),
        future_time_features=batch["future_time_features"].to(device, dtype=torch.float32),
        past_observed_mask=batch["past_observed_mask"].to(device),
    )
    forecasts.append(outputs.sequences.cpu().numpy())

  return forecasts


def invert_forecasts(forecasts, dataset_test_original):
  '''''
    Transforms forecasts to original units by inverting the min max scaler
  '''''
  forecasts = np.vstack(forecasts) #stack forecasts vertically
  forecast_median = np.median(forecasts, axis=1)
  forecast_mean = np.mean(forecasts, axis=1)
  forecast_std = np.nanstd(forecasts, axis=1)

  inverted_forecast_median = []
  inverted_forecast_mean = []
  inverted_forecast_std = []

  countries = df_scaled[['country', 'area_name']].drop_duplicates() #unique list of area_names and corresponding country encoded labels

  for item_id, ts in enumerate(dataset_test_original):
    country = countries[countries['country']==ts["item_id"]]['area_name'].values[0] #current country
    print(country)

    country_pred_median = forecast_median[item_id] #select current country's median forecasts
    country_pred_mean = forecast_mean[item_id] #select current country's median forecasts
    country_pred_std = forecast_std[item_id] #select current country's median forecasts

    scaler.fit(df[df['area_name'] == country][dep_var].values.reshape(-1, 1)) #fit scaler to selected country

    inverted_country_median = scaler.inverse_transform(country_pred_median.reshape(-1, 1)).reshape(-1) #invert the scaler
    inverted_forecast_median.append(inverted_country_median) #append to list

    inverted_country_mean = scaler.inverse_transform(country_pred_mean.reshape(-1, 1)).reshape(-1) #invert the scaler
    inverted_forecast_mean.append(inverted_country_mean) #append to list

    inverted_country_std = scaler.inverse_transform(country_pred_std.reshape(-1, 1)).reshape(-1) #invert the scaler
    inverted_forecast_std.append(inverted_country_std) #append to list

  inverted_forecast_median = np.array(inverted_forecast_median)
  inverted_forecast_mean = np.array(inverted_forecast_mean)
  inverted_forecast_std = np.array(inverted_forecast_std)

  return inverted_forecast_median, inverted_forecast_mean, inverted_forecast_std


def evaluate_model(dataset_test_original,inverted_forecast_median,forecast_start,forecast_stop,pred_var):
  mse_metrics = []
  mae_metrics = []
  r2_metrics = []
  results_data = []

  horizon=forecast_stop-forecast_start+1
  pred_vars_str = ', '.join(pred_var)

  countries = df_scaled[['country', 'area_name']].drop_duplicates() #unique list of area_names and corresponding country encoded labels

  for item_id, ts in enumerate(dataset_test_original):
    country = countries[countries['country']==ts["item_id"]]['area_name'].values[0] #current country

    ground_truth = ts["target"][-horizon:] #actuals

    mse = mse_metric.compute(
        predictions=inverted_forecast_median[item_id],
        references=np.array(ground_truth))
    mse_metrics.append(mse["mse"])

    mae = mae_metric.compute(
        predictions=inverted_forecast_median[item_id],
        references=np.array(ground_truth))
    mae_metrics.append(mae["mae"])

    r2 = r2_metric.compute(
        predictions=inverted_forecast_median[item_id],
        references=np.array(ground_truth))
    r2_metrics.append(r2)

    results_data.append({
        'Country': country,
        'Predictor Variable': pred_vars_str,
        'MSE': mse['mse'],
        'RMSE': np.sqrt(mse['mse']),
        'MAE': mae['mae'],
        'Normalized MAE': mae['mae'] / np.mean(np.array(ground_truth)),
        'R_squared': r2
    })

  results_df = pd.DataFrame(results_data)

  #calculate average results:
  avg_mse = results_df['MSE'].mean()
  avg_rmse = np.sqrt(avg_mse)
  avg_mae = results_df['MAE'].mean()
  actual_values_all = df[(df["year"] >= forecast_start) & (df["year"] <= forecast_stop)][dep_var].values
  avg_normalized_mae = avg_mae / np.mean(actual_values_all)
  avg_r2 = results_df['R_squared'].mean()
  if prediction_length==1:
    if forecast_start==2005:
      relative_mae = avg_mae / MAE_1_2005_2010
    if forecast_start==2013:
      relative_mae = avg_mae / MAE_1_2013_2019
  if prediction_length==5:
    if forecast_start==2005:
      relative_mae = avg_mae / MAE_5_2005_2010
    if forecast_start==2013:
      relative_mae = avg_mae / MAE_5_2013_2019
  if prediction_length==11:
    if forecast_start==2005:
      relative_mae = avg_mae / MAE_11_2005_2010
    if forecast_start==2013:
      relative_mae = avg_mae / MAE_11_2013_2019

  average_metrics = {"Country": 'Average - All countries','Predictor Variable': pred_vars_str,"MSE": avg_mse,"RMSE": avg_rmse,"MAE": avg_mae,"Normalized MAE": avg_normalized_mae,"R_squared": avg_r2,"MAE (relative to baseline model)": relative_mae}
  average_metrics = pd.DataFrame([average_metrics])

  #merge with by country results:
  performance_metrics = pd.concat([results_df, average_metrics], ignore_index=True)
  performance_metrics

  return performance_metrics


def create_GluonTS_dataset(data, start, stop, prediction_length, dep_var, pred_var):
  '''''
    Transforms data into GluonTS dataset (format needed for transformer model)
  '''''

  df_train = data[data['year']<=start-prediction_length].copy() #training set
  df_test = data[data['year']<=start+prediction_length].copy() #testing set
  df_val = data[data['year']<=start].copy() #validation set
  df_test_original=df[df['year']<=forecast_stop].copy()  #testing set with original units of measurement

  #reformat index:
  df_train['year'] = df_train['year'].astype(str)
  df_train['year'] = pd.to_datetime(df_train['year'], format='%Y')
  df_train.set_index('year', inplace=True)
  df_train.index = df_train.index.strftime('%Y-%m-%d %H:%M:%S')

  df_test['year'] = df_test['year'].astype(str)
  df_test['year'] = pd.to_datetime(df_test['year'], format='%Y')
  df_test.set_index('year', inplace=True)
  df_test.index = df_test.index.strftime('%Y-%m-%d %H:%M:%S')

  df_val['year'] = df_val['year'].astype(str)
  df_val['year'] = pd.to_datetime(df_val['year'], format='%Y')
  df_val.set_index('year', inplace=True)
  df_val.index = df_val.index.strftime('%Y-%m-%d %H:%M:%S')

  df_test_original['year'] = df_test_original['year'].astype(str)
  df_test_original['year'] = pd.to_datetime(df_test_original['year'], format='%Y')
  df_test_original.set_index('year', inplace=True)
  df_test_original.index = df_test_original.index.strftime('%Y-%m-%d %H:%M:%S')

  ds_kwargs = {
        "target": dep_var,
        "item_id": "country",
        "freq": freq
  }

  if pred_var:
    ds_kwargs["feat_dynamic_real"] = pred_var

  # Convert to a GluonTS pandas dataset:
  ds_train = PandasDataset.from_long_dataframe(df_train, **ds_kwargs)
  ds_test = PandasDataset.from_long_dataframe(df_test, **ds_kwargs)
  ds_val = PandasDataset.from_long_dataframe(df_val, **ds_kwargs)
  ds_test_original = PandasDataset.from_long_dataframe(df_test_original, **ds_kwargs)

  #iterates over each 'item_id' in ds and creates a 'start' timestamp and 'feat_static_cat' for each 'item_id' i.e country time series:
  process_start = ProcessStartField()
  list_ds_train = list(Map(process_start, ds_train))

  process_start = ProcessStartField()
  list_ds_test = list(Map(process_start, ds_test))

  process_start = ProcessStartField()
  list_ds_val = list(Map(process_start, ds_val))

  process_start = ProcessStartField()
  list_ds_test_original = list(Map(process_start, ds_test_original))

  #define our schema features and create our dataset from this list via the from_list function:

  feature_schema = {
        "start": Value("timestamp[s]"),
        "target": Sequence(Value("float32")),
        "feat_static_cat": Sequence(Value("uint64")),
        "item_id": Value("string"),
  }

  if pred_var:
    feature_schema["feat_dynamic_real"] = Sequence(Sequence(Value("float32")))

  features = Features(feature_schema)

  dataset_train = Dataset.from_list(list_ds_train, features=features)
  dataset_test = Dataset.from_list(list_ds_test, features=features)
  dataset_val = Dataset.from_list(list_ds_val, features=features)
  dataset_test_original = Dataset.from_list(list_ds_test_original, features=features)

  #convert the start feature of each time series to a pandas Period index using the data's freq:
  @lru_cache(10_000)
  def convert_to_pandas_period(date, freq):
      return pd.Period(date, freq)


  def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch

  dataset_train.set_transform(partial(transform_start_field, freq=freq))
  dataset_test.set_transform(partial(transform_start_field, freq=freq))
  dataset_val.set_transform(partial(transform_start_field, freq=freq))
  dataset_test_original.set_transform(partial(transform_start_field, freq=freq))

  return dataset_train, dataset_test, dataset_val, dataset_test_original


def grid_search(param_grid):
    all_names = sorted(param_grid)
    combinations = list(product(*(param_grid[name] for name in all_names)))
    best_score = float('inf')
    best_params = None

    hyperparams = []

    for combination in combinations:
        params = dict(zip(all_names, combination))

        num_dynamic_real_features=len(pred_var)
        inverted_forecast_median = [] #initialize list for forecasts
        inverted_forecast_mean = []
        inverted_forecast_std = []

        for i in range(forecast_start,forecast_stop+1):
          start=stop=i

          dataset_train, dataset_test, dataset_val, dataset_test_original = create_GluonTS_dataset (df_scaled, start, stop, prediction_length, dep_var, pred_var)

          config = TimeSeriesTransformerConfig(
          prediction_length=prediction_length,
          context_length=params['context_length'],
          lags_sequence=lags_sequence,
          num_time_features=len(time_features) + 1,
          num_static_categorical_features=1,
          cardinality=[len(dataset_train)],
          embedding_dimension=embedding_dimension,
          dropout=params['dropout'],
          num_dynamic_real_features=num_dynamic_real_features,

          encoder_layers=params['encoder_layers'],
          decoder_layers=params['decoder_layers'],
          d_model=params['d_model'],)

          forecasts = transformer_build(
            config=config,
            freq=freq,
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            dataset_val=dataset_val,
            batch_size=params['batch_size'],
            num_batches_per_epoch=params['num_batches_per_epoch'],
            learning_rate=params['learning_rate'],
            betas=params['betas'],
            weight_decay=params['weight_decay'],
            epochs=params['epochs'],
            patience=patience,
            verbose=0,
            filename=filename,
            save_path=save_path)

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
        current_score = performance[performance['Country']=='Average - All countries']['MSE'].values

        hyperparams.append(list((params, current_score)))
        hyperparams_df = pd.DataFrame(hyperparams)
        hyperparams_df.to_csv(f'/content/drive/MyDrive/master_thesis/results/Transformer/hyperparam/h_5/hyperparams_{filename}.csv', index=False)

    return hyperparams_df
