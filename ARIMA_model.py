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
#set baseline MAE (These are the MAE values resulting from the baseline model for periods 1 and 2 with horizons 1 year and 11 years):
MAE_1_2005_2010=20071.994239899
MAE_11_2005_2010=58121.5629626857

MAE_1_2013_2019=25772.3083759616
MAE_11_2013_2019=80186.8393054145

##################################################################################################################
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

#Impute CPUE and effort values for years 2011-2019:
cpue_effort_forecast = pd.read_csv("/content/drive/MyDrive/master_thesis/results/ARIMA/exogenous_forecasting/forecasts/cpue_effort.csv") #cpue and effort forecasted values years 2011-2019
#join data with forecasted cpue and effort variables years 2011-2019:
cols = ['CPUE', 'effort']
for col in cols:
    #merge predictions:
    df_hampel = pd.merge(df_hampel, cpue_effort_forecast[['year', 'area_name', f'{col}_pred']],
                         on=['year', 'area_name'], how='left')

    df_hampel[col].fillna(df_hampel[f'{col}_pred'], inplace=True) #replace NA with prediction
    df_hampel.drop(f'{col}_pred', axis=1, inplace=True)

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

#Exploratory Data Analysis:
#plots original domestic landings by country time series with ADF and Q-stat, transformed domestic landings by country with ADF and Q-stat, transformed domestic landinsg by country Probability plot and ACF and PACF plots
def plot_correlogram(x, axes, lags=None, title=None, full_plot=True):
    lags = min(10, int(len(x)/5)) if lags is None else lags

    # Time series plot
    x.plot(ax=axes[0])
    axes[0].set_title(title, fontsize=10)
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', labelrotation=45)

    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
    axes[0].text(x=.02, y=.85, s=stats, transform=axes[0].transAxes)

    if full_plot:
        probplot(x, plot=axes[1])
        mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
        s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):.2e}\nSkew: {skew:.2e}\nKurtosis: {kurtosis:.2e}'
        axes[1].text(x=.02, y=.75, s=s, transform=axes[1].transAxes)

        plot_acf(x=x, lags=lags, zero=False, ax=axes[2])
        plot_pacf(x, lags=lags, zero=False, ax=axes[3], method='ywm')
        axes[2].set_xlabel('Lag')
        axes[3].set_xlabel('Lag')

countries = df_perc["area_name"].unique()
fig, axes = plt.subplots(nrows=len(countries), ncols=5, figsize=(25, 4 * len(countries)))

for i, country in enumerate(countries):
    country_data = df_perc[df_perc["area_name"] == country]

    #domestic_fleet_landings plot (time series only)
    ax = axes[i, 0]
    plot_correlogram(country_data['domestic_fleet_landings'].dropna(), [ax], lags=10, title=f"{country} Original Time Series", full_plot=False)

    #domestic_fleet_landings_change plots (all plots)
    country_axes = axes[i, 1:5]
    plot_correlogram(country_data['domestic_fleet_landings_change'].dropna(), country_axes, lags=10, title=f"{country} Transformed Time Series")

fig.tight_layout(pad=2.0, h_pad=2.0)
plt.show()

#Correlation Matrix:
#Across all years and countries:

df_subset = df_perc[(df_perc["year"]>1996)&(df_perc["year"]<2011)] #subset data for years in which all variables are available

cols_to_use = ["domestic_fleet_landings_change", "effort_change", "CPUE_change", "foreign_landings_change", "sst_change", "chlorophyll_change", "avg_governance_score_change"]
custom_labels = ["Domestic Landings", "Effort", "CPUE",
                 "Foreign Landings",
                 "SST", "Chlorophyll", "Governance Score"]

corr = df_subset[cols_to_use].corr()

# Plot heatmap with variable names
g = sns.clustermap(corr, annot=True, xticklabels=custom_labels, yticklabels=custom_labels, vmin=-1, vmax=1)
ax = g.ax_heatmap
bottom, top = ax.get_ylim()
#ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Average Pairwise Correlations')
plt.show()

##################################################################################################################
#Period 1 (forecast years 2005-2010):

performance_metrics = pd.DataFrame()

save_dir = '/content/drive/MyDrive/master_thesis/results/ARIMA/2005_2010/models/'

data_start=1951
forecast_start=2005
forecast_stop=2010


input_configs = [
#Base
    ('base_1', 1, None, ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings']),
    ('base_11', 11, None,['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings']),
#chl:
    ('chl_1', 1, ['chlorophyll_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','chlorophyll_change','chlorophyll']),
    ('chl_11', 11, ['chlorophyll_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','chlorophyll_change','chlorophyll']),
#CPUE and effort:
    ('cpue_effort_1', 1, ['CPUE_change', 'effort_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings']),
    ('cpue_effort_11', 11, ['CPUE_change', 'effort_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings']),
#cpue, effort, foreign:
    ('cpue_effort_foreign_1', 1, ['CPUE_change', 'effort_change','foreign_landings_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
    ('cpue_effort_foreign_11', 11, ['CPUE_change', 'effort_change','foreign_landings_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
#cpue, effort, foreign rep v unrep: y
    ('cpue_effort_foreign_repvunrep_1', 1, ['CPUE_change', 'effort_change','foreign_Reported_landings_change','foreign_Unreported_landings_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
    ('cpue_effort_foreign_repvunrep_11', 11, ['CPUE_change', 'effort_change','foreign_Reported_landings_change','foreign_Unreported_landings_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
#cpue, effort, sst:
    ('cpue_effort_sst_1', 1, ['CPUE_change', 'effort_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings''sst_change','sst']),
    ('cpue_effort_sst_11', 11, ['CPUE_change', 'effort_change','sst_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','sst_change','sst']),
#cpue, effort, sst, foreign:
    ('cpue_effort_sst_foreign_1', 1, ['CPUE_change', 'effort_change','foreign_landings_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
    ('cpue_effort_sst_foreign_11', 11, ['CPUE_change', 'effort_change','foreign_landings_change','sst_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
#cpue, effort, sst, foreign rep v unrep:
    ('cpue_effort_sst_foreign_repvunrep_1', 1, ['CPUE_change', 'effort_change','foreign_Reported_landings_change','foreign_Unreported_landings_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
    ('cpue_effort_sst_foreign_repvunrep_11', 11, ['CPUE_change', 'effort_change','foreign_Reported_landings_change','foreign_Unreported_landings_change','sst_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings_change','foreign_Unreported_landings_change']),
#foreign:
    ('foreign_1', 1, ['foreign_landings_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings',]),
    ('foreign_11', 11, ['foreign_landings_change'],['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings',]),
#foreign reported v unreported:
    ('foreign_repvunrep_1', 1, ['foreign_Reported_landings_change','foreign_Unreported_landings_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
    ('foreign_repvunrep_11', 11, ['foreign_Reported_landings_change','foreign_Unreported_landings_change'],['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
#gov:
    ('gov_1', 1, ['avg_governance_score_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','avg_governance_score_change','avg_governance_score']),
    ('gov_11', 11, ['avg_governance_score_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','avg_governance_score_change','avg_governance_score']),
#sst:
    ('sst_1', 1, ['sst_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','sst_change','sst']),
    ('sst_11', 11, ['sst_change'],['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','sst_change','sst']),
#sst, foreign:
    ('sst_foreign_1', 1, ['foreign_landings_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
    ('sst_foreign_11', 11, ['foreign_landings_change','sst_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
#sst, foreign rep v unrep:
    ('sst_foreign_repvunrep_1', 1, ['foreign_Reported_landings_change','foreign_Unreported_landings_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
    ('sst_foreign_repvunrep_11', 11, ['foreign_Reported_landings_change','foreign_Unreported_landings_change','sst_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),

]

for config in input_configs:
    filename, steps, exog_var, to_keep = config

    data=df_perc
    data=data[(data['year']>=data_start)&(data['year']<=forecast_stop)]
    data=data[to_keep].dropna()

    landings_forecast, landings_model = arima_training(data, 'domestic_fleet_landings', forecast_start, forecast_stop, steps, exog_var) #model fit and forecasting

    if not os.path.exists(save_dir): #save ARIMA models
      os.makedirs(save_dir)
    for country, model in landings_model.items(): #loop through and save each country's model
      file_path = os.path.join(save_dir, f'arima_model_{filename}_{country}.pkl')

      with open(file_path, 'wb') as file:
        pickle.dump(model, file)

#calculate model metrics:
    actuals = data[(data['year'] >= forecast_start)&(data['year'] <= forecast_stop)] #select actual values
    if not exog_var:
      pred_var = ','.join(['domestic_fleet_landings_change'])
      landings_metrics = calculate_performance_metrics(landings_forecast, actuals, pred_var, steps)
    else:
      pred_var = ','.join(['domestic_fleet_landings_change'] + exog_var)
      landings_metrics = calculate_performance_metrics(landings_forecast, actuals, pred_var, steps)

    performance_metrics = pd.concat([performance_metrics, landings_metrics], ignore_index=True)
    performance_metrics.to_csv(f'/content/drive/MyDrive/master_thesis/results/ARIMA/2005_2010/metrics_ARIMA.csv', index=False)

##################################################################################################################
#Period 2 (forecast years 2013-2019):

performance_metrics = pd.DataFrame()

save_dir = '/content/drive/MyDrive/master_thesis/results/ARIMA/2013_2019/models/persistence_cpue_effort/' #save individual country's models

data_start=1951
forecast_start=2013
forecast_stop=2019
dep_var='domestic_fleet_landings'

input_configs = [
#Base
    ('base_1', 1, None, ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings']),
    ('base_5', 5, None, ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings']),
    ('base_11', 11, None,['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings']),
#chl:
    ('chl_1', 1, ['chlorophyll_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','chlorophyll_change','chlorophyll']),
    ('chl_5', 5, ['chlorophyll_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','chlorophyll_change','chlorophyll']),
    ('chl_11', 11, ['chlorophyll_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','chlorophyll_change','chlorophyll']),
#CPUE and effort:
    ('cpue_effort_1', 1, ['CPUE_change', 'effort_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings']),
    ('cpue_effort_5', 5, ['CPUE_change', 'effort_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings']),
    ('cpue_effort_11', 11, ['CPUE_change', 'effort_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings']),
#cpue, effort, foreign:
    ('cpue_effort_foreign_1', 1, ['CPUE_change', 'effort_change','foreign_landings_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
    ('cpue_effort_foreign_5', 5, ['CPUE_change', 'effort_change','foreign_landings_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
    ('cpue_effort_foreign_11', 11, ['CPUE_change', 'effort_change','foreign_landings_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
#cpue, effort, foreign rep v unrep: y
    ('cpue_effort_foreign_repvunrep_1', 1, ['CPUE_change', 'effort_change','foreign_Reported_landings_change','foreign_Unreported_landings_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
    ('cpue_effort_foreign_repvunrep_5', 5, ['CPUE_change', 'effort_change','foreign_Reported_landings_change','foreign_Unreported_landings_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
    ('cpue_effort_foreign_repvunrep_11', 11, ['CPUE_change', 'effort_change','foreign_Reported_landings_change','foreign_Unreported_landings_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
#cpue, effort, sst:
    ('cpue_effort_sst_1', 1, ['CPUE_change', 'effort_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','sst_change','sst']),
    ('cpue_effort_sst_5', 5, ['CPUE_change', 'effort_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','sst_change','sst']),
    ('cpue_effort_sst_11', 11, ['CPUE_change', 'effort_change','sst_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','sst_change','sst']),
#cpue, effort, sst, foreign:
    ('cpue_effort_sst_foreign_1', 1, ['CPUE_change', 'effort_change','foreign_landings_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
    ('cpue_effort_sst_foreign_5', 5, ['CPUE_change', 'effort_change','foreign_landings_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
    ('cpue_effort_sst_foreign_11', 11, ['CPUE_change', 'effort_change','foreign_landings_change','sst_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
#cpue, effort, sst, foreign rep v unrep:
    ('cpue_effort_sst_foreign_repvunrep_1', 1, ['CPUE_change', 'effort_change','foreign_Reported_landings_change','foreign_Unreported_landings_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
    ('cpue_effort_sst_foreign_repvunrep_5', 5, ['CPUE_change', 'effort_change','foreign_Reported_landings_change','foreign_Unreported_landings_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
    ('cpue_effort_sst_foreign_repvunrep_11', 11, ['CPUE_change', 'effort_change','foreign_Reported_landings_change','foreign_Unreported_landings_change','sst_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings_change','foreign_Unreported_landings_change']),
#foreign:
    ('foreign_1', 1, ['foreign_landings_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings',]),
    ('foreign_5', 5, ['foreign_landings_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings',]),
    ('foreign_11', 11, ['foreign_landings_change'],['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings',]),
#foreign reported v unreported:
    ('foreign_repvunrep_1', 1, ['foreign_Reported_landings_change','foreign_Unreported_landings_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
    ('foreign_repvunrep_5', 5, ['foreign_Reported_landings_change','foreign_Unreported_landings_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
    ('foreign_repvunrep_11', 11, ['foreign_Reported_landings_change','foreign_Unreported_landings_change'],['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
#gov:
    ('gov_1', 1, ['avg_governance_score_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','avg_governance_score_change','avg_governance_score']),
    ('gov_5', 5, ['avg_governance_score_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','avg_governance_score_change','avg_governance_score']),
    ('gov_11', 11, ['avg_governance_score_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','avg_governance_score_change','avg_governance_score']),
#sst:
    ('sst_1', 1, ['sst_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','sst_change','sst']),
    ('sst_5', 5, ['sst_change'], ['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','sst_change','sst']),
    ('sst_11', 11, ['sst_change'],['area_name','year','domestic_fleet_landings_change','domestic_fleet_landings','sst_change','sst']),
#sst, foreign:
    ('sst_foreign_1', 1, ['foreign_landings_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
    ('sst_foreign_5', 5, ['foreign_landings_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
    ('sst_foreign_11', 11, ['foreign_landings_change','sst_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst']),
#sst, foreign rep v unrep:
    ('sst_foreign_repvunrep_1', 1, ['foreign_Reported_landings_change','foreign_Unreported_landings_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
    ('sst_foreign_repvunrep_5', 5, ['foreign_Reported_landings_change','foreign_Unreported_landings_change','sst_change'], ['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),
    ('sst_foreign_repvunrep_11', 11, ['foreign_Reported_landings_change','foreign_Unreported_landings_change','sst_change'],['area_name','year','CPUE_change','CPUE','effort_change','effort','domestic_fleet_landings_change','domestic_fleet_landings','foreign_landings_change','foreign_landings','sst_change','sst','foreign_Reported_landings_change','foreign_Unreported_landings_change','foreign_Reported_landings','foreign_Unreported_landings']),

]

for config in input_configs:
    filename, steps, exog_var, to_keep = config

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
      pred_var = ','.join([dep_var])
      landings_metrics = calculate_performance_metrics(landings_forecast, actuals, pred_var, steps)
    else:
      pred_var = ','.join([dep_var] + exog_var)
      landings_metrics = calculate_performance_metrics(landings_forecast, actuals, pred_var, steps)

    performance_metrics = pd.concat([performance_metrics, landings_metrics], ignore_index=True)
    performance_metrics.to_csv(f'/content/drive/MyDrive/master_thesis/results/ARIMA/2013_2019/metrics_ARIMA.csv', index=False)

