# Master's Thesis: Tackling the Data Deficiencies of West Africa’s Small-Scale Fisheries

The code contained in this repository supplements my master's thesis titled 'Tackling the Data Deficiencies of West Africa's Small-Scale Fisheries'.

# Methods

ARIMA, LSTM, and Transformer models were fit to the historical domestic fleet landings of 22 countries in West Africa. Numerous exogenous variables were tested for predictive power, including: sea surface temperature (sst), chlorophyll-a concentration (chl-a), the country's average World Bank governance score (as measured across 6 different dimensions), the level of foreign landings (both in aggregate and disagreggated between reported vs. unreported landings), the domestic fleet's fishing effort by country per year, and the resulting catch per unit of effort (CPUE) by domestic fleet and year. Models were tested under 2 periods: Period 1 forecasts years 2005-2010 while Period 2 forecasts years 2013-2019. Models were evaluated against a baseline persistence model using MSE and MAE. The best performing model (ultimately, a Transformer model fit with governance and foreign landings) was used to forecast years 2020-2030. The resulting forecasts were then included in a supply and demand model in order to estimate the 2030 supply gap.

# Running instructions:

NOAA_data_prep.py : connects to the National Oceanic and Atmospheric Administration's data repositories and pulls down sst and chl-a data for years 1982-2021 and 1997-2022, respectively. The resulting data from this code can be found saved in the 'data' folder. This code does not need to be run unless adding additional years of data and refreshing analyses.

data_clean.py : contains code to clean the data for landings and exogenous variables. The  resulting data from this code can be found in the 'data' folder. This code does not need to be run unless adding additional years of data and refreshing analyses.

baseline_model.py : run this code in order to obtain the model metrics for the baseline persistence model.

ARIMA_functions.py : contains code for all of the functions used within ARIMA_model.py

ARIMA_exogenous_forecast.py : CPUE and fishing effort data were only available from 1950-2010. This code will fit an ARIMA model to CPUE and effort by country in order to forecast years 2011-2019.

ARIMA_model.py : run this code in order to obtain the model metrics of different ARIMA model configurations. Configurations must be defined in 'input_configs' before running. 

LSTM_functions.py : contains code for all of the functions used within LSTM_model.py

LSTM_hyperparameters.py : contains code to run a grid search over possible LSTM model hyperparameters. To run, you must specify which model configuration you would like to test by defining the predictor variables to include in the model with pred_vars (no exogenous variables in the model would be pred_vars=['domestic_fleet_landings','country']).

LSTM_model.py : run this code in order to obtain the model metrics of different LSTM model configurations. Configurations must be defined in 'input_configs' before running. 

Transformer_functions.py : contains code for all of the functions used within Transformer_model.py

Transformer_hyperparameters.py : contains code to run a grid search over possible Transformer model hyperparameters. To run, you must specify which model configuration you would like to test by defining the predictor variables to include in the model with pred_var (no exogenous variables in the model would be pred_var=[]).

LSTM_model.py : run this code in order to obtain the model metrics of different Transformer model configurations. Configurations must be defined in 'input_configs' before running. Code to produce final 2020-2030 forecasts can be found at the end of this file as well.

Report_analysis.py : contains the code to produce the graphs found in the report body of my master's thesis.



