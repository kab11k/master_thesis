import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from hampel import hampel
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import math
from scipy import stats
import scipy.stats as stats
drive.mount('/content/drive')

##################################################################################################################
#import landings within EZZs:
df_EZZ_landings = pd.read_csv("/content/drive/MyDrive/master_thesis/df_EZZ_landings.csv")
df_domestic_fleet_landings = pd.read_csv("/content/drive/MyDrive/master_thesis/df_domestic_fleet_landings.csv")
df_merged = pd.read_csv("/content/drive/MyDrive/master_thesis/df_clean.csv")

##################################################################################################################
#Data clean:
df_EZZ_landings=df_EZZ_landings[df_EZZ_landings['area_name']!='South Africa'].copy() #remove South Africa
df_domestic_fleet_landings=df_domestic_fleet_landings[df_domestic_fleet_landings['fishing_entity']!='South Africa']
df_merged=df_merged[df_merged['area_name']!='South Africa']

df_EZZ_landings['sector2'] = df_EZZ_landings['fishing_sector'].apply(lambda x: 'Small-scale' if x != 'Industrial' else 'Industrial') #group small-scale sectors into one category

df_EZZ_landings_ind = df_EZZ_landings[df_EZZ_landings['sector2']=='Industrial'].copy() #subset on industrial only

# Group the countries according to LME
canary_countries = ['Cape Verde', 'Gambia', 'Mauritania', 'Morocco', 'Senegal']
guinea_countries = ['Benin', 'Cameroon', 'Congo', 'Ivory Coast', 'Democratic Republic of the Congo', 'Equatorial Guinea', 'Gabon', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Liberia', 'Nigeria', 'Sao Tome and Principe', 'Sierra Leone', 'Togo']
benguela_countries = ['Angola', 'Namibia']

groups = {
    'Canary': canary_countries,
    'Guinea': guinea_countries,
    'Benguela': benguela_countries
}

#map to LME:
def map_country_to_lme(country):
    for lme, countries in groups.items():
        if country in countries:
            return lme
    return 'Other'

df_EZZ_landings['LME'] = df_EZZ_landings['area_name'].apply(map_country_to_lme)

##################################################################################################################
#Report graphs:

########################################
#Stock status:
stock_status = pd.read_csv("/content/drive/MyDrive/master_thesis/west_africa_stock_statuses.csv")

orange_tones = ['#8B4000', '#D2691E', '#E9967A', '#F4A460', '#FFDAB9']

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

#fig.suptitle('Fisheries Stock Status by LME', fontsize=16)

for idx, area in enumerate(stock_status['Area'].unique()):
    area_df = stock_status[stock_status['Area'] == area]
    axs[idx].stackplot(area_df['Year'],
                       area_df['Collapsed'],
                       area_df['Over-exploited'],
                       area_df['Exploited'],
                       area_df['Developing'],
                       area_df['Rebuilding'],
                       colors=orange_tones)
    axs[idx].set_title(f'{area}')
    axs[idx].set_xlabel('')
    axs[idx].set_ylabel('Percentage')

# Add a common legend
fig.legend(['Collapsed', 'Over-exploited', 'Fully Exploited', 'Developing', 'Rebuilding'],
           loc='lower center', ncol=5, fontsize='large')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.08, 1, 0.92])

plt.show()

########################################
#Industrial v. Small-Scale presence:
def plot_group(df_group, group_name, ax):
    df_group_agg = df_group.groupby(['year', 'sector2']).sum(numeric_only=True).reset_index() #aggregate by year and sector
    df_group_pivot = df_group_agg.pivot(index='year', columns='sector2', values='tonnes')
    df_group_pivot.plot(kind='bar', stacked=True, colormap=colormap, ax=ax)
    ax.set_title(f'{group_name} Current')
    ax.set_xlabel('')
    ax.set_ylabel('Catch (in million tonnes)')
    ax.legend(title='Sector')

    ax.set_xticks(np.arange(0, 71, step=10)) #set x-axis ticks to display every 10 years
    ax.set_xticklabels(range(min(df_group_agg['year']), max(df_group_agg['year']) + 2, 10), rotation=45) #add a 2020 tick label


#settings for subplots:
fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharex=True)
colormap = 'tab20'
#fig.suptitle('Catch by Sector', fontsize=16, ha='left', x=0.01)

# Plot for each group in a loop
for ax, (group_name, countries) in zip(axs, groups.items()):
    df_group = df_EZZ_landings[df_EZZ_landings['area_name'].isin(countries)]
    plot_group(df_group, group_name, ax)
    ax.set_ylim(0, 8000000)
    handles, labels = ax.get_legend_handles_labels()

for ax in axs:
    ax.get_legend().remove()  # This should remove the legend for each individual subplot

fig.legend(handles, labels, loc='lower left', ncol=len(labels), fontsize='large')

#plt.tight_layout()
plt.tight_layout(rect=[0, 0.08, 1, 0.92])
plt.show()

########################################
#Industrial sector: Domestic v. Foreign:

def plot_group(df_group, group_name, ax):
    df_group_agg = df_group.groupby(['year', 'fleet_origin']).sum(numeric_only=True).reset_index() #aggregate by year and sector
    df_group_pivot = df_group_agg.pivot(index='year', columns='fleet_origin', values='tonnes')
    df_group_pivot.plot(kind='bar', stacked=True, colormap=custom_colormap, ax=ax)
    ax.set_title(f'{group_name} Current')
    ax.set_xlabel('')
    ax.set_ylabel('Catch (in million tonnes)')
    ax.legend(title='Fleet Origin')

    ax.set_xticks(np.arange(0, 71, step=10)) #set x-axis ticks to display every 10 years
    ax.set_xticklabels(range(min(df_group_agg['year']), max(df_group_agg['year']) + 2, 10), rotation=45) #add a 2020 tick label


#settings for subplots:
fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharex=True)
#colormap = 'tab20'
tab20_colors = plt.cm.tab20.colors
selected_colors = [tab20_colors[i] for i in [10, 15]]  # Replace indices with your chosen ones
custom_colormap = mcolors.ListedColormap(selected_colors)
#fig.suptitle('All Sectors - Domestic vs. Foreign Catch', fontsize=16, ha='left', x=0.01)

df_EZZ_landings_ind=df_EZZ_landings[df_EZZ_landings['sector2']=='Industrial'].copy()

# Plot for each group in a loop
for ax, (group_name, countries) in zip(axs, groups.items()):
    df_group = df_EZZ_landings_ind[df_EZZ_landings_ind['area_name'].isin(countries)]
    plot_group(df_group, group_name, ax)
    ax.set_ylim(0, 8000000)
    handles, labels = ax.get_legend_handles_labels()

labels = ['Domestic', 'Foreign']

for ax in axs:
    ax.get_legend().remove()  # This should remove the legend for each individual subplot

fig.legend(handles, labels, loc='lower left', ncol=len(labels), fontsize='large')

#plt.tight_layout()
plt.tight_layout(rect=[0, 0.08, 1, 0.92])
plt.show()

########################################
#Industrial sectors: Reported v. Unreported:

def plot_group(df_group, group_name, ax):
    df_group_agg = df_group.groupby(['year', 'reporting_status']).sum(numeric_only=True).reset_index() #aggregate by year and sector
    df_group_pivot = df_group_agg.pivot(index='year', columns='reporting_status', values='tonnes')
    df_group_pivot.plot(kind='bar', stacked=True, colormap=custom_colormap, ax=ax)
    ax.set_title(f'{group_name} Current')
    ax.set_xlabel('')
    ax.set_ylabel('Catch (in million tonnes)')
    ax.legend(title='Fleet Origin')

    ax.set_xticks(np.arange(0, 71, step=10)) #set x-axis ticks to display every 10 years
    ax.set_xticklabels(range(min(df_group_agg['year']), max(df_group_agg['year']) + 2, 10), rotation=45) #add a 2020 tick label


#settings for subplots:
fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharex=True)
#colormap = 'tab20'
tab20_colors = plt.cm.tab20.colors
selected_colors = [tab20_colors[i] for i in [2, 15]]  # Replace indices with your chosen ones
custom_colormap = mcolors.ListedColormap(selected_colors)
#fig.suptitle('All Sectors - Domestic vs. Foreign Catch', fontsize=16, ha='left', x=0.01)

df_EZZ_landings_ind=df_EZZ_landings[df_EZZ_landings['sector2']=='Industrial'].copy()

# Plot for each group in a loop
for ax, (group_name, countries) in zip(axs, groups.items()):
    df_group = df_EZZ_landings_ind[df_EZZ_landings_ind['area_name'].isin(countries)]
    plot_group(df_group, group_name, ax)
    ax.set_ylim(0, 8000000)
    handles, labels = ax.get_legend_handles_labels()

#labels = ['Domestic', 'Foreign']

for ax in axs:
    ax.get_legend().remove()  # This should remove the legend for each individual subplot

fig.legend(handles, labels, loc='lower left', ncol=len(labels), fontsize='large')

#plt.tight_layout()
plt.tight_layout(rect=[0, 0.08, 1, 0.92])
plt.show()

########################################
#Angola and Namibia: Domestic v. Foreign:
benguela_countries2 = ['Angola']
benguela_countries = ['Namibia']

groups2 = {
    'Angola': benguela_countries2,
    'Namibia': benguela_countries
}

def plot_group(df_group, group_name, ax):
    df_group_agg = df_group.groupby(['year', 'fleet_origin']).sum(numeric_only=True).reset_index() #aggregate by year and sector
    df_group_pivot = df_group_agg.pivot(index='year', columns='fleet_origin', values='tonnes')
    df_group_pivot.plot(kind='bar', stacked=True, colormap=custom_colormap, ax=ax)
    ax.set_title(f'{group_name}')
    ax.set_xlabel('')
    ax.set_ylabel('Catch (in million tonnes)')
    ax.legend(title='Fleet Origin')

    ax.set_xticks(np.arange(0, 71, step=10)) #set x-axis ticks to display every 10 years
    ax.set_xticklabels(range(min(df_group_agg['year']), max(df_group_agg['year']) + 2, 10), rotation=45) #add a 2020 tick label


#settings for subplots:
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.5), sharex=True)
#colormap = 'tab20'
tab20_colors = plt.cm.tab20.colors
selected_colors = [tab20_colors[i] for i in [10, 15]]  # Replace indices with your chosen ones
custom_colormap = mcolors.ListedColormap(selected_colors)
#fig.suptitle('Benguela Current Catch', fontsize=16, ha='left', x=0.01)

# Plot for each group in a loop
for ax, (group_name, countries) in zip(axs, groups2.items()):
    df_group = df_EZZ_landings[df_EZZ_landings['area_name'].isin(countries)]
    plot_group(df_group, group_name, ax)
    ax.set_ylim(0, 8000000)
    handles, labels = ax.get_legend_handles_labels()

labels = ['Domestic', 'Foreign']

for ax in axs:
    ax.get_legend().remove()  # This should remove the legend for each individual subplot

fig.legend(handles, labels, loc='lower left', ncol=len(labels), fontsize='large')

#plt.tight_layout()
plt.tight_layout(rect=[0, 0.08, 1, 0.92])
plt.show()

########################################
#Domestic fleet landings, CPUE, effort:

def plot_group(df_group, group_name, ax):

    df_group_agg = df_group.groupby(['year', 'area_name']).sum(numeric_only=True).reset_index()
    df_group_agg['landings_in_mill'] = df_group_agg['domestic_fleet_landings']/1000000
    df_group_pivot = df_group_agg.pivot(index='year', columns='area_name', values='landings_in_mill')
    df_group_pivot.plot(kind='bar', stacked=True, colormap=custom_colormap, ax=ax, alpha=.7, zorder=1)

    sum_effort = df_group.groupby('year')['effort'].sum().reset_index() #sum the effort of all countries in the current LME
    sum_effort = sum_effort[sum_effort['year'] <= 2010] #effort data is unavailable after 2010

    sum_landings = df_group.groupby('year')['domestic_fleet_landings'].sum().reset_index() #sum the ladnings of all countries in the current LME
    sum_landings = sum_landings[sum_landings['year'] <= 2010].copy()  #will be used to calculate cpue and so, limiting to years <=2010 to match effort data avilability

    cpue = sum_landings['domestic_fleet_landings'] * 1000 / sum_effort['effort'] #calculate CPUE

    #set labels and title:
    ax.set_xlabel('')
    ax.set_ylabel('Landings (in million tonnes)')
    ax.set_title(f'{group_name} Current')
    ax.legend(loc='upper left')

    ax2 = ax.twinx() #second y-axis representing the effort scale
    ax2.plot(sum_effort['year'].astype(str), sum_effort['effort'] / 1000000, label='Fishing Effort', color='black', zorder=2)
    ax2.set_zorder(ax.get_zorder() + 1)
    ax2.set_ylabel('Effort (million kWs)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='upper right')

    ax3 = ax.twinx()#third y-axis representing the CPUE scale
    ax3.spines['right'].set_position(('axes', 1.15))
    ax3.plot(sum_effort['year'].astype(str), cpue, label='CPUE', color='red', zorder=3)
    ax2.set_zorder(ax.get_zorder() + 2)
    ax3.set_ylabel('Catch per Unit of Effort', color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    ax3.legend(loc='lower right')


#settings for subplots:
fig, axs = plt.subplots(1, 3, figsize=(22, 10), sharey=True)
#colormap = 'tab20'
tab20_colors = plt.cm.tab20.colors
selected_colors = [tab20_colors[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,16,17,18,19]]  # Replace indices with your chosen ones
custom_colormap = mcolors.ListedColormap(selected_colors)
#fig.suptitle('Landings and Effort by Domestic Fleets', fontsize=16, ha='left', x=0.01)

# Plot for each group in a loop
for ax, (group_name, countries) in zip(axs, groups.items()):
    df_group = df_merged[df_merged['area_name'].isin(countries)]
    plot_group(df_group, group_name, ax)
    ax.set_ylim(0, 5)

    ax.set_xticks(np.arange(0, 71, step=10)) #set x-axis ticks to display every 10 years
    ax.set_xticklabels(range(min(df_group['year']), max(df_group['year']) + 2, 10), rotation=45) #add a 2020 tick label

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, -0.1), ncol=2, fontsize='large')


plt.tight_layout(pad=2.0)
plt.show()

########################################
#Sea surface temp:
EZZ_shapefiles = gpd.read_file("/content/drive/MyDrive/master_thesis/EZZ_shapefiles/eez_v11.shp")

geonames = ["Guinea Bissau Exclusive Economic Zone", "Guinean Exclusive Economic Zone", "Equatorial Guinean Exclusive Economic Zone",
            "Liberian Exclusive Economic Zone", "Ivory Coast Exclusive Economic Zone", "Ghanaian Exclusive Economic Zone",
            "Togolese Exclusive Economic Zone", "Beninese Exclusive Economic Zone", "Nigerian Exclusive Economic Zone",
            "Gambian Exclusive Economic Zone", "Gabonese Exclusive Economic Zone", "Cameroonian Exclusive Economic Zone",
            "Joint regime area Nigeria / Sao Tome and Principe", "Sao Tome and Principe Exclusive Economic Zone",
            "Congolese Exclusive Economic Zone", "Democratic Republic of the Congo Exclusive Economic Zone", "Angolan Exclusive Economic Zone",
            "Namibian Exclusive Economic Zone", "South African Exclusive Economic Zone", "Moroccan Exclusive Economic Zone",
            "Overlapping claim Western Saharan Exclusive Economic Zone", "Mauritanian Exclusive Economic Zone", "Cape Verdean Exclusive Economic Zone",
            "Joint regime area Senegal / Guinea Bissau", "Senegalese Exclusive Economic Zone", "Sierra Leonian Exclusive Economic Zone"]

EZZ_shapefiles = EZZ_shapefiles[EZZ_shapefiles['GEONAME'].isin(geonames)]

df_sst_1 = df_sst_max[df_sst_max["year"]==1982][['year', 'latitude', 'longitude', 'sst']]
df_sst_2 = df_sst_max[df_sst_max["year"]==2002][['year', 'latitude', 'longitude', 'sst']]
df_sst_3 = df_sst_max[df_sst_max["year"]==2021][['year', 'latitude', 'longitude', 'sst']]

fig, axes = plt.subplots(1, 3, figsize=(16, 6), subplot_kw={'projection': ccrs.PlateCarree()}) #3 subplots
fig.suptitle('Average Sea Surface Temperatures', fontsize=16, ha='left', x=0.0)

#create common color scale for all subplots
cmap = 'coolwarm'
vmin = min(df_sst_1['sst'].min(), df_sst_2['sst'].min(), df_sst_3['sst'].min())
vmax = max(df_sst_1['sst'].max(), df_sst_2['sst'].max(), df_sst_3['sst'].max())

for i, year_df in enumerate([df_sst_1, df_sst_2, df_sst_3]): #loops through and create one subplot per year
    ax = axes[i]

    ax.add_feature(cfeature.BORDERS, linestyle=':') #adds country borders
    ax.add_feature(cfeature.COASTLINE) #adds coastline

    for _, row in EZZ_shapefiles.iterrows(): #adds EZZ borders
        ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', zorder=12)

    sc_lat_lon = ax.scatter(year_df['longitude'], year_df['latitude'], c=year_df['sst'], cmap=cmap, s=1, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), zorder=10) #adds scatter plot of sst values at .5 x .5 degree resolution

    ax.set_extent([-30, 40, -35, 37])  #set map boundaries: [min_lon, max_lon, min_lat, max_lat]
    ax.set_title(f'{year_df["year"].iloc[0]}')

cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  #fixes legend location: [left, bottom, width, height]
cbar = plt.colorbar(sc_lat_lon, cax=cbar_ax, orientation='vertical')
cbar.set_label('Sea Surface Temperature (°C)')

#plt.tight_layout()

plt.subplots_adjust(wspace=0.01, left=0.01)

plt.show()


df_merged['LME'] = df_merged['area_name'].map(lambda x: 'Canary' if x in canary_countries else ('Guinea' if x in guinea_countries else 'Benguela'))

LME_data = df_merged.groupby(['LME', 'year'])['sst'].mean().reset_index()

plt.figure(figsize=(6, 4))

for LME_name, LME_data in LME_data.groupby('LME'):
    plt.plot(LME_data['year'], LME_data['sst'], label=LME_name)

# Customize the plot
plt.xlabel('')
plt.ylabel('Average Sea Surface Temperature (°C)')
plt.title('Average Sea Surface Temperature by LME')
plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
plt.grid(True)

# Display the plot
plt.show()

################################################################################################################################################################
#Final model graphs:

#import actuals data:
df = pd.read_csv("/content/drive/MyDrive/master_thesis/df_clean.csv")
#exclude South Africa:
df=df[df['area_name']!='South Africa'].copy()

#import period 2 forecasts:
forecasts_2013_2019 = pd.read_csv("/content/drive/MyDrive/master_thesis/results/Transformer/2020_2030/forecasts_median_2013_2019.csv")

#import 2020-2030 forecasts:
results_median = pd.read_csv("/content/drive/MyDrive/master_thesis/results/Transformer/2020_2030/forecasts_median_final.csv")
results_mean = pd.read_csv("/content/drive/MyDrive/master_thesis/results/Transformer/2020_2030/forecasts_mean_final.csv")
results_std = pd.read_csv("/content/drive/MyDrive/master_thesis/results/Transformer/2020_2030/forecasts_std_final.csv")

#transform forecasted dataframes:
results_median = results_median.melt(id_vars=['area_name'],
                    var_name='year',
                    value_name='domestic_fleet_landings')
results_median['year'] = results_median['year'].astype(int)
results_mean = results_mean.melt(id_vars=['area_name'],
                    var_name='year',
                    value_name='landings_mean')
results_mean['year'] = results_mean['year'].astype(int)
results_std = results_std.melt(id_vars=['area_name'],
                    var_name='year',
                    value_name='landings_std')
results_std['year'] = results_std['year'].astype(int)
forecasts_2013_2019 = forecasts_2013_2019.melt(id_vars=['area_name'],
                    var_name='year',
                    value_name='forecasted_landings')
forecasts_2013_2019['year'] = forecasts_2013_2019['year'].astype(int)

#combine forecasts with actuals:
combined_df = pd.concat([df, results_median])
combined_df.reset_index(drop=True, inplace=True)
combined_df = combined_df.merge(forecasts_2013_2019, on=['year', 'area_name'], how='left')

########################################
#Plot actuals and forecasted years 2020-2030 by country:
threshold_year=2019

num_countries = combined_df['area_name'].nunique()
num_columns = 4
num_rows = math.ceil(num_countries / num_columns)

plt.figure(figsize=(20, num_rows * 4))

for i, country in enumerate(combined_df['area_name'].unique(), 1):
    country_data = combined_df[combined_df['area_name'] == country]
    historical_data = country_data[country_data['year'] <= threshold_year]  # define threshold_year
    forecasted_data = country_data[country_data['year'] > threshold_year]

    ax = plt.subplot(num_rows, num_columns, i)

    ax.plot(historical_data['year'], historical_data['domestic_fleet_landings']/1000000, label='Historical', color='black')
    ax.plot(forecasted_data['year'], forecasted_data['domestic_fleet_landings']/1000000, label='Forecasted', color='blue')

    ax.set_title(f"{country}")
    ax.set_xlabel('')
    ax.set_ylabel('Domestic Landings (in million tonnes)')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

########################################
#Plot actuals and forecasted years for all countries combined:

total_landings = (combined_df.groupby('year')['domestic_fleet_landings'].sum())/1000000
mean_forecast = (combined_df[combined_df['year'] >= 2020].groupby('year')['landings_mean'].sum())/1000000

variance_forecast = combined_df[combined_df['year'] >= 2020].groupby('year')['landings_std'].apply(lambda x: (x**2).sum()) / (1000000**2)
std_forecast_agg = variance_forecast.apply(lambda x: x**0.5)

ci_upper = mean_forecast + 1.645 * std_forecast_agg
ci_lower = mean_forecast - 1.645 * std_forecast_agg

plt.figure(figsize=(10, 5))

# Plot actual data
plt.plot(total_landings[total_landings.index <= 2019], color='black', label='Actual Landings')

# Plot forecasted data
plt.plot(total_landings[total_landings.index >= 2020], color='blue', label='Forecasted Landings')

# Confidence interval
#plt.fill_between(mean_forecast.index, ci_lower, ci_upper, color='blue', alpha=0.3)

plt.title('Total Domestic Fleet Landings (Actual and Forecasted)')
plt.xlabel('')
plt.ylabel('Total Domestic Landings (in million tonnes)')
plt.legend()
plt.show()

########################################
#Dot graph of predicted v. actuals by prediction model and country:

#Best models' results:

transformer_1 = pd.read_csv("/content/drive/MyDrive/master_thesis/results/best_model_forecasts/Transformer/2013_2019/to_use/forecasts_median_cpue_effort_foreign_repvunrep_1.csv")
transformer_11 = pd.read_csv("/content/drive/MyDrive/master_thesis/results/best_model_forecasts/Transformer/2013_2019/to_use/forecasts_median_gov_foreign_11.csv")
lstm_1 = pd.read_csv("/content/drive/MyDrive/master_thesis/results/best_model_forecasts/LSTM/2013_2019/forecasts_1.csv")
lstm_11 = pd.read_csv("/content/drive/MyDrive/master_thesis/results/best_model_forecasts/LSTM/2013_2019/forecasts_11.csv")
arima_1 = pd.read_csv("/content/drive/MyDrive/master_thesis/results/best_model_forecasts/ARIMA/2013_2019/forecast_base_1.csv")
arima_11 = pd.read_csv("/content/drive/MyDrive/master_thesis/results/best_model_forecasts/ARIMA/2013_2019/forecast_cpue_effort_sst_11.csv")

#Transform data:
transformer_1 = transformer_1.melt(id_vars=['area_name'],
                    var_name='year',
                    value_name='domestic_fleet_landings')
transformer_11 = transformer_11.melt(id_vars=['area_name'],
                    var_name='year',
                    value_name='domestic_fleet_landings')

transformer_1['year'] = transformer_1['year'].astype(int)
transformer_11['year'] = transformer_11['year'].astype(int)
arima_1 = arima_1[['area_name', 'year', 'domestic_fleet_landings_pred']]
arima_1 = arima_1.rename(columns={'domestic_fleet_landings_pred': 'domestic_fleet_landings'})
arima_11 = arima_11[['area_name', 'year', 'domestic_fleet_landings_pred']]
arima_11 = arima_11.rename(columns={'domestic_fleet_landings_pred': 'domestic_fleet_landings'})

#prepare data for dot grpah:

def calculate_differences(df_actual, df_forecast):
    merged = pd.merge(df_actual, df_forecast, on=['year', 'area_name'], suffixes=('_actual', '_forecast'))
    merged['difference'] = (merged['domestic_fleet_landings_forecast'] - merged['domestic_fleet_landings_actual'])/1000
    return merged[['year', 'area_name', 'difference']]

diff_arima_1 = calculate_differences(df[['area_name', 'year','domestic_fleet_landings']], arima_1)
diff_lstm_1 = calculate_differences(df[['area_name', 'year','domestic_fleet_landings']], lstm_1)
diff_transformer_1 = calculate_differences(df[['area_name', 'year','domestic_fleet_landings']], transformer_1)

diff_arima_11 = calculate_differences(df[['area_name', 'year','domestic_fleet_landings']], arima_11)
diff_lstm_11 = calculate_differences(df[['area_name', 'year','domestic_fleet_landings']], lstm_11)
diff_transformer_11 = calculate_differences(df[['area_name', 'year','domestic_fleet_landings']], transformer_11)

def label_outliers(ax, data, threshold):
    for _, row in data.iterrows():
        if abs(row['difference']) > threshold:
            ax.text(row['year'], row['difference'], row['area_name'], horizontalalignment='right', size='small', color='black')


fig, axes = plt.subplots(3, 2, figsize=(15, 15))


# y-axis limits:
all_differences_long = pd.concat([diff_arima_11['difference'], diff_lstm_11['difference'], diff_transformer_11['difference']])
y_min, y_max = all_differences_long.min(), all_differences_long.max()
y_range = y_max - y_min
y_min -= y_range * 0.1
y_max += y_range * 0.1

# Define thresholds for outliers
thresholds_short = [350, 150, 150]
thresholds_long = [250, 400, 300]

# Plotting short-term forecasts
for i, diff_df in enumerate([diff_arima_1, diff_lstm_1, diff_transformer_1]):
    sns.scatterplot(data=diff_df, x='year', y='difference', hue='area_name', ax=axes[i, 0])
    label_outliers(axes[i, 0], diff_df, thresholds_short[i])
    axes[i, 0].axhline(0, color='black', linestyle='--')
    axes[i, 0].set_ylabel('Predicted - Actual Domestic Landings (in thousand tonnes)')
    axes[i, 0].set_ylim(y_min, y_max)

# Plotting long-term forecasts
for i, diff_df in enumerate([diff_arima_11, diff_lstm_11, diff_transformer_11]):
    sns.scatterplot(data=diff_df, x='year', y='difference', hue='area_name', ax=axes[i, 1])
    label_outliers(axes[i, 1], diff_df, thresholds_long[i])
    axes[i, 1].axhline(0, color='black', linestyle='--')
    axes[i, 1].set_ylabel('')
    axes[i, 1].set_ylim(y_min, y_max)

titles_short = ['ARIMA model (Short-term)', 'LSTM model (Short-term)', 'Transformer model (Short-term)']
titles_long = ['ARIMA model (Long-term)', 'LSTM model (Long-term)', 'Transformer model (Long-term)']

for i in range(3):
    axes[i, 0].set_title(titles_short[i])
    axes[i, 1].set_title(titles_long[i])

for ax in axes.ravel(): #remove individual plot legends
    ax.get_legend().remove()
    ax.set_xlabel('')

for ax in axes[0]:
    ax.set_ylabel('')
for ax in axes[2]:
    ax.set_ylabel('')


handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.03))

plt.tight_layout(rect=[0, 0.1, 1, 0.96])

plt.show()

########################################
#Line graph actual v predicted by model type:

#Prepare data for graph:
total_actual = (df.groupby('year')['domestic_fleet_landings'].sum())/1000000
total_arima1 = (arima_1.groupby('year')['domestic_fleet_landings'].sum())/1000000
total_lstm1 = (lstm_1.groupby('year')['domestic_fleet_landings'].sum())/1000000
total_transformer1 = (transformer_1.groupby('year')['domestic_fleet_landings'].sum())/1000000
total_arima11 = (arima_11.groupby('year')['domestic_fleet_landings'].sum())/1000000
total_lstm11 = (lstm_11.groupby('year')['domestic_fleet_landings'].sum())/1000000
total_transformer11 = (transformer_11.groupby('year')['domestic_fleet_landings'].sum())/1000000

def plot_total_landings(ax, actual, arima, lstm, transformer, title):
    ax.plot(actual, marker='o', color='black', label='Actual', markersize=4)
    ax.plot(arima, marker='^', color='blue', label='ARIMA', markersize=4)
    ax.plot(lstm, marker='s', color='green', label='LSTM', markersize=4)
    ax.plot(transformer, marker='x', color='red', label='Transformer', markersize=4)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('Total Domestic Fleet Landings (in million tonnes)')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_total_landings(axes[0], total_actual, total_arima1, total_lstm1, total_transformer1, "Short-term Forecasts")

plot_total_landings(axes[1], total_actual, total_arima11, total_lstm11, total_transformer11, "Long-term Forecasts")

axes[0].legend()

plt.tight_layout()
plt.show()

########################################
#Mauritania line graph actual v predicted by model type:

actual_mauritania = (df[df['area_name']=='Mauritania'].groupby('year')['domestic_fleet_landings'].sum())/1000000

arima1_mauritania = (arima_1[arima_1['area_name']=='Mauritania'].groupby('year')['domestic_fleet_landings'].sum())/1000000
lstm_1_mauritania = (lstm_1[lstm_1['area_name']=='Mauritania'].groupby('year')['domestic_fleet_landings'].sum())/1000000
transformer1_mauritania = (transformer_1[transformer_1['area_name']=='Mauritania'].groupby('year')['domestic_fleet_landings'].sum())/1000000

arima11_mauritania = (arima_11[arima_11['area_name']=='Mauritania'].groupby('year')['domestic_fleet_landings'].sum())/1000000
lstm_11_mauritania = (lstm_11[lstm_11['area_name']=='Mauritania'].groupby('year')['domestic_fleet_landings'].sum())/1000000
transformer11_mauritania = (transformer_11[transformer_11['area_name']=='Mauritania'].groupby('year')['domestic_fleet_landings'].sum())/1000000

def plot_total_landings(ax, actual, arima, lstm, transformer, title):
    ax.plot(actual, marker='o', color='black', label='Actual', markersize=4)
    ax.plot(arima, marker='^', color='blue', label='ARIMA', markersize=4)
    ax.plot(lstm, marker='s', color='green', label='LSTM', markersize=4)
    ax.plot(transformer, marker='x', color='red', label='Transformer', markersize=4)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('Mauritania Domestic Fleet Landings (in million tonnes)')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_total_landings(axes[0], actual_mauritania, arima1_mauritania, lstm_1_mauritania, transformer1_mauritania, "Short-term Forecasts")

plot_total_landings(axes[1], actual_mauritania, arima11_mauritania, lstm_11_mauritania, transformer11_mauritania, "Long-term Forecasts")

axes[0].legend()

plt.tight_layout()
plt.show()

################################################################################################################################################################
#Supply and Demand graphs:

########################################
#Change in supply:

merged_df = pd.read_csv("/content/drive/MyDrive/master_thesis/results/results.csv")

total_2019 = merged_df['domestic_landings_2019'].sum()
total_2030 = merged_df['domestic_landings_2030'].sum()
difference = total_2030 - total_2019
change= (total_2030-total_2019)/total_2019

print(f'2019 total: {total_2019:.2f}')
print(f'2030 total: {total_2030:.2f}')
print(f'Difference: {difference:.2f}')
print(f'Change: {change:.2%}')

merged_df['Difference'] = (merged_df['domestic_landings_2030'] - merged_df['domestic_landings_2019'])/1000

muted_green = '#8FBC8F'
muted_red = '#CD5C5C'

fig, ax = plt.subplots(figsize=(8, 6))

merged_df_sorted = merged_df.sort_values(by='Difference')

ax.barh(merged_df_sorted['area_name'], merged_df_sorted['Difference'], color=(merged_df_sorted['Difference'] > 0).map({True: 'green', False: 'red'}), alpha=.5)

ax.axvline(x=0, color='black', linewidth=1.5)

ax.set_xlabel('Change in Domestic Landings (in thousands)')
ax.set_title('Change in Domestic Fleet Landings by Country (2019 to 2030)')

plt.show()

########################################
#Supply gap:

merged_df['total_supply_2019'] = (merged_df['domestic_landings_2019'] + merged_df['aquaculture_2019'])/1000
merged_df['total_supply_2030'] = (merged_df['domestic_landings_2030'] + merged_df['aquaculture_2030'])/1000
merged_df['supply_gap_2030'] = merged_df['demand_2030']/1000 - merged_df['total_supply_2030']

merged_df['Supply Change'] = merged_df['total_supply_2030'] - merged_df['total_supply_2019']
merged_df['Demand Change'] = (merged_df['demand_2030'] - merged_df['demand_2019'])/1000

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)

color1 = '#4F94CD'
color2 = '#FFA07A'
positive_color = '#CD5C5C'
negative_color = '#8FBC8F'


merged_df.plot(kind='barh', x='area_name', y=['Supply Change', 'Demand Change'], ax=axes[0],
               color=[color1, color2], alpha=0.7, legend=True)
axes[0].set_ylabel('')
axes[0].set_xlabel('Change in marine and aquaculture landings (in thousand tonnes)')
axes[0].axvline(x=0, color='black', linewidth=0.8)


gap_colors = [positive_color if value >= 0 else negative_color for value in merged_df['supply_gap_2030']]


merged_df.plot(kind='barh', x='area_name', y='supply_gap_2030', ax=axes[1], color=gap_colors, alpha=0.7, legend=False)
axes[1].set_ylabel('')
axes[1].set_xlabel('Forecasted Supply Gap (in thousand tonnes)')
axes[1].axvline(x=0, color='black', linewidth=0.8)

plt.tight_layout()
plt.show()

################################################################################################################################################################
#Period 1 v. Period 2 error differences:

dif_df = pd.read_csv("/content/drive/MyDrive/master_thesis/period_diff_normmae.csv")
dif_df2 = pd.read_csv("/content/drive/MyDrive/master_thesis/period_diff_normmae2.csv")

########################################
#ARIMA models with CPUE and effort:
mae_period_1=np.array(dif_df['2005_2010_MAE'].dropna())
mae_period_2=np.array(dif_df['2013_2019_MAE'].dropna())

print(f"MAE:")
#test for normality:
print(f"Error Normality test:")
shapiro_test = stats.shapiro(mae_period_1)
print(f"MAE period 1: Statistic={shapiro_test[0]}, p-value={shapiro_test[1]}")
shapiro_test = stats.shapiro(mae_period_2)
print(f"MAE period 2: Statistic={shapiro_test[0]}, p-value={shapiro_test[1]}")
#conclusion: errors are normally distributed

#test normality of differences:
difference_mae = mae_period_1 - mae_period_2
print(f"Error Differences Normality test:")
shapiro_test = stats.shapiro(difference_mae)
print(f"MAE difference: Statistic={shapiro_test[0]}, p-value={shapiro_test[1]}")
#conclusion: errors are normally distributed

#t-test:
t_stat_mae, p_value_mae = stats.ttest_rel(mae_period_1, mae_period_2)
print("T-test MAE: statistic =", t_stat_mae, "p-value =", p_value_mae)
#conclusion: statistically significantly different at .01 level


#wilcoxon test: only run if errors are not normally distributed
wilcoxon_statistic_mae, wilcoxon_p_value_mae = stats.wilcoxon(mae_period_1, mae_period_2)
#print("Wilcoxon MSE: statistic =", wilcoxon_statistic_mae, "p-value =", wilcoxon_p_value_mae)

########################################
#All other models:

mae_period_1=np.array(dif_df2[dif_df2['model']=='ARIMA']['2005_2010_MAE'].dropna())
mae_period_2=np.array(dif_df2[dif_df2['model']=='ARIMA']['2013_2019_MAE'].dropna())

print(f"MAE:")
#test for normality:
print(f"Error Normality test:")
shapiro_test = stats.shapiro(mae_period_1)
print(f"MAE period 1: Statistic={shapiro_test[0]}, p-value={shapiro_test[1]}")
shapiro_test = stats.shapiro(mae_period_2)
print(f"MAE period 2: Statistic={shapiro_test[0]}, p-value={shapiro_test[1]}")
#conclusion: errors are normally distributed

#test normality of differences:
difference_mae = mae_period_1 - mae_period_2
print(f"Error Differences Normality test:")
shapiro_test = stats.shapiro(difference_mae)
print(f"MAE difference: Statistic={shapiro_test[0]}, p-value={shapiro_test[1]}")
#conclusion: errors are normally distributed

#t-test:
t_stat_mae, p_value_mae = stats.ttest_rel(mae_period_1, mae_period_2)
print("T-test MAE: statistic =", t_stat_mae, "p-value =", p_value_mae)
#conclusion: Not statistically significantly different


#wilcoxon test: only run if errors are not normally distributed
wilcoxon_statistic_mae, wilcoxon_p_value_mae = stats.wilcoxon(mae_period_1, mae_period_2)
#print("Wilcoxon MSE: statistic =", wilcoxon_statistic_mae, "p-value =", wilcoxon_p_value_mae)
