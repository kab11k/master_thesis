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
drive.mount('/content/drive')

##################################################################################################################
#Import data:

#import fishing effort data:
dir = '/content/drive/MyDrive/master_thesis/CPUE/effort' #file directory path
csv_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.csv')] #list of all files in path
df_effort = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True) #combine all files into one dataframe

#import domestic fleets' landings data:
dir = '/content/drive/MyDrive/master_thesis/CPUE/landings' #file directory path
csv_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.csv')] #list of all files in path
df_domestic_fleet_landings = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True) #combine all files into one dataframe

#import landings within EZZs:
dir = '/content/drive/MyDrive/master_thesis/landings_data' #file directory path
csv_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.csv')] #list of all files in path
df_EZZ_landings = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True) #combine all files into one dataframe

#import sea surface temperature data:
df_sst = pd.read_csv("/content/drive/MyDrive/master_thesis/SST/sst_filtered.csv")

#import chlorophyll data:
dir = '/content/drive/MyDrive/master_thesis/chl/filtered' #file directory path
csv_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.csv')] #list of all files in path
df_chl = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True) #combine all files into one dataframe

#import World Bank government indicators:
df_gov_ind = pd.read_csv("/content/drive/MyDrive/master_thesis/government_indicators/world_bank_government_indicators_estimate.csv")

##################################################################################################################
#Reset country naming conventions to match throughout all datasets:

#landings:
new_names = {'Congo (ex-Zaire)': 'Congo', 'Congo, R. of': 'Democratic Republic of the Congo',
             'Morocco (Central)': 'Morocco', 'Morocco (South)': 'Morocco',
             'Sao Tome & Principe': 'Sao Tome and Principe', 'South Africa (Atlantic and Cape)': 'South Africa',
             "Côte d'Ivoire": 'Ivory Coast'}

df_effort['fishing_entity'] = df_effort['fishing_entity'].replace(new_names)
df_domestic_fleet_landings['fishing_entity'] = df_domestic_fleet_landings['fishing_entity'].replace(new_names)
df_domestic_fleet_landings['area_name'] = df_domestic_fleet_landings['area_name'].replace(new_names)
df_EZZ_landings['fishing_entity'] = df_EZZ_landings['fishing_entity'].replace(new_names)
df_EZZ_landings['area_name'] = df_EZZ_landings['area_name'].replace(new_names)

#sst:
new_names = {'Angolan Exclusive Economic Zone': 'Angola', 'Beninese Exclusive Economic Zone': 'Benin',
                 'Cameroonian Exclusive Economic Zone': 'Cameroon', 'Cape Verdean Exclusive Economic Zone': 'Cape Verde',
                 'Congolese Exclusive Economic Zone': 'Congo', 'Democratic Republic of the Congo Exclusive Economic Zone': 'Democratic Republic of the Congo',
                 'Equatorial Guinean Exclusive Economic Zone': 'Equatorial Guinea', 'Gabonese Exclusive Economic Zone': 'Gabon',
                 'Gambian Exclusive Economic Zone': 'Gambia', 'Ghanaian Exclusive Economic Zone': 'Ghana',
                 'Guinea Bissau Exclusive Economic Zone': 'Guinea-Bissau', 'Guinean Exclusive Economic Zone': 'Guinea',
                 'Ivory Coast Exclusive Economic Zone': 'Ivory Coast', 'Joint regime area Nigeria / Sao Tome and Principe': 'Sao Tome and Principe',
                 'Joint regime area Senegal / Guinea Bissau': 'Guinea-Bissau', 'Liberian Exclusive Economic Zone': 'Liberia',
                 'Mauritanian Exclusive Economic Zone': 'Mauritania', 'Moroccan Exclusive Economic Zone': 'Morocco',
                 'Namibian Exclusive Economic Zone': 'Namibia', 'Nigerian Exclusive Economic Zone': 'Nigeria',
                 'Overlapping claim Western Saharan Exclusive Economic Zone': 'Western Sahara', 'Sao Tome and Principe Exclusive Economic Zone': 'Sao Tome and Principe',
                 'Senegalese Exclusive Economic Zone': 'Senegal', 'Sierra Leonian Exclusive Economic Zone': 'Sierra Leone',
                 'South African Exclusive Economic Zone': 'South Africa', 'Togolese Exclusive Economic Zone': 'Togo'}

df_sst['GEONAME'] = df_sst['GEONAME'].replace(new_names)
df_sst = df_sst.rename(columns={'GEONAME': 'area_name'})

#chl:
df_chl['GEONAME'] = df_chl['GEONAME'].replace(new_names)
df_chl = df_chl.rename(columns={'GEONAME': 'area_name'})

#World bank:
new_names = {'Cabo Verde': 'Cape Verde', 'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
             'Congo, Rep.': 'Congo', "Cote d'Ivoire": 'Ivory Coast',
             'Gambia, The': 'Gambia'}

df_gov_ind["Country Name"] = df_gov_ind["Country Name"].replace(new_names)
df_gov_ind = df_gov_ind.rename(columns={"Country Name": 'area_name'})

##################################################################################################################
#Regroup EZZ landings

#create column to indicate fishing in an EZZ by a foreign fleet:
df_EZZ_landings['fleet_origin'] = np.where(df_EZZ_landings['area_name'] != df_EZZ_landings['fishing_entity'], 'foreign', 'domestic')

#group EZZ landings by fleet origin and reporting status:
df_EZZ_landings_pivoted = pd.pivot_table(df_EZZ_landings, values='tonnes', index=['year', 'area_name'], columns=['fleet_origin', 'reporting_status'], aggfunc='sum')
df_EZZ_landings_pivoted.columns = [f'{col[0]}_{col[1]}_landings' for col in df_EZZ_landings_pivoted.columns]#flatten the column headers
df_EZZ_landings_pivoted = df_EZZ_landings_pivoted.reset_index()

df_EZZ_landings_pivoted['foreign_landings'] = df_EZZ_landings_pivoted['foreign_Reported_landings'] + df_EZZ_landings_pivoted['foreign_Unreported_landings'] #create combined field for foreign total

#group domestic fleet landings and effort by year:
df_domestic_fleet_landings_agg = df_domestic_fleet_landings.groupby(['year', 'fishing_entity'])['tonnes'].sum().reset_index()
df_effort_agg = df_effort.groupby(['year', 'fishing_entity']).agg({'effort': 'sum', 'number_boats': 'sum'}).reset_index()

##################################################################################################################
#Join domestic landings and effort dataframes:
df_CPUE = pd.merge(df_domestic_fleet_landings_agg, df_effort_agg, on=['fishing_entity', 'year'], how='left')

#calculate CPUE
df_CPUE["CPUE"] = (df_CPUE["tonnes"]*1000) / df_CPUE["effort"] #need to convert tonnes to kg (multiply by 1000)

#Join domestic dataframe with EZZ landings dataframe:
df_CPUE = df_CPUE.rename(columns={'fishing_entity':'area_name', #to align naming with other dataset for merge
                                  'tonnes':'domestic_fleet_landings'})
df_merged = pd.merge(df_CPUE, df_EZZ_landings_pivoted[['year','area_name','foreign_landings','foreign_Reported_landings','foreign_Unreported_landings']], on=['area_name', 'year'], how='left')

##################################################################################################################
#Regroup sst, chlorophyll, and governance indicator datasets

#average sst over each EZZ per year:
df_sst_max = df_sst.groupby(['area_name', 'year', 'latitude', 'longitude'])['sst'].max().reset_index() #for data visualization
df_sst = df_sst.groupby(['area_name', 'year'])['sst'].mean().reset_index()
#average chl over each EZZ per year:
df_chl = df_chl.groupby(['area_name', 'year'])['chlorophyll'].mean().reset_index()

#average over the governance indicator rankings to create one average indicator and avoid multicolinearity:
df_gov_ind['avg_governance_score'] = df_gov_ind[['Control of Corruption', 'Government Effectiveness', 'Political Stability and Absence of Violence/Terrorism', 'Regulatory Quality', 'Rule of Law', 'Voice and Accountability']].mean(axis=1)
df_gov_ind.drop(['Control of Corruption', 'Government Effectiveness', 'Political Stability and Absence of Violence/Terrorism', 'Regulatory Quality', 'Rule of Law', 'Voice and Accountability'], axis=1, inplace=True)

##################################################################################################################
#Join landings, sst, chl, and world bank indicators dataframes:
df_merged = pd.merge(df_merged, df_sst[['area_name', 'year', 'sst']], on=['area_name', 'year'], how='left')
df_merged = pd.merge(df_merged, df_chl[['area_name', 'year', 'chlorophyll']], on=['area_name', 'year'], how='left')
df_merged = pd.merge(df_merged, df_gov_ind, on=['area_name', 'year'], how='left')

##################################################################################################################
#Handle missing years:

#interpolate governance indicators for years 1997 and 1999 (rankings were not conducted in these 2 years):
df_merged.loc[df_merged['year'] >= 1996, 'avg_governance_score'] = df_merged.loc[df_merged['year'] >= 1996].groupby('area_name', group_keys=False)['avg_governance_score'].apply(
    lambda x: x.interpolate(limit_direction='both'))

##################################################################################################################
df_merged.to_csv(f'/content/drive/MyDrive/master_thesis/df_clean.csv', index=False) #save cleaned dataset for use in prediction models
