import urllib, json, os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
drive.mount('/content/drive')

#NOAA SST data pull. Loop will pull data from NOAA's server, calculate yearly average sst, and save data as csv file on local google drive to be then loaded later
#Connect to NOAA's server and pull data in one year increments (faster this way)
for year in range(1982, 2022):
    start = year
    end = year
    name = f'df_{year}'

    # url construction:
    url = f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv?sst%5B({start}-01-01T12:00:00Z):1:({end}-12-31T12:00:00Z)%5D%5B(0.0):1:(0.0)%5D%5B(-50):1:(50)%5D%5B(-50):1:(50)%5D"

    # load data to dataframe:
    df = pd.read_csv(url, index_col='time', parse_dates=True, skiprows=[1]) #first row contains measurements units

    #calculate yearly average sea surface temp by lat and lon:
    df.index = pd.to_datetime(df.index, utc=True) #create year column to average over
    df['year'] = df.index.year
    df = df.groupby(['year', 'latitude', 'longitude'])['sst'].mean().reset_index()

    # save dataframe as csv on google drive:
    df.to_csv(f'/content/drive/MyDrive/master_thesis/SST/sst_{start}_{end}.csv')

#NOAA chlorophyll data pull (years 2017-2022):
for year in range (2017, 2022):
  start = end = year
  name = f'df_{year}'

  url = f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chlamday.csv?chlorophyll%5B({start}-01-01T12:00:00Z):1:({end}-12-31T12:00:00Z)%5D%5B(-50):1:(50)%5D%5B(-50):1:(50)%5D"

  df = pd.read_csv(url,index_col='time', parse_dates=True, skiprows=[1])

  df.index = pd.to_datetime(df.index, utc=True)
  df['year'] = df.index.year
  df = df.groupby(['year','latitude','longitude'])['chlorophyll'].mean().reset_index()

  df.to_csv(f'/content/drive/MyDrive/master_thesis/chl/chl_{start}_{end}.csv')

#NOAA chlorophyll data pull (years 1997-2022):
url = f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdSW1chlamday.csv?chlorophyll%5B(1997-09-16T00:00:00Z):1:(2003-09-16T00:00:00Z)%5D%5B(-50):1:(50)%5D%5B(-50):1:(50)%5D"

df = pd.read_csv(url,index_col='time', parse_dates=True, skiprows=[1])
df.index = pd.to_datetime(df.index, utc=True)
df['year'] = df.index.year
df = df.groupby(['year','latitude','longitude'])['chlorophyll'].mean().reset_index()

df.to_csv(f'/content/drive/MyDrive/master_thesis/chl/chl_1997_2002.csv')


#read in all NOAA sea surface temperature csv files:
dir = '/content/drive/MyDrive/master_thesis/SST' #file directory path
csv_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.csv')] #list of all files in path
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True) #combine all files into one dataframe

#read in all NOAA chlorophyll csv files:

dir = '/content/drive/MyDrive/master_thesis/chl' #file directory path
csv_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.csv')] #list of all files in path
df_chl = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True) #combine all files into one dataframe

#load EEZ shapefiles and filter on study area:
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

#prepare sst dataset to be mapped with polygon shapes:
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])] # convert the latitude and longitude columns to a point object:
df_geo = gpd.GeoDataFrame(df, crs=EZZ_shapefiles.crs, geometry=geometry) # convert to a geodataframe

#filter sst dataframe on latitudes and longitudes that map into the EZZs of our study area:
results = pd.DataFrame(columns=['year', 'latitude', 'longitude', 'sst', 'GEONAME']) # create empty dataframe to store results

#loop through each EZZ
for geoname in EZZ_shapefiles['GEONAME'].unique():

    ezz = EZZ_shapefiles[EZZ_shapefiles['GEONAME'] == geoname] #filter data on one EZZ at a time

    df_geo_subset = df_geo.loc[df_geo.within(ezz.geometry.iloc[0])].copy() #subset sst data on lat and lon within current EZZ polygon

    df_geo_subset['GEONAME'] = geoname #assign GEONAME column the name of the current EZZ

    results = pd.concat([results, df_geo_subset[['year', 'latitude', 'longitude', 'sst', 'GEONAME']]]) #append data to final results table

#save data to google drive:
results.to_csv(f'/content/drive/MyDrive/master_thesis/SST/sst_filtered.csv')

#prepare chl dataset to be mapped with polygon shapes:
geometry = [Point(xy) for xy in zip(df_chl['longitude'], df_chl['latitude'])] # convert the latitude and longitude columns to a point object
df_chl_geo = gpd.GeoDataFrame(df_chl, crs=EZZ_shapefiles.crs, geometry=geometry) # convert to a geodataframe

#filter chl dataframe on latitudes and longitudes that map into the EZZs of our study area:
results_chl = pd.DataFrame(columns=['year', 'latitude', 'longitude', 'chlorophyll', 'GEONAME']) # Create empty dataframe to store results

# Loop through each year
for year in range(2003,2023):
    # Filter chl dataframe on the current year
    df_chl_year = df_chl_geo[df_chl_geo['year'] == year]

    # Loop through each EZZ
    for geoname in EZZ_shapefiles['GEONAME'].unique():
        ezz = EZZ_shapefiles[EZZ_shapefiles['GEONAME'] == geoname] # Filter data on one EZZ at a time

        df_chl_geo_subset = df_chl_year.loc[df_chl_year.within(ezz.geometry.iloc[0])].copy() # Subset chl data on lat and lon within current EZZ polygon
        df_chl_geo_subset['GEONAME'] = geoname # Assign GEONAME column the name of the current EZZ

        results_chl = pd.concat([results_chl, df_chl_geo_subset[['year', 'latitude', 'longitude', 'chlorophyll', 'GEONAME']]]) # Append data to final results table

    # Save data to CSV for the current year
    results_chl.to_csv(f'/content/drive/MyDrive/master_thesis/chl/chl_filtered_{year}.csv', index=False)
    results_chl = pd.DataFrame(columns=['year', 'latitude', 'longitude', 'chlorophyll', 'GEONAME']) # Reset results dataframe for the next year
