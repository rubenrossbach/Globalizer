# -*- coding: utf-8 -*-
"""Downloading_country_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UVtxOZwmMwbrOM7XE4oAbg393vxX3ajy

Installations and imports
"""
import rioxarray as rxr
from rasterio.plot import show
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import sqlite3
from sqlalchemy import create_engine
import sqlite3
from sqlite3 import Error
import os


isourl = "https://www.worldpop.org/resources/docs/national_boundaries/global-input-population-data-summary.xlsx"
sheetno = 1

def get_isos(url, sheetno):
  isos = pd.read_excel(url, sheetno)
  isos.columns = isos.iloc[0]
  isos = isos[1:]
  isos = [iso for iso in isos.ISOAlpha]
  isos = isos #change this in order to download more or less data
  return isos

"""Downloading data"""

def download_data(isos):
  # base url for "individual countries 2000-2020 UN-adjusted 1 km resolution"
  base_url = "https://www.worldpop.org/rest/data/pop/wpicuadj1km"

  #loop through isos and perform API call for each of them
  country_data = []
  preprocessed_isos = []
  iso_not_found = []
  for iso in isos:
    if iso != "USA" and iso != "RUS":
        print("Downloading", iso)
        params = {"iso3": iso}
        response = requests.get(url=base_url, params=params).json()
        try:
          path = response["data"][-1]["files"]
          for element in path:
           if element.endswith("tif"):
              r = requests.get(element)
              filename = element.split('/')[-1]
              with open(filename,'wb') as output_file:
                output_file.write(r.content)
          data = rxr.open_rasterio(filename, masked=True)
          country_data.append(data)
          preprocessed_isos.append(iso)
        except:
          iso_not_found.append(iso)
          continue

  print("These iso3 codes could not be found:\n", iso_not_found)
  return country_data, preprocessed_isos

"""Preprocessing data"""

def preprocess_data(downloaded_data, isos):
  country_dataframes = []
  R = 6371
  for iso, countrydata in zip(isos, range(0, len(downloaded_data)+1)):
    print("Preprocessing country", iso)
    country = {}
    # build dataframe from Array
    df = pd.DataFrame(
          np.array(downloaded_data[countrydata][0,:,:]),
          index=downloaded_data[countrydata].y,
          columns=downloaded_data[countrydata].x)
    # turn index into column
    df.index.name = 'lat'
    df.reset_index(inplace=True)
    #split df if size > 50_000_000

    # melt
    df = df.melt(id_vars="lat",
                            var_name="lon",
                            value_name="pop")
    # drop empty rows
    df = df.dropna().astype('float32')
    # drop rows below 1 pop
    df = df[df['pop'] > 1]
    #convert to 3D dataframe (x,y,z)
    df_3d = df.copy()
    df_3d.loc[:,'lat_rad'] = df_3d.lat * np.pi / 180
    df_3d.loc[:,'lon_rad'] = df_3d.lon * np.pi / 180
    df_3d['x'] = R * np.cos(df_3d['lat_rad']) * np.cos(df_3d['lon_rad'])
    df_3d['y'] = R * np.cos(df_3d['lat_rad']) * np.sin(df_3d['lon_rad'])
    df_3d['z'] = R * np.sin(df_3d['lat_rad'])
    df_3d = df_3d[['x', 'y', 'z', 'pop']]
    country[iso] = df_3d
    country_dataframes.append(country)
    for file in os.listdir():
      if iso.lower() in file[0:3]:
        os.remove(file)
  return country_dataframes

"""Storing data as SQL"""

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"created sqlite version {sqlite3.version} database")
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

def save_data_as_sql(processed_dataframes):
  engine = create_engine('sqlite:///countrypop.db', echo=False)
  # attach the data frame to the sql
  # with a name of the table as the iso
  for country in processed_dataframes:
    iso = list(country.keys())[0]
    data = list(country.values())[0]
    print(f"Saving data for {iso} as SQL")
    data.to_sql(iso, con=engine)
  print("Please check for this file:", engine)
  return engine

"""Executing the download"""

# Commented out IPython magic to ensure Python compatibility.
#isos = get_isos(isourl, sheetno)
datasets, isos = download_data(['CHN'])
processed_dataframes = preprocess_data(datasets, isos)
#
#Deleting the downloaded datasets in order to save memory
for element in dir():
    if element[0:2] != "__" and element == "datasets":
        del globals()[element]
#This creates a database once the file is executed
if __name__ == '__main__':
    create_connection(r"countrypop.db")

sql = save_data_as_sql(processed_dataframes)
# The database will appear on the left side in the temporary data drive.
# You can download it there (Colab) or it is being saved there (VS Code)
