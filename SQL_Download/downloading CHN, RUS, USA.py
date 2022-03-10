import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sqlalchemy import create_engine
import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"created {db_file} sqlite version {sqlite3.version} database")
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

#Create SQL Connections
#create_connection(r"USA.db")
#create_connection(r"RUS.db")
create_connection(r"CHN.db")

#Load CSVs
#usadata = pd.read_csv('USA_processed')
#rusdata = pd.read_csv('RUS_processed')
chndata = pd.read_csv('CHN_processed')
chndata = chndata[6_000_001:] #Use slicing not to exceed RAM

#Save data as SQL
def save_data_as_sql(iso, data):
  engine = create_engine(f'sqlite:///{iso}.db', echo=False)
  print(f"Saving data for {iso} as SQL")
  data.to_sql(iso, con=engine)
  print("Please check for this file:", engine)
  return engine

#save_data_as_sql('USA', usadata)
#save_data_as_sql('RUS', rusdata)
save_data_as_sql('CHN', chndata)
