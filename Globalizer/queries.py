import sqlite3
import pandas as pd

# path to local database
PATH_TO_DB = 'raw_data/countrypop(2).db'

# connect to database
def get_data(country):
    '''This function get's the Countrydata from a local Database
    and converts it to a Dataframe
    WARNING: This Param substitution is, vulnerable to Injections.'''
    conn = sqlite3.connect(PATH_TO_DB)

    param = (country)
    query = (f"""
                SELECT lon, lat, pop
                FROM {param}

                """)

    data = pd.read_sql_query(query.format(param), con=conn)

    return data
