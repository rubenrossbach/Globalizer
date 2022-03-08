import psycopg2
import pandas as pd

import os
from dotenv import load_dotenv, find_dotenv

# point to .env file
env_path = find_dotenv() # automatic find

# load your api key as environment variables
load_dotenv(env_path)
password = os.getenv('PASSWORD_GOOGLE_SQL')


# path to cloud database
PATH_TO_DB = f"dbname=countrypop user=postgres password={password} host=34.159.125.98"


# not sure if we still need this function, I guess not
def get_data(country):
    '''This function get's the Countrydata from a local Database
    and converts it to a Dataframe
    WARNING: This Param substitution is, vulnerable to Injections.'''
    conn = psycopg2.connect(PATH_TO_DB)
    print(conn)
    param = (country)
    query = ("""
                SELECT x, y, z, pop
                FROM {}
                """)

    data = pd.read_sql_query(query.format(param), con=conn)
    return data


def advanced_get_data(country_lst):
    '''This function get's a list of countries and merges them into 1 dataframes'''
    conn = psycopg2.connect(PATH_TO_DB)
    print(conn)
    frames = []
    #iterating over the list, and create a new list, with dfs as elements
    for country in country_lst:
        param = (country)
        query = ("""
                    SELECT x, y, z, pop
                    FROM {}

                    """)

        data = pd.read_sql_query(query.format(param), con=conn)
        frames.append(data)
    #merging the many dataframes into only 1 dataframe
    result = pd.concat(frames)
    return result
