import sqlite3
import pandas as pd

# path to local database
PATH_TO_DB = 'raw_data/countrypop(2).db'

# not sure if we still need this function
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


def advanced_get_data(country_lst):
    '''This function get's a list of countries and merges them into 1 dataframeS'''
    print(country_lst)
    conn = sqlite3.connect(PATH_TO_DB)
    frames = []
    #iterating over the list, and create a new list, with dfs as elements
    for country in country_lst:
        param = (country)
        query = (f"""
                    SELECT lon, lat, pop
                    FROM {param}

                    """)

        data = pd.read_sql_query(query.format(param), con=conn)
        frames.append(data)
    #merging the many dataframes into only 1 dataframe
    result = pd.concat(frames)
    return result
