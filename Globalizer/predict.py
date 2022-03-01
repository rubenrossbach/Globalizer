import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# path to local database
PATH_TO_DB = '../raw_data/title-of-db.sqlite'

def predict(country, n_centers):
    """
    This function takes a country code (3 letters) and
    a number of centers n_centers.
    It returns a list of optimal coordinates.
    """
    # connect to database
    conn = sqlite3.connect(PATH_TO_DB)
    db = conn.cursor()

    # load data with pandas
    param = (country, )
    query = """
            SELECT lon, lat, pop
            FROM {}
            """
    data = pd.read_sql_query(query.format(country), con=conn)

    # do kmeans clustering with n_centers
    # data should contain the country data

    # return center coordinates
    return kmeans.cluster_centers_


## Check ##
# to check your code, run:
# python -m Globalizer.predict
if __name__=="__main__":
    print(predict(country = "ARG", n_centers = 5))
