#data sourcing
from Globalizer.queries import advanced_get_data, get_data

#preprocessing and modelling
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import MinMaxScaler

#data manipulation
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def k_clustering(country, n_centers=None):
    """This function takes a country code (3 letters) and
    a number of centers n_centers.
    It returns a list of optimal coordinates.
    """
    # get df from queries.py
    df_country = advanced_get_data(country)
    # scaling
    scaler = MinMaxScaler()
    df_country.loc[:,'pop'] = scaler.fit_transform(df_country[['pop']])
    kmeans = KMeans(n_clusters = n_centers,
                    max_iter = 300,
                    n_init = 10)
    kmeans.fit(X=df_country[['x', 'y', 'z']], sample_weight=df_country['pop'])
    # calculate avg distance
    avg_distance = (sum(np.min(cdist(df_country[['x', 'y', 'z']],
                                    kmeans.cluster_centers_,
                                    'euclidean'),
                            axis = 1)*df_country['pop'])) / df_country['pop'].sum()
    # transform 3D to 2D coordinates
    centers_3d = kmeans.cluster_centers_
    R = 6371
    centers = pd.DataFrame({
        'lat': np.arcsin(centers_3d[:,2] / R) * 180 / np.pi,
        'lon': np.arctan2(centers_3d[:,1], centers_3d[:,0]) * 180 / np.pi
    })
    centers = np.array(centers).tolist()
    return {
      "avg_distance": avg_distance,
      "centers": centers
    }

    # clustering until mean distance below threshold
def smart_clustering(country, threshold=50):

    # get df from queries.py
    df_country = advanced_get_data(country)

    # scaling
    scaler = MinMaxScaler()
    df_country.loc[:,'pop'] = scaler.fit_transform(df_country[['pop']])

    for k in range(1, 101):
        kmeans = MiniBatchKMeans(n_clusters = k,
                                    max_iter = 300,
                                    batch_size=256*4,
                                    n_init = 10)
        kmeans.fit(X=df_country[['x', 'y', 'z']], sample_weight=df_country['pop'])
        # stop when mean distance below threshold
        avg_distance = (sum(np.min(cdist(df_country[['x', 'y', 'z']],
                                    kmeans.cluster_centers_,
                                    'euclidean'),
                            axis = 1)*df_country['pop'])) / df_country['pop'].sum()
        if avg_distance < threshold:
            break

    # transform 3D to 2D coordinates
    centers_3d = kmeans.cluster_centers_
    R = 6371
    centers = pd.DataFrame({
        'lat': np.arcsin(centers_3d[:,2] / R) * 180 / np.pi,
        'lon': np.arctan2(centers_3d[:,1], centers_3d[:,0]) * 180 / np.pi
    })
    centers = np.array(centers).tolist()
    return {
      "avg_distance": avg_distance,
      "centers": centers
    }

## Check ##
# to check your code, run:
# python -m Globalizer.predict
if __name__=="__main__":
    print(k_clustering(country = ["ABW"], n_centers = 5))
