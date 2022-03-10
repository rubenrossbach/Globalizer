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
    kmeans = MiniBatchKMeans(n_clusters = n_centers,
                                        max_iter = 300,
                                        batch_size=256*4,
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

# clustering based on radius + % of population
# "I want to reach __% of the population with a max distance of __km"
def radius_clustering(country, radius=100, percent=80):

    # get df from queries.py
    df= advanced_get_data(country)

    total_pop = df['pop'].sum()
    for k in range(1, 201):
        kmeans = MiniBatchKMeans(n_clusters = k,
                                max_iter = 300,
                                batch_size=256*4,
                                n_init = 10)
        kmeans.fit(X=df[['x', 'y', 'z']], sample_weight=df['pop'])
        # calculate distance to nearest center
        distance = np.min(cdist(df[['x', 'y', 'z']],
                                kmeans.cluster_centers_,
                                'euclidean'),
                        axis=1)
        # sum population within radius
        pop_in_radius = df[distance < radius]['pop'].sum()
        # stop when __% of pop within __km
        if pop_in_radius / total_pop > percent/100:
            break

    # remove unnecessary centers
    centers_3d = kmeans.cluster_centers_
    row = 0
    while row < len(centers_3d):
        # delete center by center and check if condition still met
        try:
            centers_reduced = np.delete(arr = centers_3d, obj=row, axis=0)
            # calculate distance to nearest center
            distance_tmp = np.min(cdist(df[['x', 'y', 'z']],
                                    centers_reduced,
                                    'euclidean'),
                            axis=1)
        except:
            break
        # sum population within radius
        pop_in_radius = df[distance_tmp < radius]['pop'].sum()
        # stop when __% of pop within __km
        if pop_in_radius / total_pop >= percent/100:
            centers_3d = centers_reduced
            distance = distance_tmp
        # increase index only when no deletion
        else:
            row += 1
    # calculate average distance
    avg_distance = (sum(np.min(cdist(df[['x', 'y', 'z']],
                                        centers_3d,
                                        'euclidean'),
                                    axis = 1)*df['pop'])) / df['pop'].sum()

    # transform 3D to 2D coordinates
    R = 6371
    centers = pd.DataFrame({
        'lat': np.arcsin(centers_3d[:,2] / R) * 180 / np.pi,
        'lon': np.arctan2(centers_3d[:,1], centers_3d[:,0]) * 180 / np.pi
    })
    centers = np.array(centers).tolist()

    pop_in_radius = df[distance < radius]['pop'].sum()
    total_pop = df['pop'].sum()

    return {
        "avg_distance": avg_distance,
        "Population in Radius": pop_in_radius,
        "Percent in Radius": pop_in_radius / total_pop,
        "centers": centers
    }


## Check ##
# to check your code, run:
# python -m Globalizer.predict
if __name__=="__main__":
    #print(k_clustering(country = ["ABW"], n_centers = 5))
    print(radius_clustering(country = ["ALB"], radius=40, percent=20))
