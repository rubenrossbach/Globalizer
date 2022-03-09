#data sourcing
from Globalizer.queries import advanced_get_data, get_data

#preprocessing and modelling
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

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

def optimal_kmeans(country):
    counter = 0
    km_silhouette = []
    avg_distance_lst=[]
    n_centers_loc={}
    # get df from queries.py
    df_country = advanced_get_data(country)
    X_scaled = df_country[['x', 'y', 'z']]
    # scaling
    scaler = MinMaxScaler()
    df_country.loc[:,'pop'] = scaler.fit_transform(df_country[['pop']])

    #making an iteration with 2 - 15 centers
    for i in range(2,16):
        #instanciating the model and fitting the data
        km = KMeans(n_clusters=i, random_state=0).fit(X=X_scaled, sample_weight=df_country['pop'])
        #predict with same data?
        preds = km.predict(X_scaled)
        #computing average distance to display afterwards
        avg_distance = (sum(np.min(cdist(X_scaled,
                                    km.cluster_centers_,
                                    'euclidean'),
                            axis = 1)*df_country['pop'])) / df_country['pop'].sum()
        #appending to list
        avg_distance_lst.append(avg_distance)

        # transform 3D to 2D coordinates
        centers_3d = km.cluster_centers_
        R = 6371
        centers = pd.DataFrame({
            'lat': np.arcsin(centers_3d[:,2] / R) * 180 / np.pi,
            'lon': np.arctan2(centers_3d[:,1], centers_3d[:,0]) * 180 / np.pi
        })
        #having the 2D Centers as an nd.array
        centers = np.array(centers)
        #creating a new column with the centers as the value
        n_centers_loc[i] = centers

        #getting the silhouette score
        silhouette = silhouette_score(X_scaled,preds)
        km_silhouette.append(silhouette)
        #print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
        #print("-"*100)

        ###defining a breaking point when the score is not increasing for the next 3 iterations###
        #skip the first iteration
        if len(km_silhouette) >= 2:
            #if the new silhouette score is < than previous(-2)
            if silhouette < km_silhouette[-2]:
            # increase counter by 1
                counter += 1
            else:
                #reset counter
                counter = 0
            #if counter == 3, (similiar to patience principle)
            #break the foor loop and take the centers with respectivley max silhoutte score
            if counter == 3:
                break
    #storing the index of the value with max silhouette_score
    optimal_k = km_silhouette.index(max(km_silhouette))

####if you want to understand the return better maybe uncomment the print statements and run####

    #print(f"The Index of number of optimal Center : {optimal_k}")
    #print(f"The avg_distance: {avg_distance_lst[optimal_k]}")
    #print(f"The centers location: {n_centers_loc[optimal_k+2]}")

    return  {    "avg_distance": avg_distance_lst[optimal_k],
                 "centers": n_centers_loc[optimal_k+2]
            }
## Check ##
# to check your code, run:
# python -m Globalizer.predict
if __name__=="__main__":
    print(optimal_kmeans(country = ["ABW"]))
