#data sourcing
from Globalizer.queries import get_data

#preprocessing and modelling
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def predict(country, n_centers):
    """
    This function takes a country code (3 letters) and
    a number of centers n_centers.
    It returns a list of optimal coordinates.
    """
    # get df from queries.py
    df_country = get_data(country)


    # scaler on pop column
    scaler = MinMaxScaler()
    df_country.loc[:,'pop'] = scaler.fit_transform(df_country[['pop']])

    #fit the model
    kmeans = KMeans(n_clusters = n_centers,
                    max_iter = 300,
                    n_init = 10,
                    random_state = 0)
    kmeans.fit(X=df_country[['lat', 'lon']], sample_weight=df_country['pop'])


    # return center coordinates
    return kmeans.cluster_centers_


## Check ##
# to check your code, run:
# python -m Globalizer.predict
if __name__=="__main__":
    print(predict(country = "AFG", n_centers = 5))
