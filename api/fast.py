from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Globalizer.predict import k_clustering, smart_clustering
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"ok": True}

@app.get("/predict")
def predict(country, treshhold=50, n_centers=None):
    '''This function controls whether
    we use the Kmeans or the KmeansMinibatch. By default it's
    the Minibatch with 50km as threshold'''
    return smart_clustering(country, threshold=50) if n_centers == None else k_clustering(country, n_centers)


## Check ##
# to check your code, run:
# python -m api.fast
if __name__=="__main__":
    print(predict(country = ["ABW"], n_centers = None))
