from api.fast import predict

def testing_get_centers():
    # test if getting centers works
    # centers should have len 2 (lat, lon)
    prediction = predict(country="ABW", n_centers=1)
    assert len(prediction['centers'][0]) == 2
