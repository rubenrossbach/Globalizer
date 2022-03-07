from api.fast import index

def testing_index():
    # test if API gives response
    response = index()
    assert response['ok'] == True
