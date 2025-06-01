from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is up and running!"}
    
def test_predict_valid_ip():
    test_input = {
        "landmarks": [0.01 * i for i in range(63)]
    }
    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    assert "predicted_label" in response.json()
    
def test_predict_invalid_ip():
    test_input = {
        "landmarks": [0.01]*10
    }
    
    response = client.post("/predict", json=test_input)
    assert response.status_code == 422
    assert "Invalid number of features" in response.json()["detail"]