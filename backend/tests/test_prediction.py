from fastapi.testclient import TestClient

from app.main import app

VALID_PAYLOAD = {
    "location": "bangalore",
    "carpet_area_sqft": 1200.0,
    "floor_num": 3,
    "bathroom": 2,
    "balcony": 1,
    "furnishing": "Semi-Furnished",
    "transaction": "Resale",
    "ownership": "Freehold",
    "facing": "East",
}


def _client() -> TestClient:
    # TestClient triggers FastAPI's lifespan (startup/shutdown) as a context manager,
    # so the model is loaded exactly like it would be in production.
    return TestClient(app)


def test_health_endpoint():
    with _client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_happy_path():
    with _client() as client:
        response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_price" in body
    assert isinstance(body["predicted_price"], float)
    assert body["predicted_price"] >= 0


def test_predict_invalid_input_returns_422():
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["carpet_area_sqft"] = -50  # invalid: must be > 0
    del bad_payload["floor_num"]  # also missing a required field

    with _client() as client:
        response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_unknown_location_falls_back_to_other():
    payload = dict(VALID_PAYLOAD)
    payload["location"] = "Nonexistent Neighborhood, Nowhere City"

    with _client() as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
