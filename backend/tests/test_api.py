from fastapi.testclient import TestClient
from app.main import app

def test_health():
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()

def test_missing_files():
    assert TestClient(app).post("/api/verify").status_code == 400
