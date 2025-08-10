import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.irradiance_service import IrradianceService

client = TestClient(app)


def test_quote_handles_irradiance_failure(monkeypatch):
    # Force irradiance service to fail by returning None
    def _fail_get_irradiance(lat, lon):
        return None

    monkeypatch.setattr(IrradianceService, "get_irradiance_data", lambda self, a, b: None)

    request_data = {
        "property_details": {
            "address": "123 Solar Street, London",
            "postcode": "SW1A 1AA",
            "property_type": "semi_detached",
            "occupancy": "family_of_4"
        },
        "segmentation_result": {
            "mask": "iVBORw0KGgoAAAANSUhEUgAA...",
            "confidence": 0.95
        },
        "pitch_result": {
            "pitch_degrees": 25.5,
            "area_m2": 45.2,
            "roof_type": "gabled",
            "orientation": "south_facing"
        },
        "preferences": {
            "battery_storage": False,
            "premium_panels": False,
            "financing": "cash_purchase"
        }
    }

    response = client.post("/quote", json=request_data)
    # Current implementation returns 400 on ValueError from quote generation path
    assert response.status_code == 400
    assert "Unable to get irradiance data" in response.text


def test_quote_handles_invalid_inputs():
    # Missing fields to trigger 422
    bad_request = {
        "property_details": {"address": "", "postcode": ""},
        "segmentation_result": {"mask": "", "confidence": 0.1},
        "pitch_result": {"pitch_degrees": 100, "area_m2": 1, "roof_type": "x", "orientation": "y"},
        "preferences": {"battery_storage": False}
    }
    resp = client.post("/quote", json=bad_request)
    assert resp.status_code == 422