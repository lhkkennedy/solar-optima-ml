import base64
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def create_image(width: int = 256, height: int = 256) -> bytes:
    # Simple synthetic RGB image
    image = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_e2e_infer_pitch_quote_flow():
    # 1) /infer
    image_bytes = create_image()
    response = client.post(
        "/infer",
        files={"file": ("image.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    infer_data = response.json()
    assert "mask" in infer_data
    assert "confidence" in infer_data
    mask_b64 = infer_data["mask"]
    original_size = infer_data["original_size"]

    # 2) /pitch
    pitch_req = {
        "coordinates": {"latitude": 51.5074, "longitude": -0.1278},
        "segmentation_mask": mask_b64,
        "image_size": original_size,
    }
    pitch_response = client.post("/pitch", json=pitch_req)
    assert pitch_response.status_code == 200
    pitch_data = pitch_response.json()
    assert "pitch_degrees" in pitch_data
    assert "area_m2" in pitch_data

    # 3) /quote
    quote_req = {
        "property_details": {
            "address": "123 Solar Street, London",
            "postcode": "SW1A 1AA",
            "property_type": "semi_detached",
            "occupancy": "family_of_4",
        },
        "segmentation_result": {
            "mask": mask_b64,
            "confidence": float(infer_data["confidence"]),
        },
        "pitch_result": {
            "pitch_degrees": pitch_data["pitch_degrees"],
            "area_m2": pitch_data["area_m2"],
            "roof_type": pitch_data["roof_type"],
            "orientation": pitch_data["orientation"],
        },
        "preferences": {"battery_storage": False, "premium_panels": False, "financing": "cash_purchase"},
    }
    quote_response = client.post("/quote", json=quote_req)
    assert quote_response.status_code == 200
    quote_data = quote_response.json()
    assert "quote_id" in quote_data
    assert "system_specification" in quote_data
    assert "yield_analysis" in quote_data
    assert "financial_analysis" in quote_data


def test_request_id_header_present():
    # Verify X-Request-Id is attached
    image_bytes = create_image()
    response = client.post(
        "/infer",
        files={"file": ("image.png", image_bytes, "image/png")},
        headers={"X-Request-Id": "TEST-REQ-ID-123"},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Request-Id") == "TEST-REQ-ID-123"