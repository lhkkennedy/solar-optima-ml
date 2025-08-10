import time
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from app.main import app

client = TestClient(app)


def _create_image():
    img = Image.new("RGB", (256, 256), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.runslow
def test_perf_infer_under_one_second():
    img = _create_image()
    start = time.time()
    resp = client.post("/infer", files={"file": ("image.png", img, "image/png")})
    elapsed = time.time() - start
    assert resp.status_code == 200
    assert elapsed < 1.0


@pytest.mark.runslow
def test_perf_pitch_under_two_seconds():
    img = _create_image()
    infer = client.post("/infer", files={"file": ("image.png", img, "image/png")}).json()
    req = {
        "coordinates": {"latitude": 51.5074, "longitude": -0.1278},
        "segmentation_mask": infer["mask"],
        "image_size": infer["original_size"],
    }
    start = time.time()
    resp = client.post("/pitch", json=req)
    elapsed = time.time() - start
    assert resp.status_code == 200
    assert elapsed < 2.0


@pytest.mark.runslow
def test_perf_quote_under_three_seconds():
    img = _create_image()
    infer = client.post("/infer", files={"file": ("image.png", img, "image/png")}).json()
    pitch_req = {
        "coordinates": {"latitude": 51.5074, "longitude": -0.1278},
        "segmentation_mask": infer["mask"],
        "image_size": infer["original_size"],
    }
    pitch = client.post("/pitch", json=pitch_req).json()

    quote_req = {
        "property_details": {
            "address": "123 Solar Street, London",
            "postcode": "SW1A 1AA",
            "property_type": "semi_detached",
            "occupancy": "family_of_4",
        },
        "segmentation_result": {"mask": infer["mask"], "confidence": float(infer["confidence"])},
        "pitch_result": {
            "pitch_degrees": pitch["pitch_degrees"],
            "area_m2": pitch["area_m2"],
            "roof_type": pitch["roof_type"],
            "orientation": pitch["orientation"],
        },
        "preferences": {"battery_storage": False, "premium_panels": False, "financing": "cash_purchase"},
    }
    start = time.time()
    resp = client.post("/quote", json=quote_req)
    elapsed = time.time() - start
    assert resp.status_code == 200
    assert elapsed < 3.0