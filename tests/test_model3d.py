import base64
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def create_test_image(w: int = 256, h: int = 256) -> str:
    # simple roof-like bright rectangle on dark background
    img = Image.new("RGB", (w, h), color=(20, 20, 20))
    arr = np.array(img)
    arr[h//4:3*h//4, w//4:3*w//4] = [200, 200, 200]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_model3d_basic_request():
    payload = {
        "coordinates": {"latitude": 51.5074, "longitude": -0.1278},
        "bbox_m": 60,
        "return_mesh": False,
    }
    r = client.post("/model3d", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "planes" in data and isinstance(data["planes"], list)
    assert "edges" in data and isinstance(data["edges"], list)
    assert "summary" in data and "area_m2" in data["summary"]
    assert "bbox" in data


def test_model3d_with_image_mask():
    img_b64 = create_test_image()
    payload = {
        "coordinates": {"latitude": 51.5074, "longitude": -0.1278},
        "bbox_m": 60,
        "image_base64": img_b64,
        "return_mesh": False,
    }
    r = client.post("/model3d", json=payload)
    assert r.status_code == 200
    data = r.json()
    # planes should still be present; mask path doesn't error
    assert len(data.get("planes", [])) >= 1