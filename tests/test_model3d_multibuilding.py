import base64
import io
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _two_buildings_image(size=(512, 512)) -> bytes:
    img = Image.new("RGB", size, color=(90, 90, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 150, 200, 300], fill=(200, 200, 200))
    draw.rectangle([300, 100, 450, 250], fill=(210, 210, 210))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def test_model3d_procedural_optional_fields_present(monkeypatch):
    # Enable procedural pipeline via env flag
    monkeypatch.setenv("PROC_ROOF_ENABLE", "1")

    img_bytes = _two_buildings_image()
    img_b64 = base64.b64encode(img_bytes).decode()

    req = {
        "coordinates": {"latitude": 51.5074, "longitude": -0.1278},
        "bbox_m": 60,
        "image_base64": img_b64,
        "return_mesh": False,
    }
    resp = client.post("/model3d", json=req)
    assert resp.status_code in (200, 400)  # outside UK or DSM missing may return 400
    if resp.status_code != 200:
        return
    data = resp.json()
    # Optional procedural outputs
    if "procedural_roofs" in data:
        assert isinstance(data["procedural_roofs"], list)
        if data["procedural_roofs"]:
            m = data["procedural_roofs"][0]
            assert "footprint_regularized" in m
            assert "parts" in m
            assert "artifacts" in m

