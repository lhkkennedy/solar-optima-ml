import base64
import io

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _make_roofy_image(w: int = 256, h: int = 256) -> bytes:
    img = Image.new("RGB", (w, h), color=(30, 30, 30))
    arr = np.array(img)
    # draw a bright roof rectangle
    arr[h // 4 : 3 * h // 4, w // 5 : 4 * w // 5] = [220, 220, 220]
    return Image.fromarray(arr).tobytes("raw", "RGB")


def test_infer_mask_not_all_zeros_or_ones():
    # Construct synthetic roof-like image
    w, h = 256, 256
    img = Image.new("RGB", (w, h), color=(30, 30, 30))
    arr = np.array(img)
    arr[h // 4 : 3 * h // 4, w // 5 : 4 * w // 5] = [220, 220, 220]
    image = Image.fromarray(arr)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()

    files = {"file": ("roof.png", data, "image/png")}
    r = client.post("/infer", files=files)
    assert r.status_code == 200
    out = r.json()
    # Decode mask
    mask_b64 = out.get("mask")
    assert isinstance(mask_b64, str)
    mask_png = base64.b64decode(mask_b64)
    mimg = Image.open(io.BytesIO(mask_png)).convert("L")
    marr = np.array(mimg)
    # Assert mask is not trivially all zeros or all ones
    assert np.any(marr > 0), "mask was entirely zero"
    assert np.any(marr < 255), "mask was entirely 255"


