import argparse
import base64
import io
import os
from typing import Optional

import requests
from PIL import Image
import numpy as np


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def fetch_infer(server: str, image_path_or_url: str) -> dict:
    url = server.rstrip("/") + "/infer"
    if is_url(image_path_or_url):
        # download to memory then send
        r = requests.get(image_path_or_url, timeout=20)
        r.raise_for_status()
        files = {"file": (os.path.basename(image_path_or_url) or "image.jpg", r.content, "image/jpeg")}
    else:
        with open(image_path_or_url, "rb") as f:
            files = {"file": (os.path.basename(image_path_or_url), f.read(), "image/png")}
    resp = requests.post(url, files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()


def overlay_mask_on_image(img: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.35) -> Image.Image:
    img = img.convert("RGB")
    h, w = mask.shape
    img = img.resize((w, h), Image.BILINEAR)
    overlay = Image.new("RGB", (w, h), (0, 0, 0))
    arr = np.array(overlay)
    r, g, b = color
    arr[mask > 0.5] = [r, g, b]
    overlay = Image.fromarray(arr, mode="RGB")
    return Image.blend(img, overlay, alpha)


def main():
    ap = argparse.ArgumentParser(description="Preview /infer mask overlay")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--image", required=True, help="Path or URL to input image")
    ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--color", default="255,0,0", help="Overlay RGB, e.g., 0,255,0")
    args = ap.parse_args()

    data = fetch_infer(args.server, args.image)
    mask_b64 = data.get("mask")
    if not mask_b64:
        raise RuntimeError("No mask in response")

    # Decode mask back to array
    mask_png = base64.b64decode(mask_b64)
    mask_img = Image.open(io.BytesIO(mask_png)).convert("L")
    mask_arr = (np.array(mask_img).astype(np.float32) / 255.0)

    # Load original image again for overlay
    if is_url(args.image):
        r = requests.get(args.image, timeout=20)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
    else:
        img = Image.open(args.image)

    color = tuple(int(c) for c in args.color.split(","))
    out = overlay_mask_on_image(img, mask_arr, color=color, alpha=args.alpha)
    out.save(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()


