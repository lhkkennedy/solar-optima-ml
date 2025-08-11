import os
import argparse
import numpy as np
import cv2

from app.services.procedural_roof.pbsr import PBSRService


def main(out_dir: str, num: int = 10000, h: int = 128, w: int = 128):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(num):
        mask = np.zeros((h, w), dtype=np.uint8)
        # random bbox
        x0 = rng.integers(8, w // 4)
        y0 = rng.integers(8, h // 4)
        x1 = rng.integers(3 * w // 4, w - 8)
        y1 = rng.integers(3 * h // 4, h - 8)
        mask[y0:y1, x0:x1] = 1
        # add small noise along border
        if rng.random() < 0.5:
            rr = rng.integers(0, h, size=(50,))
            cc = rng.integers(0, w, size=(50,))
            mask[rr, cc] = 1
        # save
        cv2.imwrite(os.path.join(out_dir, f"mask_{i:06d}.png"), mask * 255)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=10000)
    args = ap.parse_args()
    main(args.out, args.num)

