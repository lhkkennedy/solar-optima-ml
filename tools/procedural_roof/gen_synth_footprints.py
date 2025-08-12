import os
import argparse
import numpy as np
import cv2

from app.services.procedural_roof.pbsr import PBSRService


def gen_synth_roof_edges(out_dir: str, num: int = 20000, h: int = 128, w: int = 128):
    """
    Generate synthetic roof edge crops for roof family classifier training.
    Files named with family keywords: flat_####.png, gable_####.png, hip_####.png, pyramid_####.png
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(123)
    for i in range(num):
        canvas = np.zeros((h, w), dtype=np.uint8)
        fam = rng.integers(0, 4)
        name = ["flat", "gable", "hip", "pyramid"][fam]
        # draw simplified edges in white
        if name == "flat":
            # no internal ridges, maybe faint border noise
            pass
        elif name == "gable":
            # horizontal or vertical ridge
            if rng.random() < 0.5:
                y = rng.integers(h // 3, 2 * h // 3)
                cv2.line(canvas, (w // 8, y), (7 * w // 8, y), color=255, thickness=1)
            else:
                x = rng.integers(w // 3, 2 * w // 3)
                cv2.line(canvas, (x, h // 8), (x, 7 * h // 8), color=255, thickness=1)
        elif name == "hip":
            # an X-like meeting at center
            cv2.line(canvas, (w // 8, h // 8), (7 * w // 8, 7 * h // 8), color=255, thickness=1)
            cv2.line(canvas, (7 * w // 8, h // 8), (w // 8, 7 * h // 8), color=255, thickness=1)
        else:  # pyramid
            # cross plus small diagonals near corners
            cv2.line(canvas, (w // 2, h // 8), (w // 2, 7 * h // 8), color=255, thickness=1)
            cv2.line(canvas, (w // 8, h // 2), (7 * w // 8, h // 2), color=255, thickness=1)
        # add random noise strokes
        for _ in range(rng.integers(0, 6)):
            x0, y0 = rng.integers(0, w), rng.integers(0, h)
            x1, y1 = rng.integers(0, w), rng.integers(0, h)
            cv2.line(canvas, (x0, y0), (x1, y1), color=int(rng.integers(0, 2) * 255), thickness=1)
        cv2.imwrite(os.path.join(out_dir, f"{name}_{i:06d}.png"), canvas)


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
    ap.add_argument("--mode", choices=["footprints", "roofs"], default="footprints")
    args = ap.parse_args()
    if args.mode == "footprints":
        main(args.out, args.num)
    else:
        gen_synth_roof_edges(args.out, args.num)

