import os
import argparse
import numpy as np
import cv2

from app.services.procedural_roof.pbsr import PBSRService, Rect


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


def _rand_affine(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random 90° rotations, flips, small translations, and mild morph ops."""
    out = img
    # rotate 0/90/180/270
    k = int(rng.integers(0, 4))
    out = np.rot90(out, k)
    # random horizontal/vertical flip
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    # small translate by up to 5%
    H, W = out.shape
    tx = int(rng.integers(-W//20, W//20))
    ty = int(rng.integers(-H//20, H//20))
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    # random erode/dilate
    if rng.random() < 0.5:
        ksz = int(rng.integers(1, 3))
        kernel = np.ones((ksz, ksz), np.uint8)
        if rng.random() < 0.5:
            out = cv2.erode(out, kernel, iterations=1)
        else:
            out = cv2.dilate(out, kernel, iterations=1)
    return out


def gen_pbsr_families(out_dir: str, num: int = 200000, h: int = 128, w: int = 128):
    """Generate labeled PBSR family masks: T11, T21, T32, T43 with strong variation."""
    os.makedirs(out_dir, exist_ok=True)
    pbsr = PBSRService(grid=8)
    rng = np.random.default_rng(42)
    families = ["T11", "T21", "T31", "T32", "T41", "T42", "T43", "T44"]
    for i in range(num):
        fam = families[int(rng.integers(0, len(families)))]
        # random bbox size and position (ensure variety of aspect and coverage)
        min_side = int(min(h, w) * 0.35)
        max_side = int(min(h, w) * 0.9)
        bw = int(rng.integers(min_side, max_side))
        bh = int(rng.integers(min_side, max_side))
        x = int(rng.integers(0, max(1, w - bw)))
        y = int(rng.integers(0, max(1, h - bh)))
        ww = x + bw
        hh = y + bh
        bbox = Rect(int(x), int(y), int(ww - x), int(hh - y))
        configs = pbsr._enumerate_topologies(bbox)
        rect_sets = configs.get(fam, [])
        if not rect_sets:
            rect_sets = configs["T11"]
        rects = rect_sets[int(rng.integers(0, len(rect_sets)))]
        # rasterize
        canvas = np.zeros((h, w), dtype=np.uint8)
        for r in rects:
            x0, y0 = r.x, r.y
            x1, y1 = x0 + r.w, y0 + r.h
            x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
            y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
            canvas[y0:y1, x0:x1] = 255
        # structured occlusions along borders to mimic imperfect masks
        if rng.random() < 0.7:
            for _ in range(int(rng.integers(1, 4))):
                ox = int(rng.integers(x, ww)); oy = int(rng.integers(y, hh))
                ow = int(rng.integers(bw//10, bw//3)); oh = int(rng.integers(bh//10, bh//3))
                canvas[max(0, oy):min(h, oy+oh), max(0, ox):min(w, ox+ow)] = 0
        # random affine and mild morph operations
        canvas = _rand_affine(canvas, rng)
        cv2.imwrite(os.path.join(out_dir, f"{fam}_{i:06d}.png"), canvas)


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
    ap.add_argument("--mode", choices=["footprints", "roofs", "pbsr"], default="footprints")
    args = ap.parse_args()
    if args.mode == "footprints":
        main(args.out, args.num)
    elif args.mode == "roofs":
        gen_synth_roof_edges(args.out, args.num)
    else:
        gen_pbsr_families(args.out, args.num)

