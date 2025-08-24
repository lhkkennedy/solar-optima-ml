import os
import argparse
import numpy as np
import cv2

import os, sys
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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


def _rand_affine(img: np.ndarray, rng: np.random.Generator, max_deg: int = 0, translate_frac: float = 0.02,
                 morph_prob: float = 0.0) -> np.ndarray:
    """Apply random flips, small translations, and mild morph ops.
    Works for 2D (grayscale) and 3D (color) arrays. Morph ops apply only to 2D.
    """
    out = img
    # random horizontal/vertical flip only (keep axis-aligned)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    # very small translate by a couple of pixels
    H, W = out.shape[:2]
    tx = int(rng.integers(int(-W*translate_frac), int(W*translate_frac)))
    ty = int(rng.integers(int(-H*translate_frac), int(H*translate_frac)))
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    border_value = 0 if out.ndim == 2 else (0, 0, 0)
    out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=border_value)
    # very mild morph (optional)
    if out.ndim == 2 and rng.random() < morph_prob:
        ksz = int(rng.integers(1, 3))
        kernel = np.ones((ksz, ksz), np.uint8)
        if rng.random() < 0.5:
            out = cv2.erode(out, kernel, iterations=1)
        else:
            out = cv2.dilate(out, kernel, iterations=1)
    return out


def _rect(x: int, y: int, w: int, h: int) -> Rect:
    return Rect(int(x), int(y), int(max(1, w)), int(max(1, h)))


def _as_rect(r: object) -> Rect:
    """Coerce various rect-like inputs into a Rect.
    Supported forms:
      - Rect
      - (x, y, w, h)
      - (Rect, color) or ((x, y, w, h), color) where color is ignored here
    """
    if isinstance(r, Rect):
        return r
    if isinstance(r, (tuple, list)):
        # Pattern: (Rect, color)
        if len(r) == 2 and isinstance(r[0], Rect):
            return r[0]
        # Pattern: ((x,y,w,h), color)
        if len(r) == 2 and isinstance(r[0], (tuple, list)) and len(r[0]) == 4:
            return _rect(r[0][0], r[0][1], r[0][2], r[0][3])
        # Pattern: (x,y,w,h)
        if len(r) == 4:
            return _rect(r[0], r[1], r[2], r[3])
    raise TypeError(f"Expected Rect or (x,y,w,h) or (Rect,color), got {type(r)}: {r}")


def _build_family_rects(fam: str, x: int, y: int, bw: int, bh: int, rng: np.random.Generator) -> list[Rect]:
    """Deterministic, crisp templates per family with slight ratio jitter.
    Produces axis-aligned parts that resemble canonical I/L/T/U/Z shapes.
    """
    rects: list[Rect] = []
    # Clamp minimal part size to avoid thin slivers and favor realistic thick strokes
    min_part = max(6, min(bw, bh) // 5)
    # thickness ~ 28–38% of the short side for bolder footprints
    thk = int(min(bw, bh) * float(rng.uniform(0.28, 0.38)))
    thk = max(min_part, thk)
    if fam == "T11":
        rects = [_rect(x, y, bw, bh)]
    elif fam == "T21":
        # L shape: one vertical leg + one horizontal leg
        t = thk
        # choose corner (0: TL, 1: TR, 2: BL, 3: BR)
        corner = int(rng.integers(0, 4))
        if corner == 0:  # top-left L
            rects = [
                _rect(x, y, t, bh),
                _rect(x, y + bh - t, bw, t)
            ]
        elif corner == 1:  # top-right L
            rects = [
                _rect(x + bw - t, y, t, bh),
                _rect(x, y + bh - t, bw, t)
            ]
        elif corner == 2:  # bottom-left L
            rects = [
                _rect(x, y, bw, t),
                _rect(x, y, t, bh)
            ]
        else:  # bottom-right L
            rects = [
                _rect(x, y, bw, t),
                _rect(x + bw - t, y, t, bh)
            ]
    elif fam == "T31":
        # T shape: top bar + bottom stem (bold strokes)
        bar_h = thk
        stem_w = thk
        rects = [
            _rect(x, y, bw, bar_h),
            _rect(x + (bw - stem_w)//2, y + bar_h, stem_w, bh - bar_h),
            _rect(x, y, 0, 0)  # placeholder removed below
        ]
        rects = rects[:2]
    elif fam == "T32":
        # U shape: top bar + two legs
        bar_h = thk
        leg_w = thk
        leg_h = bh - bar_h
        rects = [
            _rect(x, y, bw, bar_h),
            _rect(x, y + bar_h, leg_w, leg_h),
            _rect(x + bw - leg_w, y + bar_h, leg_w, leg_h)
        ]
    elif fam == "T41":
        # Z shape composed of three bold rectangles: two offset horizontal bars and one vertical connector.
        bar_h = thk
        # bar length as a fraction of width to emphasize Z limbs
        L = max(thk * 2, int(bw * float(rng.uniform(0.55, 0.8))))
        if rng.random() < 0.5:
            # orientation: top-left to bottom-right
            top_x = x
            bot_x = x + bw - L
            connector_x = x + L - thk // 2
            rects = [
                _rect(top_x, y, L, bar_h),
                _rect(bot_x, y + bh - bar_h, L, bar_h),
                _rect(connector_x, y + bar_h, thk, bh - 2 * bar_h),
            ]
        else:
            # orientation: top-right to bottom-left
            top_x = x + bw - L
            bot_x = x
            connector_x = x + bw - L - thk // 2
            rects = [
                _rect(top_x, y, L, bar_h),
                _rect(bot_x, y + bh - bar_h, L, bar_h),
                _rect(connector_x, y + bar_h, thk, bh - 2 * bar_h),
            ]
    elif fam == "T42":
        # T shape with different rects attached near the bar: use three bold stems (left/center/right) into a thick bar
        bar_h = thk
        stem_c = thk
        stem_l = int(thk * float(rng.uniform(0.8, 1.2)))
        stem_r = int(thk * float(rng.uniform(0.8, 1.2)))
        rects = [
            _rect(x, y, bw, bar_h),
            _rect(x + bw//2 - stem_c//2, y, stem_c, bh),
            _rect(x + bw//4 - stem_l//2, y, stem_l, bh//2 + bar_h//2),
            _rect(x + 3*bw//4 - stem_r//2, y, stem_r, bh//2 + bar_h//2),
        ]
    elif fam == "T43":
        # Four-quadrant ring with a clear hole (hole thickness ~= thk)
        w2 = bw // 2; h2 = bh // 2
        rects = [
            _rect(x, y, w2, h2), _rect(x + w2, y, bw - w2, h2),
            _rect(x, y + h2, w2, bh - h2), _rect(x + w2, y + h2, bw - w2, bh - h2)
        ]
    else:  # T44: step-like dangling part
        bar_h = max(min_part, int(bh * 0.7))
        step_w = max(min_part, int(bw * 0.35))
        rects = [
            _rect(x, y, bw, bar_h),
            _rect(x + bw//2 - step_w//2, y + bar_h, step_w, bh - bar_h)
        ]
    return rects


def _build_classifier_rects(shape: str, x: int, y: int, bw: int, bh: int, rng: np.random.Generator) -> list[Rect]:
    """Shapes for classifier: I(F1), L(F2), T(F2), U(F3), Z(F3), O(F4).
    Enforces one primary rectangle much larger than others (>=2x area), with
    all other rectangles being short stubs whose protrusion is < 0.5 of the
    primary rectangle height (for vertical protrusions) or width (for horizontal).
    """
    rects: list[Rect] = []
    min_part = max(6, min(bw, bh) // 6)

    # Primary size heuristics
    primary_w = max(min_part * 2, int(min(bw, int(bw * float(rng.uniform(0.45, 0.85))))))
    primary_h = max(min_part * 2, int(min(bh, int(bh * float(rng.uniform(0.45, 0.85))))))

    # Place primary, possibly centered initially; per-shape we may snap to sides
    px0 = int(x + (bw - primary_w) // 2)
    py0 = int(y + (bh - primary_h) // 2)

    # Helper to clamp stub sizes and enforce area ratio
    def _cap_area_and_create(ax: int, ay: int, aw: int, ah: int, p_area: int) -> Rect:
        sw = max(1, aw)
        sh = max(1, ah)
        # enforce primary area >= 2x stub area
        while sw * sh > max(1, p_area // 2):
            # shrink the dominating dimension first
            if sw >= sh and sw > 1:
                sw = max(1, int(sw * 0.8))
            elif sh > 1:
                sh = max(1, int(sh * 0.8))
            else:
                break
        return _rect(ax, ay, sw, sh)

    if shape == "I":  # one big block only
        rects = [_rect(x, y, bw, bh)]
        return rects

    if shape == "L":
        # Choose whether the primary is vertical or horizontal and anchor to a corner
        primary_vertical = rng.random() < 0.5
        if primary_vertical:
            # Tall primary leg hugging left or right side
            primary_w = max(min_part, int(min(bw, bh) * float(rng.uniform(0.38, 0.52))))
            primary_h = bh
            left_side = rng.random() < 0.5
            px0 = x if left_side else x + bw - primary_w
            py0 = y
            primary = _rect(px0, py0, primary_w, primary_h)
            # Horizontal stub at the bottom, protruding horizontally from the primary
            stub_h = max(min_part, int(primary_w * float(rng.uniform(0.35, 0.55))))
            # Allow a little more protrusion for L: up to 0.75 * primary width
            protr_max = max(2, int(0.75 * primary_w))
            stub_w = int(rng.integers(max(2, int(0.3 * primary_w)), protr_max + 1))
            # Enforce a minimum thickness-to-length ratio to avoid skinny protrusions
            min_ratio_h_over_w = 0.3
            min_stub_h = int(np.ceil(stub_w * min_ratio_h_over_w))
            if stub_h < min_stub_h:
                stub_h = min_stub_h
            if left_side:
                sx0 = px0 + primary_w
            else:
                sx0 = px0 - stub_w
            sy0 = y + bh - stub_h
            primary_area = primary_w * primary_h
            stub = _cap_area_and_create(sx0, sy0, stub_w, stub_h, primary_area)
            rects = [primary, stub]
        else:
            # Wide primary bar at top or bottom
            primary_h = max(min_part, int(min(bw, bh) * float(rng.uniform(0.38, 0.52))))
            primary_w = bw
            top_side = rng.random() < 0.5
            px0 = x
            py0 = y if top_side else y + bh - primary_h
            primary = _rect(px0, py0, primary_w, primary_h)
            # Vertical stub on left or right; vertical protrusion limited by 0.5 * primary height
            stub_w = max(min_part, int(primary_h * float(rng.uniform(0.35, 0.55))))
            # Allow a little more protrusion for L: up to 0.75 * primary height
            protr_max = max(2, int(0.75 * primary_h))
            stub_h = int(rng.integers(max(2, int(0.3 * primary_h)), protr_max + 1))
            # Enforce a minimum thickness-to-length ratio to avoid skinny protrusions
            min_ratio_w_over_h = 0.3
            min_stub_w = int(np.ceil(stub_h * min_ratio_w_over_h))
            if stub_w < min_stub_w:
                stub_w = min_stub_w
            left_side = rng.random() < 0.5
            sx0 = x if left_side else x + bw - stub_w
            if top_side:
                sy0 = py0 + primary_h
            else:
                sy0 = py0 - stub_h
            primary_area = primary_w * primary_h
            stub = _cap_area_and_create(sx0, sy0, stub_w, stub_h, primary_area)
            rects = [primary, stub]
        return rects

    if shape == "T":
        # Primary vertical body centered; short horizontal cap on top or bottom
        primary_w = max(min_part, int(min(bw, bh) * float(rng.uniform(0.38, 0.52))))
        primary_h = bh
        px0 = x + (bw - primary_w) // 2
        py0 = y
        primary = _rect(px0, py0, primary_w, primary_h)
        # Cap bar height (vertical protrusion) limited by 0.5 * primary height
        cap_h = int(rng.integers(max(2, int(0.15 * primary_h)), max(3, int(0.5 * primary_h))))
        cap_w = int(rng.integers(int(0.4 * bw), int(0.9 * bw)))
        cap_w = max(min_part, min(bw, cap_w))
        on_top = rng.random() < 0.5
        cx0 = x + (bw - cap_w) // 2
        cy0 = y - cap_h if not on_top and (py0 - cap_h) >= y else (py0 + primary_h)
        if on_top:
            cy0 = y
        # Enforce area ratio
        stub = _cap_area_and_create(cx0, cy0, cap_w, cap_h, primary_w * primary_h)
        rects = [primary, stub]
        return rects

    if shape == "U":
        # Primary horizontal bar at top; two short vertical legs
        primary_h = max(min_part, int(min(bw, bh) * float(rng.uniform(0.38, 0.52))))
        primary_w = bw
        px0 = x
        py0 = y
        primary = _rect(px0, py0, primary_w, primary_h)
        # Legs: vertical protrusion limited by 0.5 * primary height
        leg_h = int(rng.integers(max(2, int(0.2 * primary_h)), max(3, int(0.5 * primary_h))))
        leg_w = max(min_part, int(primary_h * float(rng.uniform(0.35, 0.55))))
        gap = int(max(2, bw * 0.15))
        lx0 = x + gap
        rx0 = x + bw - gap - leg_w
        ly0 = py0 + primary_h
        ry0 = py0 + primary_h
        p_area = primary_w * primary_h
        left_leg = _cap_area_and_create(lx0, ly0, leg_w, leg_h, p_area)
        right_leg = _cap_area_and_create(rx0, ry0, leg_w, leg_h, p_area)
        rects = [primary, left_leg, right_leg]
        return rects

    if shape == "Z":
        # Primary central block; two short horizontal stubs on opposite sides at different heights
        primary_w = max(min_part * 2, int(bw * float(rng.uniform(0.45, 0.7))))
        primary_h = max(min_part * 2, int(bh * float(rng.uniform(0.45, 0.7))))
        px0 = x + (bw - primary_w) // 2
        py0 = y + (bh - primary_h) // 2
        primary = _rect(px0, py0, primary_w, primary_h)
        # Horizontal protrusions limited by 0.5 * primary width
        stub_h = max(min_part, int(primary_h * float(rng.uniform(0.18, 0.32))))
        protr_max = max(2, int(0.5 * primary_w))
        stub_w_left = int(rng.integers(max(2, int(0.25 * primary_w)), protr_max + 1))
        stub_w_right = int(rng.integers(max(2, int(0.25 * primary_w)), protr_max + 1))
        # left stub at top-left side, extending to the left
        lsx0 = px0 - stub_w_left
        lsy0 = py0
        # right stub at bottom-right side, extending to the right
        rsx0 = px0 + primary_w
        rsy0 = py0 + primary_h - stub_h
        p_area = primary_w * primary_h
        left_stub = _cap_area_and_create(lsx0, lsy0, stub_w_left, stub_h, p_area)
        right_stub = _cap_area_and_create(rsx0, rsy0, stub_w_right, stub_h, p_area)
        rects = [primary, left_stub, right_stub]
        return rects

    # "O" F4 (ring of four rectangles) – keep uniform to preserve a clean hole
    t = max(min_part, int(min(bw, bh) * float(rng.uniform(0.28, 0.4))))
    inner_h = max(0, bh - 2 * t)
    inner_w = max(0, bw - 2 * t)
    if inner_h > 0 and inner_w > 0:
        rects = [
            _rect(x, y, bw, t),  # top
            _rect(x, y + bh - t, bw, t),  # bottom
            _rect(x, y + t, t, bh - 2 * t),  # left
            _rect(x + bw - t, y + t, t, bh - 2 * t),  # right
        ]
    else:
        rects = [_rect(x, y, bw, bh)]
    return rects


def gen_classifier_families(out_dir: str, num: int = 200000, h: int = 128, w: int = 128,
                            occl_prob: float = 0.0, morph_prob: float = 0.0,
                            shapes: list[str] | None = None) -> None:
    """Generate segmentation-like masks for classifier training with labels I/L/T/U/Z/O.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(1234)
    if not shapes:
        shapes = ["I", "L", "T", "U", "Z", "O"]
    for i in range(num):
        shp = shapes[int(rng.integers(0, len(shapes)))]
        # choose a bounding box with decent margins
        min_side = int(min(h, w) * 0.45)
        max_side = int(min(h, w) * 0.9)
        bw = int(rng.integers(min_side, max_side))
        bh = int(rng.integers(min_side, max_side))
        x = int(rng.integers(0, max(1, w - bw)))
        y = int(rng.integers(0, max(1, h - bh)))
        rects = _build_classifier_rects(shp, x, y, bw, bh, rng)
        # If any rect carries a color, render an RGB debug canvas; otherwise grayscale
        has_color = any(isinstance(r, (tuple, list)) and len(r) == 2 for r in rects)
        if has_color:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            for r in rects:
                color = (255, 255, 255)
                if isinstance(r, (tuple, list)) and len(r) == 2:
                    color = tuple(int(c) for c in r[1])
                    r = r[0]
                r = _as_rect(r)
                x0, y0 = r.x, r.y
                x1, y1 = x0 + r.w, y0 + r.h
                x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
                y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
                canvas[y0:y1, x0:x1] = color
        else:
            canvas = np.zeros((h, w), dtype=np.uint8)
            for r in rects:
                r = _as_rect(r)
                x0, y0 = r.x, r.y
                x1, y1 = x0 + r.w, y0 + r.h
                x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
                y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
                canvas[y0:y1, x0:x1] = 255
        if rng.random() < occl_prob:
            ox = int(rng.integers(x, x + bw)); oy = int(rng.integers(y, y + bh))
            ow = max(2, int(bw * 0.08)); oh = max(2, int(bh * 0.08))
            canvas[max(0, oy):min(h, oy+oh), max(0, ox):min(w, ox+ow)] = 0
        canvas = _rand_affine(canvas, rng, translate_frac=0.02, morph_prob=morph_prob)
        cv2.imwrite(os.path.join(out_dir, f"{shp}_{i:06d}.png"), canvas)


def gen_pbsr_families(out_dir: str, num: int = 200000, h: int = 128, w: int = 128,
                      occl_prob: float = 0.0, morph_prob: float = 0.0,
                      families: list[str] | None = None) -> None:
    """Generate labeled PBSR family masks with crisp axis-aligned parts.
    Use low/no occlusions to resemble target shapes; minimal jitter to avoid overfitting.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    if families is None or len(families) == 0:
        families = ["T11", "T21", "T31", "T32", "T43"]  # default subset
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
        rects = _build_family_rects(fam, x, y, bw, bh, rng)
        # rasterize
        has_color = any(isinstance(r, (tuple, list)) and len(r) == 2 for r in rects)
        if has_color:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            for r in rects:
                color = (255, 255, 255)
                if isinstance(r, (tuple, list)) and len(r) == 2:
                    color = tuple(int(c) for c in r[1])
                    r = r[0]
                r = _as_rect(r)
                x0, y0 = r.x, r.y
                x1, y1 = x0 + r.w, y0 + r.h
                x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
                y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
                canvas[y0:y1, x0:x1] = color
        else:
            canvas = np.zeros((h, w), dtype=np.uint8)
            for r in rects:
                r = _as_rect(r)
                x0, y0 = r.x, r.y
                x1, y1 = x0 + r.w, y0 + r.h
                x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
                y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
                canvas[y0:y1, x0:x1] = 255
        # optional small occlusions
        if rng.random() < occl_prob:
            ox = int(rng.integers(x, ww)); oy = int(rng.integers(y, hh))
            ow = max(2, int(bw * 0.1)); oh = max(2, int(bh * 0.1))
            canvas[max(0, oy):min(h, oy+oh), max(0, ox):min(w, ox+ow)] = 0
        # minimal affine, no rotation, optional mild morph
        canvas = _rand_affine(canvas, rng, translate_frac=0.02, morph_prob=morph_prob)
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
    ap.add_argument("--mode", choices=["footprints", "roofs", "pbsr", "classifier"], default="footprints")
    ap.add_argument("--occl_prob", type=float, default=0.0)
    ap.add_argument("--morph_prob", type=float, default=0.0)
    ap.add_argument("--families", type=str, default="", help="Comma-separated list, e.g. T11,T21,T31,T32,T43")
    ap.add_argument("--shapes", type=str, default="", help="Comma-separated list for classifier: I,L,T,U,Z,O")
    args = ap.parse_args()

    if args.mode == "footprints":
        main(args.out, args.num)
    elif args.mode == "roofs":
        gen_synth_roof_edges(args.out, args.num)
    elif args.mode == "pbsr":
        fams = [s.strip() for s in args.families.split(",") if s.strip()] if args.families else None
        gen_pbsr_families(args.out, args.num, occl_prob=args.occl_prob, morph_prob=args.morph_prob, families=fams)
    elif args.mode == "classifier":
        shp = [s.strip().upper() for s in args.shapes.split(",") if s.strip()] if args.shapes else None
        gen_classifier_families(args.out, args.num, occl_prob=args.occl_prob, morph_prob=args.morph_prob, shapes=shp)

