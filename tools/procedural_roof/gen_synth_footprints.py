import os
import argparse
import numpy as np
import cv2

import os, sys
import json
import math
import shutil
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List, Any
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.services.procedural_roof.pbsr import PBSRService, Rect


# -------------------------
# Tuning-friendly parameters
# -------------------------

@dataclass
class FootprintShapeParams:
    """Parameters controlling the internal geometry of footprint shapes.
    Adjust these to change thicknesses and proportions of roof parts.
    """
    # Minimal part thickness as a fraction of the short side of bbox
    min_part_frac: float = 0.17  # ~= 1/6
    # Primary rectangle size range as fraction of bbox
    primary_size_low: float = 0.55
    primary_size_high: float = 0.85
    # L-shape: stub thickness as fraction of primary dimension
    l_stub_thickness_low: float = 0.8
    l_stub_thickness_high: float = 0.9
    # L-shape: protrusion limit as fraction of primary dimension
    l_protrusion_limit: float = 1
    # T-shape: cap height as fraction of primary height
    t_cap_height_low: float = 0.50  # Increased from 0.15 to make caps more visible
    t_cap_height_high: float = 0.60  # Increased from 0.50 to make caps more proportional
    # T-shape: primary height as fraction of bounding box height (for vertical primary)
    t_primary_height_low: float = 0.70  # Primary height as fraction of bbox height
    t_primary_height_high: float = 0.90  # Primary height as fraction of bbox height
    # U-shape: leg thickness as fraction of primary height
    u_leg_thickness_low: float = 0.35
    u_leg_thickness_high: float = 0.45
    # U-shape: leg height as fraction of primary height
    u_leg_height_low: float = 0.30
    u_leg_height_high: float = 0.50
    # Z-shape: stub height as fraction of primary height
    z_stub_height_low: float = 0.18
    z_stub_height_high: float = 0.32
    # Z-shape: protrusion limit as fraction of primary width
    z_protrusion_limit: float = 0.50
    # O-shape: ring thickness as fraction of short side
    o_ring_thickness_low: float = 0.28
    o_ring_thickness_high: float = 0.40


@dataclass
class FootprintGenParams:
    """Top-level generation parameters for footprint masks."""
    # BBox size range (as a fraction of image min(h, w))
    bbox_min_side_frac: float = 0.65  # Increased from 0.35 to ensure larger shapes
    bbox_max_side_frac: float = 0.90
    # Noise / augmentation
    occl_prob: float = 0.0
    morph_prob: float = 0.0
    translate_frac: float = 0.02
    # RNG
    seed: int = 42
    # Shape geometry parameters
    shape: FootprintShapeParams = field(default_factory=FootprintShapeParams)


def _update_dataclass_from_dict(obj: Any, data: dict) -> Any:
    """Update a dataclass instance in-place from a flat or nested dict."""
    for k, v in (data or {}).items():
        if not hasattr(obj, k):
            continue
        cur = getattr(obj, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _update_dataclass_from_dict(cur, v)
        else:
            setattr(obj, k, v)
    return obj


def _load_footprint_params_from_config(config_path: Optional[str]) -> Optional[FootprintGenParams]:
    if not config_path:
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return None
    p = FootprintGenParams()
    _update_dataclass_from_dict(p, (cfg.get("footprint") or {}))
    return p


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





def _build_footprint_rects(shape: str, x: int, y: int, bw: int, bh: int, rng: np.random.Generator, params: FootprintShapeParams) -> list[Rect]:
    """Shapes for footprint classification: I(F1), L(F2), T(F2), U(F3), Z(F3), O(F4).
    Enforces one primary rectangle much larger than others (>=2x area), with
    all other rectangles being short stubs whose protrusion is limited relative to
    the primary rectangle dimensions.
    """
    rects: list[Rect] = []
    min_part = max(6, int(min(bw, bh) * params.min_part_frac))

    # Primary size heuristics
    primary_w = max(min_part * 2, int(min(bw, int(bw * float(rng.uniform(params.primary_size_low, params.primary_size_high))))))
    primary_h = max(min_part * 2, int(min(bh, int(bh * float(rng.uniform(params.primary_size_low, params.primary_size_high))))))

    # Place primary, possibly centered initially; per-shape we may snap to sides
    px0 = int(x + (bw - primary_w) // 2)
    py0 = int(y + (bh - primary_h) // 2)

    # Helper to clamp stub sizes and enforce area ratio
    def _cap_area_and_create(ax: int, ay: int, aw: int, ah: int, p_area: int, preserve_flush: bool = False, primary_right_edge: int = None, is_t_shape: bool = False) -> Rect:
        sw = max(1, aw)
        sh = max(1, ah)
        # For T-shapes, allow up to the full primary area (roughly equal sizes)
        # For other shapes, be more lenient - allow up to 1/2 of primary area
        if is_t_shape:
            max_stub_area = max(1, p_area)  # Allow full primary area for T-shapes
        else:
            max_stub_area = max(1, p_area // 2)  # Increased from 1/3 to 1/2
        while sw * sh > max_stub_area:
            # shrink the dominating dimension first, but preserve minimum connection requirements
            if sw >= sh and sw > 1:
                sw = max(1, int(sw * 0.9))  # Less aggressive shrinking (0.9 instead of 0.8)
            elif sh > 1:
                sh = max(1, int(sh * 0.9))  # Less aggressive shrinking
            else:
                break
        
        # If preserving flush connection, ensure stub extends to primary's edge
        if preserve_flush and primary_right_edge is not None:
            if ax + sw < primary_right_edge:
                sw = primary_right_edge - ax
        
        return _rect(ax, ay, sw, sh)
    
    # Helper to ensure L-shape connection by extending stub if needed
    def _ensure_l_connection(primary_rect: Rect, stub_rect: Rect, is_vertical_primary: bool) -> Rect:
        """Ensure stub connects to primary rectangle for proper L-shape."""
        if is_vertical_primary:
            # Primary is vertical, stub should be horizontal
            # Check horizontal alignment first
            if stub_rect.x + stub_rect.w < primary_rect.x:
                # Stub is to the left of primary - extend it
                new_w = primary_rect.x - stub_rect.x
                return _rect(stub_rect.x, stub_rect.y, new_w, stub_rect.h)
            elif stub_rect.x > primary_rect.x + primary_rect.w:
                # Stub is to the right of primary - extend it
                new_w = stub_rect.x + stub_rect.w - (primary_rect.x + primary_rect.w)
                new_x = primary_rect.x + primary_rect.w
                return _rect(new_x, stub_rect.y, new_w, stub_rect.h)
            
            # Check vertical alignment - stub should connect to primary's bottom edge
            primary_bottom = primary_rect.y + primary_rect.h
            if stub_rect.y + stub_rect.h < primary_bottom:
                # Stub doesn't reach primary's bottom - extend it down
                new_h = primary_bottom - stub_rect.y
                return _rect(stub_rect.x, stub_rect.y, stub_rect.w, new_h)
            elif stub_rect.y > primary_bottom:
                # Stub is below primary's bottom - move it up
                new_y = primary_bottom - stub_rect.h
                return _rect(stub_rect.x, new_y, stub_rect.w, stub_rect.h)
        else:
            # Primary is horizontal, stub should be vertical
            # Check vertical alignment first
            if stub_rect.y + stub_rect.h < primary_rect.y:
                # Stub is above primary - extend it
                new_h = primary_rect.y - stub_rect.y
                return _rect(stub_rect.x, stub_rect.y, stub_rect.w, new_h)
            elif stub_rect.y > primary_rect.y + primary_rect.h:
                # Stub is below primary - extend it
                new_h = stub_rect.y + stub_rect.h - (primary_rect.y + primary_rect.h)
                new_y = primary_rect.y + primary_rect.h
                return _rect(stub_rect.x, new_y, stub_rect.w, new_h)
            
            # Check horizontal alignment - stub should connect to primary's side
            if stub_rect.x + stub_rect.w < primary_rect.x:
                # Stub doesn't reach primary's left edge - extend it right
                new_w = primary_rect.x - stub_rect.x
                return _rect(stub_rect.x, stub_rect.y, new_w, stub_rect.h)
            elif stub_rect.x > primary_rect.x + primary_rect.w:
                # Stub is to the right of primary - move it left
                new_x = primary_rect.x + primary_rect.w - stub_rect.w
                return _rect(new_x, stub_rect.y, stub_rect.w, stub_rect.h)
        return stub_rect

    if shape == "I":  # one big block only
        rects = [_rect(x, y, bw, bh)]
        return rects

    if shape == "L":
        # Choose whether the primary is vertical or horizontal and anchor to a corner
        primary_vertical = rng.random() < 0.5
        if primary_vertical:
            # Tall primary leg hugging left or right side
            primary_w = max(min_part, int(min(bw, bh) * float(rng.uniform(params.primary_size_low, params.primary_size_high))))
            primary_h = bh
            left_side = rng.random() < 0.5
            px0 = x if left_side else x + bw - primary_w
            py0 = y
            primary = _rect(px0, py0, primary_w, primary_h)
            # Horizontal stub at the bottom, protruding horizontally from the primary
            stub_h = max(min_part, int(primary_w * float(rng.uniform(params.l_stub_thickness_low, params.l_stub_thickness_high))))
            # Allow protrusion up to the configured limit, but ensure minimum size
            protr_max = max(2, int(params.l_protrusion_limit * primary_w))
            min_stub_w = max(2, int(0.4 * primary_w))  # Ensure stub is at least 40% of primary width
            stub_w = int(rng.integers(min_stub_w, protr_max + 1))
            # Enforce a minimum thickness-to-length ratio to avoid skinny protrusions
            min_ratio_h_over_w = 0.4
            min_stub_h = int(np.ceil(stub_w * min_ratio_h_over_w))
            if stub_h < min_stub_h:
                stub_h = min_stub_h
            # Position stub to create proper L-shape (not T-shape)
            if left_side:
                sx0 = px0 + primary_w  # Stub extends to the right of primary
            else:
                sx0 = px0 - stub_w  # Stub extends to the left of primary
            # Ensure stub connects to primary's bottom edge
            sy0 = py0 + primary_h - stub_h
            # Ensure stub is tall enough to connect to primary's bottom
            min_stub_h_for_connection = py0 + primary_h - sy0
            if stub_h < min_stub_h_for_connection:
                stub_h = min_stub_h_for_connection
            # Ensure stub stays within bounding box
            sx0 = max(x, sx0)
            sy0 = max(y, sy0)
            stub_w = min(stub_w, x + bw - sx0)
            stub_h = min(stub_h, y + bh - sy0)
            primary_area = primary_w * primary_h
            stub = _cap_area_and_create(sx0, sy0, stub_w, stub_h, primary_area)
            # Ensure proper L-shape connection AFTER area capping
            stub = _ensure_l_connection(primary, stub, True)  # True = vertical primary
            # Final check: ensure stub connects to primary
            if stub.y + stub.h < primary.y + primary.h:
                # Extend stub to reach primary's bottom
                new_h = primary.y + primary.h - stub.y
                stub = _rect(stub.x, stub.y, stub.w, new_h)
            rects = [primary, stub]
        else:
            # Wide primary bar at top or bottom
            primary_h = max(min_part, int(min(bw, bh) * float(rng.uniform(params.primary_size_low, params.primary_size_high))))
            primary_w = bw
            top_side = rng.random() < 0.5
            px0 = x
            py0 = y if top_side else y + bh - primary_h
            primary = _rect(px0, py0, primary_w, primary_h)
            # Vertical stub on left or right; vertical protrusion limited by configured limit
            stub_w = max(min_part, int(primary_h * float(rng.uniform(params.l_stub_thickness_low, params.l_stub_thickness_high))))
            # Allow protrusion up to the configured limit, but ensure minimum size
            protr_max = max(2, int(params.l_protrusion_limit * primary_h))
            min_stub_h = max(2, int(0.4 * primary_h))  # Ensure stub is at least 40% of primary height
            stub_h = int(rng.integers(min_stub_h, protr_max + 1))
            # Enforce a minimum thickness-to-length ratio to avoid skinny protrusions
            min_ratio_w_over_h = 0.4
            min_stub_w = int(np.ceil(stub_h * min_ratio_w_over_h))
            if stub_w < min_stub_w:
                stub_w = min_stub_w
            left_side = rng.random() < 0.5
            # Position stub flush with primary's edge, not bounding box edge
            if left_side:
                sx0 = px0  # Stub starts at primary's left edge
            else:
                sx0 = px0 + primary_w - stub_w  # Stub ends at primary's right edge

            # Position stub to create proper L-shape (not T-shape)
            if top_side:
                sy0 = py0 + primary_h  # Stub extends below primary
            else:
                sy0 = py0 - stub_h  # Stub extends above primary
            # Ensure stub connects to primary vertically
            if top_side:
                # Stub should start at primary's bottom edge
                sy0 = py0 + primary_h
            else:
                # Stub should end at primary's top edge
                sy0 = py0 - stub_h
            # Ensure stub stays within bounding box, but preserve flush connection to primary
            sx0 = max(x, sx0)
            sy0 = max(y, sy0)
            # For horizontal primaries, ensure stub extends to primary's edge
            if left_side:
                # Stub should start at primary's left edge and extend right
                sx0 = px0
                stub_w = min(stub_w, x + bw - sx0)
            else:
                # Stub should end at primary's right edge and extend left
                stub_w = min(stub_w, x + bw - sx0)
                # Ensure stub ends exactly at primary's right edge
                if sx0 + stub_w < px0 + primary_w:
                    stub_w = px0 + primary_w - sx0
            stub_h = min(stub_h, y + bh - sy0)
            primary_area = primary_w * primary_h
            # For horizontal primaries, preserve flush connection to primary's edge
            preserve_flush = not left_side  # Only for right-side stubs
            primary_right_edge = px0 + primary_w if preserve_flush else None
            stub = _cap_area_and_create(sx0, sy0, stub_w, stub_h, primary_area, preserve_flush, primary_right_edge)
            # For horizontal primaries, we already positioned the stub correctly, so skip _ensure_l_connection
            # Final check: ensure stub connects to primary
            if stub.x + stub.w < primary.x:
                # Extend stub to reach primary's left edge
                new_w = primary.x - stub.x
                stub = _rect(stub.x, stub.y, new_w, stub.h)
            elif stub.x > primary.x + primary.w:
                # Move stub to connect to primary's right edge
                new_x = primary.x + primary.w - stub.w
                stub = _rect(new_x, stub.y, stub.w, stub.h)
            # Final vertical connection check
            if top_side:
                # Ensure stub starts at primary's bottom
                if stub.y > py0 + primary_h:
                    new_y = py0 + primary_h
                    stub = _rect(stub.x, new_y, stub.w, stub.h)
            else:
                # Ensure stub ends at primary's top
                if stub.y + stub.h < py0:
                    new_h = py0 - stub.y
                    stub = _rect(stub.x, stub.y, stub.w, new_h)
            rects = [primary, stub]
        return rects

    if shape == "T":
        # Choose orientation: vertical primary with horizontal cap, or horizontal primary with vertical cap
        vertical_primary = rng.random() < 0.5
        
        if vertical_primary:
            # Original T-shape: vertical primary with horizontal cap
            # For T-shapes, make primary narrower to achieve 2:3 ratio with cap
            t_primary_size_low = params.primary_size_low * 0.8  # Reduce primary width by 40%
            t_primary_size_high = params.primary_size_high * 0.8
            
            # Ensure bounding box is large enough for T-shape requirements
            # We need enough space for primary + cap with 2:3 ratio
            # Cap needs to be 1.8x-2.5x primary width, so we need at least 2.5x primary width
            min_required_bw = int(min_part * 2.5 / t_primary_size_low)  # Minimum bbox width needed
            if bw < min_required_bw:
                # Adjust primary size to fit within available space
                max_primary_w = int(bw / 2.5)  # Ensure cap can be 2.5x primary width
                primary_w = max(min_part, min(max_primary_w, int(min(bw, bh) * float(rng.uniform(t_primary_size_low, t_primary_size_high)))))
            else:
                primary_w = max(min_part, int(min(bw, bh) * float(rng.uniform(t_primary_size_low, t_primary_size_high))))
            
            # Primary height as fraction of bounding box height
            primary_h = int(bh * float(rng.uniform(params.t_primary_height_low, params.t_primary_height_high)))
            px0 = x + (bw - primary_w) // 2
            py0 = y + (bh - primary_h) // 2  # Center the primary vertically
            primary = _rect(px0, py0, primary_w, primary_h)
            
            # Cap bar height (vertical protrusion) limited by configured range
            cap_h = int(rng.integers(max(2, int(params.t_cap_height_low * primary_h)), max(3, int(params.t_cap_height_high * primary_h))))
            
            # Adaptive cap width calculation based on available space
            # Target cap width = primary_w * 1.5 to achieve the 2:3 ratio
            target_cap_w = int(primary_w * 1.5)  # 1.5 = 3/2 to achieve 2:3 ratio
            
            # Calculate available space for cap
            available_space = bw
            # Ensure we have enough space for the cap with our desired multipliers
            min_cap_w = max(min_part, int(1.8 * primary_w))  # At least 180% of primary width
            max_cap_w = min(available_space, int(2.5 * primary_w))  # Up to 250% of primary width, but not exceeding bbox
            
            # If we don't have enough space for the minimum cap width, adjust primary size
            if min_cap_w > available_space:
                # Recalculate primary width to fit within available space
                max_primary_w = int(available_space / 1.8)  # Ensure min_cap_w fits
                primary_w = max(min_part, max_primary_w)
                # Recalculate cap width range
                min_cap_w = max(min_part, int(1.8 * primary_w))
                max_cap_w = min(available_space, int(2.5 * primary_w))
                # Update primary rectangle
                px0 = x + (bw - primary_w) // 2
                primary = _rect(px0, py0, primary_w, primary_h)
            
            # Ensure valid range for random integer generation
            if min_cap_w >= max_cap_w:
                # This should not happen with proper bounding box sizing
                # If it does, it means our bounding box adjustment failed
                # In this case, we need to reduce the primary width to make room for the cap
                max_primary_w = int(bw / 1.8)  # Ensure cap can be at least 1.8x primary width
                if max_primary_w >= min_part:
                    primary_w = max(min_part, max_primary_w)
                    # Recalculate cap width range
                    min_cap_w = max(min_part, int(1.8 * primary_w))
                    max_cap_w = min(bw, int(2.5 * primary_w))
                    # Update primary rectangle
                    px0 = x + (bw - primary_w) // 2
                    primary = _rect(px0, py0, primary_w, primary_h)
                else:
                    # If we can't even fit a minimum primary, use fallback
                    min_cap_w = max(min_part, int(1.2 * primary_w))
                    max_cap_w = min(bw, int(1.8 * primary_w))
                    
                    # If still invalid, use primary width as fallback
                    if min_cap_w >= max_cap_w:
                        min_cap_w = max(min_part, primary_w)
                        max_cap_w = min(bw, primary_w + min_part)
                    
                    # Final safety check - ensure we have a valid range
                    if min_cap_w >= max_cap_w:
                        # Last resort: use min_part as the range
                        min_cap_w = min_part
                        max_cap_w = min_part + 1
            
            cap_w = int(rng.integers(min_cap_w, max_cap_w + 1))
            
            on_top = rng.random() < 0.5
            
            # Allow cap to overlap with primary for better T-shape appearance
            if on_top:
                # For top cap: align top edge of cap with top edge of primary
                cy0 = py0
            else:
                # For bottom cap: overlap with primary's bottom portion
                # Position cap so it overlaps with the bottom portion of the primary
                # The cap should extend from primary's bottom edge upward
                cy0 = py0 + primary_h - cap_h
                # Ensure the cap actually overlaps with the primary (not below it)
                if cy0 > py0 + primary_h:
                    cy0 = py0 + primary_h - cap_h
            
            # Initial positioning (will be corrected after area capping)
            cx0 = px0 + (primary_w - cap_w) // 2  # Center initially
            
            # Enforce area ratio - allow T-shape caps to be roughly equal to primary area
            stub = _cap_area_and_create(cx0, cy0, cap_w, cap_h, primary_w * primary_h, is_t_shape=True)
            
            # Now calculate the proper positioning based on the final cap dimensions
            final_cap_w = stub.w
            final_cap_h = stub.h
            
            # Ensure cap overlaps with primary to create proper T-shape
            # Cap should be centered on primary, but always overlap
            if final_cap_w <= primary_w:
                # Cap is narrower than primary - center it
                final_cx0 = px0 + (primary_w - final_cap_w) // 2
            else:
                # Cap is wider than primary - ensure it overlaps
                # Position cap so it extends beyond primary on both sides
                final_cx0 = px0 - (final_cap_w - primary_w) // 2
                # Ensure cap doesn't go outside bounding box
                if final_cx0 < x:
                    final_cx0 = x
                elif final_cx0 + final_cap_w > x + bw:
                    final_cx0 = x + bw - final_cap_w
            
            # Update the stub with the correct position
            stub = _rect(final_cx0, stub.y, final_cap_w, final_cap_h)
            
        else:
            # Rotated T-shape: horizontal primary with vertical cap
            # For horizontal T-shapes, make primary shorter to achieve 2:3 ratio with cap
            t_primary_size_low = params.primary_size_low * 0.8  # Reduce primary height by 40%
            t_primary_size_high = params.primary_size_high * 0.8
            
            # Ensure bounding box is large enough for T-shape requirements
            # We need enough space for primary + cap with 2:3 ratio
            # Cap needs to be 1.8x-2.5x primary height, so we need at least 2.5x primary height
            min_required_bh = int(min_part * 2.5 / t_primary_size_low)  # Minimum bbox height needed
            if bh < min_required_bh:
                # Adjust primary size to fit within available space
                max_primary_h = int(bh / 2.5)  # Ensure cap can be 2.5x primary height
                primary_h = max(min_part, min(max_primary_h, int(min(bw, bh) * float(rng.uniform(t_primary_size_low, t_primary_size_high)))))
            else:
                primary_h = max(min_part, int(min(bw, bh) * float(rng.uniform(t_primary_size_low, t_primary_size_high))))
            
            primary_w = bw
            px0 = x
            py0 = y + (bh - primary_h) // 2
            primary = _rect(px0, py0, primary_w, primary_h)
            
            # Cap bar width (horizontal protrusion) limited by configured range
            cap_w = int(rng.integers(max(2, int(params.t_cap_height_low * primary_w)), max(3, int(params.t_cap_height_high * primary_w))))
            
            # Adaptive cap height calculation based on available space
            # Target cap height = primary_h * 1.5 to achieve the 2:3 ratio
            target_cap_h = int(primary_h * 1.5)  # 1.5 = 3/2 to achieve 2:3 ratio
            
            # Calculate available space for cap
            available_space = bh
            # Ensure we have enough space for the cap with our desired multipliers
            min_cap_h = max(min_part, int(1.8 * primary_h))  # At least 180% of primary height
            max_cap_h = min(available_space, int(2.5 * primary_h))  # Up to 250% of primary height, but not exceeding bbox
            
            # If we don't have enough space for the minimum cap height, adjust primary size
            if min_cap_h > available_space:
                # Recalculate primary height to fit within available space
                max_primary_h = int(available_space / 1.8)  # Ensure min_cap_h fits
                primary_h = max(min_part, max_primary_h)
                # Recalculate cap height range
                min_cap_h = max(min_part, int(1.8 * primary_h))
                max_cap_h = min(available_space, int(2.5 * primary_h))
                # Update primary rectangle
                py0 = y + (bh - primary_h) // 2
                primary = _rect(px0, py0, primary_w, primary_h)
            
            # Ensure valid range for random integer generation
            if min_cap_h >= max_cap_h:
                # This should not happen with proper bounding box sizing
                # If it does, it means our bounding box adjustment failed
                # In this case, we need to reduce the primary height to make room for the cap
                max_primary_h = int(bh / 1.8)  # Ensure cap can be at least 1.8x primary height
                if max_primary_h >= min_part:
                    primary_h = max(min_part, max_primary_h)
                    # Recalculate cap height range
                    min_cap_h = max(min_part, int(1.8 * primary_h))
                    max_cap_h = min(bh, int(2.5 * primary_h))
                    # Update primary rectangle
                    py0 = y + (bh - primary_h) // 2
                    primary = _rect(px0, py0, primary_w, primary_h)
                else:
                    # If we can't even fit a minimum primary, use fallback
                    min_cap_h = max(min_part, int(1.2 * primary_h))
                    max_cap_h = min(bh, int(1.8 * primary_h))
                    
                    # If still invalid, use primary height as fallback
                    if min_cap_h >= max_cap_h:
                        min_cap_h = max(min_part, primary_h)
                        max_cap_h = min(bh, primary_h + min_part)
                    
                    # Final safety check - ensure we have a valid range
                    if min_cap_h >= max_cap_h:
                        # Last resort: use min_part as the range
                        min_cap_h = min_part
                        max_cap_h = min_part + 1
            
            cap_h = int(rng.integers(min_cap_h, max_cap_h + 1))
            
            on_left = rng.random() < 0.5
            
            # Allow cap to overlap with primary for better T-shape appearance
            if on_left:
                # For left cap: align left edge of cap with left edge of primary
                cx0 = px0
            else:
                # For right cap: overlap with primary's right portion
                # Position cap so it overlaps with the right portion of the primary
                # The cap should extend from primary's right edge leftward
                cx0 = px0 + primary_w - cap_w
                # Ensure the cap actually overlaps with the primary (not beyond it)
                if cx0 < px0:
                    cx0 = px0
            
            # Initial positioning (will be corrected after area capping)
            cy0 = py0 + (primary_h - cap_h) // 2  # Center initially
            
            # Enforce area ratio - allow T-shape caps to be roughly equal to primary area
            stub = _cap_area_and_create(cx0, cy0, cap_w, cap_h, primary_w * primary_h, is_t_shape=True)
            
            # Now calculate the proper positioning based on the final cap dimensions
            final_cap_w = stub.w
            final_cap_h = stub.h
            
            # Ensure cap overlaps with primary to create proper T-shape
            # Cap should be centered on primary, but always overlap
            if final_cap_h <= primary_h:
                # Cap is shorter than primary - center it
                final_cy0 = py0 + (primary_h - final_cap_h) // 2
            else:
                # Cap is taller than primary - ensure it overlaps
                # Position cap so it extends beyond primary on both sides
                final_cy0 = py0 - (final_cap_h - primary_h) // 2
                # Ensure cap doesn't go outside bounding box
                if final_cy0 < y:
                    final_cy0 = y
                elif final_cy0 + final_cap_h > y + bh:
                    final_cy0 = y + bh - final_cap_h
            
            # Update the stub with the correct position
            stub = _rect(stub.x, final_cy0, final_cap_w, final_cap_h)
        
        rects = [primary, stub]
        return rects

    if shape == "U":
        # Primary horizontal bar at top; two short vertical legs
        primary_h = max(min_part, int(min(bw, bh) * float(rng.uniform(params.primary_size_low, params.primary_size_high))))
        # Ensure there is room for legs below the primary within the bbox to avoid clipping
        # Reserve at least the minimum leg height (fraction of primary height)
        max_primary_h_to_fit_low_leg = max(
            min_part,
            int(bh / (1.0 + float(params.u_leg_height_low) + 1e-6))
        )
        if primary_h > max_primary_h_to_fit_low_leg:
            primary_h = max_primary_h_to_fit_low_leg
        primary_w = bw
        px0 = x
        py0 = y
        primary = _rect(px0, py0, primary_w, primary_h)
        # Legs: vertical protrusion limited by configured range
        leg_h = int(rng.integers(max(2, int(params.u_leg_height_low * primary_h)), max(3, int(params.u_leg_height_high * primary_h))))
        # Define leg width RELATIVE TO PRIMARY RECTANGLE WIDTH (not bbox)
        raw_leg_w = int(primary_w * float(rng.uniform(params.u_leg_thickness_low, params.u_leg_thickness_high)))
        # Ensure there is an inner gap between legs: reserve at least 20% of primary width or min_part
        min_inner_gap = max(min_part, int(0.20 * primary_w))
        max_leg_w = max(1, (primary_w - min_inner_gap) // 2)
        leg_w = max(1, min(max_leg_w, max(min_part, raw_leg_w)))
        # Place legs flush with the primary's side edges
        # Left leg: left edge flush with primary's left edge
        lx0 = px0
        # Right leg: right edge flush with primary's right edge
        rx0 = px0 + primary_w - leg_w
        ly0 = py0 + primary_h
        ry0 = py0 + primary_h
        # Clamp leg height so legs do not extend beyond the bbox bottom
        available_h = max(0, (y + bh) - ly0)
        if available_h <= 0:
            leg_h = 0
        else:
            leg_h = min(leg_h, available_h)
        p_area = primary_w * primary_h
        left_leg = _cap_area_and_create(lx0, ly0, leg_w, leg_h, p_area)
        # Preserve flush on the right edge even if area capping shrinks width
        right_leg = _cap_area_and_create(rx0, ry0, leg_w, leg_h, p_area, preserve_flush=True, primary_right_edge=px0 + primary_w)
        rects = [primary, left_leg, right_leg]

        # Optional 90-degree rotation to increase variation (rotate within bbox)
        if rng.random() < 0.5:
            rotate_cw = rng.random() < 0.5
            rotated: list[Rect] = []
            for r in rects:
                # local coords within bbox
                lx = r.x - x
                ly = r.y - y
                if rotate_cw:
                    # Clockwise: (x, y, w, h) -> (bh - (y + h), x, h, w)
                    nx = x + (bh - (ly + r.h))
                    ny = y + lx
                else:
                    # Counter-clockwise: (x, y, w, h) -> (y, bw - (x + w), h, w)
                    nx = x + ly
                    ny = y + (bw - (lx + r.w))
                rotated.append(_rect(nx, ny, r.h, r.w))
            rects = rotated

        return rects

    if shape == "Z":
        # Primary central block; two short horizontal stubs on opposite sides at different heights
        primary_w = max(min_part * 2, int(bw * float(rng.uniform(0.45, 0.7))))
        primary_h = max(min_part * 2, int(bh * float(rng.uniform(0.45, 0.7))))
        px0 = x + (bw - primary_w) // 2
        py0 = y + (bh - primary_h) // 2
        primary = _rect(px0, py0, primary_w, primary_h)
        # Horizontal protrusions limited by configured range
        stub_h = max(min_part, int(primary_h * float(rng.uniform(params.z_stub_height_low, params.z_stub_height_high))))
        protr_max = max(2, int(params.z_protrusion_limit * primary_w))
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
    t = max(min_part, int(min(bw, bh) * float(rng.uniform(params.o_ring_thickness_low, params.o_ring_thickness_high))))
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


def gen_footprints(
    out_dir: str,
    num: int = 200000,
    h: int = 128,
    w: int = 128,
    shapes: list[str] | None = None,
    params: Optional[FootprintGenParams] = None,
    preview_count: int = 0,
    preview_cols: int = 8,
    dump_params: bool = True,
    wipe_output: bool = False,
    debug_mode: bool = False,
) -> None:
    """Generate footprint masks for classification training with labels I/L/T/U/Z/O.
    Uses the sophisticated parameter system from PBSR but with classifier shape logic.
    """
    if wipe_output:
        _safe_wipe_directory(out_dir, force=wipe_output)
    os.makedirs(out_dir, exist_ok=True)
    p = params or FootprintGenParams()
    rng = np.random.default_rng(p.seed)
    if not shapes:
        shapes = ["I", "L", "T", "U", "Z", "O"]
    min_side = int(min(h, w) * p.bbox_min_side_frac)
    max_side = int(min(h, w) * p.bbox_max_side_frac)
    previews: list[np.ndarray] = []
    debug_info = [] if debug_mode else None
    
    for i in range(num):
        shp = shapes[int(rng.integers(0, len(shapes)))]
        # choose a bounding box with decent margins
        bw = int(rng.integers(min_side, max_side))
        bh = int(rng.integers(min_side, max_side))
        
        # For T-shapes, ensure bounding box is large enough for the 2:3 ratio requirement
        if shp == "T":
            # T-shapes can be either vertical primary (horizontal cap) or horizontal primary (vertical cap)
            # We need to ensure the bounding box is large enough for both orientations
            min_part_frac = p.shape.min_part_frac
            t_primary_size_low = p.shape.primary_size_low * 0.8
            t_primary_size_high = p.shape.primary_size_high * 0.8
            
            min_part = max(6, int(min(h, w) * min_part_frac))
            
            # For vertical primary (horizontal cap): need width >= 2.8 * min_part
            min_required_bw_vertical = int(2.8 * min_part)
            
            # For horizontal primary (vertical cap): need height >= 2.8 * min_part  
            min_required_bh_horizontal = int(2.8 * min_part)
            
            # Ensure bounding box is large enough for both orientations
            if bw < min_required_bw_vertical:
                bw = min(min_required_bw_vertical, max_side)
            if bh < min_required_bh_horizontal:
                bh = min(min_required_bh_horizontal, max_side)
            
            # Additional safety checks for both orientations
            # Vertical primary: ensure width can accommodate max cap
            max_primary_w = int(bw * t_primary_size_high)
            max_cap_w_needed = int(2.5 * max_primary_w)
            if max_cap_w_needed > bw:
                required_bw = max_cap_w_needed
                bw = min(required_bw, max_side)
            
            # Horizontal primary: ensure height can accommodate max cap
            max_primary_h = int(bh * t_primary_size_high)
            max_cap_h_needed = int(2.5 * max_primary_h)
            if max_cap_h_needed > bh:
                required_bh = max_cap_h_needed
                bh = min(required_bh, max_side)
        
        x = int(rng.integers(0, max(1, w - bw)))
        y = int(rng.integers(0, max(1, h - bh)))
        
        # Store debug info if enabled
        if debug_mode:
            debug_data = {
                "image_id": i,
                "shape": shp,
                "bbox": {"x": x, "y": y, "w": bw, "h": bh},
                "params": {
                    "primary_size_low": p.shape.primary_size_low,
                    "primary_size_high": p.shape.primary_size_high,
                    "l_stub_thickness_low": p.shape.l_stub_thickness_low,
                    "l_stub_thickness_high": p.shape.l_stub_thickness_high,
                    "l_protrusion_limit": p.shape.l_protrusion_limit,
                }
            }
        
        rects = _build_footprint_rects(shp, x, y, bw, bh, rng, p.shape)
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
        if rng.random() < p.occl_prob:
            ox = int(rng.integers(x, x + bw)); oy = int(rng.integers(y, y + bh))
            ow = max(2, int(bw * 0.08)); oh = max(2, int(bh * 0.08))
            canvas[max(0, oy):min(h, oy+oh), max(0, ox):min(w, ox+ow)] = 0
        canvas = _rand_affine(canvas, rng, translate_frac=p.translate_frac, morph_prob=p.morph_prob)
        if i < preview_count:
            previews.append(canvas.copy())
        
        # Save image
        img_filename = f"{shp}_{i:06d}.png"
        cv2.imwrite(os.path.join(out_dir, img_filename), canvas)
        
        # Store debug info if enabled
        if debug_mode:
            debug_data["filename"] = img_filename
            debug_data["rects"] = [(r.x, r.y, r.w, r.h) for r in rects]
            debug_info.append(debug_data)
    if dump_params:
        _write_params_dump(out_dir, p)
    if preview_count > 0 and len(previews) > 0:
        grid = _make_preview_grid(previews, grid_cols=preview_cols)
        cv2.imwrite(os.path.join(out_dir, "preview_grid.png"), grid)
    
    # Save debug info if enabled
    if debug_mode and debug_info:
        debug_path = os.path.join(out_dir, "debug_info.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_info, f, indent=2)


def _write_params_dump(out_dir: str, params: FootprintGenParams) -> None:
    try:
        dump_path = os.path.join(out_dir, "params_used.json")
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(asdict(params), f, indent=2)
    except Exception:
        pass


def _safe_wipe_directory(out_dir: str, force: bool = False) -> None:
    """Safely clear output directory, with optional force flag to bypass checks."""
    if not os.path.exists(out_dir):
        return
    
    if not force:
        # Check if directory contains non-image files (potential safety issue)
        items = os.listdir(out_dir)
        non_image_exts = {'.txt', '.json', '.csv', '.md', '.py', '.sh', '.bat', '.exe', '.dll', '.so', '.dylib'}
        has_non_images = any(
            any(item.lower().endswith(ext) for ext in non_image_exts) 
            for item in items if os.path.isfile(os.path.join(out_dir, item))
        )
        if has_non_images:
            raise ValueError(
                f"Output directory '{out_dir}' contains non-image files. "
                f"Use --wipe to force clear the directory."
            )
    
    # Remove all contents
    for item in os.listdir(out_dir):
        item_path = os.path.join(out_dir, item)
        if os.path.isfile(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def _make_preview_grid(images: list[np.ndarray], grid_cols: int = 8) -> np.ndarray:
    if not images:
        return np.zeros((16, 16), dtype=np.uint8)
    cell_h, cell_w = images[0].shape[:2]
    cols = max(1, grid_cols)
    rows = int(math.ceil(len(images) / cols))
    is_rgb = (images[0].ndim == 3)
    grid_shape = (rows * cell_h, cols * cell_w, 3) if is_rgb else (rows * cell_h, cols * cell_w)
    grid = np.zeros(grid_shape, dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y0, x0 = r * cell_h, c * cell_w
        if img.shape[:2] != (cell_h, cell_w):
            img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        grid[y0:y0+cell_h, x0:x0+cell_w] = img
    return grid








if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--mode", choices=["footprints", "roofs"], default="footprints")
    # Footprint generation parameters
    ap.add_argument("--shapes", type=str, default="", help="Comma-separated list for footprints: I,L,T,U,Z,O")
    ap.add_argument("--config", type=str, default="", help="Optional JSON with 'footprint' params block")
    ap.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    ap.add_argument("--bbox_min_side_frac", type=float, default=None)
    ap.add_argument("--bbox_max_side_frac", type=float, default=None)
    ap.add_argument("--occl_prob", type=float, default=None)
    ap.add_argument("--morph_prob", type=float, default=None)
    ap.add_argument("--translate_frac", type=float, default=None)
    # Shape geometry overrides
    ap.add_argument("--min_part_frac", type=float, default=None)
    ap.add_argument("--primary_size_low", type=float, default=None)
    ap.add_argument("--primary_size_high", type=float, default=None)
    ap.add_argument("--l_stub_thickness_low", type=float, default=None)
    ap.add_argument("--l_stub_thickness_high", type=float, default=None)
    ap.add_argument("--l_protrusion_limit", type=float, default=None)
    ap.add_argument("--t_cap_height_low", type=float, default=None)
    ap.add_argument("--t_cap_height_high", type=float, default=None)
    ap.add_argument("--t_primary_height_low", type=float, default=None)
    ap.add_argument("--t_primary_height_high", type=float, default=None)
    ap.add_argument("--u_leg_thickness_low", type=float, default=None)
    ap.add_argument("--u_leg_thickness_high", type=float, default=None)
    ap.add_argument("--u_leg_height_low", type=float, default=None)
    ap.add_argument("--u_leg_height_high", type=float, default=None)
    ap.add_argument("--z_stub_height_low", type=float, default=None)
    ap.add_argument("--z_stub_height_high", type=float, default=None)
    ap.add_argument("--z_protrusion_limit", type=float, default=None)
    ap.add_argument("--o_ring_thickness_low", type=float, default=None)
    ap.add_argument("--o_ring_thickness_high", type=float, default=None)
    # Quality-of-life helpers
    ap.add_argument("--preview_count", type=int, default=0, help="Save N first samples in a preview grid")
    ap.add_argument("--preview_cols", type=int, default=8, help="Preview grid columns")
    ap.add_argument("--no_dump_params", action="store_true", help="Disable writing params_used.json")
    ap.add_argument("--wipe", action="store_true", help="Clear output directory before generating (safety check for non-images)")
    ap.add_argument("--debug", action="store_true", help="Enable debug mode - save parameter info for each image")
    args = ap.parse_args()

    if args.mode == "footprints":
        shp = [s.strip().upper() for s in args.shapes.split(",") if s.strip()] if args.shapes else None
        # Load params from JSON if provided
        p = _load_footprint_params_from_config(args.config) if args.config else FootprintGenParams()
        # CLI overrides (flat)
        if args.seed is not None:
            p.seed = int(args.seed)
        if args.bbox_min_side_frac is not None:
            p.bbox_min_side_frac = float(args.bbox_min_side_frac)
        if args.bbox_max_side_frac is not None:
            p.bbox_max_side_frac = float(args.bbox_max_side_frac)
        if args.occl_prob is not None:
            p.occl_prob = float(args.occl_prob)
        if args.morph_prob is not None:
            p.morph_prob = float(args.morph_prob)
        if args.translate_frac is not None:
            p.translate_frac = float(args.translate_frac)
        # Nested shape overrides
        shape_p = p.shape
        if args.min_part_frac is not None:
            shape_p.min_part_frac = float(args.min_part_frac)
        if args.primary_size_low is not None:
            shape_p.primary_size_low = float(args.primary_size_low)
        if args.primary_size_high is not None:
            shape_p.primary_size_high = float(args.primary_size_high)
        if args.l_stub_thickness_low is not None:
            shape_p.l_stub_thickness_low = float(args.l_stub_thickness_low)
        if args.l_stub_thickness_high is not None:
            shape_p.l_stub_thickness_high = float(args.l_stub_thickness_high)
        if args.l_protrusion_limit is not None:
            shape_p.l_protrusion_limit = float(args.l_protrusion_limit)
        if args.t_cap_height_low is not None:
            shape_p.t_cap_height_low = float(args.t_cap_height_low)
        if args.t_cap_height_high is not None:
            shape_p.t_cap_height_high = float(args.t_cap_height_high)
        if args.t_primary_height_low is not None:
            shape_p.t_primary_height_low = float(args.t_primary_height_low)
        if args.t_primary_height_high is not None:
            shape_p.t_primary_height_high = float(args.t_primary_height_high)
        if args.u_leg_thickness_low is not None:
            shape_p.u_leg_thickness_low = float(args.u_leg_thickness_low)
        if args.u_leg_thickness_high is not None:
            shape_p.u_leg_thickness_high = float(args.u_leg_thickness_high)
        if args.u_leg_height_low is not None:
            shape_p.u_leg_height_low = float(args.u_leg_height_low)
        if args.u_leg_height_high is not None:
            shape_p.u_leg_height_high = float(args.u_leg_height_high)
        if args.z_stub_height_low is not None:
            shape_p.z_stub_height_low = float(args.z_stub_height_low)
        if args.z_stub_height_high is not None:
            shape_p.z_stub_height_high = float(args.z_stub_height_high)
        if args.z_protrusion_limit is not None:
            shape_p.z_protrusion_limit = float(args.z_protrusion_limit)
        if args.o_ring_thickness_low is not None:
            shape_p.o_ring_thickness_low = float(args.o_ring_thickness_low)
        if args.o_ring_thickness_high is not None:
            shape_p.o_ring_thickness_high = float(args.o_ring_thickness_high)

        gen_footprints(
            args.out,
            args.num,
            shapes=shp,
            params=p,
            h=128,
            w=128,
            preview_count=max(0, int(args.preview_count)),
            preview_cols=max(1, int(args.preview_cols)),
            dump_params=not args.no_dump_params,
            wipe_output=args.wipe,
            debug_mode=args.debug,
        )
    elif args.mode == "roofs":
        if args.wipe:
            _safe_wipe_directory(args.out, force=args.wipe)
        gen_synth_roof_edges(args.out, args.num)

