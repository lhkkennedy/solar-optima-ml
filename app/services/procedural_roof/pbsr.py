from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math

import numpy as np


@dataclass
class Rect:
    """Axis-aligned rectangle defined in image space (x,y,w,h)."""
    x: int
    y: int
    w: int
    h: int

    def as_bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass
class BuildingFamilyMatch:
    family: str  # e.g., T11, T21, T32, T43
    rects: List[Rect]
    iou_score: float


class PBSRService:
    """
    Compact Potential Building Shapes and Roofs (PBSR) matcher.

    This scaffold exposes a deterministic configuration library and a simple
    IoU-based matcher to regularize a binary building mask into a set of up to
    four rectangular parts, following the paper's F1..F4 (topologies T11, T21,
    T32, T43). For V1 we implement a heuristic enumerator over canonical
    layouts at coarse grid resolution.
    """

    def __init__(self, grid: int = 8, max_parts: int = 4):
        self.grid = max(2, grid)
        self.max_parts = max_parts

    def _mask_to_bbox(self, mask: np.ndarray) -> Rect:
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return Rect(0, 0, mask.shape[1], mask.shape[0])
        minx, maxx = xs.min(), xs.max()
        miny, maxy = ys.min(), ys.max()
        return Rect(int(minx), int(miny), int(maxx - minx + 1), int(maxy - miny + 1))

    def _rasterize_rects(self, shape: Tuple[int, int], rects: List[Rect]) -> np.ndarray:
        h, w = shape
        canvas = np.zeros((h, w), dtype=np.uint8)
        for r in rects:
            x0, y0 = r.x, r.y
            x1, y1 = x0 + r.w, y0 + r.h
            x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
            y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
            canvas[y0:y1, x0:x1] = 1
        return canvas

    def _iou(self, a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    def _enumerate_topologies(self, bbox: Rect) -> Dict[str, List[List[Rect]]]:
        """
        Produce a small set of canonical configurations per topology within bbox.
        We generate:
          - T11: one rectangle ~ full bbox
          - T21: split horizontally and vertically (L/T variants emerge by IoU fit)
          - T32: 'U' and 'Z' like splits via one main rect plus secondary
          - T43: four quadrants (closed ring)
        """
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        gx = max(self.grid, self.grid * (w // self.grid))
        gy = max(self.grid, self.grid * (h // self.grid))

        def R(px, py, pw, ph):
            return Rect(int(px), int(py), int(max(1, pw)), int(max(1, ph)))

        configs: Dict[str, List[List[Rect]]] = {"T11": [], "T21": [], "T32": [], "T43": []}

        # T11
        configs["T11"].append([R(x, y, w, h)])

        # T21: simple bi-partition (horizontal/vertical)
        for alpha in (0.4, 0.5, 0.6):
            # vertical split
            w1 = int(w * alpha)
            configs["T21"].append([R(x, y, w1, h), R(x + w1, y, w - w1, h)])
            # horizontal split
            h1 = int(h * alpha)
            configs["T21"].append([R(x, y, w, h1), R(x, y + h1, w, h - h1)])

        # T32: 'U'/'Z' like with a main plus two smaller rects
        for alpha in (0.5, 0.6):
            # U-shape (top bar + two legs)
            hb = int(h * (1 - alpha))
            wb = w
            legw = int(w * 0.35)
            legh = int(h * alpha)
            configs["T32"].append([
                R(x, y, wb, hb),
                R(x, y + hb, legw, legh),
                R(x + w - legw, y + hb, legw, legh),
            ])
            # Z-shape (diagonal bias via three bands)
            bandw = int(w * 0.6)
            bandh = int(h * 0.35)
            configs["T32"].append([
                R(x, y, bandw, bandh),
                R(x + w - bandw, y + (h // 2 - bandh // 2), bandw, bandh),
                R(x, y + h - bandh, bandw, bandh),
            ])

        # T43: four quadrants ring
        w2 = w // 2
        h2 = h // 2
        configs["T43"].append([
            R(x, y, w2, h2), R(x + w2, y, w - w2, h2),
            R(x, y + h2, w2, h - h2), R(x + w2, y + h2, w - w2, h - h2)
        ])

        return configs

    def match(self, mask: np.ndarray) -> Optional[BuildingFamilyMatch]:
        """Return best family configuration by IoU fit to the binary mask."""
        if mask.ndim != 2:
            raise ValueError("PBSRService.match expects a 2D binary mask")
        bbox = self._mask_to_bbox(mask)
        configs = self._enumerate_topologies(bbox)
        best: Optional[BuildingFamilyMatch] = None
        for family, arr in configs.items():
            for rects in arr:
                cand = self._rasterize_rects(mask.shape, rects)
                score = self._iou(mask > 0, cand > 0)
                if best is None or score > best.iou_score:
                    best = BuildingFamilyMatch(family=family, rects=rects, iou_score=score)
        return best

