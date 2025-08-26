from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from .pbsr import Rect, BuildingFamilyMatch
from .ridge_detection import RoofFamilyResult


@dataclass
class ProceduralRoofModel:
    footprint_regularized: List[Tuple[float, float]]  # lon,lat ordered ring
    parts: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, Optional[str]] = field(default_factory=lambda: {"geojson_url": None, "gltf_url": None})


class ProceduralRoofSynthesizer:
    """
    Build a vector model from PBSR rectangles and ridge results.

    Note: This version operates in image pixel space and expects the caller to
    map to lon/lat using an external transform when embedding into API outputs.
    """

    def rects_to_ring(self, rects: List[Rect]) -> List[Tuple[float, float]]:
        # Simple bounding rectangle ring around all parts (clockwise)
        minx = min(r.x for r in rects)
        miny = min(r.y for r in rects)
        maxx = max(r.x + r.w for r in rects)
        maxy = max(r.y + r.h for r in rects)
        return [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]

    def assemble(self, match: BuildingFamilyMatch, ridge_results: List[RoofFamilyResult], pixel_to_lonlat) -> ProceduralRoofModel:
        ring = self.rects_to_ring(match.rects)
        ring_ll = [pixel_to_lonlat(x, y) for x, y in ring]

        parts: List[Dict[str, Any]] = []
        for rect, rr in zip(match.rects, ridge_results):
            bbox_ring = [
                (rect.x, rect.y),
                (rect.x + rect.w, rect.y),
                (rect.x + rect.w, rect.y + rect.h),
                (rect.x, rect.y + rect.h),
                (rect.x, rect.y),
            ]
            bbox_ll = [pixel_to_lonlat(x, y) for x, y in bbox_ring]
            ridge_ll = [[pixel_to_lonlat(x0, y0), pixel_to_lonlat(x1, y1)] for (x0, y0), (x1, y1) in rr.ridges]
            parts.append({
                "rect_bbox": bbox_ll,
                "roof_family": rr.roof_family,
                "ridges": ridge_ll,
                "confidence": rr.confidence,
            })

        return ProceduralRoofModel(
            footprint_regularized=ring_ll,
            parts=parts,
            metrics=None,
        )

