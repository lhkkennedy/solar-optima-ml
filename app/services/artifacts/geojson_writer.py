from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.services.plane_fitting import Plane, Edge
from app.services.procedural_roof.synthesis import ProceduralRoofModel


def _ensure_ring_closed(ring: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not ring:
        return ring
    if ring[0] != ring[-1]:
        return [*ring, ring[0]]
    return ring


def _valid_lon_lat(lon: float, lat: float) -> bool:
    return -180.0 <= float(lon) <= 180.0 and -90.0 <= float(lat) <= 90.0


def _clip_lon_lat(lon: float, lat: float) -> Tuple[float, float]:
    lon_c = float(min(180.0, max(-180.0, lon)))
    lat_c = float(min(90.0, max(-90.0, lat)))
    return lon_c, lat_c


def _polygon_feature_2d(ring_ll: List[Tuple[float, float]], props: Dict[str, Any]) -> Dict[str, Any]:
    ring_ll = [_clip_lon_lat(lon, lat) for lon, lat in ring_ll]
    ring_ll = _ensure_ring_closed(ring_ll)
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[list(coord) for coord in ring_ll]]},
        "properties": props,
    }


def _polygon_feature_3d(ring_llz: List[Tuple[float, float, float]], props: Dict[str, Any]) -> Dict[str, Any]:
    coords = []
    for lon, lat, z in ring_llz:
        lon_c, lat_c = _clip_lon_lat(lon, lat)
        coords.append([lon_c, lat_c, float(z)])
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": props,
    }


def _line_feature_2d(a: Tuple[float, float], b: Tuple[float, float], props: Dict[str, Any]) -> Dict[str, Any]:
    a_c = _clip_lon_lat(*a)
    b_c = _clip_lon_lat(*b)
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[a_c[0], a_c[1]], [b_c[0], b_c[1]]]},
        "properties": props,
    }


def _line_feature_3d(poly_llz: List[Tuple[float, float, float]], props: Dict[str, Any]) -> Dict[str, Any]:
    coords = []
    for lon, lat, z in poly_llz:
        lon_c, lat_c = _clip_lon_lat(lon, lat)
        coords.append([lon_c, lat_c, float(z)])
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": props,
    }


def model_to_feature_collection(model: ProceduralRoofModel,
                                planes: Optional[List[Plane]] = None,
                                edges: Optional[List[Edge]] = None) -> Dict[str, Any]:
    feats: List[Dict[str, Any]] = []

    # Footprint (2D polygon)
    if model.footprint_regularized:
        feats.append(_polygon_feature_2d(model.footprint_regularized, {
            "kind": "footprint",
        }))

    # Parts: rects (2D polygons) and ridges
    for idx, part in enumerate(model.parts):
        ring = part.get("rect_bbox", [])
        if ring:
            props = {
                "kind": "part",
                "index": idx,
                "roof_family": part.get("roof_family"),
                "confidence": float(part.get("confidence", 0.0)),
            }
            if "pitch_deg" in part:
                props["pitch_deg"] = float(part["pitch_deg"])  # type: ignore[index]
            if "aspect_deg" in part:
                props["aspect_deg"] = float(part["aspect_deg"])  # type: ignore[index]
            if "residual_rmse_m" in part:
                props["residual_rmse_m"] = float(part["residual_rmse_m"])  # type: ignore[index]
            feats.append(_polygon_feature_2d(ring, props))

        # Ridges 2D
        for ridx, ridge in enumerate(part.get("ridges", [])):
            if isinstance(ridge, (list, tuple)) and len(ridge) == 2:
                a, b = ridge
                feats.append(_line_feature_2d(tuple(a), tuple(b), {"kind": "ridge2d", "part": idx, "index": ridx}))

        # Ridges 3D
        for ridx, ridge3d in enumerate(part.get("ridges_3d", [])):
            if isinstance(ridge3d, (list, tuple)) and len(ridge3d) >= 2:
                feats.append(_line_feature_3d([tuple(p) for p in ridge3d], {"kind": "ridge3d", "part": idx, "index": ridx}))

    # Planes (3D polygons) if available
    if planes:
        for p in planes:
            props = {
                "kind": "plane",
                "id": p.id,
                "pitch_deg": float(p.pitch_deg),
                "aspect_deg": float(p.aspect_deg),
                "area_m2": float(p.area_m2),
                "normal": list(p.normal),
            }
            feats.append(_polygon_feature_3d([tuple(v) for v in p.polygon], props))

    # Edges (3D lines) if available
    if edges:
        for e in edges:
            feats.append(_line_feature_3d([tuple(e.a), tuple(e.b)], {"kind": "edge"}))

    return {"type": "FeatureCollection", "features": feats}


def write_geojson(model: ProceduralRoofModel,
                  planes: Optional[List[Plane]] = None,
                  edges: Optional[List[Edge]] = None,
                  out_dir: Optional[str] = None,
                  filename: Optional[str] = None) -> str:
    """
    Write a FeatureCollection GeoJSON for the given procedural model and optional planes/edges.
    Returns the local path to the written file.
    """
    out = model_to_feature_collection(model, planes=planes, edges=edges)
    base_dir = Path(out_dir or os.getenv("ARTIFACT_DIR", "./artifacts"))
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = filename or f"roof_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    path = base_dir / f"{stem}.geojson"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return str(path)


__all__ = ["write_geojson", "model_to_feature_collection"]

