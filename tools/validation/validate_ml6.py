from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from app.services.geo_utils import to_projected
except Exception:
    # Allow running as standalone if PYTHONPATH not set
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from app.services.geo_utils import to_projected  # type: ignore


@dataclass
class Case:
    case_id: str
    latitude: float
    longitude: float
    bbox_m: float
    gt_planes: List[Dict[str, Any]]
    gt_ridges2d: List[List[List[float]]]


def load_cases(cases_dir: str) -> List[Case]:
    cases: List[Case] = []
    for p in Path(cases_dir).glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        gt = obj.get("gt", {})
        cases.append(Case(
            case_id=str(obj.get("id", p.stem)),
            latitude=float(obj["coordinates"]["latitude"]),
            longitude=float(obj["coordinates"]["longitude"]),
            bbox_m=float(obj.get("bbox_m", 60)),
            gt_planes=list(gt.get("planes", [])),
            gt_ridges2d=list(gt.get("ridges2d", [])),
        ))
    return cases


def fetch_prediction(server_url: str, lat: float, lon: float, bbox_m: float) -> Dict[str, Any]:
    url = server_url.rstrip("/") + "/model3d"
    payload = {
        "coordinates": {"latitude": lat, "longitude": lon},
        "bbox_m": bbox_m,
        "return_mesh": False,
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def _polygon_area_m2(poly_ll: List[List[float]]) -> float:
    # Rough area via projected coords in EPSG:27700 (meters)
    if len(poly_ll) < 3:
        return 0.0
    xs: List[float] = []
    ys: List[float] = []
    for lon, lat in poly_ll:
        x, y = to_projected(lat, lon, "EPSG:27700")
        xs.append(x); ys.append(y)
    area = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) * 0.5


def _segment_distance_m(a0: Tuple[float, float], a1: Tuple[float, float], b0: Tuple[float, float], b1: Tuple[float, float]) -> float:
    # Convert lon/lat to projected meters, then compute min distance between segments
    def ll_to_xy(p):
        lon, lat = p
        return to_projected(lat, lon, "EPSG:27700")
    ax0, ay0 = ll_to_xy(a0)
    ax1, ay1 = ll_to_xy(a1)
    bx0, by0 = ll_to_xy(b0)
    bx1, by1 = ll_to_xy(b1)

    def seg_pt_dist(px, py, x0, y0, x1, y1):
        vx, vy = x1 - x0, y1 - y0
        wx, wy = px - x0, py - y0
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return math.hypot(px - x0, py - y0)
        c2 = vx * vx + vy * vy
        if c2 <= c1:
            return math.hypot(px - x1, py - y1)
        t = c1 / c2
        projx, projy = x0 + t * vx, y0 + t * vy
        return math.hypot(px - projx, py - projy)

    dists = [
        seg_pt_dist(ax0, ay0, bx0, by0, bx1, by1),
        seg_pt_dist(ax1, ay1, bx0, by0, bx1, by1),
        seg_pt_dist(bx0, by0, ax0, ay0, ax1, ay1),
        seg_pt_dist(bx1, by1, ax0, ay0, ax1, ay1),
    ]
    return float(min(dists))


def _linestring_to_mask(lines: List[Tuple[Tuple[float, float], Tuple[float, float]]], bbox_ll: Tuple[float, float, float, float], size: int = 256) -> List[List[int]]:
    # Rasterize lines to a square mask for IoU computation
    min_lon, min_lat, max_lon, max_lat = bbox_ll
    mask = [[0 for _ in range(size)] for _ in range(size)]
    def to_px(p):
        lon, lat = p
        x = int(round((lon - min_lon) / (max_lon - min_lon + 1e-12) * (size - 1)))
        y = int(round((lat - min_lat) / (max_lat - min_lat + 1e-12) * (size - 1)))
        x = max(0, min(size - 1, x))
        y = max(0, min(size - 1, y))
        return x, y
    for (a, b) in lines:
        x0, y0 = to_px(a)
        x1, y1 = to_px(b)
        # Bresenham
        dx = abs(x1 - x0); dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            mask[y][x] = 1
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
    return mask


def _iou(a: List[List[int]], b: List[List[int]]) -> float:
    inter = 0
    union = 0
    H = len(a)
    W = len(a[0]) if H > 0 else 0
    for i in range(H):
        for j in range(W):
            aa = a[i][j] == 1
            bb = b[i][j] == 1
            if aa and bb:
                inter += 1
            if aa or bb:
                union += 1
    return float(inter) / float(union) if union > 0 else 0.0


def evaluate_case(gt: Case, pred: Dict[str, Any], ridge_tol_m: float = 2.0) -> Dict[str, Any]:
    # Match planes by closest area
    gt_planes = gt.gt_planes
    pred_planes = pred.get("planes", [])
    used_pred = set()
    pitch_errs: List[float] = []
    area_errs_pct: List[float] = []
    for g in gt_planes:
        g_area = float(g.get("area_m2", 0.0))
        if g_area <= 0.0:
            # fallback using polygon area
            poly = g.get("polygon", [])
            g_area = _polygon_area_m2(poly) if poly else 0.0
        best = None
        best_idx = -1
        for idx, p in enumerate(pred_planes):
            if idx in used_pred:
                continue
            a = float(p.get("area_m2", 0.0))
            diff = abs(a - g_area)
            if best is None or diff < best:
                best = diff
                best_idx = idx
        if best_idx >= 0:
            used_pred.add(best_idx)
            p = pred_planes[best_idx]
            g_pitch = float(g.get("pitch_deg", 0.0))
            p_pitch = float(p.get("pitch_deg", 0.0))
            pitch_errs.append(abs(p_pitch - g_pitch))
            if g_area > 0:
                area_errs_pct.append(100.0 * abs(float(p.get("area_m2", 0.0)) - g_area) / g_area)

    pitch_mae = sum(pitch_errs) / len(pitch_errs) if pitch_errs else None
    area_mae_pct = sum(area_errs_pct) / len(area_errs_pct) if area_errs_pct else None

    # Ridge metrics
    pred_ridges = []
    for e in pred.get("edges", []):
        a = tuple(e.get("a", [0, 0])[:2])
        b = tuple(e.get("b", [0, 0])[:2])
        pred_ridges.append((a, b))
    gt_ridges = [ (tuple(seg[0]), tuple(seg[1])) for seg in gt.gt_ridges2d ]

    matched_pred = set()
    matched_gt = set()
    for gi, (ga, gb) in enumerate(gt_ridges):
        for pi, (pa, pb) in enumerate(pred_ridges):
            if pi in matched_pred:
                continue
            d = _segment_distance_m(ga, gb, pa, pb)
            if d <= ridge_tol_m:
                matched_pred.add(pi)
                matched_gt.add(gi)
                break
    completeness = len(matched_gt) / len(gt_ridges) if gt_ridges else None
    correctness = len(matched_pred) / len(pred_ridges) if pred_ridges else None
    hit_ratio = None
    if completeness is not None and correctness is not None:
        hit_ratio = 2.0 * completeness * correctness / max(1e-6, (completeness + correctness))

    # Edge IoU using rasterized ridges
    min_lon = min([p[0] for seg in gt.gt_ridges2d for p in seg] + [pred.get("bbox", {}).get("epsg4326", [0,0,0,0])[0]])
    min_lat = min([p[1] for seg in gt.gt_ridges2d for p in seg] + [pred.get("bbox", {}).get("epsg4326", [0,0,0,0])[1]])
    max_lon = max([p[0] for seg in gt.gt_ridges2d for p in seg] + [pred.get("bbox", {}).get("epsg4326", [0,0,0,0])[2]])
    max_lat = max([p[1] for seg in gt.gt_ridges2d for p in seg] + [pred.get("bbox", {}).get("epsg4326", [0,0,0,0])[3]])
    bbox_ll = (float(min_lon), float(min_lat), float(max_lon), float(max_lat))
    mask_gt = _linestring_to_mask(gt_ridges, bbox_ll)
    mask_pr = _linestring_to_mask(pred_ridges, bbox_ll)
    edge_iou = _iou(mask_gt, mask_pr)

    return {
        "pitch_mae_deg": pitch_mae,
        "area_mae_pct": area_mae_pct,
        "ridge_completeness": completeness,
        "ridge_correctness": correctness,
        "ridge_hit_ratio": hit_ratio,
        "edge_iou": edge_iou,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="ML-6 validation harness")
    ap.add_argument("--cases_dir", required=True, help="Directory with ground-truth case JSON files")
    ap.add_argument("--server_url", default="http://localhost:8000", help="Base URL of the running API server")
    ap.add_argument("--output_json", default="ml6_report.json", help="Path to write JSON report")
    ap.add_argument("--output_csv", default="ml6_report.csv", help="Path to write CSV table")
    ap.add_argument("--ridge_tol_m", type=float, default=2.0, help="Distance threshold (meters) to match ridges")
    args = ap.parse_args()

    cases = load_cases(args.cases_dir)
    rows: List[Dict[str, Any]] = []

    for c in cases:
        try:
            pred = fetch_prediction(args.server_url, c.latitude, c.longitude, c.bbox_m)
            metrics = evaluate_case(c, pred, ridge_tol_m=args.ridge_tol_m)
        except Exception as e:
            metrics = {k: None for k in [
                "pitch_mae_deg", "area_mae_pct", "ridge_completeness", "ridge_correctness", "ridge_hit_ratio", "edge_iou"
            ]}
            metrics["error"] = str(e)

        row = {"case_id": c.case_id, **{k: (None if v is None else float(v)) for k, v in metrics.items() if k != "error"}}
        if "error" in metrics:
            row["error"] = metrics["error"]
        rows.append(row)

    # Aggregate summary
    def _avg(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    summary = {
        "pitch_mae_deg": _avg("pitch_mae_deg"),
        "area_mae_pct": _avg("area_mae_pct"),
        "ridge_completeness": _avg("ridge_completeness"),
        "ridge_correctness": _avg("ridge_correctness"),
        "ridge_hit_ratio": _avg("ridge_hit_ratio"),
        "edge_iou": _avg("edge_iou"),
        "num_cases": len(rows),
    }

    report = {"summary": summary, "cases": rows}

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["case_id"]) 
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

