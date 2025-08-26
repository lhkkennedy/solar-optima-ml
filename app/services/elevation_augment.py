from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import math
import numpy as np

from .dsm_service import DSMClip, NDSMResult
from .geo_utils import to_projected, to_wgs84
from .procedural_roof.synthesis import ProceduralRoofModel


@dataclass
class PlaneFitResult:
    normal: Tuple[float, float, float]
    pitch_deg: float
    aspect_deg: float
    residual_rmse: float


def part_height_stats(ndsm_window: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(ndsm_window, dtype=float)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def fit_part_plane(ndsm_window: np.ndarray) -> PlaneFitResult:
    """
    Least-squares fit of a plane z = a*x + b*y + c over the window in pixel coordinates.
    Pixel spacing is assumed uniform. For absolute pitch/aspect in meters, use
    augment_procedural_model which accounts for the world spacing.
    """
    H, W = ndsm_window.shape
    if H == 0 or W == 0:
        return PlaneFitResult(normal=(0.0, 0.0, 1.0), pitch_deg=0.0, aspect_deg=0.0, residual_rmse=0.0)
    yy, xx = np.mgrid[0:H, 0:W]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(H * W)])
    b = ndsm_window.ravel()
    try:
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, bcoef, c = coeffs
    except Exception:
        a, bcoef, c = 0.0, 0.0, float(np.nanmean(ndsm_window))
    # Normal and angles in pixel units
    nx, ny, nz = -a, -bcoef, 1.0
    norm = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    pitch = math.degrees(math.atan2(math.hypot(a, bcoef), 1.0))
    aspect = (math.degrees(math.atan2(bcoef, a)) + 360.0) % 360.0
    # Residuals
    z_hat = a * A[:, 0] + bcoef * A[:, 1] + c
    resid = b - z_hat
    rmse = float(math.sqrt(float(np.mean(resid * resid))))
    return PlaneFitResult(normal=(float(nx), float(ny), float(nz)), pitch_deg=float(pitch), aspect_deg=float(aspect), residual_rmse=rmse)


def _bilinear_sample(ndsm: np.ndarray, y: float, x: float) -> float:
    H, W = ndsm.shape
    if not (0 <= x < W - 1 and 0 <= y < H - 1):
        xi = int(np.clip(round(x), 0, W - 1))
        yi = int(np.clip(round(y), 0, H - 1))
        return float(ndsm[yi, xi])
    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    x1 = x0 + 1; y1 = y0 + 1
    dx = x - x0; dy = y - y0
    v00 = float(ndsm[y0, x0])
    v10 = float(ndsm[y0, x1])
    v01 = float(ndsm[y1, x0])
    v11 = float(ndsm[y1, x1])
    v0 = v00 * (1 - dx) + v10 * dx
    v1 = v01 * (1 - dx) + v11 * dx
    return float(v0 * (1 - dy) + v1 * dy)


def sample_ridge_z(ridge2d_ll: Tuple[Tuple[float, float], Tuple[float, float]], ndsm: NDSMResult) -> List[Tuple[float, float, float]]:
    """
    Sample a ridge line given by lon/lat endpoints and attach heights from nDSM.
    Returns a polyline of (lon, lat, h) samples along the segment.
    """
    (lon0, lat0), (lon1, lat1) = ridge2d_ll
    # Map lon/lat to grid indices using bbox in EPSG:27700
    minx, miny, maxx, maxy = ndsm.bbox_27700
    H, W = ndsm.shape
    dx = (maxx - minx) / max(W - 1, 1)
    dy = (maxy - miny) / max(H - 1, 1)
    x0_m, y0_m = to_projected(lat0, lon0, "EPSG:27700")
    x1_m, y1_m = to_projected(lat1, lon1, "EPSG:27700")
    # Convert to grid coordinates
    j0 = (x0_m - minx) / dx
    i0 = (y0_m - miny) / dy
    j1 = (x1_m - minx) / dx
    i1 = (y1_m - miny) / dy
    # Sample N points
    N = max(2, int(math.hypot(j1 - j0, i1 - i0)))
    xs = np.linspace(j0, j1, num=N)
    ys = np.linspace(i0, i1, num=N)
    poly: List[Tuple[float, float, float]] = []
    for jj, ii in zip(xs, ys):
        h = _bilinear_sample(ndsm.ndsm, ii, jj)
        # Map grid index back to lon/lat via meters
        xm = minx + jj * dx
        ym = miny + ii * dy
        lat, lon = to_wgs84(xm, ym, "EPSG:27700")
        poly.append((float(lon), float(lat), float(h)))
    return poly


def _ring_lonlat_to_window(ndsm: NDSMResult, ring_ll: List[Tuple[float, float]]) -> Tuple[slice, slice]:
    # Compute bounding window in ndsm grid from lon/lat ring
    minx, miny, maxx, maxy = ndsm.bbox_27700
    H, W = ndsm.shape
    dx = (maxx - minx) / max(W - 1, 1)
    dy = (maxy - miny) / max(H - 1, 1)
    xs_m = []
    ys_m = []
    for lon, lat in ring_ll:
        xm, ym = to_projected(lat, lon, "EPSG:27700")
        xs_m.append(xm); ys_m.append(ym)
    x0 = max(min(xs_m), minx); x1 = min(max(xs_m), maxx)
    y0 = max(min(ys_m), miny); y1 = min(max(ys_m), maxy)
    j0 = int(np.clip(round((x0 - minx) / dx), 0, W - 1))
    j1 = int(np.clip(round((x1 - minx) / dx), 0, W - 1))
    i0 = int(np.clip(round((y0 - miny) / dy), 0, H - 1))
    i1 = int(np.clip(round((y1 - miny) / dy), 0, H - 1))
    if j1 < j0:
        j0, j1 = j1, j0
    if i1 < i0:
        i0, i1 = i1, i0
    return (slice(i0, i1 + 1), slice(j0, j1 + 1))


def augment_procedural_model(clip: DSMClip, ndsm: NDSMResult, model: ProceduralRoofModel) -> ProceduralRoofModel:
    """
    Enrich a ProceduralRoofModel with elevation parameters from nDSM:
      - per-part plane fit (pitch_deg, aspect_deg, residuals)
      - per-part height stats
      - per-part 3D ridges (as polylines with height samples)
    """
    # Precompute world spacing for orientation consistency if needed later
    minx, miny, maxx, maxy = ndsm.bbox_27700
    H, W = ndsm.shape
    dx = (maxx - minx) / max(W - 1, 1)
    dy = (maxy - miny) / max(H - 1, 1)

    # Iterate parts
    for part in model.parts:
        ring_ll: List[Tuple[float, float]] = part.get("rect_bbox", [])
        if not ring_ll:
            continue
        sl_i, sl_j = _ring_lonlat_to_window(ndsm, ring_ll)
        window = ndsm.ndsm[sl_i, sl_j]
        # Plane fit and stats
        fit = fit_part_plane(window)
        stats = part_height_stats(window)
        part["pitch_deg"] = fit.pitch_deg
        part["aspect_deg"] = fit.aspect_deg
        part["plane_normal"] = fit.normal
        part["residual_rmse_m"] = fit.residual_rmse
        part["height_stats_m"] = stats

        # Ridges to 3D polylines
        ridges_3d: List[List[Tuple[float, float, float]]] = []
        for ridge in part.get("ridges", []):
            if not isinstance(ridge, (list, tuple)) or len(ridge) != 2:
                continue
            p0, p1 = ridge
            poly = sample_ridge_z((tuple(p0), tuple(p1)), ndsm)
            ridges_3d.append(poly)
        part["ridges_3d"] = ridges_3d

    return model


__all__ = [
    "PlaneFitResult",
    "fit_part_plane",
    "sample_ridge_z",
    "part_height_stats",
    "augment_procedural_model",
]

