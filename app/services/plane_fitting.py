from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np

from app.services.dsm_service import DSMClip, NDSMResult
from app.services.geo_utils import to_wgs84


@dataclass
class Plane:
    id: str
    normal: Tuple[float, float, float]
    pitch_deg: float
    aspect_deg: float
    polygon: List[Tuple[float, float, float]]  # (lon, lat, h)
    area_m2: float


@dataclass
class Edge:
    a: Tuple[float, float, float]
    b: Tuple[float, float, float]


class PlaneFitting:
    """Planar decomposition utilities.
    Provides RANSAC-based multi‑plane extraction; falls back to simple least‑squares split.
    """

    # --- Simple synthetic fit from clip (legacy placeholder) ---
    def fit(self, clip: DSMClip) -> Tuple[List[Plane], List[Edge]]:
        min_lon, min_lat, max_lon, max_lat = clip.bbox_4326
        width_m = clip.bbox_27700[2] - clip.bbox_27700[0]
        height_m = clip.bbox_27700[3] - clip.bbox_27700[1]
        area_m2 = float(width_m * height_m)
        h_base = 10.0
        mid_lon = (min_lon + max_lon) / 2.0
        mid_lat = (min_lat + max_lat) / 2.0
        pitch = 25.0
        p1_poly = [
            (min_lon, min_lat, h_base),
            (max_lon, min_lat, h_base + 1.5),
            (mid_lon, max_lat, h_base + 3.0),
            (min_lon, max_lat, h_base + 1.5),
        ]
        p2_poly = [
            (min_lon, min_lat, h_base + 1.5),
            (max_lon, min_lat, h_base),
            (max_lon, max_lat, h_base + 1.5),
            (mid_lon, max_lat, h_base + 3.0),
        ]
        planes = [
            Plane(id="p1", normal=(0.1, -0.9, 0.4), pitch_deg=pitch, aspect_deg=180.0, polygon=p1_poly, area_m2=area_m2/2.0),
            Plane(id="p2", normal=(-0.1, 0.9, 0.4), pitch_deg=pitch, aspect_deg=0.0, polygon=p2_poly, area_m2=area_m2/2.0),
        ]
        edges = [Edge(a=p1_poly[-1], b=p2_poly[-1])]
        return planes, edges

    # --- Least‑squares split fit from nDSM ---
    def fit_from_ndsm(self, clip: DSMClip, ndsm: NDSMResult) -> Tuple[List[Plane], List[Edge]]:
        try:
            planes, edges = self._ransac_planes(clip, ndsm)
            if planes:
                return planes, edges
        except Exception:
            pass
        # fallback: two‑half LSQ
        return self._lsq_split(clip, ndsm)

    def _lsq_split(self, clip: DSMClip, ndsm: NDSMResult) -> Tuple[List[Plane], List[Edge]]:
        minx, miny, maxx, maxy = clip.bbox_27700
        H, W = ndsm.ndsm.shape
        dx = (maxx - minx) / max(W - 1, 1)
        dy = (maxy - miny) / max(H - 1, 1)
        mid_row = H // 2
        planes: List[Plane] = []
        for pid, (r0, r1) in enumerate([(0, mid_row), (mid_row, H)]):
            zz = ndsm.ndsm[r0:r1, :]
            yy, xx = np.mgrid[r0:r1, 0:W]
            X = minx + xx * dx
            Y = miny + yy * dy
            A = np.column_stack([X.ravel(), Y.ravel(), np.ones(X.size)])
            b = zz.ravel()
            try:
                coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
                a, bcoef, c = coeffs
            except Exception:
                a, bcoef, c = 0.0, 0.0, float(np.mean(zz))
            nx, ny, nz = -a, -bcoef, 1.0
            norm = math.sqrt(nx*nx + ny*ny + nz*nz) or 1.0
            nx, ny, nz = nx/norm, ny/norm, nz/norm
            pitch = math.degrees(math.atan2(math.hypot(a, bcoef), 1.0))
            aspect = (math.degrees(math.atan2(bcoef, a)) + 360.0) % 360.0
            x0, x1 = minx, maxx
            y0, y1 = (miny + r0*dy, miny + (r1-1)*dy)
            corners_xy = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            poly_llh: List[Tuple[float, float, float]] = []
            for (x, y) in corners_xy:
                z = a*x + bcoef*y + c
                lat, lon = to_wgs84(x, y)
                poly_llh.append((lon, lat, float(z)))
            area_m2 = float((x1 - x0) * (y1 - y0))
            planes.append(Plane(id=f"p{pid+1}", normal=(nx, ny, nz), pitch_deg=pitch, aspect_deg=aspect, polygon=poly_llh, area_m2=area_m2))
        y_mid = miny + (mid_row * dy)
        a_pt = to_wgs84(minx, y_mid)
        b_pt = to_wgs84(maxx, y_mid)
        edges = [Edge(a=(a_pt[1], a_pt[0], planes[0].polygon[0][2]), b=(b_pt[1], b_pt[0], planes[0].polygon[1][2]))]
        return planes, edges

    def _ransac_planes(self, clip: DSMClip, ndsm: NDSMResult, max_planes: int = 3, thresh_m: float = 0.25,
                        min_support: int = 500, max_iter: int = 500) -> Tuple[List[Plane], List[Edge]]:
        minx, miny, maxx, maxy = clip.bbox_27700
        H, W = ndsm.ndsm.shape
        dx = (maxx - minx) / max(W - 1, 1)
        dy = (maxy - miny) / max(H - 1, 1)
        yy, xx = np.mgrid[0:H, 0:W]
        X = (minx + xx * dx).ravel()
        Y = (miny + yy * dy).ravel()
        Z = ndsm.ndsm.ravel()
        used = np.zeros_like(Z, dtype=bool)
        planes: List[Plane] = []
        for pid in range(max_planes):
            best_inliers = None
            best_coeffs = None
            for _ in range(max_iter):
                idx = np.random.choice(np.where(~used)[0], size=3, replace=False)
                x3, y3, z3 = X[idx], Y[idx], Z[idx]
                # Fit plane through 3 points
                A = np.column_stack([x3, y3, np.ones(3)])
                try:
                    coeffs = np.linalg.lstsq(A, z3, rcond=None)[0]
                except Exception:
                    continue
                a, bcoef, c = coeffs
                # Residuals on unused points
                z_hat = a*X + bcoef*Y + c
                resid = np.abs(z_hat - Z)
                inliers = (~used) & (resid < thresh_m)
                support = int(np.sum(inliers))
                if support > (int(np.sum(best_inliers)) if best_inliers is not None else 0):
                    best_inliers = inliers
                    best_coeffs = (a, bcoef, c)
            if best_inliers is None or int(np.sum(best_inliers)) < min_support:
                break
            used |= best_inliers
            a, bcoef, c = best_coeffs  # type: ignore
            # Normal, pitch, aspect
            nx, ny, nz = -a, -bcoef, 1.0
            norm = math.sqrt(nx*nx + ny*ny + nz*nz) or 1.0
            nx, ny, nz = nx/norm, ny/norm, nz/norm
            pitch = math.degrees(math.atan2(math.hypot(a, bcoef), 1.0))
            aspect = (math.degrees(math.atan2(bcoef, a)) + 360.0) % 360.0
            # Bounding rectangle of inliers
            x_in = X[best_inliers]
            y_in = Y[best_inliers]
            x0, x1 = float(np.min(x_in)), float(np.max(x_in))
            y0, y1 = float(np.min(y_in)), float(np.max(y_in))
            corners_xy = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            poly_llh: List[Tuple[float, float, float]] = []
            for (x, y) in corners_xy:
                z = a*x + bcoef*y + c
                lat, lon = to_wgs84(x, y)
                poly_llh.append((lon, lat, float(z)))
            area_m2 = float((x1 - x0) * (y1 - y0))
            planes.append(Plane(id=f"p{pid+1}", normal=(nx, ny, nz), pitch_deg=pitch, aspect_deg=aspect, polygon=poly_llh, area_m2=area_m2))
        # Edges: connect centroids of first two planes if available
        edges: List[Edge] = []
        if len(planes) >= 2:
            def centroid(p: Plane) -> Tuple[float, float, float]:
                xs = [v[0] for v in p.polygon]
                ys = [v[1] for v in p.polygon]
                zs = [v[2] for v in p.polygon]
                return (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
            c1 = centroid(planes[0])
            c2 = centroid(planes[1])
            edges.append(Edge(a=c1, b=c2))
        return planes, edges