import math
from dataclasses import dataclass
from typing import Tuple, Dict

try:
    # Optional precise transforms; code falls back to simple meters-per-degree
    from pyproj import Transformer  # type: ignore
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False


@dataclass
class BBox4326:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


@dataclass
class BBoxProjected:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    crs: str


def to_projected(lat: float, lon: float, target_epsg: str = "EPSG:27700") -> Tuple[float, float]:
    """Convert WGS84 lat/lon to projected CRS (default: British National Grid EPSG:27700).
    Falls back to a rough meters-per-degree approximation if pyproj is unavailable.
    """
    if _HAS_PYPROJ:
        t = Transformer.from_crs("EPSG:4326", target_epsg, always_xy=True)
        x, y = t.transform(lon, lat)
        return float(x), float(y)
    # Fallback: equirectangular approx (meters)
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111_000.0
    m_per_deg_lon = 111_000.0 * math.cos(lat_rad)
    x = lon * m_per_deg_lon
    y = lat * m_per_deg_lat
    return x, y


def to_wgs84(x: float, y: float, source_epsg: str = "EPSG:27700") -> Tuple[float, float]:
    """Convert projected CRS to WGS84 lat/lon.

    When precise transforms are unavailable, use a consistent equirectangular
    approximation by first deriving latitude from Y and then deriving longitude
    from X using the cosine of that latitude. This avoids systematic longitude
    drift introduced by using a fixed reference latitude.
    """
    if _HAS_PYPROJ:
        t = Transformer.from_crs(source_epsg, "EPSG:4326", always_xy=True)
        lon, lat = t.transform(x, y)
        return float(lat), float(lon)  # keep (lat, lon)
    # Fallback: derive lat from Y, then use lat to compute meters-per-degree for lon
    m_per_deg_lat = 111_000.0
    lat = y / m_per_deg_lat
    m_per_deg_lon = 111_000.0 * math.cos(math.radians(lat))
    # Guard against extreme latitudes causing near-zero cos(lat)
    if abs(m_per_deg_lon) < 1e-6:
        m_per_deg_lon = 1e-6
    lon = x / m_per_deg_lon
    return float(lat), float(lon)


def bbox_projected_around(lat: float, lon: float, size_m: float, target_epsg: str = "EPSG:27700") -> BBoxProjected:
    """Square bbox centered on lat/lon with side length `size_m` in meters (projected CRS)."""
    cx, cy = to_projected(lat, lon, target_epsg)
    half = size_m / 2.0
    return BBoxProjected(min_x=cx - half, min_y=cy - half, max_x=cx + half, max_y=cy + half, crs=target_epsg)


def bbox_to_wgs84(bbox: BBoxProjected) -> BBox4326:
    """Convert projected bbox to WGS84 bbox (lon/lat)."""
    min_lat, min_lon = to_wgs84(bbox.min_x, bbox.min_y, bbox.crs)
    max_lat, max_lon = to_wgs84(bbox.max_x, bbox.max_y, bbox.crs)
    # Ensure ordering
    return BBox4326(min_lon=min(min_lon, max_lon), min_lat=min(min_lat, max_lat),
                    max_lon=max(min_lon, max_lon), max_lat=max(min_lat, max_lat))


def clamp_bbox_size(bbox_m: float, min_m: float = 40.0, max_m: float = 120.0) -> float:
    return float(max(min_m, min(max_m, bbox_m)))