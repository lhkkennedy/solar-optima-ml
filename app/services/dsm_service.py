import os
import tempfile
import numpy as np
import requests
from typing import Tuple, Optional, List
import logging
from dataclasses import dataclass
from pathlib import Path
import hashlib

from app.services.geo_utils import clamp_bbox_size, bbox_projected_around, bbox_to_wgs84
from app.services.dsm_index import DsmIndexer, TileRef
from app.services.footprint_service import FootprintService, Footprint

logger = logging.getLogger(__name__)

try:
    import rasterio  # type: ignore
    from rasterio.merge import merge as rio_merge  # type: ignore
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False

@dataclass
class DSMData:
    """DSM data structure for a location"""
    latitude: float
    longitude: float
    elevation_m: float
    resolution_m: float
    data_source: str
    confidence: float

@dataclass
class DSMClip:
    bbox_27700: Tuple[float, float, float, float]
    bbox_4326: Tuple[float, float, float, float]
    resolution_m: float
    dsm_paths: List[Path]
    dtm_paths: List[Path]
    footprint: Optional[Footprint]

@dataclass
class NDSMResult:
    ndsm: np.ndarray
    resolution_m: float
    shape: Tuple[int, int]
    bbox_27700: Tuple[float, float, float, float]

class DSMService:
    """
    Service for accessing UK LIDAR DSM (Digital Surface Model) data
    Uses Environment Agency LIDAR data for roof height information
    """
    
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self._cache = {}
        self.base_url = os.getenv("EA_DSM_BASE_URL", "https://environment.data.gov.uk/ds/survey")
        self.uk_bounds = {"min_lat": 49.9, "max_lat": 60.9, "min_lon": -8.6, "max_lon": 1.8}
        # Step‑1 additions
        # Determine writable cache dir
        default_cache = os.getenv("DSM_CACHE_DIR")
        if not default_cache:
            default_cache = os.path.join(tempfile.gettempdir(), "dsm_cache")
        self.dsm_cache_dir = Path(default_cache)
        try:
            self.dsm_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback to local project-relative cache
            self.dsm_cache_dir = Path("./.dsm_cache")
            self.dsm_cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_resolution_m = 1.0
        self.indexer = DsmIndexer(self.base_url, resolution_m=self.default_resolution_m)
        self.footprints = FootprintService()
        # WCS endpoints (optional)
        self.wcs_dsm = os.getenv("EA_WCS_DSM", "https://environment.data.gov.uk/spatialdata/lidar-composite-dsm-1m/wcs")
        self.wcs_dtm = os.getenv("EA_WCS_DTM", "https://environment.data.gov.uk/spatialdata/lidar-composite-dtm-1m/wcs")
        self.layer_dsm = os.getenv("EA_LAYER_DSM", "lidar-composite-dsm-1m")
        self.layer_dtm = os.getenv("EA_LAYER_DTM", "lidar-composite-dtm-1m")
    
    def is_within_uk(self, latitude: float, longitude: float) -> bool:
        return (self.uk_bounds["min_lat"] <= latitude <= self.uk_bounds["max_lat"] and
                self.uk_bounds["min_lon"] <= longitude <= self.uk_bounds["max_lon"])    

    def get_cache_key(self, latitude: float, longitude: float) -> str:
        lat_rounded = round(latitude, 3)
        lon_rounded = round(longitude, 3)
        return f"{lat_rounded}_{lon_rounded}"

    # === ML-6 step 1: locate + fetch ===
    def locate_and_fetch(self, latitude: float, longitude: float, bbox_m: float = 60.0) -> Optional[DSMClip]:
        if not self.is_within_uk(latitude, longitude):
            logger.warning("Coordinates outside UK bounds")
            return None
        bbox_m = clamp_bbox_size(bbox_m)
        bbox_proj = bbox_projected_around(latitude, longitude, bbox_m)
        bbox_wgs = bbox_to_wgs84(bbox_proj)
        bbox_27700_tuple = (bbox_proj.min_x, bbox_proj.min_y, bbox_proj.max_x, bbox_proj.max_y)
        bbox_4326_tuple = (bbox_wgs.min_lon, bbox_wgs.min_lat, bbox_wgs.max_lon, bbox_wgs.max_lat)
        # Determine tiles
        tiles: List[TileRef] = self.indexer.tiles_for_bbox(bbox_27700_tuple)
        # Fetch/cache tiles (try HTTP download, fallback to placeholder touch)
        dsm_paths: List[Path] = []
        dtm_paths: List[Path] = []
        if tiles:
            for t in tiles:
                dsm_path = self.dsm_cache_dir / f"dsm_{t.id}.tif"
                dtm_path = self.dsm_cache_dir / f"dtm_{t.id}.tif"
                if not dsm_path.exists():
                    self._download_to(t.url_dsm, dsm_path)
                if not dtm_path.exists() and t.url_dtm:
                    self._download_to(t.url_dtm, dtm_path)
                if not dsm_path.exists():
                    dsm_path.touch()
                if not dtm_path.exists():
                    dtm_path.touch()
                dsm_paths.append(dsm_path)
                dtm_paths.append(dtm_path)
        else:
            # No index tiles: try WCS on-demand for this bbox
            dsm_path = self._fetch_wcs("dsm", bbox_27700_tuple)
            dtm_path = self._fetch_wcs("dtm", bbox_27700_tuple)
            if dsm_path is None or dtm_path is None:
                # Create placeholders to keep pipeline running; fusion will fall back
                dsm_path = self.dsm_cache_dir / f"wcs_dsm_{self._bbox_key(bbox_27700_tuple)}.tif"
                dtm_path = self.dsm_cache_dir / f"wcs_dtm_{self._bbox_key(bbox_27700_tuple)}.tif"
                dsm_path.touch()
                dtm_path.touch()
            dsm_paths.append(dsm_path)
            dtm_paths.append(dtm_path)
        # Optional footprint
        fp = self.footprints.get_building_polygon(*bbox_4326_tuple)
        return DSMClip(bbox_27700=bbox_27700_tuple, bbox_4326=bbox_4326_tuple,
                       resolution_m=self.default_resolution_m, dsm_paths=dsm_paths,
                       dtm_paths=dtm_paths, footprint=fp)

    # === ML-6 step 3: elevation fusion (nDSM) ===
    def fuse_elevation(self, clip: DSMClip, grid_size: int = 128) -> NDSMResult:
        """Compute nDSM (DSM − DTM) over the clipped area.
        If rasterio is unavailable or tiles are placeholders, generate a synthetic slope.
        """
        # Derived cache check
        cached = self._load_ndsm_cache(clip, grid_size)
        if cached is not None:
            return cached
        width_m = clip.bbox_27700[2] - clip.bbox_27700[0]
        height_m = clip.bbox_27700[3] - clip.bbox_27700[1]
        if _HAS_RASTERIO:
            try:
                dsm_arrays = []
                dsm_transforms = []
                for p in clip.dsm_paths:
                    if p.exists() and p.stat().st_size > 0:
                        src = rasterio.open(p)  # may fail on placeholder
                        dsm_arrays.append(src.read(1))
                        dsm_transforms.append(src.transform)
                        src.close()
                dtm_arrays = []
                dtm_transforms = []
                for p in clip.dtm_paths:
                    if p.exists() and p.stat().st_size > 0:
                        src = rasterio.open(p)
                        dtm_arrays.append(src.read(1))
                        dtm_transforms.append(src.transform)
                        src.close()
                if dsm_arrays and dtm_arrays:
                    dsm_merged, _ = rio_merge(dsm_arrays)
                    dtm_merged, _ = rio_merge(dtm_arrays)
                    ndsm_full = dsm_merged[0] - dtm_merged[0]
                    # Simple clip to grid_size
                    ndsm = _resize_to_grid(ndsm_full, grid_size, grid_size)
                    ndsm = np.clip(ndsm, 0, None)
                    out = NDSMResult(ndsm=ndsm, resolution_m=clip.resolution_m, shape=ndsm.shape, bbox_27700=clip.bbox_27700)
                    self._save_ndsm_cache(clip, grid_size, out)
                    return out
            except Exception as e:
                logger.debug(f"fuse_elevation raster path failed, falling back: {e}")
        # Fallback synthetic slope
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        xx, yy = np.meshgrid(x, y)
        ndsm = 3.0 * (xx + 0.5 * yy)  # meters
        out = NDSMResult(ndsm=ndsm.astype(np.float32), resolution_m=clip.resolution_m, shape=ndsm.shape, bbox_27700=clip.bbox_27700)
        self._save_ndsm_cache(clip, grid_size, out)
        return out

    # === legacy/placeholder methods kept for ML-2 ===
    def get_dsm_data(self, latitude: float, longitude: float) -> Optional[DSMData]:
        if not self.is_within_uk(latitude, longitude):
            logger.warning(f"Coordinates ({latitude}, {longitude}) outside UK bounds")
            return None
        cache_key = self.get_cache_key(latitude, longitude)
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        dsm_data = self._get_placeholder_dsm_data(latitude, longitude)
        if self.cache_enabled and dsm_data:
            self._cache[cache_key] = dsm_data
        return dsm_data

    def _get_placeholder_dsm_data(self, latitude: float, longitude: float) -> DSMData:
        base_elevation = 50.0
        lat_factor = (latitude - 50.0) * 100
        random_factor = np.random.normal(0, 20)
        elevation = max(0, base_elevation + lat_factor + random_factor)
        confidence = float(np.clip(0.85 + np.random.normal(0, 0.1), 0.0, 1.0))
        return DSMData(
            latitude=latitude,
            longitude=longitude,
            elevation_m=float(elevation),
            resolution_m=self.default_resolution_m,
            data_source="Environment Agency LIDAR (placeholder)",
            confidence=confidence,
        )

    def get_height_profile(self, latitude: float, longitude: float, radius_m: float = 50.0, points: int = 9) -> Optional[np.ndarray]:
        if not self.is_within_uk(latitude, longitude):
            return None
        lat_offset = radius_m / 111000.0
        lon_offset = radius_m / (111000.0 * np.cos(np.radians(latitude)))
        heights = np.zeros((points, points))
        for i in range(points):
            for j in range(points):
                lat_i = latitude + (i - points//2) * lat_offset / (points//2)
                lon_j = longitude + (j - points//2) * lon_offset / (points//2)
                d = self.get_dsm_data(lat_i, lon_j)
                if d:
                    heights[i, j] = d.elevation_m
                else:
                    c = self.get_dsm_data(latitude, longitude)
                    heights[i, j] = c.elevation_m if c else 0.0
        return heights


def _resize_to_grid(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Lightweight resize using numpy (nearest neighbor)."""
    sh, sw = arr.shape
    y_idx = (np.linspace(0, sh - 1, h)).astype(int)
    x_idx = (np.linspace(0, sw - 1, w)).astype(int)
    return arr[y_idx][:, x_idx] 

    
def _bbox_key(bbox: Tuple[float, float, float, float]) -> str:
    m = hashlib.md5()
    m.update(str([round(v, 2) for v in bbox]).encode("utf-8"))
    return m.hexdigest()[:12]


DSMService._bbox_key = staticmethod(_bbox_key)  # type: ignore


def _wcs_query(url: str, layer: str, bbox: Tuple[float, float, float, float], res_m: float) -> str:
    minx, miny, maxx, maxy = bbox
    params = {
        "SERVICE": "WCS",
        "REQUEST": "GetCoverage",
        "VERSION": "1.0.0",
        "COVERAGE": layer,
        "CRS": "EPSG:27700",
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "RESX": f"{res_m}m",
        "RESY": f"{res_m}m",
        "FORMAT": "GeoTIFF",
    }
    return url + "?" + "&".join(f"{k}={v}" for k, v in params.items())


def _download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 15):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False


def _valid_gtiff(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 4096
    except Exception:
        return False


def _safe_unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def _fetch_wcs(self: 'DSMService', kind: str, bbox: Tuple[float, float, float, float]) -> Optional[Path]:
    url_base = self.wcs_dsm if kind == "dsm" else self.wcs_dtm
    layer = self.layer_dsm if kind == "dsm" else self.layer_dtm
    url = _wcs_query(url_base, layer, bbox, self.default_resolution_m)
    dest = self.dsm_cache_dir / f"wcs_{kind}_{self._bbox_key(bbox)}.tif"
    ok = _download_file(url, dest)
    if not ok or not _valid_gtiff(dest):
        _safe_unlink(dest)
        logger.debug(f"WCS fetch failed for {kind} bbox {bbox}")
        return None
    return dest


def _ndsm_cache_path(self: 'DSMService', clip: DSMClip, grid_size: int) -> Path:
    key = f"ndsm_{self._bbox_key(clip.bbox_27700)}_{grid_size}"
    return self.dsm_cache_dir / f"{key}.npy"


def _load_ndsm_cache(self: 'DSMService', clip: DSMClip, grid_size: int) -> Optional[NDSMResult]:
    p = _ndsm_cache_path(self, clip, grid_size)
    try:
        if p.exists():
            arr = np.load(p)
            return NDSMResult(ndsm=arr, resolution_m=clip.resolution_m, shape=arr.shape, bbox_27700=clip.bbox_27700)
    except Exception:
        pass
    return None


def _save_ndsm_cache(self: 'DSMService', clip: DSMClip, grid_size: int, out: NDSMResult) -> None:
    p = _ndsm_cache_path(self, clip, grid_size)
    try:
        np.save(p, out.ndsm.astype(np.float32))
    except Exception:
        pass


DSMService._fetch_wcs = _fetch_wcs  # type: ignore
DSMService._ndsm_cache_path = _ndsm_cache_path  # type: ignore
DSMService._load_ndsm_cache = _load_ndsm_cache  # type: ignore
DSMService._save_ndsm_cache = _save_ndsm_cache  # type: ignore

    
def _stream_download(url: str, dest: Path, timeout: int = 20) -> bool:
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False


def _is_valid_file(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 1024  # >1KB heuristic
    except Exception:
        return False


def _safe_unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


# attach as method for clarity but defined after helpers for readability
def _download_to(self, url: str, dest: Path) -> None:
    ok = _stream_download(url, dest)
    if not ok or not _is_valid_file(dest):
        _safe_unlink(dest)
        logger.debug(f"Download failed or invalid for {url}; will use placeholder")

# bind helper as instance method
DSMService._download_to = _download_to  # type: ignore 