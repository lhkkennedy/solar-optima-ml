import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import rasterio
    from rasterio.transform import Affine
    _HAS_RASTERIO = True
except Exception:  # pragma: no cover
    _HAS_RASTERIO = False

from app.services.dsm_service import DSMService
from app.services.dsm_index import TileRef

pytestmark = pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio not available")


def _write_tif(path: Path, arr: np.ndarray, minx: float, miny: float, res: float, crs: str = "EPSG:27700") -> None:
    h, w = arr.shape
    transform = Affine(res, 0, minx, 0, -res, miny + h * res)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr, 1)


def test_ndsm_fusion_real_raster(monkeypatch):
    tmpdir = tempfile.TemporaryDirectory()
    cache_dir = Path(tmpdir.name)
    monkeypatch.setenv("DSM_CACHE_DIR", str(cache_dir))

    svc = DSMService()

    # Create small DSM/DTM tiles aligned on a simple bbox
    # bbox (EPSG:27700)
    minx, miny, size = 532000.0, 180000.0, 100.0
    arr_dtm = np.zeros((20, 20), dtype=np.float32) + 5.0
    yy, xx = np.mgrid[0:20, 0:20]
    arr_dsm = arr_dtm + 0.1 * (xx + yy)  # gentle slope above terrain

    dsm_path = cache_dir / "dsm_test.tif"
    dtm_path = cache_dir / "dtm_test.tif"
    _write_tif(dsm_path, arr_dsm, minx, miny, size / 20)
    _write_tif(dtm_path, arr_dtm, minx, miny, size / 20)

    # Monkeypatch tile indexer to return our tile id and avoid downloads
    def tiles_for_bbox_stub(_bbox):
        return [TileRef(id="test", url_dsm="", url_dtm="", resolution_m=1.0, bbox_27700=(minx, miny, minx+size, miny+size))]

    svc.indexer.tiles_for_bbox = tiles_for_bbox_stub  # type: ignore

    # Also monkeypatch locate_and_fetch to map id->existing files
    orig_locate = svc.locate_and_fetch

    def locate_and_fetch_with_files(lat, lon, bbox_m):
        clip = orig_locate(lat, lon, bbox_m)
        # Overwrite paths with our test files
        clip.dsm_paths = [dsm_path]
        clip.dtm_paths = [dtm_path]
        return clip

    svc.locate_and_fetch = locate_and_fetch_with_files  # type: ignore

    # Use central London coords (only used for UK bounds)
    clip = svc.locate_and_fetch(51.5074, -0.1278, 60.0)
    ndsm = svc.fuse_elevation(clip, grid_size=64)
    assert ndsm.ndsm.shape == (64, 64)
    assert float(np.mean(ndsm.ndsm)) > 0.0

    tmpdir.cleanup()