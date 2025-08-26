import numpy as np

from app.services.elevation_augment import fit_part_plane, part_height_stats, sample_ridge_z
from app.services.dsm_service import NDSMResult


def test_fit_part_plane_and_stats():
    ndsm = np.zeros((32, 32), dtype=np.float32)
    for i in range(32):
        ndsm[i, :] = i * 0.1
    fit = fit_part_plane(ndsm)
    stats = part_height_stats(ndsm)
    assert fit.pitch_deg >= 0.0
    assert stats["max"] > stats["min"]


def test_sample_ridge_z_basic():
    arr = np.zeros((16, 16), dtype=np.float32)
    for j in range(16):
        arr[:, j] = j * 0.2
    ndsm = NDSMResult(ndsm=arr, resolution_m=1.0, shape=arr.shape, bbox_27700=(0.0, 0.0, 15.0, 15.0))
    # Endpoints in lon/lat derived from bbox corners treated as projected back via to_wgs84 in the function
    # Here we just ensure it runs and returns >=2 points
    # Using fake lon/lat: function will project and clip; this is a smoke test only
    poly = sample_ridge_z(((0.0, 0.0), (0.001, 0.001)), ndsm)
    assert isinstance(poly, list)
    assert len(poly) >= 2

