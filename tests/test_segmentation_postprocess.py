import os
import numpy as np

from app.models.segmentation import SegmentationModel


def test_postprocess_removes_small_blobs(monkeypatch):
    # Configure aggressive small-blob filtering
    monkeypatch.setenv("SEG_BACKEND", "placeholder")
    monkeypatch.setenv("SEG_MIN_BLOB_FRAC", "0.05")  # 5% min area
    monkeypatch.setenv("SEG_POST_KERNEL", "3")

    m = SegmentationModel(model_path=None)

    h, w = 100, 100
    mask = np.zeros((h, w), dtype=np.float32)
    # Large blob (should remain)
    mask[10:60, 10:60] = 1.0
    # Small blob (area 3x3 = 9 < 5% of 10_000)
    mask[80:83, 80:83] = 1.0

    out = m._postprocess(mask)
    assert out.shape == mask.shape
    # Large blob preserved
    assert out[20, 20] == 1.0
    # Small blob removed
    assert float(out[81, 81]) == 0.0


def test_postprocess_is_binary(monkeypatch):
    monkeypatch.setenv("SEG_BACKEND", "placeholder")
    m = SegmentationModel(model_path=None)
    mask = np.random.rand(64, 64).astype(np.float32)
    out = m._postprocess(mask)
    assert out.dtype == np.float32
    vals = np.unique(out)
    # Binary values {0.0, 1.0}
    assert set(map(float, vals.tolist())).issubset({0.0, 1.0})


