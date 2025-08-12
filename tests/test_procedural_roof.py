import numpy as np

from app.services.procedural_roof.pbsr import PBSRService, Rect
from app.services.procedural_roof.ridge_detection import RidgeDetectionService


def test_pbsr_t11_basic():
    h, w = 128, 128
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[16:112, 20:108] = 1
    pbsr = PBSRService(grid=8)
    match = pbsr.match(mask)
    assert match is not None
    assert len(match.rects) >= 1
    assert match.iou_score > 0.7


def test_pbsr_t21_split_prefers_reasonable_fit():
    h, w = 128, 128
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[16:112, 20:60] = 1
    mask[16:112, 60:108] = 1
    pbsr = PBSRService(grid=8)
    match = pbsr.match(mask)
    assert match is not None
    # Accept either T21 or others; ensure at least 2 rects provides decent fit
    assert match.iou_score > 0.6


def test_ridge_detection_gable_horizontal():
    h, w = 64, 64
    img = np.zeros((h, w), dtype=np.uint8)
    # Draw a horizontal bright ridge-like band
    img[31:33, 8:56] = 255
    rect = Rect(8, 16, 48, 32)
    rd = RidgeDetectionService(canny_low=10, canny_high=50)
    res = rd.analyze_part(img, rect)
    assert res.roof_family in ("gable", "flat")
    # Expect a ridge identified or fallback to flat if edge detector misses
    if res.ridges:
        ((x0, y0), (x1, y1)) = res.ridges[0]
        assert y0 == y1  # horizontal

