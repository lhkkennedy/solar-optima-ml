import numpy as np
import torch

from tools.segmentation.sweep_threshold_deeplab import compute_iou_binary as iou_dl
from tools.segmentation.sweep_threshold_segformer import compute_iou_binary as iou_sf


def test_iou_binary_consistency():
    tgt = torch.zeros((1, 16, 16), dtype=torch.uint8)
    tgt[:, 2:10, 2:10] = 1
    pred = tgt.clone()
    assert abs(iou_dl(pred, tgt) - 1.0) < 1e-6
    assert abs(iou_sf(pred, tgt) - 1.0) < 1e-6

    # Partial overlap
    pred2 = torch.zeros_like(tgt)
    pred2[:, 5:14, 5:14] = 1
    i1 = iou_dl(pred2, tgt)
    i2 = iou_sf(pred2, tgt)
    assert 0.0 < i1 < 1.0
    assert abs(i1 - i2) < 1e-6


