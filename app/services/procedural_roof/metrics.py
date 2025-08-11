from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FootprintMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    er: Optional[float] = None  # regularization error, optional


@dataclass
class RidgeMetrics:
    hit_ratio: float
    correctness: float
    completeness: float


@dataclass
class ProceduralMetrics:
    footprint: Optional[FootprintMetrics]
    ridges: Optional[RidgeMetrics]


def bin_stats(y_true: np.ndarray, y_pred: np.ndarray) -> FootprintMetrics:
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * prec * rec / max(1e-8, (prec + rec))
    return FootprintMetrics(accuracy=float(acc), precision=float(prec), recall=float(rec), f1=float(f1))

