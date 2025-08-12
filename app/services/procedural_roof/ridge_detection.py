from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2

from .pbsr import Rect
import os
from .classifiers import OnnxClassifier, ClassifierConfig, load_proc_roof_classifiers


@dataclass
class RoofFamilyResult:
    roof_family: str  # flat|gable|hip|pyramid|half_hip
    ridges: List[Tuple[Tuple[int, int], Tuple[int, int]]]  # ((x0,y0),(x1,y1))
    confidence: float


class RidgeDetectionService:
    """
    Simple ridge detector scaffold.

    For V1, run Canny edges and detect ridge candidates based on rectangle
    geometry and edge support. This is a deterministic placeholder until small
    ONNX classifiers are introduced.
    """

    def __init__(self, canny_low: int = 50, canny_high: int = 150, support_tol: float = 0.05, onnx_path: str | None = None, classifier: OnnxClassifier | None = None):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.support_tol = support_tol
        self.clf = classifier
        if self.clf is None and onnx_path:
            try:
                self.clf = OnnxClassifier(ClassifierConfig(onnx_path=onnx_path))
            except Exception:
                self.clf = None
        # Env-based auto-load
        if self.clf is None:
            use = str(os.getenv("PROC_ROOF_USE_CLASSIFIER", "0")).strip() in {"1", "true", "True"}
            if use:
                _, roof = load_proc_roof_classifiers()
                self.clf = roof

    def _edge_map(self, img_gray: np.ndarray) -> np.ndarray:
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        edges = cv2.Canny(img_blur, self.canny_low, self.canny_high)
        return edges

    def _line_support(self, edges: np.ndarray, p0: Tuple[int, int], p1: Tuple[int, int], tol: int) -> int:
        # Sample K points along the line, count edge pixels within +-tol (Manhattan box)
        x0, y0 = p0; x1, y1 = p1
        length = int(np.hypot(x1 - x0, y1 - y0))
        if length <= 0:
            return 0
        xs = np.linspace(x0, x1, num=length, dtype=np.int32)
        ys = np.linspace(y0, y1, num=length, dtype=np.int32)
        h, w = edges.shape
        count = 0
        for xi, yi in zip(xs, ys):
            x0b = max(0, xi - tol); x1b = min(w, xi + tol + 1)
            y0b = max(0, yi - tol); y1b = min(h, yi + tol + 1)
            if edges[y0b:y1b, x0b:x1b].any():
                count += 1
        return count

    def analyze_part(self, gray_crop: np.ndarray, rect: Rect) -> RoofFamilyResult:
        h, w = gray_crop.shape
        edges = self._edge_map(gray_crop)

        # Candidate ridge orientations: horizontal and vertical across rect center
        cx = rect.x + rect.w // 2
        cy = rect.y + rect.h // 2
        # Clamp within image
        cx = max(0, min(w - 1, cx)); cy = max(0, min(h - 1, cy))
        tol = max(1, int(self.support_tol * max(rect.w, rect.h)))

        # Horizontal ridge candidate
        p0_h = (rect.x, cy)
        p1_h = (rect.x + rect.w - 1, cy)
        s_h = self._line_support(edges, p0_h, p1_h, tol)

        # Vertical ridge candidate
        p0_v = (cx, rect.y)
        p1_v = (cx, rect.y + rect.h - 1)
        s_v = self._line_support(edges, p0_v, p1_v, tol)

        # Optional ONNX classifier path (expects NCHW float in [0,1])
        if self.clf is not None:
            crop = gray_crop[max(0, rect.y):rect.y+rect.h, max(0, rect.x):rect.x+rect.w]
            if crop.size > 0:
                inp = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                nchw = inp[None, None, :, :]
                logits = self.clf.predict_logits(nchw)
                k = int(np.argmax(logits, axis=1)[0])
                families = ["flat", "gable", "hip", "pyramid"]
                fam = families[k] if 0 <= k < len(families) else "flat"
                return RoofFamilyResult(roof_family=fam, ridges=[], confidence=0.8)

        # Heuristic family selection fallback
        ridges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        conf = 0.0
        if s_h == 0 and s_v == 0:
            family = "flat"
            conf = 0.5
        elif s_h >= s_v:
            family = "gable"
            ridges.append((p0_h, p1_h))
            conf = min(1.0, s_h / max(1, rect.w))
        else:
            family = "gable"
            ridges.append((p0_v, p1_v))
            conf = min(1.0, s_v / max(1, rect.h))

        return RoofFamilyResult(roof_family=family, ridges=ridges, confidence=float(conf))

