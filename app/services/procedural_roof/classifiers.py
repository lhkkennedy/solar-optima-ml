from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None  # runtime optional


@dataclass
class ClassifierConfig:
    onnx_path: str
    input_name: Optional[str] = None
    output_name: Optional[str] = None


class OnnxClassifier:
    def __init__(self, cfg: ClassifierConfig):
        if ort is None:
            raise RuntimeError("onnxruntime not available")
        if not os.path.exists(cfg.onnx_path):
            raise FileNotFoundError(cfg.onnx_path)
        self.session = ort.InferenceSession(cfg.onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])  # prefers GPU
        self.input_name = cfg.input_name or self.session.get_inputs()[0].name
        self.output_name = cfg.output_name or self.session.get_outputs()[0].name

    def predict_logits(self, nchw_float: np.ndarray) -> np.ndarray:
        out = self.session.run([self.output_name], {self.input_name: nchw_float})[0]
        return out  # (N,C)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    expx = np.exp(x)
    return expx / np.sum(expx, axis=axis, keepdims=True)


def load_proc_roof_classifiers() -> Tuple[Optional[OnnxClassifier], Optional[OnnxClassifier]]:
    """
    Load optional PBSR family and roof family classifiers based on env vars.
    - PROC_ROOF_USE_CLASSIFIER=1 enables attempts
    - PROC_ROOF_ONNX_DIR sets base directory (default /models)
    Expected filenames:
      - family: family_classifier.onnx
      - roof: roof_family.onnx
    Returns (family_classifier, roof_classifier)
    """
    use = str(os.getenv("PROC_ROOF_USE_CLASSIFIER", "0")).strip() in {"1", "true", "True"}
    if not use or ort is None:
        return None, None
    base = os.getenv("PROC_ROOF_ONNX_DIR", "/models")
    fam_path = os.path.join(base, "family_classifier.onnx")
    roof_path = os.path.join(base, "roof_family.onnx")
    fam = None
    roo = None
    try:
        if os.path.exists(fam_path):
            fam = OnnxClassifier(ClassifierConfig(onnx_path=fam_path))
    except Exception:
        fam = None
    try:
        if os.path.exists(roof_path):
            roo = OnnxClassifier(ClassifierConfig(onnx_path=roof_path))
    except Exception:
        roo = None
    return fam, roo


__all__ = [
    "ClassifierConfig",
    "OnnxClassifier",
    "load_proc_roof_classifiers",
]

