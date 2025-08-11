from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
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

