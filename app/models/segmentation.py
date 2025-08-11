import os
from typing import Tuple, Any

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False
from PIL import Image
import numpy as np
# Optional backends
try:
    import onnxruntime as ort  # type: ignore
    _HAS_ONNX = True
except Exception:
    _HAS_ONNX = False


class SegmentationModel:
    """
    Segmentation model wrapper (SegFormer-B0 or ONNX) with graceful fallback to placeholder.
    Reads config from env:
      - SEG_BACKEND=torch|onnx (default: torch)
      - SEG_MODEL_PATH=/models/segformer-b0 (default: None -> placeholder)
    """

    def __init__(self, model_path: str | None = None):
        if _HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[attr-defined]
        else:
            self.device = None
        self.backend = os.getenv("SEG_BACKEND", "torch").lower()
        if self.backend == "torch" and not _HAS_TORCH:
            # Fall back when torch is unavailable
            self.backend = "placeholder"
        self.model_path = model_path or os.getenv("SEG_MODEL_PATH")
        self.model = None
        self.session = None  # ONNX session
        self.input_size = (256, 256)
        self._load_model(self.model_path)

    def _load_model(self, model_path: str | None = None):
        try:
            if self.backend == "onnx" and _HAS_ONNX and model_path and os.path.exists(model_path):
                # Expect a single-file ONNX; adjust I/O names accordingly in predict
                self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])  # type: ignore
                print(f"Loaded ONNX model from {model_path}")
                return
            if self.backend == "torch" and _HAS_TORCH and model_path and os.path.exists(model_path):
                # Stub: wire actual SegFormer loading here when weights are available
                print("Torch backend selected but model loading is not implemented; falling back to placeholder")
                self.model = None
                return
            print("Using placeholder segmentation model (no weights configured)")
            self.model = None
            self.session = None
        except Exception as e:
            print(f"Error loading model: {e}; using placeholder")
            self.model = None
            self.session = None

    def predict(self, image: Image.Image) -> Tuple[np.ndarray, float]:
        if self.session is not None:
            return self._predict_onnx(image)
        if self.model is not None:
            return self._predict_torch(image)
        # Placeholder
        image_resized = image.resize(self.input_size)
        image_array = np.array(image_resized)
        mask = self._generate_placeholder_mask(image_array)
        confidence = float(np.clip(0.85 + np.random.normal(0, 0.05), 0.0, 1.0))
        return mask, confidence

    def _predict_torch(self, image: Image.Image) -> Tuple[np.ndarray, float]:
        # Stub; return placeholder until a real torch model is wired
        image_resized = image.resize(self.input_size)
        image_array = np.array(image_resized)
        mask = self._generate_placeholder_mask(image_array)
        return mask, 0.8

    def _predict_onnx(self, image: Image.Image) -> Tuple[np.ndarray, float]:
        # Very generic ONNX path: assume CHW float32 feed and single-channel logits out
        img = image.resize(self.input_size)
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        x = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
        try:
            inputs = {self.session.get_inputs()[0].name: x}
            outputs = self.session.run(None, inputs)
            logits = outputs[0]
            if logits.ndim == 4 and logits.shape[1] > 1:
                # take roof class as channel 1 by convention
                probs = 1 / (1 + np.exp(-logits[:, 1]))
            else:
                probs = 1 / (1 + np.exp(-logits[:, 0]))
            mask = (probs[0] > 0.5).astype(np.float32)
            conf = float(np.clip(np.mean(probs), 0.0, 1.0))
            return mask, conf
        except Exception as e:
            print(f"ONNX inference failed: {e}; using placeholder")
            return self._predict_torch(image)

    def _generate_placeholder_mask(self, image_array: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.input_size, dtype=np.float32)
        center_x, center_y = self.input_size[0] // 2, self.input_size[1] // 2
        roof_width, roof_height = int(self.input_size[0] * 0.6), int(self.input_size[1] * 0.5)
        x1 = max(0, center_x - roof_width // 2)
        x2 = min(self.input_size[0], center_x + roof_width // 2)
        y1 = max(0, center_y - roof_height // 2)
        y2 = min(self.input_size[1], center_y + roof_height // 2)
        mask[y1:y2, x1:x2] = 1.0
        noise = np.random.normal(0, 0.1, mask.shape).astype(np.float32)
        mask = np.clip(mask + noise, 0, 1).astype(np.float32)
        return mask

    def preprocess_image(self, image: Image.Image) -> Any:
        if not _HAS_TORCH:
            return np.transpose(np.array(image.resize(self.input_size)).astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        img = image.resize(self.input_size)
        arr = np.array(img)
        tensor = torch.from_numpy(arr).float() / 255.0  # type: ignore[attr-defined]
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor