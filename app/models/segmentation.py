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
import cv2
# Optional transformers (SegFormer)
try:
    from transformers import SegformerForSemanticSegmentation  # type: ignore
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False
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
        # Input size configurable (use 512 for SegFormer runs)
        try:
            inp = int(os.getenv("SEG_INPUT_SIZE", "256"))
        except Exception:
            inp = 256
        self.input_size = (inp, inp)
        # Inference options
        try:
            self.thresh = float(os.getenv("SEG_THRESH", "0.5"))
        except Exception:
            self.thresh = 0.5
        try:
            self.post_kernel = int(os.getenv("SEG_POST_KERNEL", "3"))
        except Exception:
            self.post_kernel = 3
        try:
            self.min_blob_frac = float(os.getenv("SEG_MIN_BLOB_FRAC", "0.002"))  # 0.2% of pixels
        except Exception:
            self.min_blob_frac = 0.002
        self._load_model(self.model_path)

    def _load_model(self, model_path: str | None = None):
        try:
            if self.backend == "onnx" and _HAS_ONNX and model_path and os.path.exists(model_path):
                # Expect a single-file ONNX; adjust I/O names accordingly in predict
                self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])  # type: ignore
                print(f"Loaded ONNX model from {model_path}")
                return
            if self.backend == "torch" and _HAS_TORCH and model_path and os.path.exists(model_path):
                # Expect a DeepLabv3-ResNet50 fine-tuned checkpoint with 2 classes (background, roof)
                from torchvision.models.segmentation import deeplabv3_resnet50  # type: ignore
                model = deeplabv3_resnet50(weights=None, num_classes=2)
                sd = torch.load(model_path, map_location="cpu")  # type: ignore[attr-defined]
                model.load_state_dict(sd, strict=False)
                model.eval()
                self.model = model
                print(f"Loaded Torch segmentation checkpoint from {model_path}")
                return
            if self.backend == "segformer" and _HAS_TRANSFORMERS and _HAS_TORCH and model_path and os.path.exists(model_path):
                # Initialize SegFormer-B0 head for 2 classes then load state dict
                seg = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512", num_labels=2, ignore_mismatched_sizes=True
                )
                sd = torch.load(model_path, map_location="cpu")  # type: ignore[attr-defined]
                seg.load_state_dict(sd, strict=False)
                seg.eval()
                self.model = seg
                print(f"Loaded SegFormer checkpoint from {model_path}")
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
            mask, conf = self._predict_onnx(image)
            return self._postprocess(mask), conf
        if self.model is not None:
            if self.backend == "segformer":
                mask, conf = self._predict_segformer(image)
            else:
                mask, conf = self._predict_torch(image)
            return self._postprocess(mask), conf
        # Placeholder
        image_resized = image.resize(self.input_size)
        image_array = np.array(image_resized)
        mask = self._generate_placeholder_mask(image_array)
        confidence = float(np.clip(0.85 + np.random.normal(0, 0.05), 0.0, 1.0))
        return self._postprocess(mask), confidence

    def _predict_torch(self, image: Image.Image) -> Tuple[np.ndarray, float]:
        assert _HAS_TORCH and self.model is not None
        # Resize to network input for now
        img = image.resize(self.input_size)
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        # Normalize as torchvision models expect ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        x = np.transpose(arr, (2, 0, 1))[None, ...]
        xt = torch.from_numpy(x)  # type: ignore[attr-defined]
        with torch.inference_mode():  # type: ignore[attr-defined]
            out = self.model(xt)["out"]  # type: ignore[index]
            # out: (N, C=2, H, W)
            logits = out.detach().cpu().numpy()
        # Foreground probability via softmax channel 1
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        fg = probs[0, 1]
        mask = (fg > self.thresh).astype(np.float32)
        conf = float(np.clip(np.mean(fg), 0.0, 1.0))
        return mask, conf

    def _predict_segformer(self, image: Image.Image) -> Tuple[np.ndarray, float]:
        assert _HAS_TORCH and self.model is not None and self.backend == "segformer"
        # Match training preprocessing used in our trainer: resize + ImageNet norm
        img = image.resize(self.input_size)
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        x = np.transpose(arr, (2, 0, 1))[None, ...]
        xt = torch.from_numpy(x)
        if self.device is not None:
            xt = xt.to(self.device)
            self.model.to(self.device)  # type: ignore[union-attr]
        with torch.inference_mode():
            out = self.model(pixel_values=xt)  # type: ignore[operator]
            logits = out.logits
            logits = F.interpolate(logits, size=self.input_size, mode="bilinear", align_corners=False)
            probs = torch.softmax(logits, dim=1)[0, 1]
            mask = (probs > self.thresh).float().cpu().numpy()
            conf = float(torch.clamp(probs.mean(), 0, 1).item())
        return mask.astype(np.float32), conf

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

    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        """Clean small speckles and smooth edges using morphology; keep main blobs."""
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        bin_img = (mask > 0.5).astype(np.uint8) * 255
        k = max(1, self.post_kernel)
        kernel = np.ones((k, k), np.uint8)
        # open then close to remove speckle and fill small holes
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        # remove very small blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
        if num_labels > 1:
            h, w = bin_img.shape
            min_area = max(1, int(self.min_blob_frac * h * w))
            keep = np.zeros(num_labels, dtype=bool)
            # keep largest and any above min_area
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            keep[largest] = True
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    keep[i] = True
            bin_img = np.where(keep[labels], 255, 0).astype(np.uint8)
        return (bin_img > 0).astype(np.float32)

    def preprocess_image(self, image: Image.Image) -> Any:
        if not _HAS_TORCH:
            return np.transpose(np.array(image.resize(self.input_size)).astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        img = image.resize(self.input_size)
        arr = np.array(img)
        tensor = torch.from_numpy(arr).float() / 255.0  # type: ignore[attr-defined]
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor