from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
try:
    import torch  # type: ignore
    from torchvision import transforms as T  # type: ignore
    from torchvision.models.detection import maskrcnn_resnet50_fpn  # type: ignore
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    T = None  # type: ignore
    maskrcnn_resnet50_fpn = None  # type: ignore
    MaskRCNN_ResNet50_FPN_Weights = None  # type: ignore
    _HAS_TORCH = False
from PIL import Image


@dataclass
class BBox:
    """Axis-aligned bounding box in image pixel space."""
    x: int
    y: int
    w: int
    h: int

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass
class Instance:
    """Detected building instance."""
    mask: np.ndarray  # HxW, dtype=bool
    bbox: BBox
    score: float
    crop: np.ndarray  # h x w x 3, uint8
    crop_to_full_affine: np.ndarray  # 2x3 float32


class InstanceService:
    """
    Mask R-CNN based instance detection.

    Environment variables:
    - MASKRCNN_SCORE_THR (float, default 0.5)
    - MASKRCNN_MASK_THR (float, default 0.5)
    - PROC_ROOF_MAX_BUILDINGS (int, default 5)
    """

    def __init__(self,
                 score_threshold: Optional[float] = None,
                 mask_threshold: Optional[float] = None,
                 max_buildings: Optional[int] = None,
                 device: Optional[str] = None) -> None:
        self.score_threshold: float = float(os.getenv("MASKRCNN_SCORE_THR", "0.5")) if score_threshold is None else float(score_threshold)
        self.mask_threshold: float = float(os.getenv("MASKRCNN_MASK_THR", "0.5")) if mask_threshold is None else float(mask_threshold)
        self.max_buildings: int = int(os.getenv("PROC_ROOF_MAX_BUILDINGS", "5")) if max_buildings is None else int(max_buildings)

        if device is None and _HAS_TORCH:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # type: ignore[attr-defined]
        elif _HAS_TORCH:
            self.device = torch.device(device)  # type: ignore[attr-defined]
        else:
            self.device = None

        self._model: Optional[object] = None
        # Basic to-tensor transform without resizing to keep coordinate consistency
        if _HAS_TORCH:
            self._to_tensor = T.Compose([
                T.ToTensor(),  # converts HxWxC uint8 [0,255] to CxHxW float [0,1]
            ])
        else:
            # Fallback placeholder; never used when _HAS_TORCH is False
            self._to_tensor = lambda img: img

    def _ensure_model(self) -> None:
        if self._model is not None or not _HAS_TORCH:
            return
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT  # type: ignore[assignment]
        model = maskrcnn_resnet50_fpn(weights=weights)  # type: ignore[operator]
        model.eval()
        model.to(self.device)  # type: ignore[union-attr]
        self._model = model

    def warmup(self, image_size: Tuple[int, int] = (512, 512)) -> None:
        """Run a dummy forward pass to load weights and JIT kernels."""
        if not _HAS_TORCH:
            return
        self._ensure_model()
        dummy = torch.zeros((3, image_size[0], image_size[1]), dtype=torch.float32, device=self.device)  # type: ignore[attr-defined]
        with torch.inference_mode():  # type: ignore[attr-defined]
            _ = self._model([dummy])  # type: ignore[arg-type]

    def _to_pil(self, image: np.ndarray | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if not isinstance(image, np.ndarray):
            raise TypeError("InstanceService.detect expects a numpy array or PIL.Image")
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return Image.fromarray(image.astype(np.uint8), mode="RGB")

    def detect(self, image: np.ndarray | Image.Image) -> List[Instance]:
        """Detect building instances in the given RGB image.

        Returns:
            List[Instance]: up to `max_buildings` instances, sorted by mask area desc.
        """
        # If torch is unavailable, return empty
        if not _HAS_TORCH:
            return []
        self._ensure_model()

        pil_img = self._to_pil(image)
        width, height = pil_img.size
        tensor = self._to_tensor(pil_img).to(self.device)  # type: ignore[union-attr]

        with torch.inference_mode():  # type: ignore[attr-defined]
            outputs = self._model([tensor])  # type: ignore[arg-type]

        if not outputs:
            return []

        out = outputs[0]
        boxes: torch.Tensor = out.get("boxes", torch.empty((0, 4), device=tensor.device))
        scores: torch.Tensor = out.get("scores", torch.empty((0,), device=tensor.device))
        masks: torch.Tensor = out.get("masks", torch.empty((0, 1, height, width), device=tensor.device))

        # Filter by score threshold
        keep = scores >= self.score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        masks = masks[keep]

        # Convert to CPU numpy for post-processing
        boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
        scores_np = scores.detach().cpu().numpy().astype(np.float32)
        masks_np = masks.detach().cpu().numpy()  # (N,1,H,W) float [0,1]

        # Threshold masks
        bin_masks = masks_np[:, 0] >= float(self.mask_threshold)

        # Compute areas to sort
        areas = bin_masks.reshape(bin_masks.shape[0], -1).sum(axis=1) if bin_masks.size > 0 else np.zeros((0,), dtype=np.int64)
        order = np.argsort(-areas)  # descending by mask area

        instances: List[Instance] = []
        for idx in order[: self.max_buildings]:
            if idx < 0 or idx >= boxes_np.shape[0]:
                continue
            x1, y1, x2, y2 = boxes_np[idx]
            # Clip and convert to ints
            x0i = max(0, min(int(np.floor(x1)), width - 1))
            y0i = max(0, min(int(np.floor(y1)), height - 1))
            x1i = max(0, min(int(np.ceil(x2)), width))
            y1i = max(0, min(int(np.ceil(y2)), height))

            w = max(1, x1i - x0i)
            h = max(1, y1i - y0i)
            bbox = BBox(x=x0i, y=y0i, w=w, h=h)

            mask_full = bin_masks[idx]
            # Crop extraction from original image
            np_img = np.array(pil_img)
            crop = np_img[y0i:y0i + h, x0i:x0i + w].copy()

            # Affine matrix that maps crop coords (xc, yc) to full image coords
            # [x, y]^T_full = [ [1,0,tx],[0,1,ty] ] * [x, y, 1]^T
            affine = np.array([[1.0, 0.0, float(x0i)],
                               [0.0, 1.0, float(y0i)]], dtype=np.float32)

            inst = Instance(
                mask=mask_full.astype(bool),
                bbox=bbox,
                score=float(scores_np[idx]),
                crop=crop,
                crop_to_full_affine=affine,
            )
            instances.append(inst)

        return instances


__all__ = [
    "BBox",
    "Instance",
    "InstanceService",
]

