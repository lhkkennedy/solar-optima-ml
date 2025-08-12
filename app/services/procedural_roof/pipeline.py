from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import numpy as np
import cv2

from .pbsr import PBSRService, BuildingFamilyMatch, Rect
from .ridge_detection import RidgeDetectionService, RoofFamilyResult
from .classifiers import load_proc_roof_classifiers
from .synthesis import ProceduralRoofSynthesizer, ProceduralRoofModel
from ..instance_service import Instance


PixelToLonLat = Callable[[float, float], Tuple[float, float]]


@dataclass
class ProceduralPipelineConfig:
    use_classifiers: bool = False  # reserved flag; heuristics by default


class ProceduralPipeline:
    """
    Procedural roof pipeline:
      - PBSR family matching on instance mask (crop region)
      - Ridge detection per part on instance crop
      - Synthesis of vector model in lon/lat using provided pixel->lon/lat mapping
    """

    def __init__(self,
                 pbsr: Optional[PBSRService] = None,
                 ridge: Optional[RidgeDetectionService] = None,
                 config: Optional[ProceduralPipelineConfig] = None) -> None:
        self.pbsr = pbsr or PBSRService()
        if ridge is not None:
            self.ridge = ridge
        else:
            # Load classifiers if enabled
            fam, roof = load_proc_roof_classifiers()
            self.ridge = RidgeDetectionService(classifier=roof)
        self.config = config or ProceduralPipelineConfig()
        self.synth = ProceduralRoofSynthesizer()

    def _to_gray(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim == 2:
            return rgb
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def run(self,
            instance: Instance,
            pixel_to_lonlat: Optional[PixelToLonLat] = None) -> Optional[ProceduralRoofModel]:
        """
        Execute the pipeline for a single detected instance.

        Args:
            instance: Instance returned by InstanceService
            pixel_to_lonlat: function mapping full-image pixel (x,y) -> (lon,lat)

        Returns:
            ProceduralRoofModel or None if matching failed.
        """
        x0, y0, w, h = instance.bbox.x, instance.bbox.y, instance.bbox.w, instance.bbox.h

        # Crop mask to the instance bbox for stable PBSR coordinates
        mask_crop = instance.mask[y0:y0 + h, x0:x0 + w]
        if mask_crop.size == 0:
            return None

        # PBSR matching on cropped mask (expects 2D binary)
        match_local: Optional[BuildingFamilyMatch] = self.pbsr.match(mask_crop.astype(np.uint8))
        if match_local is None:
            return None

        # Ridge detection per part using the instance crop (RGB -> gray)
        crop_rgb = instance.crop
        gray_crop = self._to_gray(crop_rgb)
        ridge_results_local: List[RoofFamilyResult] = []
        for rect in match_local.rects:
            rr = self.ridge.analyze_part(gray_crop, rect)
            ridge_results_local.append(rr)

        # Translate rects and ridge lines back to full-image coordinates
        rects_full: List[Rect] = [Rect(x=rect.x + x0, y=rect.y + y0, w=rect.w, h=rect.h) for rect in match_local.rects]
        ridge_results_full: List[RoofFamilyResult] = []
        for rr in ridge_results_local:
            ridges_full = [((p0[0] + x0, p0[1] + y0), (p1[0] + x0, p1[1] + y0)) for p0, p1 in rr.ridges]
            ridge_results_full.append(RoofFamilyResult(roof_family=rr.roof_family, ridges=ridges_full, confidence=rr.confidence))

        match_full = BuildingFamilyMatch(family=match_local.family, rects=rects_full, iou_score=match_local.iou_score)

        # Default pixel->lonlat mapping is identity (x,y)->(x,y)
        if pixel_to_lonlat is None:
            def pixel_to_lonlat(x: float, y: float) -> Tuple[float, float]:  # type: ignore[no-redef]
                return float(x), float(y)

        model = self.synth.assemble(match_full, ridge_results_full, pixel_to_lonlat)
        return model


__all__ = [
    "ProceduralPipeline",
    "ProceduralPipelineConfig",
]

