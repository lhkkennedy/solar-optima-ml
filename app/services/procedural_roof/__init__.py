"""
Procedural roof reconstruction services

This package implements a lightweight scaffold for the EUROGRAPHICS 2022
paper "Procedural Roof Generation From a Single Satellite Image" by
Zhang & Aliaga. It exposes services to:

- classify building shape families and match compact configurations (PBSR)
- detect roof families and ridge configurations per roof part
- synthesize a procedural (vector) roof model
- compute evaluation metrics (optionally)

The heavy ML bits (training/export of small classifiers) are intentionally
omitted in this initial scaffold. The runtime defaults to deterministic,
parameter-free heuristics so the rest of the system remains stable while we
iterate. When feature-flagged off, the package is idle and has no side-effects.
"""

from .pbsr import PBSRService, BuildingFamilyMatch  # lightweight
from .ridge_detection import RidgeDetectionService, RoofFamilyResult  # lightweight

__all__ = ["PBSRService", "BuildingFamilyMatch", "RidgeDetectionService", "RoofFamilyResult"]

