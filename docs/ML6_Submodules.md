# ML-6 Sub-modules (Procedural + Elevation 3D Pipeline)

This document lists concrete sub-modules to complete ML-6 per the spec (procedural parameterization first, elevation required for 3D). Use as a checklist during implementation.

## 1) Instance Detection
- File: `app/services/instance_service.py`
- Purpose: Detect per-building instances (masks/bboxes) within a bbox image.
- Approach: TorchVision Mask R-CNN (COCO weights to start). Thresholds and top-K selection by area.
- Env:
  - `MASKRCNN_SCORE_THR` (default 0.5)
  - `MASKRCNN_MASK_THR` (default 0.5)
  - `PROC_ROOF_MAX_BUILDINGS` (default 5)
- Outputs: `List[Instance{mask, bbox, score, crop, crop_to_full_affine}]`
- Tests: synthetic shapes; verify K-limit behavior and coordinate transforms.

## 2) Procedural Core Pipeline
- File: `app/services/procedural_roof/pipeline.py`
- Purpose: For each instance, run PBSR matching + ridge detection to create a parametric roof model.
- Integrations:
  - `PBSRService.match(mask)`
  - `RidgeDetectionService.detect(image_crop, rects)`
  - Optional ONNX classifiers when available (`PROC_ROOF_USE_CLASSIFIER=1`, `PROC_ROOF_ONNX_DIR`).
- Output: `ProceduralRoofModel{footprint_regularized, parts[{rect_bbox, roof_family, ridges2d, confidence}]}`
- Tests: IoU(footprint vs mask), ridge existence, classifier override path.

## 3) Elevation Augment (required for 3D)
- File: `app/services/elevation_augment.py`
- Purpose: Attach heights and pitches to procedural parts using DSM/DTM (nDSM).
- Functions:
  - `fit_part_plane(ndsm_window) -> normal, pitch_deg, aspect_deg, residuals`
  - `sample_ridge_z(ridge2d, ndsm) -> polyline3d`
  - `part_height_stats(ndsm_window) -> {min,max,mean}`
- Inputs: `DSMClip`, `NDSMResult`, `ProceduralRoofModel`, bbox transforms.
- Outputs: enriched `ProceduralRoofModel` with `ridges_3d`, `pitch_deg`, `aspect_deg`, `height_stats`.
- Tests: use dummy GeoTIFFs pattern from `tests/test_ndsm_fusion.py` to validate pitch/height.

## 4) Artifacts: GeoJSON Writer
- File: `app/services/artifacts/geojson_writer.py`
- Purpose: Emit FeatureCollection in EPSG:4326 for footprint, parts (rects), ridges 2D/3D (LineStringZ), and planes if available.
- Notes: Use `geo_utils` to convert from image/27700 -> lon/lat; ensure valid rings and properties.
- Output: local path; return via storage layer as URL.
- Tests: JSON schema validity; geometry counts; coordinate ranges.

## 5) Artifacts: glTF Writer
- File: `app/services/artifacts/gltf_writer.py`
- Purpose: Produce simple 3D mesh (extruded parts with pitched faces) and ridge lines.
- Library: `pygltflib` (or minimal writer).
- Output: `.glb` file; return via storage layer as URL.
- Tests: generate minimal scene; ensure glTF loads (basic sanity).

## 6) Artifact Storage
- File: `app/services/artifacts/storage.py`
- Purpose: Write artifacts locally or to GCS and return URLs.
- Local env:
  - `ARTIFACT_DIR=./artifacts`
  - `ARTIFACT_BASE_URL=http://localhost:8000/static` (optional)
- GCS env (optional):
  - `GCS_ARTIFACTS_BUCKET`
- Tests: local path write; mock GCS upload.

## 7) API Integration (/model3d)
- File: `app/main.py`
- Flow:
  1. Locate+fetch DSM/DTM -> `DSMClip`.
  2. Instance detection (Mask R-CNN) -> top-K instances by area.
  3. Per instance: procedural pipeline -> `ProceduralRoofModel`.
  4. Elevation augment: per-part plane fit + ridge z sampling -> 3D params.
  5. Artifacts: write GeoJSON + glTF -> URLs.
- Response:
  - `procedural_roofs: [ ... up to K ]`
  - `total_count`, `truncated`
  - `artifacts: {geojson_url, gltf_url}` (per building and/or combined)
  - Keep `planes`, `edges`, `summary` for backward-compat.
- Env additions: `PROC_ROOF_MAX_BUILDINGS`, instance thresholds.
- Tests: `tests/test_model3d_multibuilding.py` for K-limit, fields, and artifact URLs.

## 8) Classifier Runtime Wiring
- Files: `app/services/procedural_roof/classifiers.py`, `.../pipeline.py`, `.../ridge_detection.py`
- Purpose: Use ONNX classifiers to prune PBSR template search and decide roof family.
- Env: `PROC_ROOF_USE_CLASSIFIER`, `PROC_ROOF_ONNX_DIR`
- Fallback: clean heuristics when models are absent.
- Tests: mock ONNX outputs; verify selection path.

## 9) Validation & Metrics Harness
- File: `tools/validation/validate_ml6.py`
- Inputs: golden set with LiDAR planes and reference areas/ridges.
- Metrics: pitch MAE, area MAE, ridge completeness/correctness, edge IoU.
- Targets: pitch MAE <= 3-5 deg; area MAE <= 5-10%; ridges completeness >= 80%, correctness >= 85%.
- Output: JSON/CSV report.

## 10) Tests
- `tests/test_instances.py`: Mask R-CNN smoke test; K-limit; transforms.
- `tests/test_model3d_multibuilding.py`: response schema; `total_count/truncated`.
- `tests/test_artifacts.py`: GeoJSON valid; glTF file exists.
- `tests/test_procedural_elevation.py`: per-part pitch/height on synthetic ndsm.

## 11) Environment Variables (summary)
- Procedural: `PROC_ROOF_ENABLE=1` (default), `PROC_ROOF_MAX_BUILDINGS=5`, `PROC_ROOF_USE_CLASSIFIER=0|1`, `PROC_ROOF_ONNX_DIR=/models`
- Instances: `MASKRCNN_SCORE_THR=0.5`, `MASKRCNN_MASK_THR=0.5`
- Elevation (required for 3D): `EA_WCS_DSM`, `EA_WCS_DTM`, `EA_LAYER_DSM`, `EA_LAYER_DTM`, `DSM_CACHE_DIR`
- Artifacts: `ARTIFACT_DIR`, `ARTIFACT_BASE_URL`, `GCS_ARTIFACTS_BUCKET`

## 12) Delivery Order
1. Instance service + K-limit
2. Procedural pipeline integration (heuristics)
3. Elevation augment (per-part planes + ridge z)
4. GeoJSON writer + local storage
5. API schema updates + tests
6. glTF writer
7. Classifier runtime wiring
8. Validation harness

---

Owner: ML-6
Last updated: TBD