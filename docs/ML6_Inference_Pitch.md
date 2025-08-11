# ML-6: Production Inference & Elevations (3D Roof Parametrization)

Type: enhancement • Priority: Critical • Depends on: ML-1..5 ✅ • ETA: 5–7 days

Reference (GPU tuning on Cloud Run): https://cloud.google.com/run/docs/configuring/services/gpu-best-practices

## Objective
Given coordinates and overhead imagery, return an accurate 3D parametrization of the roof:
- Planes with normals, pitch°, aspect°, polygon outlines, and areas (m²)
- Ridges/edges (plane intersections) with heights
- Summary (total area, dominant pitches/orientations, height, roof type)
- Artifacts: GeoJSON (EPSG:4326) and optional glTF/OBJ mesh

## API
- POST `/model3d`
  - Request
    ```json
    {
      "coordinates": {"latitude": 51.5074, "longitude": -0.1278},
      "bbox_m": 60,
      "image_base64": "..." ,
      "image_url": "https://...",  
      "provider_hint": "client|xyz|google|bing",
      "return_mesh": true
    }
    ```
  - Response (abridged)
    ```json
    {
      "planes": [
        {"id":"p1","normal":[0.1, -0.9, 0.4],"pitch_deg":23.4,"aspect_deg":178.0,
         "polygon":[[lon,lat,h],...],"area_m2":45.2}
      ],
      "edges":[{"a":[lon,lat,h],"b":[lon,lat,h]}],
      "summary":{"area_m2":140.3,"max_height_m":11.8,"roof_type":"gabled"},
      "confidence":0.89,
      "artifacts":{"geojson_url":"...","gltf_url":"..."}
    }
    ```
- Backward‐compatible: `/pitch` continues to return summary-only values.

## Pipeline (≤3–5 s end‑to‑end)
1) Locate + fetch
   - BBox ~40–80 m square around (lat,lon)
   - DSM/DTM (EA LIDAR) tiles → cache by bbox (GCS or local)
   - Optional building footprint (OS/OSM/OpenBuildings) to constrain roof region
2) Segmentation (GPU)
   - Model: SegFormer‑B0 (Torch or ONNX)
   - Output: roof mask (binary/instance); refine with footprint overlap
3) Elevation fusion
   - nDSM = DSM − DTM; 3×3 median denoise; vegetation removal by morphology + height threshold
   - Align mask to DSM grid (affine by bbox)
4) Planar decomposition
   - Robust multi‑plane fitting (RANSAC or PEAC/region‑growing in normal space)
   - Merge planes by angle (<5°) & distance (<0.15 m)
   - Extract per‑plane polygon (marching squares) and clip to inliers
   - Edges/ridges = intersections of adjacent planes; classify roof type from plane graph
5) Outputs + confidence
   - Confidence f(seg_conf, DSM flags, inlier ratio, plane residuals)
   - Emit GeoJSON + optional glTF; summary metrics

## Accuracy tactics
- Use building footprint as prior; reject tiny planes; MAD outlier pruning
- Snap plane borders to footprint if ≤0.2 m away
- Validate normals with photo metadata/sun shadows when available (optional)

## Performance tactics (Cloud Run GPU)
- Model storage: GCS bucket mounted read‑only via Cloud Storage volume (gcsfuse) at `/models`
- Warm model on startup; low initial concurrency (1–2) then tune after load tests
- Cache DSM tiles, footprints, geocodes keyed by bbox in GCS/Supabase
- Vectorized NumPy/numba for plane fitting inner loops
- Follow Cloud Run GPU guidance for concurrency/startup/caching (see docs above)

## Configuration / Secrets
- `SEG_BACKEND=torch|onnx` (default torch)
- `SEG_MODEL_PATH=/models/segformer-b0`
- `DSM_CACHE_DIR=/var/cache/dsm` (ephemeral) or `gs://...` if remote cache
- `EA_DSM_BASE_URL=...` (tiles index)
- `CORS_ALLOW_ORIGINS=...`
- Cloud Run deploy flags:
  - `--gpu=type=nvidia-tesla-t4,count=1` (or L4 if available)
  - `--add-volume name=models,type=cloud-storage,bucket=$GCS_MODELS_BUCKET`
  - `--add-volume-mount volume=models,mount-path=/models,read-only`
  - `--concurrency=2 --cpu=2 --memory=4Gi` (tune after tests)

## Benchmarks & validation
- Golden set: 50–100 roofs with LiDAR truth
- Metrics: plane residual RMSE (m); pitch MAE (°); area MAE (%); edge IoU
- Targets: P50 ≤2.5 s, P95 ≤5 s; pitch MAE ≤3–5° typical; area MAE ≤5–10%

## Risks & mitigations
- DSM gaps → fallback planes with low confidence
- GPU cold start → warmup routine; min instances if needed later
- Provider imagery licensing → prefer client-supplied images

---

## Changes required in repo to enable ML‑6

Code
- `app/models/segmentation.py`
  - Add real loader: Torch or ONNXRuntime; read from `SEG_MODEL_PATH`
  - Start-up warm load; clear error if path missing
- `app/services/dsm_service.py`
  - Implement EA LIDAR DSM/DTM fetch + cache; grid align; nDSM computation
- New `app/services/footprint_service.py` (optional)
  - Fetch/cache building footprints (OS/OSM/OpenBuildings)
- New `app/services/plane_fitting.py`
  - RANSAC/region‑growing plane fitter; polygon extraction; edge intersections
- `app/models/quote.py`
  - No change required now; keep `/pitch` behavior; option to reference new outputs later
- `app/main.py`
  - Add POST `/model3d` endpoint returning planes/edges/summary/artifacts
  - Ensure server binds to `PORT` env (Cloud Run)

Container/Deploy
- Dockerfile
  - Ensure app listens on `$PORT` (default 8080) instead of fixed 8000
  - Keep models out of the image; mount `/models` via GCS volume
- GitHub Actions CD (`.github/workflows/cd.yml`)
  - Add `--gpu` and `--add-volume*` flags
  - New secret per env: `GCS_MODELS_BUCKET`

Data & Caching
- Supabase (optional now; formalized in ML‑8)
  - Tables used for caches: `irradiance_cache`, `geocode_cache` (read‑only here)
  - Artifacts metadata table later (`roof_measurements`) when persisting

Tests
- Add fixtures: small test image, DSM/DTM GeoTIFF snippet
- Unit tests: plane fitting determinism; segmentation smoke test
- Integration test: full `/model3d` flow on a tiny bbox

Docs
- README: model setup (upload to GCS), deploy flags, GPU notes, expected outputs

Deliverables
- Code, tests, docs; /model3d endpoint; Cloud Run GPU deploy with GCS volume.