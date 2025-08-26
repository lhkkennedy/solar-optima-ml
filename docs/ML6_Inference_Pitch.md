# ML-6: Procedural Roof Generation (Primary) & Elevations (Required for 3D)

Type: enhancement • Priority: Critical • Depends on: ML-1..5 ✅ • ETA: 5–7 days

Reference (GPU tuning on Cloud Run): https://cloud.google.com/run/docs/configuring/services/gpu-best-practices

## Objective
Given coordinates and overhead imagery, return an accurate 3D parametrization of the roof:
- Planes with normals, pitch°, aspect°, polygon outlines, and areas (m²)
- Ridges/edges (plane intersections) with heights
- Summary (total area, dominant pitches/orientations, height, roof type)
- Artifacts: GeoJSON (EPSG:4326) and glTF/OBJ mesh

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

## Pipeline (procedural-first, elevation-required)
1) Locate + fetch imagery; optional building footprint (OSM/OpenBuildings) for crop.
2) Instance segmentation or client mask (preferred for scalability) to isolate building image.
3) Building decomposition (PBSR): classify compact building shape family (I/L/T/U/Z within up to 4 parts; topologies T11, T21, T32, T43) and match a regularized configuration via IoU over a compact template set.
4) Roof ridge detection per part: compute edge map (Canny for V1, or ONNX classifier) and infer roof family (flat/gable/hip/pyramid/half-hip) and ridge configuration maximizing edge support.
5) Elevation fusion (DSM/DTM) → nDSM; denoise; align footprint/mask; vegetation removal.
6) Planar decomposition (RANSAC/region‑growing): merge planes; extract polygons; intersect planes; estimate per‑part plane normals.
7) Snap procedural ridge axes to plane aspects/pitches; sample nDSM along ridges and within parts to assign heights and compute pitch°. Produce full 3D parametrization (ridges_3d, part planes) and assemble artifacts (GeoJSON + glTF).

#### Training small classifiers (RTX 3060 friendly)
We can replace the V1 deterministic heuristics with small ONNX classifiers:

- Family classifier (PBSR): ResNet18 (input 128×128), predicts {T11,T21,T32,T43}.
- Roof family classifier: ResNet18 (input 128×128), predicts {flat,gable,hip,pyramid}.

Workflow:
1) Generate synthetic data (footprints/roofs) with transformations (occlusions, noise, rotations).
2) Train on GPU (5–20 epochs; seconds to minutes per epoch on 3060).
3) Export to ONNX; mount under `/models`; set `PROC_ROOF_USE_CLASSIFIER=1`.

Paths:
- Scripts under `tools/procedural_roof/`: `gen_synth_footprints.py`, `train_family.py`, `export_onnx.py`.
- Runtime looks for ONNX in `PROC_ROOF_ONNX_DIR`.

## Accuracy tactics
- Use building footprint as prior; reject tiny planes; MAD outlier pruning
- Snap plane borders to footprint if ≤0.2 m away
- Validate normals with photo metadata/sun shadows when available (optional)

## Wishlist
- Street‑view façade texturing: fetch street‑level images (e.g., Google Street View) to texture façades of the 3D building models. This would "paint" building sides for realistic visualization. Requires camera pose estimation and photo-to-model UV projection; licensing and caching to be considered.

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
- Elevation (required for 3D):
  - `EA_WCS_DSM=https://.../wcs` (Environment Agency DSM WCS)
  - `EA_WCS_DTM=https://.../wcs` (Environment Agency DTM WCS)
  - `EA_LAYER_DSM=...` (coverage name)
  - `EA_LAYER_DTM=...` (coverage name)
- `CORS_ALLOW_ORIGINS=...`
- Cloud Run deploy flags:
  - `--gpu=type=nvidia-tesla-t4,count=1` (or L4 if available)
  - `--add-volume name=models,type=cloud-storage,bucket=$GCS_MODELS_BUCKET`
  - `--add-volume-mount volume=models,mount-path=/models,read-only`
  - `--concurrency=2 --cpu=2 --memory=4Gi` (tune after tests)
  - App binds to `$PORT`; Cloud Run sets this automatically

## Benchmarks & validation
- Golden set: 50–100 roofs with LiDAR truth
- Metrics: plane residual RMSE (m); pitch MAE (°); area MAE (%); ridge completeness/correctness; edge IoU
- Targets: P50 ≤2.5 s, P95 ≤5 s; pitch MAE ≤3–5°; area MAE ≤5–10%; ridge completeness ≥80%, correctness ≥85%

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
  - App now listens on `$PORT` (default 8080) instead of fixed 8000
  - Keep models out of the image; mount `/models` via GCS volume
- GitHub Actions CD (`.github/workflows/deploy.yml`)
  - Tests + build + deploy to dev on branch `dev`, manual staging via workflow dispatch
  - Add `--gpu` and `--add-volume*` flags
  - Secrets per env: `GCP_PROJECT_ID_*`, `GCP_REGION_*`, `GCP_SA_KEY_*`, `GCS_MODELS_BUCKET_*`, `GCS_ARTIFACTS_BUCKET_*`

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