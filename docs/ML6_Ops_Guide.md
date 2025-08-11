# ML‑6 Ops Guide

## Suggested Git commit message

```text
ML-6: Implement submodules 1–12, artifacts, API, validation, and deploy workflow

Code
- Submodule 1: Add app/services/instance_service.py (Mask R-CNN scaffold) + exports
- Submodule 2: Add app/services/procedural_roof/pipeline.py + exports
- Submodule 3: Add app/services/elevation_augment.py (fit_part_plane, sample_ridge_z, augment)
- Submodule 4: Add app/services/artifacts/geojson_writer.py (FeatureCollection)
- Submodule 5: Add app/services/artifacts/gltf_writer.py (with pygltflib fallback)
- Submodule 6: Add app/services/artifacts/storage.py (local/GCS URLs)
- Submodule 7: Integrate /model3d pipeline in app/main.py (instances→procedural→elevation→artifacts)
- Submodule 8: Add classifier wiring (load_proc_roof_classifiers) + ridge detection auto-load
- Submodule 9: Add tools/validation/validate_ml6.py (pitch/area/ridge/IoU)
- Submodule 10: Add tests: instances, procedural_elevation, artifacts, model3d_multibuilding
- Submodules 11–12: Dockerfile binds $PORT; README env/Docker notes
- Robust fallbacks for missing torch/torchvision and pygltflib

Infra
- Add .github/workflows/deploy.yml (tests + build + deploy to dev, manual staging)

Deps
- requirements.txt: add pygltflib

Result
- All tests green locally (65 passed, 1 skipped)
```

## Ops checklist

- **Environments and services**
  - Create Cloud Run services: `solaroptima-ml-dev`, `solaroptima-ml-staging`.
  - Enable APIs: Cloud Run, Cloud Build, Artifact Registry.
  - Choose and standardize region (e.g., `europe-west2`).

- **Storage**
  - Create GCS buckets per env:
    - Models: e.g., `gs://ml6-models-dev`, `gs://ml6-models-staging`.
    - Artifacts: e.g., `gs://ml6-artifacts-dev`, `gs://ml6-artifacts-staging`.
  - Grant deploy service accounts read (models) and write (artifacts).

- **GitHub environments and secrets**
  - Create GitHub Environments: `dev`, `staging`.
  - Dev secrets: `GCP_SA_KEY_DEV` (or WIF), `GCP_PROJECT_ID_DEV`, `GCP_REGION_DEV`,
    `GCS_MODELS_BUCKET_DEV`, `GCS_ARTIFACTS_BUCKET_DEV`, `EA_WCS_DSM`, `EA_WCS_DTM`, `EA_LAYER_DSM`, `EA_LAYER_DTM`.
  - Staging secrets: same with `_STAGING` suffixes.
  - Optional: `CORS_ALLOW_ORIGINS`.

- **CI/CD workflow** (added at `.github/workflows/deploy.yml`, name: `deploy`)
  - Tests on PR/push; build via Cloud Build.
  - Auto-deploy to dev on push to branch `dev`.
  - Manual deploy to staging via `workflow_dispatch`.
  - Trigger staging:
    - UI: Actions → `deploy` → Run workflow → target=`staging`.
    - CLI: `gh workflow run deploy -f target=staging`.

- **Cloud Run configuration**
  - Common flags: `--cpu=2 --memory=4Gi --concurrency=2`.
  - Dockerfile binds `$PORT` (Cloud Run sets it).
  - Env vars per env:
    - **Procedural**: `PROC_ROOF_ENABLE=1`, `PROC_ROOF_MAX_BUILDINGS`, `PROC_ROOF_USE_CLASSIFIER` (0/1), `PROC_ROOF_ONNX_DIR=/models`.
    - **Elevation**: `EA_WCS_DSM`, `EA_WCS_DTM`, `EA_LAYER_DSM`, `EA_LAYER_DTM`, `DSM_CACHE_DIR=/var/cache/dsm`.
    - **Artifacts**: `ARTIFACT_DIR=/var/artifacts`, `GCS_ARTIFACTS_BUCKET`, optional `ARTIFACT_BASE_URL`.
    - **CORS**: `CORS_ALLOW_ORIGINS` (comma-separated).
  - Optional models mount: add GCS volume → mount at `/models`.

- **GPU (optional)**
  - If needed: install CUDA wheels, add `--gpu=type=nvidia-tesla-t4,count=1` (or L4), test cold start.

- **Artifact URLs**
  - Decide local static vs. GCS hosting.
  - If GCS: public objects and https URLs, or implement signed URLs.

- **Observability**
  - Logging: request ID already included.
  - Health: `GET /health`.
  - Add uptime checks, error/latency alerts, basic dashboard.

- **Security**
  - Tighten CORS to known frontends.
  - Add auth (API key/JWT) if needed; store secrets in Cloud Run/Secret Manager.

- **Performance/cost**
  - Start with min instances = 0; adjust after measuring cold starts.
  - Benchmark `/model3d`; tune CPU/memory/concurrency.
  - Consider caching DSM/nDSM in GCS or persistent volume if needed.

- **Post-deploy validation**
  - Smoke: `/health`, `/model3d` with small bbox + `image_base64`.
  - Verify artifacts download/view (GeoJSON/glTF).

## Optional accuracy improvements (post-ops)

- Replace bbox-based footprint mask with polygon rasterization for nDSM.
- Scale plane fitting by world spacing for exact pitch/aspect.
- Upgrade glTF meshes to pitched/extruded faces per part/plane.
