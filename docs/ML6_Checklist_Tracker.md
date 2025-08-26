# ML-6 Ops Checklist Tracker

## 📋 Implementation Status

### ✅ **Completed**
- [x] CI/CD workflow (`.github/workflows/deploy.yml`)
- [x] Dockerfile with proper configuration
- [x] Health endpoint (`/health`)
- [x] Basic Cloud Run deployment structure
- [x] Request ID middleware
- [x] CORS configuration
- [x] Environment variable structure

### 🔧 **In Progress / Needs Setup**

#### 1. **Environments and Services**
- [ ] Enable required GCP APIs:
  ```bash
  gcloud services enable run.googleapis.com
  gcloud services enable cloudbuild.googleapis.com
  gcloud services enable artifactregistry.googleapis.com
  ```
- [ ] Verify Cloud Run services: `solaroptima-ml-dev`, `solaroptima-ml-staging`
- [ ] Standardize region (recommended: `europe-west2`)

#### 2. **Storage Setup**
- [ ] Run storage setup script:
  ```bash
  chmod +x tools/setup_storage.sh
  ./tools/setup_storage.sh YOUR_PROJECT_ID europe-west2
  ```
- [ ] Create GCS buckets:
  - [ ] `gs://ml6-models-dev`
  - [ ] `gs://ml6-artifacts-dev`
  - [ ] `gs://ml6-models-staging`
  - [ ] `gs://ml6-artifacts-staging`
- [ ] Set up IAM permissions for service accounts
- [ ] Upload ML models to models buckets

#### 3. **GitHub Environments and Secrets**
- [ ] Create GitHub environments: `dev`, `staging`
- [ ] Set up service accounts:
  - [ ] `ml6-deploy-dev`
  - [ ] `ml6-deploy-staging`
- [ ] Configure secrets (see `docs/github_setup_guide.md`):
  - [ ] `GCP_SA_KEY_DEV`
  - [ ] `GCP_PROJECT_ID_DEV`
  - [ ] `GCP_REGION_DEV`
  - [ ] `GCS_MODELS_BUCKET_DEV`
  - [ ] `GCS_ARTIFACTS_BUCKET_DEV`
  - [ ] `EA_WCS_DSM`
  - [ ] `EA_WCS_DTM`
  - [ ] `EA_LAYER_DSM`
  - [ ] `EA_LAYER_DTM`
  - [ ] Same secrets for staging with `_STAGING` suffix
  - [ ] Optional: `CORS_ALLOW_ORIGINS`

#### 4. **Cloud Run Configuration** ✅ (Updated)
- [x] CPU/Memory: `--cpu=2 --memory=4Gi --concurrency=2`
- [x] Environment variables configured
- [x] GCS volume mounts for models
- [x] CORS configuration
- [ ] Optional: GPU configuration if needed

#### 5. **Observability**
- [x] Health endpoint: `GET /health`
- [x] Request ID logging
- [ ] Set up uptime checks
- [ ] Configure error/latency alerts
- [ ] Create basic monitoring dashboard

#### 6. **Security**
- [ ] Tighten CORS to known frontends
- [ ] Consider API key authentication
- [ ] Store secrets in Cloud Run/Secret Manager

#### 7. **Performance/Cost Optimization**
- [ ] Start with min instances = 0
- [ ] Measure cold start times
- [ ] Benchmark `/model3d` endpoint
- [ ] Tune CPU/memory/concurrency based on load
- [ ] Consider DSM/nDSM caching in GCS

#### 8. **Post-Deploy Validation**
- [ ] Run validation script:
  ```bash
  chmod +x tools/validate_deployment.sh
  ./tools/validate_deployment.sh YOUR_SERVICE_URL dev
  ```
- [ ] Test `/health` endpoint
- [ ] Test `/model3d` with small bbox + `image_base64`
- [ ] Verify artifacts download/view (GeoJSON/glTF)
- [ ] Check GCS bucket for generated artifacts

## 🚀 **Deployment Commands**

### First-time setup:
```bash
# 1. Set up storage
./tools/setup_storage.sh YOUR_PROJECT_ID europe-west2

# 2. Create GitHub environments and secrets
# (Follow docs/github_setup_guide.md)

# 3. Upload models to GCS
gsutil cp -r /path/to/models/* gs://ml6-models-dev/
gsutil cp -r /path/to/models/* gs://ml6-models-staging/
```

### Deploy to dev:
```bash
# Push to dev branch (auto-deploys)
git push origin dev
```

### Deploy to staging:
```bash
# Via GitHub UI: Actions → deploy → Run workflow → target=staging
# Or via CLI:
gh workflow run deploy -f target=staging
```

### Validate deployment:
```bash
./tools/validate_deployment.sh https://solaroptima-ml-dev-xxxxx-ew.a.run.app dev
```

## 📊 **Monitoring Checklist**

### Health Checks
- [ ] `/health` returns 200
- [ ] Response time < 1s
- [ ] Service is ready in Cloud Run

### Model Endpoints
- [ ] `/model3d` accepts valid requests
- [ ] Artifacts are generated in GCS
- [ ] GeoJSON/glTF files are accessible

### Performance Metrics
- [ ] Cold start time < 30s
- [ ] Average response time < 5s
- [ ] Error rate < 1%

### Cost Monitoring
- [ ] Cloud Run costs within budget
- [ ] GCS storage costs reasonable
- [ ] Cloud Build costs minimal

## 🔧 **Troubleshooting**

### Common Issues:
1. **Service won't start**: Check environment variables and secrets
2. **Models not found**: Verify GCS bucket permissions and model uploads
3. **Artifacts not generated**: Check artifact bucket permissions
4. **CORS errors**: Verify `CORS_ALLOW_ORIGINS` configuration
5. **High latency**: Consider increasing CPU/memory or adding caching

### Debug Commands:
```bash
# Check service logs
gcloud run services logs read solaroptima-ml-dev --region=europe-west2

# Check service status
gcloud run services describe solaroptima-ml-dev --region=europe-west2

# Test endpoint locally
curl -X POST "http://localhost:8000/health"
```

## 📈 **Next Steps (Post-ML6)**

### Optional Accuracy Improvements:
- [ ] Replace bbox-based footprint mask with polygon rasterization
- [ ] Scale plane fitting by world spacing for exact pitch/aspect
- [ ] Upgrade glTF meshes to pitched/extruded faces per part/plane

### Future Enhancements:
- [ ] GPU acceleration for inference
- [ ] Advanced caching strategies
- [ ] Multi-region deployment
- [ ] Blue-green deployment strategy 