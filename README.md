# SolarOptima ML Service

ML micro-service for automated solar panel assessment and quotation for UK residential properties.

## Features

- **Roof Segmentation**: SegFormer-B0 model for aerial imagery analysis
- **Pitch Estimation**: DSM-based roof pitch calculation
- **Solar Quotation**: MCS-compliant yield and cost calculations
- **FastAPI**: Modern, async API with automatic documentation

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the service**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Access the API**:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Environment Variables (key)

- `SEG_BACKEND=torch|onnx`
- `SEG_MODEL_PATH=/models/segformer-b0`
- `PROC_ROOF_ENABLE=0|1`: include procedural roof output in `/model3d`
- `PROC_ROOF_USE_CLASSIFIER=0|1`: use ONNX classifiers for roof family if present
- `PROC_ROOF_ONNX_DIR=/models`: directory containing ONNX files
 - `EA_WCS_DSM`, `EA_WCS_DTM`, `EA_LAYER_DSM`, `EA_LAYER_DTM`, `DSM_CACHE_DIR`: elevation config
 - `ARTIFACT_DIR`, `ARTIFACT_BASE_URL`, `GCS_ARTIFACTS_BUCKET`: artifact storage
 - `PORT`: port to bind (Cloud Run sets this automatically)

### Docker

1. **Build the image**:
   ```bash
   docker build -t solaroptima-ml .
   ```

2. **Run the container**:
   ```bash
   # Locally
   docker run -e PORT=8000 -p 8000:8000 solaroptima-ml
   # On Cloud Run, the service will listen on $PORT automatically
   ```

## API Usage

### Segmentation Endpoint

Analyze aerial imagery to segment roof areas:

```bash
curl -X POST "http://localhost:8000/infer" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@aerial_image.png"
```

### Procedural + Elevation (3D required)

We reconstruct a vector, parametric roof model (PBSR + ridge detection) and then require DSM/DTM elevation to produce full 3D (heights, pitches):

- Enable procedural (default): `PROC_ROOF_ENABLE=1`
- Optional ONNX classifiers: `PROC_ROOF_USE_CLASSIFIER=1`, `PROC_ROOF_ONNX_DIR=/models`
- Elevation for 3D (required): set EA WCS env vars (`EA_WCS_DSM`, `EA_WCS_DTM`, `EA_LAYER_DSM`, `EA_LAYER_DTM`) and `DSM_CACHE_DIR`.
- `/model3d` includes `procedural_roofs` (footprint, parts, ridges) and returns 3D (ridges_3d, per‑part pitch/aspect) by sampling nDSM and fitting per‑part planes.

**Response**:
```json
{
  "mask": "iVBORw0KGgoAAAANSUhEUgAA...",
  "confidence": 0.85,
  "original_size": [512, 512],
  "mask_size": [256, 256]
}
```

## Procedural Roofs + Elevation (3D)

Enable a vector roof model from a single image (inspired by Zhang & Aliaga, EUROGRAPHICS 2022):

- Set env:
  - `PROC_ROOF_ENABLE=1`
  - Elevation (required for 3D): `EA_WCS_DSM`, `EA_WCS_DTM`, `EA_LAYER_DSM`, `EA_LAYER_DTM`, `DSM_CACHE_DIR`
  - optional classifiers: `PROC_ROOF_USE_CLASSIFIER=1`, `PROC_ROOF_ONNX_DIR=/models`
- Response from `/model3d` will include a `procedural_roofs` key:
  ```json
  {
    "footprint_regularized": [[lon,lat], ...],
    "parts": [
      {"rect_bbox": [[lon,lat],...], "roof_family": "gable", "ridges": [[[lon,lat],[lon,lat]]], "confidence": 0.8}
    ]
  }
  ```

Training small classifiers (optional):

```bash
python tools/procedural_roof/gen_synth_footprints.py --out data/pbsr_masks --num 50000
python tools/procedural_roof/train_family.py --data data/pbsr_masks --out runs/family_resnet18.pt --epochs 10
python tools/procedural_roof/export_onnx.py --pt runs/family_resnet18.pt --onnx models/proc_roof_family.onnx
```

### Dev Preview Frontend (CORS-friendly launch)

To test `/model3d` quickly with a bbox and image from a simple frontend, use the static preview under `web/preview/`.

Serve it locally so the origin is http://localhost (avoids `Origin: null` CORS issues):

```bash
# Linux/macOS
python -m http.server -d web/preview 8081

# Windows PowerShell: open the browser then start the server
powershell -Command "Start-Process http://localhost:8081"; python -m http.server -d web/preview 8081
```

Then set your dev environment secret `CORS_ALLOW_ORIGINS` to `http://localhost:8081` and redeploy so the API accepts calls from the preview.

### Python Example

```python
import requests
from PIL import Image
import base64

# Load image
with open("aerial_image.png", "rb") as f:
    image_data = f.read()

# Send request
response = requests.post(
    "http://localhost:8000/infer",
    files={"file": ("image.png", image_data, "image/png")}
)

# Process response
if response.status_code == 200:
    result = response.json()
    
    # Decode mask
    mask_data = base64.b64decode(result["mask"])
    mask_image = Image.open(io.BytesIO(mask_data))
    
    print(f"Confidence: {result['confidence']}")
    print(f"Mask size: {result['mask_size']}")
else:
    print(f"Error: {response.text}")
```

### Quote Generation Endpoint

Generate complete solar quotes with MCS-compliant calculations:

```bash
curl -X POST "http://localhost:8000/quote" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "property_details": {
      "address": "123 Solar Street, London",
      "postcode": "SW1A 1AA",
      "property_type": "semi_detached",
      "occupancy": "family_of_4"
    },
    "segmentation_result": {
      "mask": "iVBORw0KGgoAAAANSUhEUgAA...",
      "confidence": 0.95
    },
    "pitch_result": {
      "pitch_degrees": 25.5,
      "area_m2": 45.2,
      "roof_type": "gabled",
      "orientation": "south_facing"
    },
    "preferences": {
      "battery_storage": true,
      "premium_panels": false,
      "financing": "cash_purchase"
    }
  }'
```

**Response**:
```json
{
  "quote_id": "SOL-2025-ABC12345",
  "generated_date": "2025-01-15T10:30:00",
  "property_details": {
    "address": "123 Solar Street, London",
    "postcode": "SW1A 1AA",
    "property_type": "semi_detached",
    "occupancy": "family_of_4"
  },
  "system_specification": {
    "optimal_kwp": 3.5,
    "panel_count": 8,
    "panel_type": "400W Monocrystalline Panel",
    "inverter_type": "3.6kW Hybrid Inverter",
    "battery_capacity": "5.2kWh",
    "panel_efficiency": 0.2,
    "inverter_efficiency": 0.96
  },
  "yield_analysis": {
    "estimated_yearly_kwh": 3150,
    "solar_fraction": 0.75,
    "co2_savings_kg": 1250,
    "mcs_compliant": true,
    "performance_ratio": 0.75,
    "monthly_yield": {
      "jan": 95, "feb": 120, "mar": 220, "apr": 280,
      "may": 350, "jun": 380, "jul": 380, "aug": 350,
      "sep": 280, "oct": 220, "nov": 120, "dec": 95
    }
  },
  "financial_analysis": {
    "total_cost_gbp": 8750,
    "installation_cost": 6500,
    "battery_cost": 2250,
    "annual_savings": 1050,
    "payback_years": 8.2,
    "roi_percentage": 12.1,
    "feed_in_tariff": 0,
    "export_benefit": 157.5
  },
  "itemized_breakdown": [
    {
      "sku": "PANEL-400W",
      "description": "400W Monocrystalline Panel",
      "quantity": 8,
      "unit_cost": 180,
      "total_cost": 1440
    },
    {
      "sku": "INV-3.6KW",
      "description": "3.6kW Hybrid Inverter",
      "quantity": 1,
      "unit_cost": 1200,
      "total_cost": 1200
    }
  ],
  "warranties": {
    "panels": "25 years",
    "inverter": "10 years",
    "battery": "10 years",
    "installation": "2 years"
  },
  "next_steps": [
    "Schedule site survey",
    "Apply for planning permission (if required)",
    "Arrange financing",
    "Choose installation date"
  ],
  "valid_until": "2025-02-14",
  "mcs_compliant": true,
  "confidence_score": 0.92
}
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_infer.py -v
pytest tests/test_pitch.py -v
pytest tests/test_quote.py -v
pytest tests/test_integration.py -v
pytest tests/test_contract.py -v
pytest tests/test_resilience.py -v
# Performance (optional)
pytest -m runslow tests/test_performance.py -v
```

## Project Structure

```
solar-optima-ml/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   ├── segmentation.py  # SegFormer-B0 model
│   │   ├── pitch_estimator.py # Roof pitch estimator
│   │   └── quote.py         # Quote generation model
│   └── services/
│       ├── dsm_service.py   # DSM data service
│       ├── irradiance_service.py # PVGIS API integration
│       ├── cost_service.py  # BEIS cost database
│       └── quote_calculator.py # MCS yield calculations
├── tests/
│   ├── test_infer.py        # Segmentation tests
│   ├── test_pitch.py        # Pitch estimation tests
│   ├── test_quote.py        # Quote generation tests
│   ├── test_integration.py  # End-to-end flow tests
│   ├── test_contract.py     # OpenAPI contract tests
│   ├── test_resilience.py   # Failure mode tests
│   └── test_performance.py  # Performance tests (runslow)
├── models/                  # Pre-trained model weights
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Development Tasks

This project follows the task breakdown from `DESIGN.md`:

- **ML-1**: ✅ Dockerfile + FastAPI skeleton with `/infer` endpoint
- **ML-2**: ✅ Roof pitch estimator (`/pitch` endpoint)
- **ML-3**: ✅ Quote generation (`/quote` endpoint)
- **ML-4**: ✅ Integration testing
- **ML-5**: ⏳ CI/CD pipeline

## Current Status

**ML-3 and ML-4 are complete. ML-5 CI/CD is implemented (Cloud Run), pending first release tag and package visibility.**

- ✅ `/infer` (segmentation) API and tests
- ✅ `/pitch` (roof pitch) API and tests
- ✅ `/quote` (quote generation) API and tests
- ✅ Integration/contract/resilience tests (ML-4)
- ✅ CI (pytest + Docker build)
- ✅ CD (tag-based deploy to Google Cloud Run for `dev`/`staging`/`prod`)
- ⏳ First production release pending: add environment secrets and tag `vX.Y.Z`

**Test Results:**
- Unit tests: 9/9 (segmentation) + 8/8 (pitch) + 29/29 (quote)
- Integration/contract/resilience: 6/6
- Optional performance (local): pass under target budgets

## Feature completeness vs. placeholders

- Segmentation (`/infer`)
  - Status: API complete; placeholder model used (no transformers runtime).
  - Ready for: demos, integration tests.
  - Not yet: real SegFormer inference weights, accuracy benchmarking.

- Pitch Estimation (`/pitch`)
  - Status: API complete with UK-bounds validation; integration tests in place.
  - Placeholder: DSM data source uses synthetic values; planar decomposition simplified.
  - Not yet: Environment Agency LIDAR DSM retrieval, tile caching.

- Quote Generation (`/quote`)
  - Status: API complete; MCS-style calculations implemented; itemized costs + ROI/payback.
  - Placeholders: PVGIS irradiance and BEIS/pricing are simulated; address→geocoding not wired (uses placeholder coords in model).
  - Not yet: real PVGIS v5.2 integration, live pricing, VAT/overheads, geocoding from postcode/address.

- Platform & Ops
  - Logging/Tracing: Request ID middleware added; basic logging in place.
  - CI: GitHub Actions for tests and Docker build.
  - CD: GitHub Actions to Google Cloud Run (env-gated deployments).
  - Docker: Image builds locally and in CI.

## Known gaps and placeholders (to replace)

- Segmentation model is a placeholder (no transformers runtime or real weights).
- DSMService returns synthetic values; replace with Environment Agency LIDAR DSM tiles + caching.
- IrradianceService uses placeholder data; integrate PVGIS v5.2 with retry/backoff and caching.
- CostService uses static sample costs; connect to BEIS/vendor price feeds; add VAT/overheads/margins.
- QuoteModel uses placeholder lat/lon for yield; add geocoding from address/postcode.
- Persistence: no database for quotes; no PDF export; no email pipeline.
- Security: no authn/authz, coarse CORS, no API keys/rate limiting.
- Observability: no metrics/tracing dashboards.

## Next steps and improvements

- Replace placeholders with live data/services
  - PVGIS v5.2 integration + caching; handle rate limits.
  - Environment Agency LIDAR DSM ingestion + tile cache (e.g., local/Cloud Storage).
  - Real segmentation model (SegFormer-B0) inference with transformers; evaluate accuracy.
  - Cost/pricing source (BEIS or supplier feed); include install overheads, scaffolding, margins, VAT.
  - Geocoding for address→(lat,lon) and postcode validation.

- Productization
  - Persist quotes (e.g., Postgres) + PDF quote generation + email delivery.
  - Authentication (API key/JWT), rate limiting, stricter CORS.
  - Monitoring: structured logs with request_id, error budgets, uptime checks.
  - Performance hardening and load testing (k6/Gatling); tune Cloud Run min/max instances.

- CI/CD & DevEx
  - Add status badges, coverage, mypy/flake8 (optional) and nightly perf tests.
  - Optionally publish/pull image from Google Artifact Registry (private) instead of public GHCR.
  - Blue/green deploy strategy per environment.

## Environment Variables

```bash
# Model paths
SEGMENTATION_MODEL_PATH=/models/segformer_b0_inria.pth

# Service config
LOG_LEVEL=INFO
MAX_IMAGE_SIZE=1024
DSM_CACHE_ENABLED=true
```

### Segmentation Backend (ML‑6)
- Configure via env:
  - `SEG_BACKEND`: `torch` or `onnx` (default `torch`)
  - `SEG_MODEL_PATH`: path to model (e.g., `/models/segformer-b0.onnx` mounted from GCS)
- Cloud Run GPU (example flags already in CD):
  - `--gpu=type=nvidia-tesla-t4,count=1`
  - `--add-volume name=models,type=cloud-storage,bucket=$GCS_MODELS_BUCKET`
  - `--add-volume-mount volume=models,mount-path=/models,read-only`

## ML-6 (EA LiDAR) usage notes

Start with on-demand WCS (no mirror):
- Set (optional) environment variables:
  - `EA_WCS_DSM` / `EA_WCS_DTM` (defaults point to EA 1 m composite WCS)
  - `EA_LAYER_DSM` / `EA_LAYER_DTM` (defaults: `lidar-composite-dsm-1m` / `lidar-composite-dtm-1m`)
- Ensure `DSM_CACHE_DIR` exists (defaults to `/var/cache/dsm`); derived nDSM windows are cached there as `.npy` files.

Later, when you add a JSON tile index or mirror to GCS:
- Provide `LIDAR_INDEX_JSON` (path to a JSON list of tile records with `bbox_27700`, `res_m`, `dsm_url`, `dtm_url`).
- The service will prefer the mirror/index tiles; WCS becomes a fallback only.

## Performance

- **Segmentation Time**: <1 second for 256×256 images
- **Pitch Estimation Time**: <2 seconds end-to-end
- **Memory Usage**: <512MB RAM
- **Throughput**: 10+ requests/second on CPU

## Contributing

1. Create a GitHub issue with label `cursor-task`
2. Follow the task specifications in `DESIGN.md`
3. Keep diffs under 150 lines
4. Add/extend tests in `tests/` directory
5. Update README usage examples

## License

MIT License - see LICENSE file for details. 