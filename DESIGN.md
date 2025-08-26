# SolarOptima ML Micro-Service Design

## Project Overview

SolarOptima ML micro-service provides automated solar panel assessment and quotation capabilities for UK residential properties. The service analyzes aerial imagery to segment roofs, estimate optimal panel configurations, and generate MCS-compliant yield calculations.

## 4.1 Architecture Decisions

| Decision | Lean Default | Rationale |
|----------|--------------|-----------|
| **Framework** | FastAPI + Pydantic | Small footprint, async, first-class typing, plays nicely with Python ML stacks |
| **Model** | SegFormer-B0 pre-trained on Inria Aerial Roof Segmentation, fine-tuned with 10–20 labelled samples of UK roofs if available | Lightweight (≈ 15 MB), runs on CPU within <1 s for 256×256 tiles |
| **3-D shape** | Simple height-by-lookup from UK LIDAR DSM (Environment Agency) + Planar decomposition | Produces roof pitch & area good enough for MCS yield calcs |
| **Irradiance** | Call PVGIS v5 JSON API (https://re.jrc.ec.europa.eu/api/v5_2/) | Free, MCS-referenced source |
| **Cost tables** | Postgres table `components(pricing_date, sku, cost_gbp)` loaded from latest BEIS small-scale PV cost dataset | Government-backed pricing data |
| **Compliance** | Implement MIS 3001 Issue 5.1 formulas for kWh yield & payback | MCS certification requirements |

## 4.2 Task Breakdown

### ML-1: Dockerfile + FastAPI Skeleton
**Description**: Create basic FastAPI service with `/infer` endpoint accepting PNG overhead imagery, returns segmentation mask  
**Done when**: `docker run` responds 200 with 256×256 mask  
**Files to touch**:
- `Dockerfile`
- `app/main.py`
- `app/models/segmentation.py`
- `requirements.txt`
- `tests/test_infer.py`

### ML-2: Roof Pitch Estimator
**Description**: Add `/pitch` endpoint using DSM lookup for roof pitch estimation  
**Done when**: Unit test passes with synthetic roof  
**Files to touch**:
- `app/models/pitch_estimator.py`
- `app/services/dsm_service.py`
- `app/main.py` (add pitch endpoint)
- `tests/test_pitch.py`
- `README.md` (update usage)

### ML-3: Quote Generation
**Description**: Combine segmentation and pitch into `/quote` returning JSON with area, optimal kWp, and itemized list  
**Done when**: JSON schema validated  
**Files to touch**:
- `app/models/quote.py`
- `app/services/irradiance_service.py`
- `app/services/cost_service.py`
- `app/main.py` (add quote endpoint)
- `tests/test_quote.py`
- `README.md` (update usage)

### ML-4: Integration Testing
**Description**: Integration test hitting `/quote` from Next.js app via `/api/quote` proxy  
**Done when**: Playwright test passes  
**Files to touch**:
- `tests/integration/test_api_quote.py`
- `playwright.config.js`
- `tests/integration/fixtures/`

### ML-5: CI/CD Pipeline
**Description**: GitHub Action to build & push Docker image, run unit + integration tests  
**Done when**: Green build  
**Files to touch**:
- `.github/workflows/ml-service.yml`
- `docker-compose.test.yml`
- `.dockerignore`

### ML-6: Procedural Roof – Shape Configuration & Ridge Detection
**Description**: Detect precise roof part arrangements (Configuration Recognizer) and estimate roof ridge topology for each building. This unlocks parameterised procedural roof models used later for 3-D synthesis and costing.
**Sub-Tasks**:
- **ML-6a**: *Synthetic Dataset Generator* – Script to create ≥120 k labelled edge-map images for supported roof family types (I, L, T, U, Z). Includes noise augmentation (random curve scribbles, edge dropout) and basic train/val/test split.
- **ML-6b**: *Roof Family Classifier* – ResNet-18 backbone adapted to 5–8 roof families. Target ≥95 % top-1 accuracy on synthetic validation set.
- **ML-6c**: *Configuration Recognizer* – Exhaustive but de-duplicated search across 4520 canonical configurations using IOU-based matching. Implements preprocessing (crop, resize to 120×120, padding, flips, rotations).
- **ML-6d**: *Ridge Configuration Recognizer* – For each roof part, maximise supporting edge-points within 5 % distance of candidate ridges; refine overlaps via binary search to 90 % coverage threshold.
- **ML-6e**: *Pipeline Integration* – Expose `/procedural_roof` service endpoint that consumes segmentation mask and returns parameter set \(\{x,y,w,h\}\) plus ridge graph.

**Done when**:
1. Unit tests cover classifier accuracy, recogniser IOU matching, and ridge detection logic.
2. End-to-end test returning correct parameters for synthetic multi-roof example passes.
3. Documentation (this file, README, Ops guide) updated with training instructions and environment variables `RIDGE_MODEL_PATH`, `SYNTH_DATA_PATH`.
4. GitHub Action extended to train classifier nightly when `SYNTH_DATA_VERSION` changes.

**Files to touch**:
- `tools/procedural_roof/gen_synth_footprints.py` (extend for ridge dataset)
- `app/services/procedural_roof/pipeline.py` (add recogniser + ridges)
- `app/services/procedural_roof/ridge_detection.py` (new)
- `tests/test_procedural_roof.py` (expand)
- `README.md` (usage)

---

## Technical Specifications

### API Endpoints

#### POST `/infer`
- **Input**: PNG image (256×256 or larger)
- **Output**: JSON with segmentation mask (256×256)
- **Response**: `