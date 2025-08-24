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

## Technical Specifications

### API Endpoints

#### POST `/infer`
- **Input**: PNG image (256×256 or larger)
- **Output**: JSON with segmentation mask (256×256)
- **Response**: `{ "mask": "base64_encoded_mask", "confidence": 0.95 }`

#### POST `/pitch`
- **Input**: JSON with coordinates and segmentation mask
- **Output**: Roof pitch estimation
- **Response**: `{ "pitch_degrees": 25.5, "area_m2": 45.2, "confidence": 0.88 }`

#### POST `/quote`
- **Input**: JSON with property details and segmentation results
- **Output**: Complete solar quote
- **Response**: 
```json
{
  "area_m2": 45.2,
  "optimal_kwp": 3.5,
  "estimated_yearly_kwh": 3150,
  "payback_years": 8.2,
  "total_cost_gbp": 8750,
  "items": [
    {
      "sku": "PANEL-400W",
      "quantity": 8,
      "unit_cost": 180,
      "total_cost": 1440
    }
  ]
}
```

### Data Models

#### Segmentation Model
- **Framework**: SegFormer-B0
- **Input**: 256×256 RGB aerial imagery
- **Output**: Binary mask (roof vs. background)
- **Performance**: <1s inference on CPU

#### Pitch Estimator
- **Method**: DSM height lookup + planar decomposition
- **Data Source**: UK LIDAR DSM (Environment Agency)
- **Accuracy**: ±5° for typical UK roofs

#### Irradiance Calculator
- **API**: PVGIS v5.2
- **Coverage**: UK-wide solar resource data
- **Compliance**: MCS MIS 3001 Issue 5.1

### Database Schema

```sql
CREATE TABLE components (
    id SERIAL PRIMARY KEY,
    pricing_date DATE NOT NULL,
    sku VARCHAR(50) NOT NULL,
    cost_gbp DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_components_sku_date ON components(sku, pricing_date);
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/solaroptima

# External APIs
PVGIS_API_URL=https://re.jrc.ec.europa.eu/api/v5_2/
DSM_DATA_PATH=/data/lidar/

# Model paths
SEGMENTATION_MODEL_PATH=/models/segformer_b0_inria.pth
PITCH_MODEL_PATH=/models/pitch_estimator.pkl

# Service config
LOG_LEVEL=INFO
MAX_IMAGE_SIZE=1024
```

## Development Workflow

1. **Task Implementation**: Each task (ML-1 through ML-5) should be implemented as a separate GitHub issue with label `cursor-task`
2. **Code Review**: Human review for hidden side-effects and `.cursorrules` compliance
3. **Testing**: Unit tests for each component, integration tests for API endpoints
4. **Documentation**: Update README.md with usage examples for each new endpoint

## Performance Requirements

- **Inference Time**: <2 seconds end-to-end for quote generation
- **Throughput**: 10 requests/second on single CPU instance
- **Memory**: <512MB RAM usage
- **Accuracy**: >90% roof segmentation accuracy on UK aerial imagery

## Security Considerations

- Input validation for all image uploads
- Rate limiting on API endpoints
- Secure handling of cost data
- CORS configuration for Next.js frontend
- Environment variable management for sensitive data

## Monitoring & Observability

- Structured logging with correlation IDs
- Health check endpoint (`/health`)
- Metrics collection (request count, latency, error rates)
- Model performance monitoring (segmentation accuracy, pitch estimation error)

## Future Enhancements

- Multi-roof property support
- Shading analysis from surrounding buildings
- Seasonal variation in yield calculations
- Integration with property databases (Rightmove, Zoopla)
- Mobile app API support 