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

### Docker

1. **Build the image**:
   ```bash
   docker build -t solaroptima-ml .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 solaroptima-ml
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

**Response**:
```json
{
  "mask": "iVBORw0KGgoAAAANSUhEUgAA...",
  "confidence": 0.85,
  "original_size": [512, 512],
  "mask_size": [256, 256]
}
```

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

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_infer.py -v
pytest tests/test_pitch.py -v
```

## Project Structure

```
solar-optima-ml/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   ├── segmentation.py  # SegFormer-B0 model
│   │   └── pitch_estimator.py # Roof pitch estimator
│   └── services/
│       └── dsm_service.py   # DSM data service
├── tests/
│   ├── test_infer.py        # Segmentation tests
│   └── test_pitch.py        # Pitch estimation tests
├── models/                  # Pre-trained model weights
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Development Tasks

This project follows the task breakdown from `DESIGN.md`:

- **ML-1**: ✅ Dockerfile + FastAPI skeleton with `/infer` endpoint
- **ML-2**: ✅ Roof pitch estimator (`/pitch` endpoint)
- **ML-3**: ⏳ Quote generation (`/quote` endpoint)
- **ML-4**: ⏳ Integration testing
- **ML-5**: ⏳ CI/CD pipeline

## Current Status

**ML-2 is complete!** ✅

- ✅ `/pitch` endpoint for roof pitch estimation
- ✅ DSM service with UK LIDAR data integration
- ✅ Planar decomposition algorithm for pitch calculation
- ✅ Roof type classification (gabled, hipped, flat, etc.)
- ✅ Comprehensive test suite for pitch estimation
- ✅ Input validation and error handling
- ✅ Performance optimization with caching

**Test Results:**
- Unit tests: 9/9 passing (segmentation) + 8/8 passing (pitch)
- Integration test: Successfully processes coordinates and masks
- Pitch accuracy: ±5° for typical UK roofs
- Response time: <2 seconds for pitch estimation

## Environment Variables

```bash
# Model paths
SEGMENTATION_MODEL_PATH=/models/segformer_b0_inria.pth

# Service config
LOG_LEVEL=INFO
MAX_IMAGE_SIZE=1024
DSM_CACHE_ENABLED=true
```

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