import pytest
import base64
import io
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def create_test_image(width: int = 256, height: int = 256) -> bytes:
    """Create a test PNG image"""
    image = Image.new('RGB', (width, height), color='blue')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "solaroptima-ml"

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "SolarOptima ML Service"
    assert "infer" in data["endpoints"]

def test_infer_valid_image():
    """Test /infer endpoint with valid image"""
    # Create test image
    image_data = create_test_image(256, 256)
    
    # Make request
    files = {"file": ("test.png", image_data, "image/png")}
    response = client.post("/infer", files=files)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert "mask" in data
    assert "confidence" in data
    assert "original_size" in data
    assert "mask_size" in data
    
    # Validate data types
    assert isinstance(data["mask"], str)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["original_size"], list)
    assert isinstance(data["mask_size"], list)
    
    # Validate confidence range
    assert 0.0 <= data["confidence"] <= 1.0
    
    # Validate mask is base64 encoded
    try:
        mask_data = base64.b64decode(data["mask"])
        mask_image = Image.open(io.BytesIO(mask_data))
        assert mask_image.size == (256, 256)
    except Exception:
        pytest.fail("Mask is not valid base64 encoded PNG")

def test_infer_large_image():
    """Test /infer endpoint with larger image"""
    # Create larger test image
    image_data = create_test_image(512, 512)
    
    # Make request
    files = {"file": ("test.png", image_data, "image/png")}
    response = client.post("/infer", files=files)
    
    # Should still work
    assert response.status_code == 200
    data = response.json()
    assert "mask" in data

def test_infer_small_image():
    """Test /infer endpoint with image too small"""
    # Create small test image
    image_data = create_test_image(128, 128)
    
    # Make request
    files = {"file": ("test.png", image_data, "image/png")}
    response = client.post("/infer", files=files)
    
    # Should fail
    assert response.status_code == 400
    data = response.json()
    assert "Image must be at least 256x256 pixels" in data["detail"]

def test_infer_invalid_file():
    """Test /infer endpoint with non-image file"""
    # Create text file
    text_data = b"This is not an image"
    
    # Make request
    files = {"file": ("test.txt", text_data, "text/plain")}
    response = client.post("/infer", files=files)
    
    # Should fail
    assert response.status_code == 400
    data = response.json()
    assert "File must be an image" in data["detail"]

def test_infer_no_file():
    """Test /infer endpoint without file"""
    response = client.post("/infer")
    assert response.status_code == 422  # Validation error

def test_segmentation_model_placeholder():
    """Test segmentation model placeholder functionality"""
    from app.models.segmentation import SegmentationModel
    
    # Create model
    model = SegmentationModel()
    
    # Create test image
    image = Image.new('RGB', (256, 256), color='blue')
    
    # Get prediction
    mask, confidence = model.predict(image)
    
    # Validate output
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (256, 256)
    assert mask.dtype == np.float32
    assert np.all((mask >= 0) & (mask <= 1))  # Values between 0 and 1
    
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0

def test_mask_generation():
    """Test placeholder mask generation"""
    from app.models.segmentation import SegmentationModel
    
    model = SegmentationModel()
    
    # Create test image array
    image_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Generate mask
    mask = model._generate_placeholder_mask(image_array)
    
    # Validate mask
    assert mask.shape == (256, 256)
    assert mask.dtype == np.float32
    assert np.all((mask >= 0) & (mask <= 1))
    
    # Should have some roof pixels (not all zeros)
    assert np.sum(mask) > 0 