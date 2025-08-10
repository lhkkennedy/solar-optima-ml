import pytest
import base64
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
from app.models.pitch_estimator import PitchEstimator
from app.services.dsm_service import DSMService

client = TestClient(app)

def create_test_mask(width: int = 256, height: int = 256) -> str:
    """Create a test segmentation mask"""
    # Create a simple rectangular "roof" mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Add a rectangular roof area (center 60% of image)
    center_x, center_y = width // 2, height // 2
    roof_width, roof_height = int(width * 0.6), int(height * 0.5)
    
    x1 = max(0, center_x - roof_width // 2)
    x2 = min(width, center_x + roof_width // 2)
    y1 = max(0, center_y - roof_height // 2)
    y2 = min(height, center_y + roof_height // 2)
    
    mask[y1:y2, x1:x2] = 255
    
    # Convert to base64
    mask_image = Image.fromarray(mask)
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    mask_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return mask_base64

def test_pitch_endpoint_valid_request():
    """Test /pitch endpoint with valid request"""
    # Create test data
    mask_base64 = create_test_mask()
    
    request_data = {
        "coordinates": {
            "latitude": 51.5074,
            "longitude": -0.1278
        },
        "segmentation_mask": mask_base64,
        "image_size": [256, 256]
    }
    
    # Make request
    response = client.post("/pitch", json=request_data)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert "pitch_degrees" in data
    assert "area_m2" in data
    assert "confidence" in data
    assert "roof_type" in data
    assert "orientation" in data
    assert "height_m" in data
    assert "slope_percentage" in data
    
    # Validate data types and ranges
    assert isinstance(data["pitch_degrees"], float)
    assert isinstance(data["area_m2"], float)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["roof_type"], str)
    assert isinstance(data["orientation"], str)
    assert isinstance(data["height_m"], float)
    assert isinstance(data["slope_percentage"], float)
    
    # Validate reasonable ranges
    assert 5.0 <= data["pitch_degrees"] <= 60.0
    assert 10.0 <= data["area_m2"] <= 300.0
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["roof_type"] in ["flat", "low_pitch", "gabled", "hipped", "steep_pitch"]
    assert data["orientation"] in ["south_facing", "south_east", "south_west", "east_facing", "west_facing"]

def test_pitch_endpoint_outside_uk():
    """Test /pitch endpoint with coordinates outside UK"""
    mask_base64 = create_test_mask()
    
    request_data = {
        "coordinates": {
            "latitude": 40.7128,  # New York
            "longitude": -74.0060
        },
        "segmentation_mask": mask_base64,
        "image_size": [256, 256]
    }
    
    response = client.post("/pitch", json=request_data)
    assert response.status_code == 400
    assert "within UK bounds" in response.json()["detail"]

def test_pitch_endpoint_invalid_mask():
    """Test /pitch endpoint with invalid mask"""
    request_data = {
        "coordinates": {
            "latitude": 51.5074,
            "longitude": -0.1278
        },
        "segmentation_mask": "invalid_base64",
        "image_size": [256, 256]
    }
    
    response = client.post("/pitch", json=request_data)
    assert response.status_code == 400

def test_pitch_endpoint_small_image():
    """Test /pitch endpoint with small image size"""
    mask_base64 = create_test_mask()
    
    request_data = {
        "coordinates": {
            "latitude": 51.5074,
            "longitude": -0.1278
        },
        "segmentation_mask": mask_base64,
        "image_size": [128, 128]  # Too small
    }
    
    response = client.post("/pitch", json=request_data)
    assert response.status_code == 400
    assert "at least 256x256 pixels" in response.json()["detail"]

def test_dsm_service_uk_bounds():
    """Test DSM service UK bounds checking"""
    dsm_service = DSMService()
    
    # Test UK coordinates
    assert dsm_service.is_within_uk(51.5074, -0.1278)  # London
    assert dsm_service.is_within_uk(55.9533, -3.1883)  # Edinburgh
    assert dsm_service.is_within_uk(51.4816, -3.1791)  # Cardiff
    
    # Test non-UK coordinates
    assert not dsm_service.is_within_uk(40.7128, -74.0060)  # New York
    assert not dsm_service.is_within_uk(48.8566, 2.3522)    # Paris

def test_pitch_estimator_integration():
    """Test pitch estimator integration"""
    dsm_service = DSMService()
    estimator = PitchEstimator(dsm_service)
    
    # Test with valid data
    mask_base64 = create_test_mask()
    
    pitch_estimate = estimator.estimate_pitch(
        latitude=51.5074,
        longitude=-0.1278,
        segmentation_mask=mask_base64,
        image_size=(256, 256)
    )
    
    # Validate estimate
    assert 5.0 <= pitch_estimate.pitch_degrees <= 60.0
    assert 10.0 <= pitch_estimate.area_m2 <= 300.0
    assert 0.0 <= pitch_estimate.confidence <= 1.0
    assert pitch_estimate.roof_type in ["flat", "low_pitch", "gabled", "hipped", "steep_pitch"]

def test_dsm_service_height_profile():
    """Test DSM service height profile generation"""
    dsm_service = DSMService()
    
    height_profile = dsm_service.get_height_profile(51.5074, -0.1278)
    
    assert height_profile is not None
    assert height_profile.shape == (9, 9)  # Default grid size
    assert np.all(height_profile >= 0)  # Heights should be non-negative

def test_pitch_estimator_mask_decoding():
    """Test pitch estimator mask decoding"""
    estimator = PitchEstimator()
    
    # Create test mask
    mask_base64 = create_test_mask()
    
    # Decode mask
    mask = estimator._decode_mask(mask_base64, (256, 256))
    
    # Validate mask
    assert mask.shape == (256, 256)
    assert mask.dtype == np.float32
    assert np.all((mask >= 0) & (mask <= 1))
    assert np.sum(mask) > 0  # Should have some roof pixels 