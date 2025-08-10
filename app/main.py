from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Tuple
import base64
import io
from PIL import Image
import numpy as np

from app.models.segmentation import SegmentationModel
from app.models.pitch_estimator import PitchEstimator

app = FastAPI(
    title="SolarOptima ML Service",
    description="ML micro-service for solar panel assessment and quotation",
    version="1.0.0"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://solaroptima.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
segmentation_model = SegmentationModel()
pitch_estimator = PitchEstimator()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "solaroptima-ml"}

@app.post("/infer")
async def infer_segmentation(file: UploadFile = File(...)):
    """
    Accept PNG overhead imagery and return segmentation mask
    
    Args:
        file: PNG image file (256x256 or larger)
    
    Returns:
        JSON with base64-encoded segmentation mask and confidence score
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and validate image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Validate image size
        if image.size[0] < 256 or image.size[1] < 256:
            raise HTTPException(
                status_code=400, 
                detail="Image must be at least 256x256 pixels"
            )
        
        # Perform segmentation
        mask, confidence = segmentation_model.predict(image)
        
        # Convert mask to base64
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_buffer = io.BytesIO()
        mask_pil.save(mask_buffer, format="PNG")
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
        
        return {
            "mask": mask_base64,
            "confidence": float(confidence),
            "original_size": image.size,
            "mask_size": mask.shape
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

# Add new Pydantic models for pitch endpoint
class Coordinates(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")

class PitchRequest(BaseModel):
    coordinates: Coordinates
    segmentation_mask: str = Field(..., description="Base64 encoded segmentation mask")
    image_size: List[int] = Field(..., min_length=2, max_length=2, description="Original image size [width, height]")

class PitchResponse(BaseModel):
    pitch_degrees: float
    area_m2: float
    confidence: float
    roof_type: str
    orientation: str
    height_m: float
    slope_percentage: float

@app.post("/pitch", response_model=PitchResponse)
async def estimate_pitch(request: PitchRequest):
    """
    Estimate roof pitch from coordinates and segmentation mask
    
    Args:
        request: PitchRequest with coordinates and segmentation mask
    
    Returns:
        PitchResponse with pitch estimation results
    """
    try:
        # Validate coordinates are within UK
        if not pitch_estimator.dsm_service.is_within_uk(
            request.coordinates.latitude, 
            request.coordinates.longitude
        ):
            raise HTTPException(
                status_code=400, 
                detail="Coordinates must be within UK bounds"
            )
        
        # Validate image size
        if len(request.image_size) != 2 or request.image_size[0] < 256 or request.image_size[1] < 256:
            raise HTTPException(
                status_code=400,
                detail="Image size must be at least 256x256 pixels"
            )
        
        # Estimate pitch
        pitch_estimate = pitch_estimator.estimate_pitch(
            latitude=request.coordinates.latitude,
            longitude=request.coordinates.longitude,
            segmentation_mask=request.segmentation_mask,
            image_size=tuple(request.image_size)
        )
        
        return PitchResponse(
            pitch_degrees=round(pitch_estimate.pitch_degrees, 1),
            area_m2=round(pitch_estimate.area_m2, 1),
            confidence=round(pitch_estimate.confidence, 2),
            roof_type=pitch_estimate.roof_type,
            orientation=pitch_estimate.orientation,
            height_m=round(pitch_estimate.height_m, 1),
            slope_percentage=round(pitch_estimate.slope_percentage, 1)
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pitch estimation failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "SolarOptima ML Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "infer": "/infer",
            "pitch": "/pitch",
            "docs": "/docs"
        }
    } 