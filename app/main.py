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
from app.models.quote import QuoteModel, PropertyDetails, SegmentationResult, PitchResult, CustomerPreferences

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
quote_model = QuoteModel()

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

# Add Pydantic models for quote endpoint
class PropertyDetailsRequest(BaseModel):
    address: str = Field(..., min_length=5, description="Property address")
    postcode: str = Field(..., min_length=5, description="UK postcode")
    property_type: str = Field(..., description="Property type (semi_detached, detached, terraced, flat, etc.)")
    occupancy: str = Field(..., description="Occupancy type (family_of_4, couple, single, etc.)")

class SegmentationResultRequest(BaseModel):
    mask: str = Field(..., description="Base64 encoded segmentation mask")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Segmentation confidence score")

class PitchResultRequest(BaseModel):
    pitch_degrees: float = Field(..., ge=0, le=90, description="Roof pitch angle in degrees")
    area_m2: float = Field(..., ge=5, le=200, description="Roof area in square meters")
    roof_type: str = Field(..., description="Roof type (gabled, hipped, flat, mansard, complex)")
    orientation: str = Field(..., description="Roof orientation (south_facing, south_east, south_west, east_facing, west_facing, north_facing)")

class CustomerPreferencesRequest(BaseModel):
    battery_storage: bool = Field(default=False, description="Include battery storage")
    premium_panels: bool = Field(default=False, description="Use premium panels")
    financing: str = Field(default="cash_purchase", description="Financing option (cash_purchase, finance, lease)")

class QuoteRequest(BaseModel):
    property_details: PropertyDetailsRequest
    segmentation_result: SegmentationResultRequest
    pitch_result: PitchResultRequest
    preferences: CustomerPreferencesRequest

@app.post("/quote")
async def generate_quote(request: QuoteRequest):
    """
    Generate complete solar quote from property details, segmentation, and pitch data
    
    Args:
        request: QuoteRequest with all required data
    
    Returns:
        Complete solar quote with system specification, yield analysis, and financial breakdown
    """
    try:
        # Convert request to internal models
        property_details = PropertyDetails(
            address=request.property_details.address,
            postcode=request.property_details.postcode,
            property_type=request.property_details.property_type,
            occupancy=request.property_details.occupancy
        )
        
        segmentation_result = SegmentationResult(
            mask=request.segmentation_result.mask,
            confidence=request.segmentation_result.confidence
        )
        
        pitch_result = PitchResult(
            pitch_degrees=request.pitch_result.pitch_degrees,
            area_m2=request.pitch_result.area_m2,
            roof_type=request.pitch_result.roof_type,
            orientation=request.pitch_result.orientation
        )
        
        preferences = CustomerPreferences(
            battery_storage=request.preferences.battery_storage,
            premium_panels=request.preferences.premium_panels,
            financing=request.preferences.financing
        )
        
        # Validate request data
        validation_errors = quote_model.validate_quote_request(
            property_details, segmentation_result, pitch_result, preferences
        )
        
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail=f"Validation errors: {', '.join(validation_errors)}"
            )
        
        # Generate quote
        quote_response = quote_model.generate_quote(
            property_details=property_details,
            segmentation_result=segmentation_result,
            pitch_result=pitch_result,
            preferences=preferences
        )
        
        # Convert to dict for JSON response
        quote_dict = {
            "quote_id": quote_response.quote_id,
            "generated_date": quote_response.generated_date,
            "property_details": {
                "address": quote_response.property_details.address,
                "postcode": quote_response.property_details.postcode,
                "property_type": quote_response.property_details.property_type,
                "occupancy": quote_response.property_details.occupancy
            },
            "system_specification": quote_response.system_specification,
            "yield_analysis": quote_response.yield_analysis,
            "financial_analysis": quote_response.financial_analysis,
            "itemized_breakdown": quote_response.itemized_breakdown,
            "warranties": quote_response.warranties,
            "next_steps": quote_response.next_steps,
            "valid_until": quote_response.valid_until,
            "mcs_compliant": quote_response.mcs_compliant,
            "confidence_score": quote_response.confidence_score
        }
        
        return quote_dict
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quote generation failed: {str(e)}")

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
            "quote": "/quote",
            "docs": "/docs"
        }
    } 