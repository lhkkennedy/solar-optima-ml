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
from app.middleware.request_id import add_request_id_middleware
from app.services.dsm_service import DSMService
from app.services.plane_fitting import PlaneFitting
from typing import Optional
from app.settings import get_settings
from app.services import InstanceService
from app.services.procedural_roof import ProceduralPipeline
from app.services.elevation_augment import augment_procedural_model
from app.services.artifacts.geojson_writer import write_geojson
from app.services.artifacts.gltf_writer import write_gltf
from app.services.artifacts.storage import ArtifactStorage
import os

SETTINGS = get_settings()

app = FastAPI(
    title="SolarOptima ML Service",
    description="ML micro-service for solar panel assessment and quotation",
    version="1.0.0"
)

# Register request ID middleware
add_request_id_middleware(app)

# CORS middleware (configurable via CORS_ALLOW_ORIGINS)
# Accept comma-separated list of origins, e.g.:
#   CORS_ALLOW_ORIGINS=http://localhost:8081,http://127.0.0.1:8081
cors_env = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
if cors_env:
    allow_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
else:
    # Sensible defaults for local dev and production
    allow_origins = [
        "http://localhost:3000",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "https://solaroptima.com",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],  # includes OPTIONS
    allow_headers=["*"],
)

# Initialize models
segmentation_model = SegmentationModel()
pitch_estimator = PitchEstimator()
quote_model = QuoteModel()
plane_fitting = PlaneFitting()
ml6_dsm = DSMService()
try:
    instance_service = InstanceService()
except Exception:
    instance_service = None  # allow tests to import app without torch/torchvision
proc_pipeline = ProceduralPipeline()
artifact_storage = ArtifactStorage()

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

# === ML-6: /model3d (scaffold using step-1 and placeholder plane fitting) ===
class Coordinates(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")

class Model3DRequest(BaseModel):
    coordinates: Coordinates  # uses existing Coordinates model above
    bbox_m: float = Field(60, ge=40, le=120)
    image_base64: str | None = None
    image_url: str | None = None
    provider_hint: str | None = None
    return_mesh: bool = False
    segmentation_only: bool = False


class SegMaskRequest(BaseModel):
    image_base64: str | None = None
    image_url: str | None = None


@app.post("/segmask")
async def segmask(req: SegMaskRequest):
    """Return segmentation mask only from a provided image (JSON input)."""
    try:
        if not (req.image_base64 or req.image_url):
            raise HTTPException(status_code=400, detail="Provide image_base64 or image_url")
        # Load image
        if req.image_base64:
            try:
                img_bytes = base64.b64decode(req.image_base64)
            except Exception:
                raise HTTPException(status_code=400, detail="image_base64 must be valid base64")
            img = Image.open(io.BytesIO(img_bytes))
        else:
            import requests as _requests
            r = _requests.get(req.image_url, timeout=10)  # type: ignore[arg-type]
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Optional: basic size check similar to /infer
        if img.size[0] < 256 or img.size[1] < 256:
            raise HTTPException(status_code=400, detail="Image must be at least 256x256 pixels")
        # Run segmentation
        seg_mask, seg_conf = segmentation_model.predict(img)
        try:
            mask_pil = Image.fromarray((seg_mask * 255).astype(np.uint8))
        except Exception:
            mask_pil = Image.fromarray(((seg_mask > 0.5).astype(np.uint8) * 255))
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        mask_base64 = base64.b64encode(buf.getvalue()).decode()
        return {
            "mask": mask_base64,
            "confidence": float(seg_conf or 0.0),
            "original_size": list(img.size),
            "mask_size": list(seg_mask.shape),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"segmask failed: {str(e)}")

@app.post("/model3d")
async def model3d(req: Model3DRequest):
    try:
        # If only segmentation output is requested, ensure an image source is provided
        if req.segmentation_only and not (req.image_base64 or req.image_url):
            raise HTTPException(status_code=400, detail="segmentation_only requires image_base64 or image_url")
        lat = req.coordinates.latitude
        lon = req.coordinates.longitude
        clip = ml6_dsm.locate_and_fetch(lat, lon, req.bbox_m)
        if not clip:
            raise HTTPException(status_code=400, detail="Coordinates must be within UK bounds or DSM unavailable")
        # Elevation fusion (nDSM) prior to plane fitting
        ndsm = ml6_dsm.fuse_elevation(clip)
        # Optional segmentation masking from client image/url
        try:
            mask_arr = None
            seg_conf: Optional[float] = None
            if req.image_base64:
                img_bytes = base64.b64decode(req.image_base64)
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                seg_mask, seg_conf = segmentation_model.predict(img)
                # If caller only wants the segmentation mask, return it now
                if req.segmentation_only:
                    try:
                        mask_pil_only = Image.fromarray((seg_mask * 255).astype(np.uint8))
                    except Exception:
                        mask_pil_only = Image.fromarray(((seg_mask > 0.5).astype(np.uint8) * 255))
                    _buf = io.BytesIO()
                    mask_pil_only.save(_buf, format="PNG")
                    _mask_b64 = base64.b64encode(_buf.getvalue()).decode()
                    return {"mask": _mask_b64, "confidence": float(seg_conf or 0.0), "mask_size": list(seg_mask.shape)}
                mimg = Image.fromarray((seg_mask * 255).astype(np.uint8))
                mimg = mimg.resize((ndsm.ndsm.shape[1], ndsm.ndsm.shape[0]), Image.NEAREST)
                mask_arr = (np.array(mimg) > 127).astype(np.float32)
            elif req.image_url:
                import requests as _requests
                r = _requests.get(req.image_url, timeout=10)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                seg_mask, seg_conf = segmentation_model.predict(img)
                if req.segmentation_only:
                    try:
                        mask_pil_only = Image.fromarray((seg_mask * 255).astype(np.uint8))
                    except Exception:
                        mask_pil_only = Image.fromarray(((seg_mask > 0.5).astype(np.uint8) * 255))
                    _buf = io.BytesIO()
                    mask_pil_only.save(_buf, format="PNG")
                    _mask_b64 = base64.b64encode(_buf.getvalue()).decode()
                    return {"mask": _mask_b64, "confidence": float(seg_conf or 0.0), "mask_size": list(seg_mask.shape)}
                mimg = Image.fromarray((seg_mask * 255).astype(np.uint8))
                mimg = mimg.resize((ndsm.ndsm.shape[1], ndsm.ndsm.shape[0]), Image.NEAREST)
                mask_arr = (np.array(mimg) > 127).astype(np.float32)
            # Footprint mask (optional)
            fp_mask = None
            if clip.footprint and isinstance(clip.footprint.geojson, dict):
                try:
                    coords = clip.footprint.geojson.get("geometry", {}).get("coordinates", [])
                    if coords:
                        # very lightweight rasterization: use bbox of footprint
                        lon_vals = [c[0] for c in coords[0]]
                        lat_vals = [c[1] for c in coords[0]]
                        min_lon_fp, max_lon_fp = min(lon_vals), max(lon_vals)
                        min_lat_fp, max_lat_fp = min(lat_vals), max(lat_vals)
                        H, W = ndsm.ndsm.shape
                        min_lon, min_lat, max_lon, max_lat = clip.bbox_4326
                        xs = np.linspace(min_lon, max_lon, W)
                        ys = np.linspace(min_lat, max_lat, H)
                        within_x = (xs >= min_lon_fp) & (xs <= max_lon_fp)
                        within_y = (ys >= min_lat_fp) & (ys <= max_lat_fp)
                        fp_mask = np.outer(within_y.astype(np.float32), within_x.astype(np.float32))
                except Exception:
                    fp_mask = None
            # Apply masks if present
            if mask_arr is not None:
                ndsm.ndsm *= mask_arr
            if fp_mask is not None:
                ndsm.ndsm *= fp_mask
        except Exception:
            pass
        # Fit planes over nDSM
        try:
            planes, edges = plane_fitting.fit_from_ndsm(clip, ndsm)
        except Exception:
            planes, edges = plane_fitting.fit(clip)
        # Confidence heuristic
        base_conf = 0.65
        if mask_arr is not None and seg_conf is not None:
            base_conf = max(base_conf, 0.6 + 0.4 * float(seg_conf))
        if clip.footprint is not None:
            base_conf = min(1.0, base_conf + 0.05)
        # Elevation energy check
        energy = float(np.mean(np.abs(ndsm.ndsm))) if isinstance(ndsm.ndsm, np.ndarray) else 0.0
        if energy < 0.1:
            base_conf = min(base_conf, 0.7)
        response = {
            "planes": [
                {
                    "id": p.id,
                    "normal": list(p.normal),
                    "pitch_deg": p.pitch_deg,
                    "aspect_deg": p.aspect_deg,
                    "polygon": [[lon, lat, h] for (lon, lat, h) in p.polygon],
                    "area_m2": p.area_m2,
                }
                for p in planes
            ],
            "edges": [{"a": list(e.a), "b": list(e.b)} for e in edges],
            "summary": {
                "area_m2": sum(p.area_m2 for p in planes),
                "max_height_m": max(max(v[2] for v in p.polygon) for p in planes),
                "roof_type": "gabled",
            },
            "confidence": round(base_conf, 2),
            "artifacts": {"geojson_url": None, "gltf_url": None},
            "bbox": {"epsg27700": list(clip.bbox_27700), "epsg4326": list(clip.bbox_4326)},
        }
        # Optional: procedural roof reconstruction (feature-flagged)
        if bool(int(str(int(SETTINGS.__dict__.get("proc_roof_enable", 0))) )):
            try:
                procedural_roofs = []
                total_count = 0
                truncated = False
                image_rgb = None
                W = H = None
                # Use provided image as instance detection source
                if 'img' in locals():
                    image_rgb = img
                # If no image was provided via base64/url but segmentation produced an image, reuse it
                if image_rgb is not None and instance_service is not None:
                    if image_rgb.mode != "RGB":
                        image_rgb = image_rgb.convert("RGB")
                    np_img = np.array(image_rgb)
                    H, W = np_img.shape[0], np_img.shape[1]
                    # Pixel to lon/lat mapping across bbox
                    min_lon, min_lat, max_lon, max_lat = clip.bbox_4326
                    def p2ll(px: float, py: float) -> Tuple[float, float]:
                        lon = min_lon + (max_lon - min_lon) * (px / max(1, W - 1))
                        lat = min_lat + (max_lat - min_lat) * (py / max(1, H - 1))
                        return float(lon), float(lat)

                    instances = instance_service.detect(np_img)
                    total_count = len(instances)
                    for inst in instances:
                        model = proc_pipeline.run(inst, pixel_to_lonlat=p2ll)
                        if model is None:
                            continue
                        # Elevation augment with nDSM
                        model = augment_procedural_model(clip, ndsm, model)
                        # Artifacts
                        gj_path = write_geojson(model)
                        glb_path = write_gltf(model)
                        model.artifacts["geojson_url"] = artifact_storage.store(gj_path)
                        model.artifacts["gltf_url"] = artifact_storage.store(glb_path)
                        # Append to response
                        procedural_roofs.append({
                            "footprint_regularized": model.footprint_regularized,
                            "parts": model.parts,
                            "metrics": model.metrics,
                            "artifacts": model.artifacts,
                        })
                if procedural_roofs:
                    response["procedural_roofs"] = procedural_roofs
                    response["total_count"] = total_count
                    response["truncated"] = truncated
            except Exception:
                # Do not fail core endpoint if procedural step errors
                pass
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model3d failed: {str(e)}")

# Add new Pydantic models for pitch endpoint

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
            "model3d": "/model3d",
            "pitch": "/pitch",
            "quote": "/quote",
            "docs": "/docs"
        }
    } 