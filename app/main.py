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

SETTINGS = get_settings()

app = FastAPI(
    title="SolarOptima ML Service",
    description="ML micro-service for solar panel assessment and quotation",
    version="1.0.0"
)

# Register request ID middleware
add_request_id_middleware(app)

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
plane_fitting = PlaneFitting()
ml6_dsm = DSMService()

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

@app.post("/model3d")
async def model3d(req: Model3DRequest):
    try:
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
                # Lightweight per-building: use segmentation mask if provided, else skip
                from app.services.procedural_roof import PBSRService, RidgeDetectionService, ProceduralRoofSynthesizer
                pbsr = PBSRService()
                ridge = RidgeDetectionService()
                synth = ProceduralRoofSynthesizer()
                # Derive a single-building mask from seg_mask if present; fallback to whole bbox
                H, W = ndsm.ndsm.shape
                if 'mask_arr' in locals() and mask_arr is not None:
                    bin_mask = (mask_arr > 0.5).astype(np.uint8)
                else:
                    bin_mask = np.ones((H, W), dtype=np.uint8)

                match = pbsr.match(bin_mask)
                ridge_results = []
                # Create a dummy grayscale from nDSM energy for edge detection
                nd_norm = ndsm.ndsm
                if isinstance(nd_norm, np.ndarray):
                    nd_norm = (255.0 * (nd_norm - np.nanmin(nd_norm)) / (np.nanmax(nd_norm) - np.nanmin(nd_norm) + 1e-6)).astype(np.uint8)
                else:
                    nd_norm = np.zeros((H, W), dtype=np.uint8)

                for rect in (match.rects if match else []):
                    crop = nd_norm[max(0, rect.y):rect.y+rect.h, max(0, rect.x):rect.x+rect.w]
                    if crop.size == 0:
                        ridge_results.append(ridge.analyze_part(nd_norm, rect))
                    else:
                        ridge_results.append(ridge.analyze_part(nd_norm, rect))

                # pixel to lonlat mapper
                min_lon, min_lat, max_lon, max_lat = clip.bbox_4326
                def p2ll(px, py):
                    lon = min_lon + (max_lon - min_lon) * (px / max(1, W - 1))
                    lat = min_lat + (max_lat - min_lat) * (py / max(1, H - 1))
                    return [float(lon), float(lat)]

                # Assemble 2D procedural model first
                proc_model = synth.assemble(match, ridge_results, p2ll) if match else None

                # Augment with elevation: sample heights along ridges and estimate per-part pitch/aspect
                if proc_model and isinstance(ndsm.ndsm, np.ndarray):
                    # compute 3D ridges and add per-part pitch_deg/aspect_deg
                    parts_aug = []
                    for rect, rr, part in zip(match.rects if match else [], ridge_results, proc_model.parts):
                        # Ridge 3D
                        ridges3d = []
                        for seg in part.get("ridges", []):
                            (lon0, lat0), (lon1, lat1) = seg
                            # map lon/lat back to pixel indices
                            def ll2p(lon, lat):
                                x = (lon - min_lon) / (max_lon - min_lon + 1e-12) * (W - 1)
                                y = (lat - min_lat) / (max_lat - min_lat + 1e-12) * (H - 1)
                                return int(round(x)), int(round(y))
                            x0, y0 = ll2p(lon0, lat0)
                            x1, y1 = ll2p(lon1, lat1)
                            x0 = max(0, min(W - 1, x0)); x1 = max(0, min(W - 1, x1))
                            y0 = max(0, min(H - 1, y0)); y1 = max(0, min(H - 1, y1))
                            z0 = float(ndsm.ndsm[y0, x0])
                            z1 = float(ndsm.ndsm[y1, x1])
                            ridges3d.append([[lon0, lat0, z0], [lon1, lat1, z1]])

                        # Plane fit on nDSM within rect to get pitch/aspect
                        window = ndsm.ndsm[max(0, rect.y):rect.y+rect.h, max(0, rect.x):rect.x+rect.w]
                        pitch_deg = None; aspect_deg = None
                        if window.size > 0:
                            yy, xx = np.mgrid[0:window.shape[0], 0:window.shape[1]]
                            z = window.astype(np.float32)
                            # mask zero (outside) values to reduce bias
                            m = z > 0
                            if m.sum() > 50:
                                X = np.column_stack([xx[m].ravel(), yy[m].ravel(), np.ones(int(m.sum()))])
                                yv = z[m].ravel()
                                # least squares fit z = a*x + b*y + c
                                try:
                                    coeffs, *_ = np.linalg.lstsq(X, yv, rcond=None)
                                    a, b = float(coeffs[0]), float(coeffs[1])
                                    # normal ~ ( -a, -b, 1 )
                                    nx, ny, nz = -a, -b, 1.0
                                    norm = (nx**2 + ny**2 + nz**2) ** 0.5
                                    nx /= norm; ny /= norm; nz /= norm
                                    pitch_deg = float(np.degrees(np.arccos(abs(nz))))
                                    # aspect: downslope direction
                                    aspect_rad = float(np.arctan2(nx, ny))
                                    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
                                except Exception:
                                    pass

                        part_aug = dict(part)
                        part_aug["ridges_3d"] = ridges3d
                        if pitch_deg is not None and aspect_deg is not None:
                            part_aug["pitch_deg"] = round(pitch_deg, 1)
                            part_aug["aspect_deg"] = round(aspect_deg, 1)
                        parts_aug.append(part_aug)
                    proc_model.parts = parts_aug

                if proc_model:
                    response["procedural_roofs"] = [{
                        "footprint_regularized": proc_model.footprint_regularized,
                        "parts": proc_model.parts,
                        "metrics": proc_model.metrics,
                        "artifacts": proc_model.artifacts,
                    }]
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