from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import io
from PIL import Image
import numpy as np

from app.models.segmentation import SegmentationModel

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

# Initialize segmentation model
segmentation_model = SegmentationModel()

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

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "SolarOptima ML Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "infer": "/infer",
            "docs": "/docs"
        }
    } 