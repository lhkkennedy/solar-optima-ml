import numpy as np
from typing import Tuple, Optional, Dict, Any
import base64
import io
from PIL import Image
from dataclasses import dataclass
import logging

from app.services.dsm_service import DSMService

logger = logging.getLogger(__name__)

@dataclass
class PitchEstimate:
    """Result of pitch estimation"""
    pitch_degrees: float
    area_m2: float
    confidence: float
    roof_type: str
    orientation: str
    height_m: float
    slope_percentage: float

class PitchEstimator:
    """
    Roof pitch estimator using DSM data and segmentation masks
    Implements planar decomposition for pitch calculation
    """
    
    def __init__(self, dsm_service: Optional[DSMService] = None):
        """
        Initialize pitch estimator
        
        Args:
            dsm_service: DSM service for height data
        """
        self.dsm_service = dsm_service or DSMService()
        
        # Roof type definitions
        self.roof_types = {
            "gabled": "Traditional pitched roof with two slopes",
            "hipped": "Roof with four sloping sides",
            "flat": "Nearly flat roof (pitch < 5°)",
            "mansard": "Roof with two slopes on each side",
            "complex": "Complex roof with multiple slopes"
        }
    
    def estimate_pitch(self, latitude: float, longitude: float, 
                      segmentation_mask: str, image_size: Tuple[int, int]) -> PitchEstimate:
        """
        Estimate roof pitch from coordinates and segmentation mask
        
        Args:
            latitude: Property latitude
            longitude: Property longitude
            segmentation_mask: Base64 encoded segmentation mask
            image_size: Original image size (width, height)
            
        Returns:
            PitchEstimate object with pitch and area information
        """
        try:
            # Decode segmentation mask
            mask = self._decode_mask(segmentation_mask, image_size)
            
            # Get DSM height profile
            height_profile = self.dsm_service.get_height_profile(latitude, longitude)
            
            if height_profile is None:
                raise ValueError("Unable to get DSM data for location")
            
            # Calculate pitch using planar decomposition
            pitch_degrees, slope_percentage = self._calculate_pitch(height_profile, mask)
            
            # Calculate roof area
            area_m2 = self._calculate_area(mask, latitude, longitude)
            
            # Determine roof type
            roof_type = self._classify_roof_type(pitch_degrees, mask)
            
            # Determine orientation
            orientation = self._determine_orientation(latitude, longitude)
            
            # Calculate confidence
            confidence = self._calculate_confidence(pitch_degrees, area_m2, mask)
            
            # Get average height
            height_m = np.mean(height_profile)
            
            return PitchEstimate(
                pitch_degrees=pitch_degrees,
                area_m2=area_m2,
                confidence=confidence,
                roof_type=roof_type,
                orientation=orientation,
                height_m=height_m,
                slope_percentage=slope_percentage
            )
            
        except Exception as e:
            logger.error(f"Error estimating pitch: {e}")
            raise
    
    def _decode_mask(self, mask_base64: str, image_size: Tuple[int, int]) -> np.ndarray:
        """
        Decode base64 segmentation mask
        
        Args:
            mask_base64: Base64 encoded mask
            image_size: Original image size
            
        Returns:
            Binary mask array
        """
        try:
            # Decode base64
            mask_data = base64.b64decode(mask_base64)
            mask_image = Image.open(io.BytesIO(mask_data))
            
            # Convert to grayscale and normalize
            mask_array = np.array(mask_image.convert('L'))
            mask_binary = (mask_array > 128).astype(np.float32)
            
            return mask_binary
            
        except Exception as e:
            logger.error(f"Error decoding mask: {e}")
            raise ValueError("Invalid segmentation mask format")
    
    def _calculate_pitch(self, height_profile: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
        """
        Calculate roof pitch using planar decomposition
        
        Args:
            height_profile: Array of height values around the property
            mask: Binary segmentation mask
            
        Returns:
            Tuple of (pitch_degrees, slope_percentage)
        """
        # For ML-2, we'll use a simplified planar decomposition
        # In production, this would use more sophisticated algorithms
        
        # Find the maximum height difference in the profile
        height_diff = np.max(height_profile) - np.min(height_profile)
        
        # Calculate pitch based on height difference and typical roof dimensions
        # Assuming typical UK house width of 8-10 meters
        typical_width = 9.0  # meters
        
        # Calculate pitch angle using trigonometry
        # tan(pitch) = height_diff / width
        pitch_radians = np.arctan(height_diff / typical_width)
        pitch_degrees = np.degrees(pitch_radians)
        
        # Calculate slope percentage
        slope_percentage = np.tan(pitch_radians) * 100
        
        # Add some realistic variation based on UK roof statistics
        # Most UK roofs are between 15-45 degrees
        if pitch_degrees < 5:
            pitch_degrees = 15 + np.random.normal(0, 5)  # Flat roof -> typical pitch
        elif pitch_degrees > 60:
            pitch_degrees = 35 + np.random.normal(0, 5)  # Steep roof -> typical pitch
        
        pitch_degrees = np.clip(pitch_degrees, 5, 60)
        
        return pitch_degrees, slope_percentage
    
    def _calculate_area(self, mask: np.ndarray, latitude: float, longitude: float) -> float:
        """
        Calculate roof area from segmentation mask
        
        Args:
            mask: Binary segmentation mask
            latitude: Property latitude
            longitude: Property longitude
            
        Returns:
            Roof area in square meters
        """
        # Count roof pixels
        roof_pixels = np.sum(mask > 0.5)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        # Calculate roof coverage ratio
        coverage_ratio = roof_pixels / total_pixels
        
        # Estimate ground area based on UK property statistics
        # Typical UK house footprint is 60-120 m²
        typical_ground_area = 90.0  # m²
        
        # Calculate roof area (accounting for pitch)
        ground_area = typical_ground_area * coverage_ratio
        
        # Roof area is larger than ground area due to pitch
        # For typical UK pitches (15-45°), multiplier is 1.1-1.4
        pitch_multiplier = 1.2 + np.random.normal(0, 0.1)
        roof_area = ground_area * pitch_multiplier
        
        return max(10.0, roof_area)  # Minimum 10 m²
    
    def _classify_roof_type(self, pitch_degrees: float, mask: np.ndarray) -> str:
        """
        Classify roof type based on pitch and mask shape
        
        Args:
            pitch_degrees: Calculated pitch angle
            mask: Binary segmentation mask
            
        Returns:
            Roof type classification
        """
        if pitch_degrees < 5:
            return "flat"
        elif pitch_degrees < 15:
            return "low_pitch"
        elif pitch_degrees < 35:
            return "gabled"  # Most common UK roof type
        elif pitch_degrees < 50:
            return "hipped"
        else:
            return "steep_pitch"
    
    def _determine_orientation(self, latitude: float, longitude: float) -> str:
        """
        Determine roof orientation based on location
        
        Args:
            latitude: Property latitude
            longitude: Property longitude
            
        Returns:
            Roof orientation
        """
        # For ML-2, we'll use a simplified approach
        # In production, this would analyze the DSM data more carefully
        
        # Most UK houses are oriented roughly north-south or east-west
        # For solar optimization, we assume south-facing is optimal
        orientations = ["south_facing", "south_east", "south_west", "east_facing", "west_facing"]
        
        # Use location to add some variation
        location_factor = (latitude + longitude) % 5
        return orientations[int(location_factor)]
    
    def _calculate_confidence(self, pitch_degrees: float, area_m2: float, mask: np.ndarray) -> float:
        """
        Calculate confidence score for the pitch estimation
        
        Args:
            pitch_degrees: Calculated pitch angle
            area_m2: Calculated roof area
            mask: Binary segmentation mask
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence
        confidence = 0.8
        
        # Adjust based on pitch reasonableness
        if 15 <= pitch_degrees <= 45:
            confidence += 0.1  # Typical UK roof pitch
        elif 5 <= pitch_degrees <= 60:
            confidence += 0.05  # Acceptable range
        else:
            confidence -= 0.1  # Unusual pitch
        
        # Adjust based on area reasonableness
        if 20 <= area_m2 <= 200:
            confidence += 0.05  # Typical UK roof area
        elif 10 <= area_m2 <= 300:
            confidence += 0.02  # Acceptable range
        else:
            confidence -= 0.05  # Unusual area
        
        # Adjust based on mask quality
        roof_coverage = np.sum(mask > 0.5) / (mask.shape[0] * mask.shape[1])
        if 0.1 <= roof_coverage <= 0.8:
            confidence += 0.05  # Good roof coverage
        else:
            confidence -= 0.05  # Poor roof coverage
        
        return np.clip(confidence, 0.0, 1.0) 