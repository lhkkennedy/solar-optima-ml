import numpy as np
import requests
from typing import Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DSMData:
    """DSM data structure for a location"""
    latitude: float
    longitude: float
    elevation_m: float
    resolution_m: float
    data_source: str
    confidence: float

class DSMService:
    """
    Service for accessing UK LIDAR DSM (Digital Surface Model) data
    Uses Environment Agency LIDAR data for roof height information
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize DSM service
        
        Args:
            cache_enabled: Whether to cache DSM data for performance
        """
        self.cache_enabled = cache_enabled
        self._cache = {}  # Simple in-memory cache
        self.base_url = "https://environment.data.gov.uk/ds/survey"
        
        # UK coordinate bounds (approximate)
        self.uk_bounds = {
            "min_lat": 49.9,
            "max_lat": 60.9,
            "min_lon": -8.6,
            "max_lon": 1.8
        }
    
    def is_within_uk(self, latitude: float, longitude: float) -> bool:
        """
        Check if coordinates are within UK bounds
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            True if coordinates are within UK
        """
        return (
            self.uk_bounds["min_lat"] <= latitude <= self.uk_bounds["max_lat"] and
            self.uk_bounds["min_lon"] <= longitude <= self.uk_bounds["max_lon"]
        )
    
    def get_cache_key(self, latitude: float, longitude: float) -> str:
        """Generate cache key for coordinates"""
        # Round to 3 decimal places (~100m precision)
        lat_rounded = round(latitude, 3)
        lon_rounded = round(longitude, 3)
        return f"{lat_rounded}_{lon_rounded}"
    
    def get_dsm_data(self, latitude: float, longitude: float) -> Optional[DSMData]:
        """
        Get DSM data for given coordinates
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            DSMData object or None if not available
        """
        # Validate coordinates
        if not self.is_within_uk(latitude, longitude):
            logger.warning(f"Coordinates ({latitude}, {longitude}) outside UK bounds")
            return None
        
        # Check cache first
        cache_key = self.get_cache_key(latitude, longitude)
        if self.cache_enabled and cache_key in self._cache:
            logger.debug(f"DSM data found in cache for {cache_key}")
            return self._cache[cache_key]
        
        try:
            # For ML-2, we'll use a placeholder implementation
            # In production, this would call the actual Environment Agency API
            dsm_data = self._get_placeholder_dsm_data(latitude, longitude)
            
            # Cache the result
            if self.cache_enabled and dsm_data:
                self._cache[cache_key] = dsm_data
                logger.debug(f"Cached DSM data for {cache_key}")
            
            return dsm_data
            
        except Exception as e:
            logger.error(f"Error getting DSM data for ({latitude}, {longitude}): {e}")
            return None
    
    def _get_placeholder_dsm_data(self, latitude: float, longitude: float) -> DSMData:
        """
        Generate placeholder DSM data for ML-2 development
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Placeholder DSMData object
        """
        # Generate realistic elevation based on UK topography
        # Higher elevations in Scotland and Wales, lower in England
        base_elevation = 50.0  # Base elevation in meters
        
        # Add variation based on latitude (higher in north)
        lat_factor = (latitude - 50.0) * 100  # ~100m per degree latitude
        
        # Add some random variation for realism
        random_factor = np.random.normal(0, 20)
        
        elevation = base_elevation + lat_factor + random_factor
        elevation = max(0, elevation)  # Ensure non-negative
        
        # Generate confidence based on data quality
        confidence = 0.85 + np.random.normal(0, 0.1)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return DSMData(
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation,
            resolution_m=1.0,
            data_source="Environment Agency LIDAR (placeholder)",
            confidence=confidence
        )
    
    def get_height_profile(self, latitude: float, longitude: float, 
                          radius_m: float = 50.0, points: int = 9) -> np.ndarray:
        """
        Get height profile around a point for pitch calculation
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            radius_m: Radius in meters to sample
            points: Number of points to sample (creates a grid)
            
        Returns:
            Array of height values in a grid around the point
        """
        if not self.is_within_uk(latitude, longitude):
            return None
        
        # Convert radius to approximate lat/lon offsets
        # Rough approximation: 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
        lat_offset = radius_m / 111000.0
        lon_offset = radius_m / (111000.0 * np.cos(np.radians(latitude)))
        
        # Create grid of points
        heights = np.zeros((points, points))
        
        for i in range(points):
            for j in range(points):
                # Calculate offset from center
                lat_offset_i = (i - points//2) * lat_offset / (points//2)
                lon_offset_j = (j - points//2) * lon_offset / (points//2)
                
                # Get DSM data for this point
                dsm_data = self.get_dsm_data(
                    latitude + lat_offset_i,
                    longitude + lon_offset_j
                )
                
                if dsm_data:
                    heights[i, j] = dsm_data.elevation_m
                else:
                    # Use center point elevation if this point fails
                    center_dsm = self.get_dsm_data(latitude, longitude)
                    heights[i, j] = center_dsm.elevation_m if center_dsm else 0.0
        
        return heights 