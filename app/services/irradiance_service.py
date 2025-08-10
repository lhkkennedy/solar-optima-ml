import requests
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import time
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class IrradianceData:
    """Solar irradiance data for a location"""
    latitude: float
    longitude: float
    annual_irradiance_kwh_m2: float
    monthly_irradiance: Dict[str, float]
    optimal_angle: float
    data_source: str
    confidence: float

class IrradianceService:
    """
    Service for accessing UK solar irradiance data via PVGIS v5.2 API
    Provides location-based solar resource data for yield calculations
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize irradiance service
        
        Args:
            cache_enabled: Whether to cache irradiance data for performance
        """
        self.cache_enabled = cache_enabled
        self._cache = {}  # Simple in-memory cache
        self.base_url = "https://re.jrc.ec.europa.eu/api/v5_2"
        
        # UK coordinate bounds (same as DSM service)
        self.uk_bounds = {
            "min_lat": 49.9,
            "max_lat": 60.9,
            "min_lon": -8.6,
            "max_lon": 1.8
        }
        
        # Rate limiting (PVGIS has limits)
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
    
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
        # Round to 2 decimal places (~1km precision)
        lat_rounded = round(latitude, 2)
        lon_rounded = round(longitude, 2)
        return f"irradiance_{lat_rounded}_{lon_rounded}"
    
    def get_irradiance_data(self, latitude: float, longitude: float) -> Optional[IrradianceData]:
        """
        Get solar irradiance data for given coordinates
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            IrradianceData object or None if not available
        """
        # Validate coordinates
        if not self.is_within_uk(latitude, longitude):
            logger.warning(f"Coordinates ({latitude}, {longitude}) outside UK bounds")
            return None
        
        # Check cache first
        cache_key = self.get_cache_key(latitude, longitude)
        if self.cache_enabled and cache_key in self._cache:
            logger.debug(f"Irradiance data found in cache for {cache_key}")
            return self._cache[cache_key]
        
        try:
            # Rate limiting
            self._respect_rate_limit()
            
            # For ML-3, we'll use a placeholder implementation
            # In production, this would call the actual PVGIS API
            irradiance_data = self._get_placeholder_irradiance_data(latitude, longitude)
            
            # Cache the result
            if self.cache_enabled and irradiance_data:
                self._cache[cache_key] = irradiance_data
                logger.debug(f"Cached irradiance data for {cache_key}")
            
            return irradiance_data
            
        except Exception as e:
            logger.error(f"Error getting irradiance data for ({latitude}, {longitude}): {e}")
            return None
    
    def _respect_rate_limit(self):
        """Respect PVGIS API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_placeholder_irradiance_data(self, latitude: float, longitude: float) -> IrradianceData:
        """
        Generate placeholder irradiance data for ML-3 development
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Placeholder IrradianceData object
        """
        # Generate realistic irradiance based on UK solar resource
        # UK average annual irradiance is ~1000-1200 kWh/m²
        base_irradiance = 1100.0  # kWh/m²/year
        
        # Add variation based on latitude (higher in south)
        lat_factor = (55.0 - latitude) * 20  # ~20 kWh/m² per degree latitude
        
        # Add some random variation for realism
        import random
        random_factor = random.uniform(-50, 50)
        
        annual_irradiance = base_irradiance + lat_factor + random_factor
        annual_irradiance = max(800, min(1400, annual_irradiance))  # Realistic range
        
        # Generate monthly breakdown (UK seasonal variation)
        monthly_irradiance = {
            "jan": annual_irradiance * 0.03,  # Winter months
            "feb": annual_irradiance * 0.04,
            "mar": annual_irradiance * 0.07,
            "apr": annual_irradiance * 0.09,
            "may": annual_irradiance * 0.11,
            "jun": annual_irradiance * 0.12,  # Summer peak
            "jul": annual_irradiance * 0.12,
            "aug": annual_irradiance * 0.11,
            "sep": annual_irradiance * 0.09,
            "oct": annual_irradiance * 0.07,
            "nov": annual_irradiance * 0.04,
            "dec": annual_irradiance * 0.03
        }
        
        # Calculate optimal tilt angle (roughly latitude * 0.76 + 3.1)
        optimal_angle = latitude * 0.76 + 3.1
        
        # Generate confidence based on data quality
        confidence = 0.90 + random.uniform(-0.05, 0.05)
        confidence = max(0.0, min(1.0, confidence))
        
        return IrradianceData(
            latitude=latitude,
            longitude=longitude,
            annual_irradiance_kwh_m2=annual_irradiance,
            monthly_irradiance=monthly_irradiance,
            optimal_angle=optimal_angle,
            data_source="PVGIS v5.2 (placeholder)",
            confidence=confidence
        )
    
    def get_optimal_tilt_angle(self, latitude: float, longitude: float) -> Optional[float]:
        """
        Get optimal panel tilt angle for location
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Optimal tilt angle in degrees
        """
        irradiance_data = self.get_irradiance_data(latitude, longitude)
        return irradiance_data.optimal_angle if irradiance_data else None
    
    def get_monthly_irradiance(self, latitude: float, longitude: float) -> Optional[Dict[str, float]]:
        """
        Get monthly irradiance breakdown for location
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Monthly irradiance data
        """
        irradiance_data = self.get_irradiance_data(latitude, longitude)
        return irradiance_data.monthly_irradiance if irradiance_data else None
    
    def calculate_yield_factor(self, latitude: float, longitude: float, 
                              pitch_degrees: float, orientation: str) -> Optional[float]:
        """
        Calculate yield factor based on roof pitch and orientation
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            pitch_degrees: Roof pitch angle
            orientation: Roof orientation (south_facing, etc.)
            
        Returns:
            Yield factor (0.0 to 1.0)
        """
        irradiance_data = self.get_irradiance_data(latitude, longitude)
        if not irradiance_data:
            return None
        
        # Base yield factor from optimal angle
        optimal_angle = irradiance_data.optimal_angle
        angle_factor = 1.0 - abs(pitch_degrees - optimal_angle) / 90.0
        angle_factor = max(0.0, min(1.0, angle_factor))
        
        # Orientation factor
        orientation_factors = {
            "south_facing": 1.0,
            "south_east": 0.95,
            "south_west": 0.95,
            "east_facing": 0.85,
            "west_facing": 0.85,
            "north_facing": 0.60
        }
        
        orientation_factor = orientation_factors.get(orientation, 0.8)
        
        # Combined yield factor
        yield_factor = angle_factor * orientation_factor
        
        return max(0.0, min(1.0, yield_factor)) 