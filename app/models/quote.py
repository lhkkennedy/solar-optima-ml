import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, date
import uuid

from app.services.quote_calculator import QuoteCalculator, SystemSpecification, YieldAnalysis, FinancialAnalysis
from app.services.irradiance_service import IrradianceService
from app.services.cost_service import CostService

logger = logging.getLogger(__name__)

@dataclass
class PropertyDetails:
    """Property information for quote generation"""
    address: str
    postcode: str
    property_type: str
    occupancy: str

@dataclass
class SegmentationResult:
    """Segmentation analysis results"""
    mask: str  # Base64 encoded mask
    confidence: float

@dataclass
class PitchResult:
    """Pitch estimation results"""
    pitch_degrees: float
    area_m2: float
    roof_type: str
    orientation: str

@dataclass
class CustomerPreferences:
    """Customer preferences for quote customization"""
    battery_storage: bool
    premium_panels: bool
    financing: str

@dataclass
class QuoteResponse:
    """Complete solar quote response"""
    quote_id: str
    generated_date: str
    property_details: PropertyDetails
    system_specification: Dict[str, Any]
    yield_analysis: Dict[str, Any]
    financial_analysis: Dict[str, Any]
    itemized_breakdown: List[Dict[str, Any]]
    warranties: Dict[str, str]
    next_steps: List[str]
    valid_until: str
    mcs_compliant: bool
    confidence_score: float

class QuoteModel:
    """
    Quote model that orchestrates all services to generate complete solar quotes
    Combines segmentation, pitch estimation, irradiance, and cost data
    """
    
    def __init__(self, quote_calculator: Optional[QuoteCalculator] = None):
        """
        Initialize quote model
        
        Args:
            quote_calculator: Quote calculator service
        """
        self.quote_calculator = quote_calculator or QuoteCalculator()
        
        # Generate quote ID prefix
        self.quote_prefix = "SOL"
        self.year = datetime.now().year
    
    def generate_quote(self, property_details: PropertyDetails,
                      segmentation_result: SegmentationResult,
                      pitch_result: PitchResult,
                      preferences: CustomerPreferences) -> QuoteResponse:
        """
        Generate complete solar quote
        
        Args:
            property_details: Property information
            segmentation_result: Segmentation analysis results
            pitch_result: Pitch estimation results
            preferences: Customer preferences
            
        Returns:
            Complete QuoteResponse object
        """
        try:
            # Generate unique quote ID
            quote_id = self._generate_quote_id()
            
            # Calculate optimal system size
            optimal_kwp = self.quote_calculator.calculate_optimal_system_size(
                roof_area_m2=pitch_result.area_m2,
                pitch_degrees=pitch_result.pitch_degrees,
                orientation=pitch_result.orientation
            )
            
            # Calculate yield analysis
            yield_analysis = self.quote_calculator.calculate_yield(
                latitude=51.5074,  # Placeholder - would come from address geocoding
                longitude=-0.1278,  # Placeholder - would come from address geocoding
                system_size_kwp=optimal_kwp,
                pitch_degrees=pitch_result.pitch_degrees,
                orientation=pitch_result.orientation
            )
            
            # Calculate financial analysis
            financial_analysis = self.quote_calculator.calculate_financial_analysis(
                system_size_kwp=optimal_kwp,
                estimated_yearly_kwh=yield_analysis.estimated_yearly_kwh,
                roof_type=pitch_result.roof_type,
                has_battery=preferences.battery_storage
            )
            
            # Generate system specification
            system_spec = self.quote_calculator.generate_system_specification(
                system_size_kwp=optimal_kwp,
                has_battery=preferences.battery_storage
            )
            
            # Generate itemized breakdown
            itemized_breakdown = self.quote_calculator.generate_itemized_breakdown(
                system_size_kwp=optimal_kwp,
                has_battery=preferences.battery_storage
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                segmentation_result.confidence,
                pitch_result.area_m2,
                yield_analysis.mcs_compliant
            )
            
            # Generate warranties
            warranties = self._generate_warranties(system_spec)
            
            # Generate next steps
            next_steps = self._generate_next_steps(property_details, optimal_kwp)
            
            # Calculate quote validity
            valid_until = self._calculate_valid_until()
            
            return QuoteResponse(
                quote_id=quote_id,
                generated_date=datetime.now().isoformat(),
                property_details=property_details,
                system_specification=asdict(system_spec),
                yield_analysis=asdict(yield_analysis),
                financial_analysis=asdict(financial_analysis),
                itemized_breakdown=itemized_breakdown,
                warranties=warranties,
                next_steps=next_steps,
                valid_until=valid_until,
                mcs_compliant=yield_analysis.mcs_compliant,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error generating quote: {e}")
            raise ValueError(f"Quote generation failed: {str(e)}")
    
    def _generate_quote_id(self) -> str:
        """Generate unique quote ID"""
        unique_id = str(uuid.uuid4())[:8].upper()
        return f"{self.quote_prefix}-{self.year}-{unique_id}"
    
    def _calculate_confidence_score(self, segmentation_confidence: float,
                                  roof_area: float, mcs_compliant: bool) -> float:
        """
        Calculate overall confidence score for the quote
        
        Args:
            segmentation_confidence: Segmentation confidence (0-1)
            roof_area: Roof area in m²
            mcs_compliant: Whether system is MCS compliant
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from segmentation
        base_confidence = segmentation_confidence
        
        # Roof area factor (larger roofs are more reliable)
        area_factor = min(1.0, roof_area / 50.0)  # Normalize to 50m²
        
        # MCS compliance bonus
        mcs_bonus = 0.1 if mcs_compliant else 0.0
        
        # Calculate weighted confidence
        confidence = (base_confidence * 0.7 + area_factor * 0.2 + mcs_bonus)
        
        return min(1.0, max(0.0, confidence))
    
    def _generate_warranties(self, system_spec: SystemSpecification) -> Dict[str, str]:
        """
        Generate warranty information for system components
        
        Args:
            system_spec: System specification
            
        Returns:
            Warranty dictionary
        """
        warranties = {
            "panels": "25 years",
            "inverter": "10 years",
            "installation": "2 years"
        }
        
        # Add battery warranty if present
        if system_spec.battery_capacity:
            warranties["battery"] = "10 years"
        
        return warranties
    
    def _generate_next_steps(self, property_details: PropertyDetails, 
                           system_size_kwp: float) -> List[str]:
        """
        Generate next steps for customer
        
        Args:
            property_details: Property information
            system_size_kwp: System size in kWp
            
        Returns:
            List of next steps
        """
        next_steps = [
            "Schedule site survey",
            "Apply for planning permission (if required)",
            "Arrange financing"
        ]
        
        # Add system-specific steps
        if system_size_kwp > 3.68:
            next_steps.append("Apply for DNO permission (system > 3.68kW)")
        
        if property_details.property_type in ["listed", "conservation_area"]:
            next_steps.append("Apply for listed building consent")
        
        next_steps.extend([
            "Choose installation date",
            "Sign installation contract",
            "Prepare for installation day"
        ])
        
        return next_steps
    
    def _calculate_valid_until(self) -> str:
        """Calculate quote validity date (30 days from generation)"""
        from datetime import timedelta
        valid_date = datetime.now() + timedelta(days=30)
        return valid_date.strftime("%Y-%m-%d")
    
    def validate_quote_request(self, property_details: PropertyDetails,
                             segmentation_result: SegmentationResult,
                             pitch_result: PitchResult,
                             preferences: CustomerPreferences) -> List[str]:
        """
        Validate quote request data
        
        Args:
            property_details: Property information
            segmentation_result: Segmentation results
            pitch_result: Pitch results
            preferences: Customer preferences
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate property details
        if not property_details.address or len(property_details.address.strip()) < 5:
            errors.append("Property address must be at least 5 characters")
        
        if not property_details.postcode or len(property_details.postcode.strip()) < 5:
            errors.append("Valid postcode required")
        
        # Validate segmentation results
        if segmentation_result.confidence < 0.5:
            errors.append("Segmentation confidence too low (minimum 0.5)")
        
        if not segmentation_result.mask:
            errors.append("Segmentation mask required")
        
        # Validate pitch results
        if pitch_result.pitch_degrees < 0 or pitch_result.pitch_degrees > 90:
            errors.append("Pitch angle must be between 0 and 90 degrees")
        
        if pitch_result.area_m2 < 5:
            errors.append("Roof area too small (minimum 5m²)")
        
        if pitch_result.area_m2 > 200:
            errors.append("Roof area too large (maximum 200m²)")
        
        # Validate roof type
        valid_roof_types = ["gabled", "hipped", "flat", "mansard", "complex"]
        if pitch_result.roof_type not in valid_roof_types:
            errors.append(f"Invalid roof type. Must be one of: {', '.join(valid_roof_types)}")
        
        # Validate orientation
        valid_orientations = ["south_facing", "south_east", "south_west", 
                            "east_facing", "west_facing", "north_facing"]
        if pitch_result.orientation not in valid_orientations:
            errors.append(f"Invalid orientation. Must be one of: {', '.join(valid_orientations)}")
        
        return errors
    
    def get_quote_summary(self, quote_response: QuoteResponse) -> Dict[str, Any]:
        """
        Generate quote summary for quick overview
        
        Args:
            quote_response: Complete quote response
            
        Returns:
            Summary dictionary
        """
        return {
            "quote_id": quote_response.quote_id,
            "system_size_kwp": quote_response.system_specification["optimal_kwp"],
            "estimated_yearly_kwh": quote_response.yield_analysis["estimated_yearly_kwh"],
            "total_cost_gbp": quote_response.financial_analysis["total_cost_gbp"],
            "payback_years": quote_response.financial_analysis["payback_years"],
            "mcs_compliant": quote_response.mcs_compliant,
            "confidence_score": quote_response.confidence_score,
            "valid_until": quote_response.valid_until
        } 