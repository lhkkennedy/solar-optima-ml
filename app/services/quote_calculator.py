import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, date
import math

from app.services.irradiance_service import IrradianceService
from app.services.cost_service import CostService

logger = logging.getLogger(__name__)

@dataclass
class SystemSpecification:
    """Solar system specification"""
    optimal_kwp: float
    panel_count: int
    panel_type: str
    inverter_type: str
    battery_capacity: Optional[str]
    panel_efficiency: float
    inverter_efficiency: float

@dataclass
class YieldAnalysis:
    """Solar yield analysis results"""
    estimated_yearly_kwh: float
    solar_fraction: float
    co2_savings_kg: float
    mcs_compliant: bool
    performance_ratio: float
    monthly_yield: Dict[str, float]

@dataclass
class FinancialAnalysis:
    """Financial analysis results"""
    total_cost_gbp: float
    installation_cost: float
    battery_cost: float
    annual_savings: float
    payback_years: float
    roi_percentage: float
    feed_in_tariff: float
    export_benefit: float

class QuoteCalculator:
    """
    Quote calculator implementing MCS MIS 3001 Issue 5.1 yield calculations
    Combines irradiance, cost, and system data to generate complete quotes
    """
    
    def __init__(self, irradiance_service: Optional[IrradianceService] = None,
                 cost_service: Optional[CostService] = None):
        """
        Initialize quote calculator
        
        Args:
            irradiance_service: Service for solar irradiance data
            cost_service: Service for component costs
        """
        self.irradiance_service = irradiance_service or IrradianceService()
        self.cost_service = cost_service or CostService()
        
        # MCS constants
        self.mcs_performance_ratio = 0.75  # Standard MCS performance ratio
        self.co2_factor = 0.233  # kg CO2 per kWh (UK grid average)
        self.electricity_price = 0.34  # £/kWh (UK average)
        self.export_price = 0.15  # £/kWh (Smart Export Guarantee)
        
        # System efficiency factors
        self.panel_efficiency = 0.20  # 20% efficient panels
        self.inverter_efficiency = 0.96  # 96% efficient inverters
        self.system_losses = 0.05  # 5% system losses
    
    def calculate_optimal_system_size(self, roof_area_m2: float, 
                                    pitch_degrees: float, 
                                    orientation: str) -> float:
        """
        Calculate optimal system size based on roof characteristics
        
        Args:
            roof_area_m2: Available roof area in square meters
            pitch_degrees: Roof pitch angle
            orientation: Roof orientation
            
        Returns:
            Optimal system size in kWp
        """
        # Calculate usable roof area (accounting for pitch and orientation)
        usable_factor = self._calculate_usable_factor(pitch_degrees, orientation)
        usable_area = roof_area_m2 * usable_factor
        
        # Panel area (400W panel is ~1.8m²)
        panel_area = 1.8  # m² per panel
        
        # Maximum panels that can fit
        max_panels = int(usable_area / panel_area)
        
        # Calculate kWp (400W panels)
        max_kwp = (max_panels * 400) / 1000
        
        # Apply practical limits
        max_kwp = min(max_kwp, 10.0)  # Max 10kWp for residential
        max_kwp = max(max_kwp, 1.0)   # Min 1kWp
        
        return round(max_kwp, 1)
    
    def _calculate_usable_factor(self, pitch_degrees: float, orientation: str) -> float:
        """
        Calculate usable roof area factor
        
        Args:
            pitch_degrees: Roof pitch angle
            orientation: Roof orientation
            
        Returns:
            Usable area factor (0.0 to 1.0)
        """
        # Pitch factor (optimal around 30-35 degrees)
        pitch_factor = 1.0 - abs(pitch_degrees - 32.5) / 90.0
        pitch_factor = max(0.0, min(1.0, pitch_factor))
        
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
        
        # Combined factor
        usable_factor = pitch_factor * orientation_factor
        
        # Apply minimum threshold
        return max(0.3, usable_factor)
    
    def calculate_yield(self, latitude: float, longitude: float,
                       system_size_kwp: float, pitch_degrees: float,
                       orientation: str) -> YieldAnalysis:
        """
        Calculate solar yield using MCS MIS 3001 Issue 5.1 formulas
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            system_size_kwp: System size in kWp
            pitch_degrees: Roof pitch angle
            orientation: Roof orientation
            
        Returns:
            YieldAnalysis object
        """
        # Get irradiance data
        irradiance_data = self.irradiance_service.get_irradiance_data(latitude, longitude)
        if not irradiance_data:
            raise ValueError("Unable to get irradiance data for location")
        
        # Calculate yield factor
        yield_factor = self.irradiance_service.calculate_yield_factor(
            latitude, longitude, pitch_degrees, orientation
        )
        
        if not yield_factor:
            yield_factor = 0.8  # Default fallback
        
        # MCS yield calculation formula
        # Annual yield = System size × Annual irradiance × Performance ratio × Yield factor
        annual_irradiance = irradiance_data.annual_irradiance_kwh_m2
        performance_ratio = self.mcs_performance_ratio
        
        estimated_yearly_kwh = (system_size_kwp * annual_irradiance * 
                               performance_ratio * yield_factor)
        
        # Calculate monthly breakdown
        monthly_yield = {}
        for month, monthly_irradiance in irradiance_data.monthly_irradiance.items():
            monthly_yield[month] = (system_size_kwp * monthly_irradiance * 
                                   performance_ratio * yield_factor)
        
        # Calculate solar fraction (assumes 4,000 kWh annual consumption)
        annual_consumption = 4000  # kWh
        solar_fraction = min(1.0, estimated_yearly_kwh / annual_consumption)
        
        # Calculate CO2 savings
        co2_savings_kg = estimated_yearly_kwh * self.co2_factor
        
        # Check MCS compliance
        mcs_compliant = self._check_mcs_compliance(system_size_kwp, estimated_yearly_kwh)
        
        return YieldAnalysis(
            estimated_yearly_kwh=estimated_yearly_kwh,
            solar_fraction=solar_fraction,
            co2_savings_kg=co2_savings_kg,
            mcs_compliant=mcs_compliant,
            performance_ratio=performance_ratio,
            monthly_yield=monthly_yield
        )
    
    def _check_mcs_compliance(self, system_size_kwp: float, 
                            estimated_yearly_kwh: float) -> bool:
        """
        Check if system meets MCS compliance requirements
        
        Args:
            system_size_kwp: System size in kWp
            estimated_yearly_kwh: Estimated annual yield
            
        Returns:
            True if MCS compliant
        """
        # MCS minimum yield requirements
        min_yield_ratio = 0.8  # 80% of expected yield
        
        # Expected yield (rough estimate: 850 kWh/kWp for UK)
        expected_yield = system_size_kwp * 850
        
        # Check if actual yield meets minimum
        if estimated_yearly_kwh < (expected_yield * min_yield_ratio):
            return False
        
        # Check system size limits
        if system_size_kwp < 1.0 or system_size_kwp > 10.0:
            return False
        
        return True
    
    def calculate_financial_analysis(self, system_size_kwp: float,
                                   estimated_yearly_kwh: float,
                                   roof_type: str, has_battery: bool = False) -> FinancialAnalysis:
        """
        Calculate financial analysis including payback and ROI
        
        Args:
            system_size_kwp: System size in kWp
            estimated_yearly_kwh: Estimated annual yield
            roof_type: Type of roof
            has_battery: Whether system includes battery storage
            
        Returns:
            FinancialAnalysis object
        """
        # Calculate costs
        cost_breakdown = self.cost_service.calculate_total_cost(
            system_size_kwp, roof_type, has_battery
        )
        
        total_cost = cost_breakdown["total_cost"]
        installation_cost = cost_breakdown["installation_cost"]
        battery_cost = cost_breakdown["battery_cost"]
        
        # Calculate savings
        # Assume 70% self-consumption, 30% export
        self_consumption_ratio = 0.7
        export_ratio = 0.3
        
        self_consumption_kwh = estimated_yearly_kwh * self_consumption_ratio
        export_kwh = estimated_yearly_kwh * export_ratio
        
        # Financial benefits
        electricity_savings = self_consumption_kwh * self.electricity_price
        export_benefit = export_kwh * self.export_price
        feed_in_tariff = 0  # No longer available for new installations
        
        annual_savings = electricity_savings + export_benefit + feed_in_tariff
        
        # Calculate payback period
        if annual_savings > 0:
            payback_years = total_cost / annual_savings
        else:
            payback_years = float('inf')
        
        # Calculate ROI
        if total_cost > 0:
            roi_percentage = (annual_savings / total_cost) * 100
        else:
            roi_percentage = 0.0
        
        return FinancialAnalysis(
            total_cost_gbp=total_cost,
            installation_cost=installation_cost,
            battery_cost=battery_cost,
            annual_savings=annual_savings,
            payback_years=payback_years,
            roi_percentage=roi_percentage,
            feed_in_tariff=feed_in_tariff,
            export_benefit=export_benefit
        )
    
    def generate_system_specification(self, system_size_kwp: float,
                                    has_battery: bool = False) -> SystemSpecification:
        """
        Generate system specification with optimal components
        
        Args:
            system_size_kwp: System size in kWp
            has_battery: Whether to include battery storage
            
        Returns:
            SystemSpecification object
        """
        # Get optimal components
        components = self.cost_service.get_optimal_components(system_size_kwp, has_battery)
        
        # Calculate panel count
        panel_watts = int(components["panel"].sku.split("-")[1].replace("W", ""))
        panel_count = int((system_size_kwp * 1000) / panel_watts)
        
        # Battery capacity
        battery_capacity = None
        if "battery" in components:
            battery_capacity = components["battery"].sku.split("-")[1]
        
        return SystemSpecification(
            optimal_kwp=system_size_kwp,
            panel_count=panel_count,
            panel_type=components["panel"].description,
            inverter_type=components["inverter"].description,
            battery_capacity=battery_capacity,
            panel_efficiency=self.panel_efficiency,
            inverter_efficiency=self.inverter_efficiency
        )
    
    def generate_itemized_breakdown(self, system_size_kwp: float,
                                  has_battery: bool = False) -> List[Dict]:
        """
        Generate itemized cost breakdown
        
        Args:
            system_size_kwp: System size in kWp
            has_battery: Whether system includes battery storage
            
        Returns:
            List of itemized components
        """
        components = self.cost_service.get_optimal_components(system_size_kwp, has_battery)
        cost_breakdown = self.cost_service.calculate_total_cost(system_size_kwp, "gabled", has_battery)
        
        itemized = []
        
        # Panels
        panel_count = cost_breakdown["panel_count"]
        itemized.append({
            "sku": components["panel"].sku,
            "description": components["panel"].description,
            "quantity": panel_count,
            "unit_cost": components["panel"].unit_cost_gbp,
            "total_cost": cost_breakdown["panel_cost"]
        })
        
        # Inverter
        itemized.append({
            "sku": components["inverter"].sku,
            "description": components["inverter"].description,
            "quantity": 1,
            "unit_cost": components["inverter"].unit_cost_gbp,
            "total_cost": cost_breakdown["inverter_cost"]
        })
        
        # Battery (if included)
        if "battery" in components:
            itemized.append({
                "sku": components["battery"].sku,
                "description": components["battery"].description,
                "quantity": 1,
                "unit_cost": components["battery"].unit_cost_gbp,
                "total_cost": cost_breakdown["battery_cost"]
            })
        
        # Accessories
        itemized.append({
            "sku": "MOUNTING-KIT",
            "description": "Roof Mounting Kit",
            "quantity": panel_count,
            "unit_cost": 25.0,
            "total_cost": panel_count * 25.0
        })
        
        itemized.append({
            "sku": "CABLING-KIT",
            "description": "DC/AC Cabling Kit",
            "quantity": 2,
            "unit_cost": 15.0,
            "total_cost": 30.0
        })
        
        # Installation
        itemized.append({
            "sku": "INSTALLATION",
            "description": "Professional Installation",
            "quantity": 1,
            "unit_cost": cost_breakdown["installation_cost"],
            "total_cost": cost_breakdown["installation_cost"]
        })
        
        return itemized 