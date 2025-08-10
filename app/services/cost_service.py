import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, date
import random

logger = logging.getLogger(__name__)

@dataclass
class ComponentCost:
    """Component cost information"""
    sku: str
    description: str
    category: str
    unit_cost_gbp: float
    installation_cost_gbp: float
    warranty_years: int
    supplier: str
    last_updated: date

@dataclass
class InstallationCost:
    """Installation cost breakdown"""
    base_installation: float
    scaffolding: float
    electrical_work: float
    certification: float
    total_installation: float

class CostService:
    """
    Service for accessing BEIS small-scale PV cost database
    Provides component pricing and installation cost estimation
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize cost service
        
        Args:
            cache_enabled: Whether to cache cost data for performance
        """
        self.cache_enabled = cache_enabled
        self._cache = {}  # Simple in-memory cache
        
        # Load placeholder cost data
        self._load_placeholder_cost_data()
    
    def _load_placeholder_cost_data(self):
        """Load placeholder cost data for ML-3 development"""
        # Solar panels (per panel)
        self.panel_costs = {
            "PANEL-400W": ComponentCost(
                sku="PANEL-400W",
                description="400W Monocrystalline Panel",
                category="solar_panel",
                unit_cost_gbp=180.0,
                installation_cost_gbp=50.0,
                warranty_years=25,
                supplier="SolarTech UK",
                last_updated=date.today()
            ),
            "PANEL-450W": ComponentCost(
                sku="PANEL-450W",
                description="450W Monocrystalline Panel",
                category="solar_panel",
                unit_cost_gbp=220.0,
                installation_cost_gbp=50.0,
                warranty_years=25,
                supplier="SolarTech UK",
                last_updated=date.today()
            ),
            "PANEL-500W": ComponentCost(
                sku="PANEL-500W",
                description="500W Monocrystalline Panel",
                category="solar_panel",
                unit_cost_gbp=260.0,
                installation_cost_gbp=50.0,
                warranty_years=25,
                supplier="SolarTech UK",
                last_updated=date.today()
            )
        }
        
        # Inverters
        self.inverter_costs = {
            "INV-3.6KW": ComponentCost(
                sku="INV-3.6KW",
                description="3.6kW Hybrid Inverter",
                category="inverter",
                unit_cost_gbp=1200.0,
                installation_cost_gbp=300.0,
                warranty_years=10,
                supplier="InverterPro",
                last_updated=date.today()
            ),
            "INV-5.0KW": ComponentCost(
                sku="INV-5.0KW",
                description="5.0kW Hybrid Inverter",
                category="inverter",
                unit_cost_gbp=1600.0,
                installation_cost_gbp=300.0,
                warranty_years=10,
                supplier="InverterPro",
                last_updated=date.today()
            ),
            "INV-6.0KW": ComponentCost(
                sku="INV-6.0KW",
                description="6.0kW Hybrid Inverter",
                category="inverter",
                unit_cost_gbp=2000.0,
                installation_cost_gbp=300.0,
                warranty_years=10,
                supplier="InverterPro",
                last_updated=date.today()
            )
        }
        
        # Battery storage
        self.battery_costs = {
            "BAT-5.2KWH": ComponentCost(
                sku="BAT-5.2KWH",
                description="5.2kWh Lithium Battery",
                category="battery",
                unit_cost_gbp=2250.0,
                installation_cost_gbp=400.0,
                warranty_years=10,
                supplier="BatteryStore",
                last_updated=date.today()
            ),
            "BAT-10.4KWH": ComponentCost(
                sku="BAT-10.4KWH",
                description="10.4kWh Lithium Battery",
                category="battery",
                unit_cost_gbp=4200.0,
                installation_cost_gbp=400.0,
                warranty_years=10,
                supplier="BatteryStore",
                last_updated=date.today()
            )
        }
        
        # Mounting and accessories
        self.accessory_costs = {
            "MOUNTING-KIT": ComponentCost(
                sku="MOUNTING-KIT",
                description="Roof Mounting Kit",
                category="accessory",
                unit_cost_gbp=25.0,
                installation_cost_gbp=0.0,
                warranty_years=10,
                supplier="MountPro",
                last_updated=date.today()
            ),
            "CABLING-KIT": ComponentCost(
                sku="CABLING-KIT",
                description="DC/AC Cabling Kit",
                category="accessory",
                unit_cost_gbp=15.0,
                installation_cost_gbp=0.0,
                warranty_years=5,
                supplier="CableTech",
                last_updated=date.today()
            )
        }
    
    def get_component_cost(self, sku: str) -> Optional[ComponentCost]:
        """
        Get component cost by SKU
        
        Args:
            sku: Component SKU
            
        Returns:
            ComponentCost object or None if not found
        """
        # Check all categories
        for category in [self.panel_costs, self.inverter_costs, 
                        self.battery_costs, self.accessory_costs]:
            if sku in category:
                return category[sku]
        
        return None
    
    def get_panel_options(self) -> List[ComponentCost]:
        """Get available panel options"""
        return list(self.panel_costs.values())
    
    def get_inverter_options(self) -> List[ComponentCost]:
        """Get available inverter options"""
        return list(self.inverter_costs.values())
    
    def get_battery_options(self) -> List[ComponentCost]:
        """Get available battery options"""
        return list(self.battery_costs.values())
    
    def calculate_installation_cost(self, system_size_kwp: float, 
                                  roof_type: str, has_battery: bool = False) -> InstallationCost:
        """
        Calculate installation cost based on system size and configuration
        
        Args:
            system_size_kwp: System size in kWp
            roof_type: Type of roof (gabled, hipped, flat, etc.)
            has_battery: Whether system includes battery storage
            
        Returns:
            InstallationCost breakdown
        """
        # Base installation cost (£/kWp)
        base_cost_per_kwp = 800.0
        
        # Roof complexity factors
        roof_factors = {
            "gabled": 1.0,
            "hipped": 1.1,
            "flat": 1.2,
            "mansard": 1.3,
            "complex": 1.4
        }
        
        roof_factor = roof_factors.get(roof_type, 1.0)
        
        # Calculate base installation
        base_installation = system_size_kwp * base_cost_per_kwp * roof_factor
        
        # Scaffolding cost (varies with system size)
        scaffolding = max(500, system_size_kwp * 200)
        
        # Electrical work
        electrical_work = 300 + (system_size_kwp * 100)
        if has_battery:
            electrical_work += 200  # Additional electrical work for battery
        
        # Certification and paperwork
        certification = 150 + (system_size_kwp * 50)
        
        # Total installation
        total_installation = base_installation + scaffolding + electrical_work + certification
        
        return InstallationCost(
            base_installation=base_installation,
            scaffolding=scaffolding,
            electrical_work=electrical_work,
            certification=certification,
            total_installation=total_installation
        )
    
    def get_optimal_components(self, system_size_kwp: float, 
                             has_battery: bool = False) -> Dict[str, ComponentCost]:
        """
        Get optimal component selection for system size
        
        Args:
            system_size_kwp: System size in kWp
            has_battery: Whether to include battery storage
            
        Returns:
            Dictionary of optimal components
        """
        # Select optimal panel (400W panels for most systems)
        panel_watts = 400
        if system_size_kwp > 6.0:
            panel_watts = 450
        elif system_size_kwp > 8.0:
            panel_watts = 500
        
        panel_sku = f"PANEL-{panel_watts}W"
        optimal_panel = self.panel_costs.get(panel_sku, self.panel_costs["PANEL-400W"])
        
        # Select optimal inverter
        if system_size_kwp <= 3.6:
            inverter_sku = "INV-3.6KW"
        elif system_size_kwp <= 5.0:
            inverter_sku = "INV-5.0KW"
        else:
            inverter_sku = "INV-6.0KW"
        
        optimal_inverter = self.inverter_costs.get(inverter_sku, self.inverter_costs["INV-3.6KW"])
        
        # Select battery if requested
        optimal_battery = None
        if has_battery:
            if system_size_kwp <= 4.0:
                battery_sku = "BAT-5.2KWH"
            else:
                battery_sku = "BAT-10.4KWH"
            optimal_battery = self.battery_costs.get(battery_sku, self.battery_costs["BAT-5.2KWH"])
        
        components = {
            "panel": optimal_panel,
            "inverter": optimal_inverter
        }
        
        if optimal_battery:
            components["battery"] = optimal_battery
        
        return components
    
    def calculate_total_cost(self, system_size_kwp: float, roof_type: str,
                           has_battery: bool = False) -> Dict[str, float]:
        """
        Calculate total system cost
        
        Args:
            system_size_kwp: System size in kWp
            roof_type: Type of roof
            has_battery: Whether system includes battery storage
            
        Returns:
            Cost breakdown dictionary
        """
        # Get optimal components
        components = self.get_optimal_components(system_size_kwp, has_battery)
        
        # Calculate panel count
        panel_watts = int(components["panel"].sku.split("-")[1].replace("W", ""))
        panel_count = int((system_size_kwp * 1000) / panel_watts)
        
        # Component costs
        panel_cost = components["panel"].unit_cost_gbp * panel_count
        inverter_cost = components["inverter"].unit_cost_gbp
        
        battery_cost = 0
        if "battery" in components:
            battery_cost = components["battery"].unit_cost_gbp
        
        # Accessory costs
        mounting_cost = self.accessory_costs["MOUNTING-KIT"].unit_cost_gbp * panel_count
        cabling_cost = self.accessory_costs["CABLING-KIT"].unit_cost_gbp * 2  # DC and AC cables
        
        # Installation costs
        installation = self.calculate_installation_cost(system_size_kwp, roof_type, has_battery)
        
        # Total cost
        total_cost = (panel_cost + inverter_cost + battery_cost + 
                     mounting_cost + cabling_cost + installation.total_installation)
        
        return {
            "panel_cost": panel_cost,
            "inverter_cost": inverter_cost,
            "battery_cost": battery_cost,
            "accessory_cost": mounting_cost + cabling_cost,
            "installation_cost": installation.total_installation,
            "total_cost": total_cost,
            "panel_count": panel_count
        } 