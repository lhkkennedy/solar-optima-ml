import pytest
import base64
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.main import app
from app.models.quote import QuoteModel, PropertyDetails, SegmentationResult, PitchResult, CustomerPreferences
from app.services.quote_calculator import QuoteCalculator
from app.services.irradiance_service import IrradianceService, IrradianceData
from app.services.cost_service import CostService, ComponentCost
from datetime import date

client = TestClient(app)

def create_test_mask(width: int = 256, height: int = 256) -> str:
    """Create a test segmentation mask"""
    # Create a simple test mask (mostly roof area)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[50:200, 50:200] = 255  # Roof area
    mask_pil = Image.fromarray(mask)
    mask_buffer = io.BytesIO()
    mask_pil.save(mask_buffer, format="PNG")
    return base64.b64encode(mask_buffer.getvalue()).decode()

class TestIrradianceService:
    """Test irradiance service functionality"""
    
    def test_irradiance_service_initialization(self):
        """Test irradiance service initialization"""
        service = IrradianceService()
        assert service.cache_enabled == True
        assert service.base_url == "https://re.jrc.ec.europa.eu/api/v5_2"
        assert service.uk_bounds["min_lat"] == 49.9
    
    def test_is_within_uk_valid_coordinates(self):
        """Test UK bounds validation with valid coordinates"""
        service = IrradianceService()
        assert service.is_within_uk(51.5074, -0.1278) == True  # London
        assert service.is_within_uk(55.9533, -3.1883) == True   # Edinburgh
        assert service.is_within_uk(53.4084, -2.9916) == True   # Liverpool
    
    def test_is_within_uk_invalid_coordinates(self):
        """Test UK bounds validation with invalid coordinates"""
        service = IrradianceService()
        assert service.is_within_uk(40.7128, -74.0060) == False  # New York
        assert service.is_within_uk(48.8566, 2.3522) == False     # Paris
        assert service.is_within_uk(60.0, -10.0) == False         # Outside UK
    
    def test_get_irradiance_data_uk_location(self):
        """Test getting irradiance data for UK location"""
        service = IrradianceService()
        data = service.get_irradiance_data(51.5074, -0.1278)
        
        assert data is not None
        assert data.latitude == 51.5074
        assert data.longitude == -0.1278
        assert 800 <= data.annual_irradiance_kwh_m2 <= 1400
        assert len(data.monthly_irradiance) == 12
        assert "jan" in data.monthly_irradiance
        assert "jun" in data.monthly_irradiance
        assert data.optimal_angle > 0
        assert 0.0 <= data.confidence <= 1.0
    
    def test_get_irradiance_data_outside_uk(self):
        """Test getting irradiance data for non-UK location"""
        service = IrradianceService()
        data = service.get_irradiance_data(40.7128, -74.0060)  # New York
        assert data is None
    
    def test_calculate_yield_factor(self):
        """Test yield factor calculation"""
        service = IrradianceService()
        factor = service.calculate_yield_factor(51.5074, -0.1278, 25.0, "south_facing")
        
        assert factor is not None
        assert 0.0 <= factor <= 1.0
    
    def test_get_optimal_tilt_angle(self):
        """Test optimal tilt angle calculation"""
        service = IrradianceService()
        angle = service.get_optimal_tilt_angle(51.5074, -0.1278)
        
        assert angle is not None
        assert 30 <= angle <= 45  # Reasonable range for UK

class TestCostService:
    """Test cost service functionality"""
    
    def test_cost_service_initialization(self):
        """Test cost service initialization"""
        service = CostService()
        assert service.cache_enabled == True
        assert len(service.panel_costs) > 0
        assert len(service.inverter_costs) > 0
        assert len(service.battery_costs) > 0
    
    def test_get_component_cost_valid_sku(self):
        """Test getting component cost with valid SKU"""
        service = CostService()
        panel_cost = service.get_component_cost("PANEL-400W")
        
        assert panel_cost is not None
        assert panel_cost.sku == "PANEL-400W"
        assert panel_cost.description == "400W Monocrystalline Panel"
        assert panel_cost.unit_cost_gbp > 0
        assert panel_cost.warranty_years == 25
    
    def test_get_component_cost_invalid_sku(self):
        """Test getting component cost with invalid SKU"""
        service = CostService()
        cost = service.get_component_cost("INVALID-SKU")
        assert cost is None
    
    def test_calculate_installation_cost(self):
        """Test installation cost calculation"""
        service = CostService()
        installation = service.calculate_installation_cost(3.5, "gabled", False)
        
        assert installation.base_installation > 0
        assert installation.scaffolding > 0
        assert installation.electrical_work > 0
        assert installation.certification > 0
        assert installation.total_installation > 0
    
    def test_get_optimal_components(self):
        """Test optimal component selection"""
        service = CostService()
        components = service.get_optimal_components(3.5, False)
        
        assert "panel" in components
        assert "inverter" in components
        assert "battery" not in components
        
        # Test with battery
        components_with_battery = service.get_optimal_components(3.5, True)
        assert "battery" in components_with_battery
    
    def test_calculate_total_cost(self):
        """Test total cost calculation"""
        service = CostService()
        cost_breakdown = service.calculate_total_cost(3.5, "gabled", False)
        
        assert cost_breakdown["panel_cost"] > 0
        assert cost_breakdown["inverter_cost"] > 0
        assert cost_breakdown["battery_cost"] == 0  # No battery
        assert cost_breakdown["installation_cost"] > 0
        assert cost_breakdown["total_cost"] > 0
        assert cost_breakdown["panel_count"] > 0

class TestQuoteCalculator:
    """Test quote calculator functionality"""
    
    def test_quote_calculator_initialization(self):
        """Test quote calculator initialization"""
        calculator = QuoteCalculator()
        assert calculator.mcs_performance_ratio == 0.75
        assert calculator.co2_factor == 0.233
        assert calculator.electricity_price == 0.34
    
    def test_calculate_optimal_system_size(self):
        """Test optimal system size calculation"""
        calculator = QuoteCalculator()
        system_size = calculator.calculate_optimal_system_size(50.0, 25.0, "south_facing")
        
        assert system_size > 0
        assert system_size <= 10.0  # Max residential size
    
    def test_calculate_yield(self):
        """Test yield calculation"""
        calculator = QuoteCalculator()
        yield_analysis = calculator.calculate_yield(51.5074, -0.1278, 3.5, 25.0, "south_facing")
        
        assert yield_analysis.estimated_yearly_kwh > 0
        assert 0.0 <= yield_analysis.solar_fraction <= 1.0
        assert yield_analysis.co2_savings_kg > 0
        assert isinstance(yield_analysis.mcs_compliant, bool)
        assert len(yield_analysis.monthly_yield) == 12
    
    def test_calculate_financial_analysis(self):
        """Test financial analysis calculation"""
        calculator = QuoteCalculator()
        financial = calculator.calculate_financial_analysis(3.5, 3000, "gabled", False)
        
        assert financial.total_cost_gbp > 0
        assert financial.installation_cost > 0
        assert financial.annual_savings > 0
        assert financial.payback_years > 0
        assert financial.roi_percentage > 0
    
    def test_generate_system_specification(self):
        """Test system specification generation"""
        calculator = QuoteCalculator()
        spec = calculator.generate_system_specification(3.5, False)
        
        assert spec.optimal_kwp == 3.5
        assert spec.panel_count > 0
        assert spec.panel_type is not None
        assert spec.inverter_type is not None
        assert spec.battery_capacity is None  # No battery
    
    def test_generate_itemized_breakdown(self):
        """Test itemized breakdown generation"""
        calculator = QuoteCalculator()
        breakdown = calculator.generate_itemized_breakdown(3.5, False)
        
        assert len(breakdown) > 0
        for item in breakdown:
            assert "sku" in item
            assert "description" in item
            assert "quantity" in item
            assert "unit_cost" in item
            assert "total_cost" in item

class TestQuoteModel:
    """Test quote model functionality"""
    
    def test_quote_model_initialization(self):
        """Test quote model initialization"""
        model = QuoteModel()
        assert model.quote_prefix == "SOL"
        assert model.year == 2025  # Current year
    
    def test_generate_quote_id(self):
        """Test quote ID generation"""
        model = QuoteModel()
        quote_id = model._generate_quote_id()
        
        assert quote_id.startswith("SOL-2025-")
        assert len(quote_id) == 17  # SOL-2025-XXXXXXXX (8 chars)
    
    def test_validate_quote_request_valid(self):
        """Test quote request validation with valid data"""
        model = QuoteModel()
        
        property_details = PropertyDetails(
            address="123 Solar Street, London",
            postcode="SW1A 1AA",
            property_type="semi_detached",
            occupancy="family_of_4"
        )
        
        segmentation_result = SegmentationResult(
            mask="base64_encoded_mask",
            confidence=0.95
        )
        
        pitch_result = PitchResult(
            pitch_degrees=25.0,
            area_m2=45.0,
            roof_type="gabled",
            orientation="south_facing"
        )
        
        preferences = CustomerPreferences(
            battery_storage=False,
            premium_panels=False,
            financing="cash_purchase"
        )
        
        errors = model.validate_quote_request(
            property_details, segmentation_result, pitch_result, preferences
        )
        
        assert len(errors) == 0
    
    def test_validate_quote_request_invalid(self):
        """Test quote request validation with invalid data"""
        model = QuoteModel()
        
        property_details = PropertyDetails(
            address="",  # Invalid: empty address
            postcode="",  # Invalid: empty postcode
            property_type="semi_detached",
            occupancy="family_of_4"
        )
        
        segmentation_result = SegmentationResult(
            mask="",  # Invalid: empty mask
            confidence=0.3  # Invalid: low confidence
        )
        
        pitch_result = PitchResult(
            pitch_degrees=95.0,  # Invalid: > 90 degrees
            area_m2=2.0,  # Invalid: too small
            roof_type="invalid_type",  # Invalid: unknown type
            orientation="invalid_orientation"  # Invalid: unknown orientation
        )
        
        preferences = CustomerPreferences(
            battery_storage=False,
            premium_panels=False,
            financing="cash_purchase"
        )
        
        errors = model.validate_quote_request(
            property_details, segmentation_result, pitch_result, preferences
        )
        
        assert len(errors) > 0
        assert any("address" in error.lower() for error in errors)
        assert any("postcode" in error.lower() for error in errors)
        assert any("confidence" in error.lower() for error in errors)
        assert any("pitch angle" in error.lower() for error in errors)

class TestQuoteAPI:
    """Test quote API endpoint"""
    
    def test_quote_endpoint_valid_request(self):
        """Test /quote endpoint with valid request"""
        mask_base64 = create_test_mask()
        
        request_data = {
            "property_details": {
                "address": "123 Solar Street, London",
                "postcode": "SW1A 1AA",
                "property_type": "semi_detached",
                "occupancy": "family_of_4"
            },
            "segmentation_result": {
                "mask": mask_base64,
                "confidence": 0.95
            },
            "pitch_result": {
                "pitch_degrees": 25.5,
                "area_m2": 45.2,
                "roof_type": "gabled",
                "orientation": "south_facing"
            },
            "preferences": {
                "battery_storage": True,
                "premium_panels": False,
                "financing": "cash_purchase"
            }
        }
        
        response = client.post("/quote", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "quote_id" in data
        assert "generated_date" in data
        assert "property_details" in data
        assert "system_specification" in data
        assert "yield_analysis" in data
        assert "financial_analysis" in data
        assert "itemized_breakdown" in data
        assert "warranties" in data
        assert "next_steps" in data
        assert "valid_until" in data
        assert "mcs_compliant" in data
        assert "confidence_score" in data
        
        # Validate specific fields
        assert data["quote_id"].startswith("SOL-2025-")
        assert data["system_specification"]["optimal_kwp"] > 0
        assert data["yield_analysis"]["estimated_yearly_kwh"] > 0
        assert data["financial_analysis"]["total_cost_gbp"] > 0
        assert data["financial_analysis"]["payback_years"] > 0
        assert 0.0 <= data["confidence_score"] <= 1.0
    
    def test_quote_endpoint_invalid_request(self):
        """Test /quote endpoint with invalid request"""
        request_data = {
            "property_details": {
                "address": "",  # Invalid: empty address
                "postcode": "",  # Invalid: empty postcode
                "property_type": "semi_detached",
                "occupancy": "family_of_4"
            },
            "segmentation_result": {
                "mask": "",  # Invalid: empty mask
                "confidence": 0.3  # Invalid: low confidence
            },
            "pitch_result": {
                "pitch_degrees": 95.0,  # Invalid: > 90 degrees
                "area_m2": 2.0,  # Invalid: too small
                "roof_type": "invalid_type",  # Invalid: unknown type
                "orientation": "invalid_orientation"  # Invalid: unknown orientation
            },
            "preferences": {
                "battery_storage": False,
                "premium_panels": False,
                "financing": "cash_purchase"
            }
        }
        
        response = client.post("/quote", json=request_data)
        assert response.status_code == 422  # Validation error for missing fields
        
        data = response.json()
        assert "detail" in data
        # Pydantic validation errors are returned as a list, not a string
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
    
    def test_quote_endpoint_missing_fields(self):
        """Test /quote endpoint with missing required fields"""
        request_data = {
            "property_details": {
                "address": "123 Solar Street, London",
                "postcode": "SW1A 1AA"
                # Missing property_type and occupancy
            },
            "segmentation_result": {
                "mask": create_test_mask(),
                "confidence": 0.95
            },
            "pitch_result": {
                "pitch_degrees": 25.5,
                "area_m2": 45.2,
                "roof_type": "gabled",
                "orientation": "south_facing"
            },
            "preferences": {
                "battery_storage": False,
                "premium_panels": False,
                "financing": "cash_purchase"
            }
        }
        
        response = client.post("/quote", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_quote_endpoint_with_battery(self):
        """Test /quote endpoint with battery storage"""
        mask_base64 = create_test_mask()
        
        request_data = {
            "property_details": {
                "address": "123 Solar Street, London",
                "postcode": "SW1A 1AA",
                "property_type": "semi_detached",
                "occupancy": "family_of_4"
            },
            "segmentation_result": {
                "mask": mask_base64,
                "confidence": 0.95
            },
            "pitch_result": {
                "pitch_degrees": 25.5,
                "area_m2": 45.2,
                "roof_type": "gabled",
                "orientation": "south_facing"
            },
            "preferences": {
                "battery_storage": True,  # Include battery
                "premium_panels": False,
                "financing": "cash_purchase"
            }
        }
        
        response = client.post("/quote", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["system_specification"]["battery_capacity"] is not None
        assert data["financial_analysis"]["battery_cost"] > 0
        assert "battery" in data["warranties"]
    
    def test_quote_endpoint_large_system(self):
        """Test /quote endpoint with large system (>3.68kW)"""
        mask_base64 = create_test_mask()
        
        request_data = {
            "property_details": {
                "address": "123 Solar Street, London",
                "postcode": "SW1A 1AA",
                "property_type": "semi_detached",
                "occupancy": "family_of_4"
            },
            "segmentation_result": {
                "mask": mask_base64,
                "confidence": 0.95
            },
            "pitch_result": {
                "pitch_degrees": 25.5,
                "area_m2": 100.0,  # Large roof area
                "roof_type": "gabled",
                "orientation": "south_facing"
            },
            "preferences": {
                "battery_storage": False,
                "premium_panels": False,
                "financing": "cash_purchase"
            }
        }
        
        response = client.post("/quote", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        system_size = data["system_specification"]["optimal_kwp"]
        
        # Check if DNO permission step is included for large systems
        if system_size > 3.68:
            assert any("DNO permission" in step for step in data["next_steps"])
    
    def test_quote_endpoint_root_includes_quote(self):
        """Test that root endpoint includes quote endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "quote" in data["endpoints"]
        assert data["endpoints"]["quote"] == "/quote" 