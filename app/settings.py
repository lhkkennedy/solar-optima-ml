import os
from dataclasses import dataclass
try:
    # Ensure .env from project root is loaded even if CWD differs
    from dotenv import load_dotenv  # type: ignore
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False

if _HAS_DOTENV:
    try:
        _BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        _DOTENV_PATH = os.path.join(_BASE_DIR, ".env")
        load_dotenv(_DOTENV_PATH)
    except Exception:
        # Non-fatal if dotenv isn't available or file missing
        pass

@dataclass
class Settings:
    use_placeholder_pvgis: bool = os.getenv("USE_PLACEHOLDER_PVGIS", "true").lower() == "true"
    use_placeholder_beis: bool = os.getenv("USE_PLACEHOLDER_BEIS", "true").lower() == "true"
    enable_request_id_logging: bool = os.getenv("ENABLE_REQUEST_ID_LOGGING", "true").lower() == "true"
    # ML-6 procedural roof feature flag (0/1)
    # Default on: procedural pipeline is primary
    proc_roof_enable: int = int(os.getenv("PROC_ROOF_ENABLE", "1"))
    proc_roof_use_classifier: int = int(os.getenv("PROC_ROOF_USE_CLASSIFIER", "0"))
    proc_roof_onnx_dir: str = os.getenv("PROC_ROOF_ONNX_DIR", "/models")


def get_settings() -> Settings:
    return Settings()