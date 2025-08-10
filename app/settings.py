import os
from dataclasses import dataclass

@dataclass
class Settings:
    use_placeholder_pvgis: bool = os.getenv("USE_PLACEHOLDER_PVGIS", "true").lower() == "true"
    use_placeholder_beis: bool = os.getenv("USE_PLACEHOLDER_BEIS", "true").lower() == "true"
    enable_request_id_logging: bool = os.getenv("ENABLE_REQUEST_ID_LOGGING", "true").lower() == "true"


def get_settings() -> Settings:
    return Settings()