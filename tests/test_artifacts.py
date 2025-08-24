import os
from pathlib import Path

from app.services.artifacts.geojson_writer import write_geojson
from app.services.artifacts.gltf_writer import write_gltf
from app.services.artifacts.storage import ArtifactStorage
from app.services.procedural_roof.synthesis import ProceduralRoofModel


def _dummy_model() -> ProceduralRoofModel:
    return ProceduralRoofModel(
        footprint_regularized=[(-0.1, 51.5), (-0.09, 51.5), (-0.09, 51.51), (-0.1, 51.51), (-0.1, 51.5)],
        parts=[{
            "rect_bbox": [(-0.1, 51.5), (-0.095, 51.5), (-0.095, 51.505), (-0.1, 51.505), (-0.1, 51.5)],
            "roof_family": "gable",
            "ridges": [[(-0.099, 51.502), (-0.096, 51.502)]],
            "confidence": 0.9,
            "height_stats_m": {"mean": 2.0},
        }],
    )


def test_geojson_and_gltf_written(tmp_path):
    os.environ["ARTIFACT_DIR"] = str(tmp_path)
    model = _dummy_model()
    geo_path = write_geojson(model)
    glb_path = write_gltf(model)
    assert Path(geo_path).exists()
    assert Path(glb_path).exists()

    storage = ArtifactStorage(artifact_dir=str(tmp_path), base_url="http://localhost:8000/static")
    url = storage.store(geo_path)
    assert url.startswith("http://") or url.startswith("file:") or url.startswith("gs://")

