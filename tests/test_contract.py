import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _resolve_schema_ref(spec: dict, schema_node: dict) -> dict:
    ref = schema_node.get("$ref")
    if not ref:
        return schema_node
    # Expect format: #/components/schemas/Name
    parts = ref.split("/")
    if len(parts) >= 4 and parts[1] == "components" and parts[2] == "schemas":
        name = parts[3]
        return spec.get("components", {}).get("schemas", {}).get(name, {})
    return {}


def test_openapi_contains_endpoints():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    spec = resp.json()

    paths = spec.get("paths", {})
    assert "/infer" in paths
    assert "/pitch" in paths
    assert "/quote" in paths


def test_quote_schema_has_expected_fields():
    resp = client.get("/openapi.json")
    spec = resp.json()

    quote_path = spec["paths"]["/quote"]["post"]
    assert "requestBody" in quote_path
    schema_node = quote_path["requestBody"]["content"]["application/json"]["schema"]

    # Resolve $ref if present
    schema = _resolve_schema_ref(spec, schema_node)

    # Ensure top-level properties exist
    props = schema.get("properties", {})
    assert "property_details" in props
    assert "segmentation_result" in props
    assert "pitch_result" in props
    assert "preferences" in props