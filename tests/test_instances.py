import numpy as np

from app.services.instance_service import InstanceService


def test_instance_service_smoke():
    # Create a simple image with a bright square building-like region
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[64:192, 64:192, :] = 255
    svc = InstanceService()
    # Do not require GPU; CPU inference is fine for smoke test
    svc.warmup((256, 256))
    instances = svc.detect(img)
    assert isinstance(instances, list)
    # We do not guarantee detections with COCO weights; just ensure it runs without error
    assert len(instances) >= 0

