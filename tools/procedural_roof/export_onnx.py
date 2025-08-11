import os
import argparse
import torch
from torchvision import models


def export_resnet18(pt_path: str, onnx_path: str, num_classes: int = 4, input_size: int = 128):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    sd = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"input": {0: "N", 2: "H", 3: "W"}, "logits": {0: "N"}},
    )
    print("Exported to", onnx_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--onnx", required=True)
    args = ap.parse_args()
    export_resnet18(args.pt, args.onnx)

