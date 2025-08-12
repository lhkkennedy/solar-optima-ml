import os
import argparse
import torch
from torchvision import models


def export_resnet18(pt_path: str, onnx_path: str, num_classes: int = 4, input_size: int = 128, in_channels: int = 3):
    model = models.resnet18(weights=None)
    # Adjust first conv for desired input channels (1 for roof, 3 for family)
    if in_channels != 3:
        old_conv = model.conv1
        model.conv1 = torch.nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                      stride=old_conv.stride, padding=old_conv.padding, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    sd = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()

    dummy = torch.randn(1, in_channels, input_size, input_size, dtype=torch.float32)
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
    ap.add_argument("--in_channels", type=int, default=3)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--input_size", type=int, default=128)
    args = ap.parse_args()
    export_resnet18(args.pt, args.onnx, num_classes=args.num_classes, input_size=args.input_size, in_channels=args.in_channels)

