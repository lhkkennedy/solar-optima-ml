import os
import argparse
import torch
import torch.nn as nn


class FamilyClassifierNet(nn.Module):
    """
    Tiny 3-channel classifier producing 4 logits (T11, T21, T32, T43).
    Input: (N, 3, H, W)
    Output: (N, 4)
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class RoofFamilyNet(nn.Module):
    """
    Tiny 1-channel classifier producing 4 logits (flat, gable, hip, pyramid).
    Input: (N, 1, H, W)
    Output: (N, 4)
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def export_dummy_family(out_dir: str, input_size: int = 128) -> str:
    model = FamilyClassifierNet()
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "family_classifier.onnx")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"input": {0: "N", 2: "H", 3: "W"}, "logits": {0: "N"}},
    )
    return onnx_path


def export_dummy_roof(out_dir: str, input_size: int = 128) -> str:
    model = RoofFamilyNet()
    model.eval()
    dummy = torch.randn(1, 1, input_size, input_size, dtype=torch.float32)
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "roof_family.onnx")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"input": {0: "N", 2: "H", 3: "W"}, "logits": {0: "N"}},
    )
    return onnx_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate minimal ONNX classifiers for procedural roof testing")
    ap.add_argument("--out", required=True, help="Output directory (e.g., C:/models)")
    ap.add_argument("--size", type=int, default=128, help="Input size H=W")
    args = ap.parse_args()
    fam = export_dummy_family(args.out, args.size)
    roo = export_dummy_roof(args.out, args.size)
    print("Wrote:", fam)
    print("Wrote:", roo)


if __name__ == "__main__":
    main()


