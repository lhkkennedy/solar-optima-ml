import argparse
from typing import List
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50

# Ensure project root is on sys.path when running as a script
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tools.segmentation.train_deeplab import HFDatasetTorch, load_hf_datasets


def compute_iou_binary(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (pred & target).float().sum().item()
    union = (pred | target).float().sum().item()
    return float((inter + eps) / (union + eps))


def sweep_threshold(
    ckpt: str,
    dataset1: str,
    dataset2: str,
    image_size: int,
    batch_size: int,
    thr_min: float,
    thr_max: float,
    thr_step: float,
    num_workers: int = 2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_hf = load_hf_datasets(dataset1, dataset2)
    val_ds = HFDatasetTorch(val_hf, image_size, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = deeplabv3_resnet50(weights=None, num_classes=2)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    thresholds: List[float] = [round(x, 4) for x in np.arange(thr_min, thr_max + 1e-9, thr_step).tolist()]
    ious = {t: 0.0 for t in thresholds}
    counts = 0

    with torch.inference_mode():
        for batch in val_loader:
            x = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)
            out = model(x)["out"]
            probs = torch.softmax(out, dim=1)[:, 1, :, :]
            for t in thresholds:
                pred = (probs > t).to(torch.uint8)
                iou = compute_iou_binary(pred, (y == 1).to(torch.uint8))
                ious[t] += iou
            counts += 1

    for t in thresholds:
        ious[t] /= max(1, counts)
    best_thr = max(ious.items(), key=lambda kv: kv[1])
    print("Threshold sweep (IoU) - DeepLab:")
    for t in thresholds:
        print(f"  t={t:.3f}: IoU={ious[t]:.4f}")
    print(f"\nRecommended SEG_THRESH={best_thr[0]:.3f} (IoU={best_thr[1]:.4f})")


def parse_args():
    ap = argparse.ArgumentParser(description="Sweep threshold (DeepLab) on validation set to maximize IoU")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset1", default="dpanangian/roof-segmentation-control-net")
    ap.add_argument("--dataset2", default="dpanangian/roof3d-segmentation-control-net")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--thr_min", type=float, default=0.30)
    ap.add_argument("--thr_max", type=float, default=0.70)
    ap.add_argument("--thr_step", type=float, default=0.02)
    ap.add_argument("--num_workers", type=int, default=2)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sweep_threshold(
        ckpt=args.ckpt,
        dataset1=args.dataset1,
        dataset2=args.dataset2,
        image_size=args.image_size,
        batch_size=args.batch_size,
        thr_min=args.thr_min,
        thr_max=args.thr_max,
        thr_step=args.thr_step,
        num_workers=args.num_workers,
    )


