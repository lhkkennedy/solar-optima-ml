import argparse
from typing import List
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

# Ensure project root on sys.path
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tools.segmentation.train_segformer import HFDatasetTorch, load_hf_datasets


def compute_iou_binary(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (pred & target).float().sum().item()
    union = (pred | target).float().sum().item()
    return float((inter + eps) / (union + eps))


def sweep_threshold(ckpt: str,
                    dataset1: str,
                    dataset2: str,
                    image_size: int,
                    batch_size: int,
                    thr_min: float,
                    thr_max: float,
                    thr_step: float,
                    num_workers: int = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_hf = load_hf_datasets(dataset1, dataset2)
    val_ds = HFDatasetTorch(val_hf, image_size, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512", num_labels=2, ignore_mismatched_sizes=True
    )
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
            out = model(pixel_values=x)
            logits = out.logits
            logits = torch.nn.functional.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.softmax(logits, dim=1)[:, 1, :, :]
            for t in thresholds:
                pred = (probs > t).to(torch.uint8)
                iou = compute_iou_binary(pred, (y == 1).to(torch.uint8))
                ious[t] += iou
            counts += 1

    for t in thresholds:
        ious[t] /= max(1, counts)
    best_thr = max(ious.items(), key=lambda kv: kv[1])
    print("Threshold sweep (IoU) - SegFormer:")
    for t in thresholds:
        print(f"  t={t:.3f}: IoU={ious[t]:.4f}")
    print(f"\nRecommended SEG_THRESH={best_thr[0]:.3f} (IoU={best_thr[1]:.4f})")


def parse_args():
    ap = argparse.ArgumentParser(description="Sweep threshold (SegFormer) on validation set to maximize IoU")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset1", default="dpanangian/roof-segmentation-control-net")
    ap.add_argument("--dataset2", default="dpanangian/roof3d-segmentation-control-net")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=12)
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


