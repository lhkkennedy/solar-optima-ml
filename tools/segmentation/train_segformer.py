import os
import argparse
from typing import Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import SegformerForSemanticSegmentation


def _first(x: Any) -> Any:
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return x[0]
    return x


def _to_pil(img: Any) -> Image.Image:
    img = _first(img)
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return Image.fromarray(img.astype(np.uint8), mode="L").convert("RGB")
        if img.shape[-1] == 4:
            img = img[..., :3]
        return Image.fromarray(img.astype(np.uint8), mode="RGB")
    try:
        return img.convert("RGB")
    except Exception:
        raise TypeError(f"Unsupported image type: {type(img)}")


def _to_mask_pil(mask: Any) -> Image.Image:
    mask = _first(mask)
    if isinstance(mask, Image.Image):
        return mask.convert("L")
    if isinstance(mask, np.ndarray):
        if mask.ndim == 3:
            mask = mask[..., 0]
        return Image.fromarray(mask.astype(np.uint8), mode="L")
    try:
        return mask.convert("L")
    except Exception:
        raise TypeError(f"Unsupported mask type: {type(mask)}")


def build_transforms(image_size: int, is_train: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    aug = []
    if is_train:
        aug.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02))
    img_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        *aug,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
        transforms.PILToTensor(),
    ])
    return img_tf, mask_tf


def load_hf_datasets(name1: str, name2: str, train_split: str = "train", val_split: Optional[str] = None, seed: int = 42):
    from datasets import load_dataset, concatenate_datasets

    ds1 = load_dataset(name1)
    ds2 = load_dataset(name2)

    def get_split(ds, split):
        if split in ds:
            return ds[split]
        return ds[list(ds.keys())[0]].train_test_split(test_size=0.1, seed=seed)["train" if split == "train" else "test"]

    train1 = get_split(ds1, train_split)
    train2 = get_split(ds2, train_split)
    train = concatenate_datasets([train1, train2])

    if val_split is not None:
        val1 = get_split(ds1, val_split)
        val2 = get_split(ds2, val_split)
        val = concatenate_datasets([val1, val2])
    else:
        split = train.train_test_split(test_size=0.05, seed=seed)
        train, val = split["train"], split["test"]
    return train, val


class HFDatasetTorch(Dataset):
    def __init__(self, hf_ds, image_size: int, is_train: bool = False, dup_factor: int = 1,
                 p_hflip: float = 0.5, p_vflip: float = 0.0, p_rot90: float = 0.5,
                 max_affine_deg: float = 10.0, max_affine_translate: float = 0.05,
                 scale_min: float = 0.95, scale_max: float = 1.05, add_noise_std: float = 0.0):
        self.hf_ds = hf_ds
        self.img_tf, self.mask_tf = build_transforms(image_size, is_train)
        ex = hf_ds[0]
        self.image_field = "image" if "image" in ex else next((k for k in ex.keys() if "image" in k), None)
        if self.image_field is None:
            raise KeyError("Could not infer image field name in dataset")
        mask_fields = ["conditioning_image", "mask", "label", "segmentation", "annotation"]
        self.mask_field = next((k for k in mask_fields if k in ex), None)
        if self.mask_field is None:
            self.mask_field = next((k for k in ex.keys() if "mask" in k.lower()), None)
        if self.mask_field is None:
            raise KeyError("Could not infer mask field name in dataset")
        self.is_train = is_train
        self.dup_factor = max(1, int(dup_factor))
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90
        self.max_affine_deg = max_affine_deg
        self.max_affine_translate = max_affine_translate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.add_noise_std = add_noise_std

    def __len__(self) -> int:
        return len(self.hf_ds) * self.dup_factor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx = idx % len(self.hf_ds)
        example = self.hf_ds[base_idx]
        img = _to_pil(example[self.image_field])
        m = _to_mask_pil(example[self.mask_field])
        if self.is_train:
            if np.random.rand() < self.p_hflip:
                img = TF.hflip(img); m = TF.hflip(m)
            if np.random.rand() < self.p_vflip:
                img = TF.vflip(img); m = TF.vflip(m)
            if np.random.rand() < self.p_rot90:
                k = int(np.random.choice([1, 2, 3]))
                img = TF.rotate(img, 90 * k, interpolation=TF.InterpolationMode.BILINEAR)
                m = TF.rotate(m, 90 * k, interpolation=TF.InterpolationMode.NEAREST)
            deg = float(np.random.uniform(-self.max_affine_deg, self.max_affine_deg))
            tx = float(np.random.uniform(-self.max_affine_translate, self.max_affine_translate))
            ty = float(np.random.uniform(-self.max_affine_translate, self.max_affine_translate))
            scale = float(np.random.uniform(self.scale_min, self.scale_max))
            img = TF.affine(img, angle=deg, translate=(int(tx * img.width), int(ty * img.height)),
                            scale=scale, shear=[0.0, 0.0], interpolation=TF.InterpolationMode.BILINEAR)
            m = TF.affine(m, angle=deg, translate=(int(tx * m.width), int(ty * m.height)),
                          scale=scale, shear=[0.0, 0.0], interpolation=TF.InterpolationMode.NEAREST)
        xi = self.img_tf(img)
        if self.is_train and self.add_noise_std > 0.0:
            noise = torch.randn_like(xi) * float(self.add_noise_std)
            xi = torch.clamp(xi + noise, -3.0, 3.0)
        xm = self.mask_tf(m)
        xm = (xm > 127).long().squeeze(0)
        return {"pixel_values": xi, "labels": xm}


def compute_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (pred & target).float().sum().item()
    union = (pred | target).float().sum().item()
    return float((inter + eps) / (union + eps))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
    except Exception:
        pass

    train_hf, val_hf = load_hf_datasets(args.dataset1, args.dataset2)
    train_ds = HFDatasetTorch(train_hf, args.image_size, is_train=True, dup_factor=args.aug_dup,
                              p_hflip=0.5, p_vflip=0.1, p_rot90=0.5, max_affine_deg=10.0, max_affine_translate=0.05,
                              scale_min=0.95, scale_max=1.05, add_noise_std=args.aug_noise)
    val_ds = HFDatasetTorch(val_hf, args.image_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type=="cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    # Pretrained SegFormer-B0 with adapted head
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512", num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion_ce = nn.CrossEntropyLoss()

    best_iou = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))  # type: ignore[attr-defined]

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        total_loss = 0.0
        for batch in pbar:
            x = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):  # type: ignore[attr-defined]
                out = model(pixel_values=x)
                logits = out.logits  # (N,2,h,w)
                # upsample to GT size
                logits = torch.nn.functional.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion_ce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})

        # validation
        model.eval()
        val_iou = 0.0
        cnt = 0
        with torch.inference_mode():
            for batch in val_loader:
                x = batch["pixel_values"].to(device)
                y = batch["labels"].to(device)
                out = model(pixel_values=x)
                logits = out.logits
                logits = torch.nn.functional.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                pred = logits.softmax(dim=1)[:,1] > 0.5
                val_iou += compute_iou(pred.to(torch.uint8), (y==1).to(torch.uint8))
                cnt += 1
        val_iou /= max(1, cnt)
        print(f"val IoU={val_iou:.4f}")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), args.out)
            print(f"Saved best checkpoint to {args.out} (IoU={best_iou:.4f})")


def parse_args():
    ap = argparse.ArgumentParser(description="Train SegFormer-B0 roof segmentation using HF datasets")
    ap.add_argument("--dataset1", default="dpanangian/roof-segmentation-control-net")
    ap.add_argument("--dataset2", default="dpanangian/roof3d-segmentation-control-net")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--aug_dup", type=int, default=4)
    ap.add_argument("--aug_noise", type=float, default=0.02)
    ap.add_argument("--out", default="C:/models/segformer_b0_roof.pt")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)


