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
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR


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
    # datasets.Image type supports .convert
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
            # take one channel
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
        transforms.PILToTensor(),  # uint8 [0..255]
    ])
    return img_tf, mask_tf


def load_hf_datasets(name1: str, name2: str, train_split: str = "train", val_split: Optional[str] = None, seed: int = 42):
    from datasets import load_dataset, concatenate_datasets

    ds1 = load_dataset(name1)
    ds2 = load_dataset(name2)

    def get_split(ds, split):
        if split in ds:
            return ds[split]
        # fall back to creating a split
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
    def __init__(self, hf_ds, image_size: int, is_train: bool = False,
                 dup_factor: int = 1,
                 p_hflip: float = 0.5,
                 p_vflip: float = 0.0,
                 p_rot90: float = 0.5,
                 max_affine_deg: float = 10.0,
                 max_affine_translate: float = 0.05,
                 scale_min: float = 0.95,
                 scale_max: float = 1.05,
                 add_noise_std: float = 0.0):
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
        # paired geometric augmentations
        if self.is_train:
            # flips
            if np.random.rand() < self.p_hflip:
                img = TF.hflip(img); m = TF.hflip(m)
            if np.random.rand() < self.p_vflip:
                img = TF.vflip(img); m = TF.vflip(m)
            # 90-degree rotations
            if np.random.rand() < self.p_rot90:
                k = int(np.random.choice([1, 2, 3]))
                img = TF.rotate(img, 90 * k, interpolation=TF.InterpolationMode.BILINEAR)
                m = TF.rotate(m, 90 * k, interpolation=TF.InterpolationMode.NEAREST)
            # small affine
            deg = float(np.random.uniform(-self.max_affine_deg, self.max_affine_deg))
            tx = float(np.random.uniform(-self.max_affine_translate, self.max_affine_translate))
            ty = float(np.random.uniform(-self.max_affine_translate, self.max_affine_translate))
            scale = float(np.random.uniform(self.scale_min, self.scale_max))
            shear = [0.0, 0.0]
            img = TF.affine(img, angle=deg, translate=(int(tx * img.width), int(ty * img.height)),
                            scale=scale, shear=shear, interpolation=TF.InterpolationMode.BILINEAR)
            m = TF.affine(m, angle=deg, translate=(int(tx * m.width), int(ty * m.height)),
                          scale=scale, shear=shear, interpolation=TF.InterpolationMode.NEAREST)
        xi = self.img_tf(img)
        if self.is_train and self.add_noise_std > 0.0:
            noise = torch.randn_like(xi) * float(self.add_noise_std)
            xi = torch.clamp(xi + noise, -3.0, 3.0)
        xm = self.mask_tf(m)
        xm = (xm > 127).long().squeeze(0)
        return {"pixel_values": xi, "labels": xm}


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2, eps: float = 1e-6) -> float:
    # pred, target: (N,H,W) long
    iou = 0.0
    for cls in range(1, num_classes):  # ignore background in IoU
        pred_c = (pred == cls).float()
        tgt_c = (target == cls).float()
        inter = (pred_c * tgt_c).sum()
        union = pred_c.sum() + tgt_c.sum() - inter
        iou_c = float((inter + eps) / (union + eps))
        iou += iou_c
    return iou / max(1, num_classes - 1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Enable backend autotuning for fixed input sizes
    try:
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("medium")  # torch>=2.0
    except Exception:
        pass

    train_hf, val_hf = load_hf_datasets(args.dataset1, args.dataset2, args.train_split, args.val_split)
    train_ds = HFDatasetTorch(
        train_hf, args.image_size, is_train=True, dup_factor=args.aug_dup,
        p_hflip=0.5, p_vflip=0.1, p_rot90=0.5,
        max_affine_deg=10.0, max_affine_translate=0.05,
        scale_min=0.95, scale_max=1.05, add_noise_std=args.aug_noise
    )
    val_ds = HFDatasetTorch(val_hf, args.image_size, is_train=False)

    # On Windows, multiprocessing with dataset transforms can fail to pickle closures.
    # Use single-process data loading for reliability.
    pin_mem = device.type == "cuda"
    nw = max(0, int(args.num_workers))
    common_dl_kwargs = dict(pin_memory=pin_mem)
    if nw > 0:
        common_dl_kwargs.update(dict(num_workers=nw, prefetch_factor=2, persistent_workers=True))
    else:
        common_dl_kwargs.update(dict(num_workers=0))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **common_dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **common_dl_kwargs,
    )

    model = deeplabv3_resnet50(weights=None, num_classes=2)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion_ce = nn.CrossEntropyLoss()
    # LR scheduler selection
    if args.scheduler == "cosine_wr":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, args.t0), T_mult=max(1, args.tmult), eta_min=args.min_lr)
        scheduler_per_batch = False
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=1, pct_start=0.3)
        # We'll step per batch with correct steps_per_epoch below
        scheduler_per_batch = True
    else:  # cosine
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)
        scheduler_per_batch = False
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # type: ignore[attr-defined]

    def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # logits: (N,2,H,W), target: (N,H,W) in {0,1}
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]  # foreground
        target_f = target.float()
        inter = (probs * target_f).sum(dim=(1, 2))
        denom = probs.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        dice = (2.0 * inter + eps) / (denom + eps)
        return 1.0 - dice.mean()

    def focal_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.75, eps: float = 1e-6) -> torch.Tensor:
        # logits: (N,2,H,W), target: (N,H,W)
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        # Gather foreground probs according to targets
        target_flat = target.view(logits.size(0), 1, *target.shape[1:])
        log_pt = torch.gather(log_probs, 1, target_flat).squeeze(1)  # (N,H,W)
        pt = torch.gather(probs, 1, target_flat).squeeze(1)
        alpha_t = torch.where(target == 1, torch.tensor(alpha, device=logits.device), torch.tensor(1.0 - alpha, device=logits.device))
        loss = -alpha_t * (1 - pt).pow(gamma) * log_pt
        return loss.mean()

    # Lovasz-Softmax helpers (binary compatible)
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        gts = gt_sorted.sum()
        inter = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - inter / torch.clamp(union, min=1e-6)
        if gt_sorted.numel() > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def _flatten_probas(probas: torch.Tensor, labels: torch.Tensor):
        C = probas.size(1)
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        return probas, labels

    def lovasz_softmax_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probas = torch.softmax(logits, dim=1)
        probas, labels = _flatten_probas(probas, labels)
        C = probas.size(1)
        losses = []
        for c in range(C):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probas[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = _lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))
        if not losses:
            return torch.tensor(0.0, device=logits.device)
        return torch.mean(torch.stack(losses))

    best_iou = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        total_loss = 0.0
        for i, batch in enumerate(pbar, start=1):
            x = batch["pixel_values"].to(device, non_blocking=pin_mem)
            y = batch["labels"].to(device, non_blocking=pin_mem)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                    out = model(x)["out"]
                    ce = criterion_ce(out, y)
                    dl = dice_loss_from_logits(out, y) if args.dice_weight > 0 else torch.tensor(0.0, device=out.device)
                    fl = focal_loss_from_logits(out, y, gamma=args.focal_gamma, alpha=args.focal_alpha) if args.focal_weight > 0 else torch.tensor(0.0, device=out.device)
                    lv = lovasz_softmax_loss(out, y) if args.lovasz_weight > 0 else torch.tensor(0.0, device=out.device)
                    base_w = max(0.0, 1.0 - args.dice_weight - args.focal_weight - args.lovasz_weight)
                    loss = base_w * ce + args.dice_weight * dl + args.focal_weight * fl + args.lovasz_weight * lv
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)["out"]  # (N,2,H,W)
                ce = criterion_ce(out, y)
                dl = dice_loss_from_logits(out, y) if args.dice_weight > 0 else torch.tensor(0.0, device=out.device)
                fl = focal_loss_from_logits(out, y, gamma=args.focal_gamma, alpha=args.focal_alpha) if args.focal_weight > 0 else torch.tensor(0.0, device=out.device)
                lv = lovasz_softmax_loss(out, y) if args.lovasz_weight > 0 else torch.tensor(0.0, device=out.device)
                base_w = max(0.0, 1.0 - args.dice_weight - args.focal_weight - args.lovasz_weight)
                loss = base_w * ce + args.dice_weight * dl + args.focal_weight * fl + args.lovasz_weight * lv
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            if args.scheduler == "onecycle":
                # step per batch with correct steps_per_epoch
                # reset steps_per_epoch at start of epoch
                if i == 1:
                    scheduler._step_count = 0  # reset internal counter
                    scheduler.total_steps = len(train_loader)  # type: ignore[attr-defined]
                scheduler.step()
            pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})

        # validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        count = 0
        with torch.inference_mode():
            for batch in val_loader:
                x = batch["pixel_values"].to(device, non_blocking=pin_mem)
                y = batch["labels"].to(device, non_blocking=pin_mem)
                if use_amp:
                    with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                        out = model(x)["out"]
                        ce = criterion_ce(out, y)
                        dl = dice_loss_from_logits(out, y) if args.dice_weight > 0 else torch.tensor(0.0, device=out.device)
                        fl = focal_loss_from_logits(out, y, gamma=args.focal_gamma, alpha=args.focal_alpha) if args.focal_weight > 0 else torch.tensor(0.0, device=out.device)
                        lv = lovasz_softmax_loss(out, y) if args.lovasz_weight > 0 else torch.tensor(0.0, device=out.device)
                        base_w = max(0.0, 1.0 - args.dice_weight - args.focal_weight - args.lovasz_weight)
                        loss = base_w * ce + args.dice_weight * dl + args.focal_weight * fl + args.lovasz_weight * lv
                else:
                    out = model(x)["out"]
                    ce = criterion_ce(out, y)
                    dl = dice_loss_from_logits(out, y) if args.dice_weight > 0 else torch.tensor(0.0, device=out.device)
                    fl = focal_loss_from_logits(out, y, gamma=args.focal_gamma, alpha=args.focal_alpha) if args.focal_weight > 0 else torch.tensor(0.0, device=out.device)
                    lv = lovasz_softmax_loss(out, y) if args.lovasz_weight > 0 else torch.tensor(0.0, device=out.device)
                    base_w = max(0.0, 1.0 - args.dice_weight - args.focal_weight - args.lovasz_weight)
                    loss = base_w * ce + args.dice_weight * dl + args.focal_weight * fl + args.lovasz_weight * lv
                val_loss += float(loss.item())
                pred = out.argmax(dim=1)
                val_iou += compute_iou(pred, y)
                count += 1
        val_loss /= max(1, count)
        val_iou /= max(1, count)
        print(f"val: loss={val_loss:.4f} IoU={val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), args.out)
            print(f"Saved best checkpoint to {args.out} (IoU={best_iou:.4f})")
        if not scheduler_per_batch:
            scheduler.step()


def parse_args():
    ap = argparse.ArgumentParser(description="Train DeepLabv3 (ResNet50) roof segmentation using HF datasets")
    ap.add_argument("--dataset1", default="dpanangian/roof-segmentation-control-net")
    ap.add_argument("--dataset2", default="dpanangian/roof3d-segmentation-control-net")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default=None)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="C:/models/seg_deeplabv3_resnet50_roof.pt")
    ap.add_argument("--amp", type=int, default=1, help="Enable mixed precision (1/0)")
    ap.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (Windows-safe if dataset is picklable)")
    ap.add_argument("--aug_dup", type=int, default=1, help="Oversample factor with random augments")
    ap.add_argument("--aug_noise", type=float, default=0.0, help="Stddev of Gaussian noise added to images after normalization")
    ap.add_argument("--dice_weight", type=float, default=0.3, help="Weight for Dice loss term [0..1]")
    ap.add_argument("--min_lr", type=float, default=1e-6, help="Cosine annealing minimum LR")
    ap.add_argument("--focal_weight", type=float, default=0.0, help="Weight for focal loss term [0..1]")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_alpha", type=float, default=0.75)
    ap.add_argument("--lovasz_weight", type=float, default=0.0, help="Weight for Lovasz-Softmax loss term [0..1]")
    ap.add_argument("--scheduler", choices=["cosine", "cosine_wr", "onecycle"], default="cosine")
    ap.add_argument("--t0", type=int, default=5, help="Cosine warm restarts T_0")
    ap.add_argument("--tmult", type=int, default=2, help="Cosine warm restarts T_mult")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)


