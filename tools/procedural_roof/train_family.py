import os
import argparse
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class MaskDataset(Dataset):
    def __init__(self, root: str, size: int = 128, classes: list[str] | None = None):
        self.paths = sorted(glob.glob(os.path.join(root, "*.png")))
        self.size = size
        self.tf = transforms.Compose([
            transforms.ToTensor(),  # HWC uint8 -> CHW float [0,1]
        ])
        # Supported families (8 classes)
        self.classes = classes or [
            "T11", "T21", "T31", "T32", "T41", "T42", "T43", "T44"
        ]
        self.label_map = {name: idx for idx, name in enumerate(self.classes)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        x = self.tf(img)
        # Label from filename prefix
        name = os.path.basename(p)
        prefix = name.split("_")[0]
        y_idx = self.label_map.get(prefix, 0)
        return x.repeat(3, 1, 1), torch.tensor(int(y_idx), dtype=torch.long)


def train(root: str, out: str, epochs: int = 5, bs: int = 64, lr: float = 1e-3, classes: list[str] | None = None, size: int = 128, eval_data: str | None = None, test_data: str | None = None):
    ds = MaskDataset(root, size=size, classes=classes)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2)
    model = models.resnet18(weights=None)
    num_classes = len(ds.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def _evaluate(split_root: str, tag: str):
        if not split_root:
            return
        dse = MaskDataset(split_root, size=size, classes=ds.classes)
        dle = DataLoader(dse, batch_size=bs, shuffle=False, num_workers=2)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dle:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        acc = correct / max(1, total)
        print(f"{tag} accuracy: {acc:.4f} ({correct}/{total}) on classes {ds.classes}")

    # Training loop with graceful interrupt
    model.train()
    try:
        for ep in range(epochs):
            tot = 0.0
            seen = 0
            correct = 0
            for i, (x, y) in enumerate(dl, start=1):
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                tot += float(loss.item())
                seen += int(y.numel())
                correct += int((logits.argmax(dim=1) == y).sum().item())
                if i % 50 == 0 or i == len(dl):
                    print(f"ep {ep+1}/{epochs} step {i}/{len(dl)} loss {tot/i:.4f} acc {correct/max(1,seen):.4f}")
            print(f"epoch {ep+1}/{epochs} loss {tot/len(dl):.4f} acc {correct/max(1,seen):.4f}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model and running evaluation...")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save(model.state_dict(), out)
        print(f"saved: {out}")
        _evaluate(eval_data or "", "val")
        _evaluate(test_data or "", "test")
        return

    # Normal save + evaluation after completing all epochs
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"saved: {out}")
    _evaluate(eval_data or "", "val")
    _evaluate(test_data or "", "test")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--classes", type=str, default="", help="Comma list of class names to map by filename prefix (e.g., I,L,T,U,Z,O)")
    ap.add_argument("--eval_data", type=str, default="", help="Optional validation split path")
    ap.add_argument("--test_data", type=str, default="", help="Optional test split path")
    args = ap.parse_args()
    classes = [s.strip() for s in args.classes.split(",") if s.strip()] if args.classes else None
    train(args.data, args.out, epochs=args.epochs, bs=args.bs, lr=args.lr, classes=classes, size=args.size, eval_data=args.eval_data, test_data=args.test_data)

