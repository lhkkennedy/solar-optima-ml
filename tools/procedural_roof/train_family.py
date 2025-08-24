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


def train(root: str, out: str, epochs: int = 5, bs: int = 64, lr: float = 1e-3, num_classes: int = 8, size: int = 128):
    ds = MaskDataset(root, size=size)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tot += float(loss.item())
        print(f"epoch {ep+1}/{epochs} loss {tot/len(dl):.4f}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(model.state_dict(), out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_classes", type=int, default=8)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()
    train(args.data, args.out, epochs=args.epochs, bs=args.bs, lr=args.lr, num_classes=args.num_classes, size=args.size)

