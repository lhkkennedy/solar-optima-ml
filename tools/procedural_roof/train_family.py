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
    def __init__(self, root: str, size: int = 128):
        self.paths = sorted(glob.glob(os.path.join(root, "*.png")))
        self.size = size
        self.tf = transforms.Compose([
            transforms.ToTensor(),  # HWC uint8 -> CHW float [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        x = self.tf(img)
        # Label from filename prefix: T11, T21, T32, T43
        name = os.path.basename(p)
        label_map = {"T11": 0, "T21": 1, "T32": 2, "T43": 3}
        y_idx = 0
        for k, v in label_map.items():
            if name.startswith(k + "_"):
                y_idx = v
                break
        return x.repeat(3, 1, 1), torch.tensor(y_idx, dtype=torch.long)


def train(root: str, out: str, epochs: int = 5, bs: int = 64, lr: float = 1e-3):
    ds = MaskDataset(root)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)  # T11,T21,T32,T43
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
    args = ap.parse_args()
    train(args.data, args.out, epochs=args.epochs)

