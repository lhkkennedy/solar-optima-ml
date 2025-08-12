import os
import argparse
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models


class EdgeCropDataset(Dataset):
    def __init__(self, root: str, size: int = 128):
        # expects grayscale PNG crops of roof parts, with optional noise
        self.paths = sorted(glob.glob(os.path.join(root, "*.png")))
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        # 1-channel input expected by modified resnet18
        x = img.astype(np.float32) / 255.0
        x = x[None, :, :]  # CHW
        # placeholder label mapping by filename convention: flat/gable/hip/pyramid
        name = os.path.basename(p).lower()
        label_map = {"flat": 0, "gable": 1, "hip": 2, "pyramid": 3}
        y = 0
        for k, v in label_map.items():
            if k in name:
                y = v
                break
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def train(root: str, out: str, epochs: int = 5, bs: int = 64, lr: float = 1e-3):
    ds = EdgeCropDataset(root)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2)
    model = models.resnet18(weights=None)
    # make it accept 1-channel
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                            stride=old_conv.stride, padding=old_conv.padding, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 4)  # flat,gable,hip,pyramid
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


