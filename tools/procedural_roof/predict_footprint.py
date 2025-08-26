import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models


def load_and_prepare_mask(path: str, target_size: int = 128, auto_crop: bool = True) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    # Binarize if needed (handles overlays or soft masks)
    if img.max() > 1:
        # Otsu to separate FG/BG robustly
        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        img_bin = (img > 0).astype(np.uint8) * 255

    # Optional bbox crop with small padding
    if auto_crop:
        ys, xs = np.where(img_bin > 0)
        if len(xs) > 0:
            pad = 5
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(img_bin.shape[1] - 1, x1 + pad); y1 = min(img_bin.shape[0] - 1, y1 + pad)
            img_bin = img_bin[y0:y1 + 1, x0:x1 + 1]

    # Resize to model input
    img_bin = cv2.resize(img_bin, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return img_bin


def predict(mask_path: str, ckpt_path: str, classes: list[str], size: int = 128, no_crop: bool = False, dump_proc: str | None = None, tta_rot90: bool = False) -> tuple[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_bin = load_and_prepare_mask(mask_path, target_size=size, auto_crop=not no_crop)
    if dump_proc:
        cv2.imwrite(dump_proc, img_bin)
    x_base = torch.from_numpy(img_bin).float().div(255.0).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    with torch.no_grad():
        if not tta_rot90:
            logits = model(x_base)
            prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        else:
            # Test-time augmentation over 0/90/180/270 degrees
            probs = []
            for k in range(4):
                xk = torch.rot90(x_base, k=k, dims=(2, 3))
                logits = model(xk)
                pk = torch.softmax(logits, dim=1)[0]
                probs.append(pk)
            prob = torch.stack(probs, dim=0).mean(dim=0).detach().cpu().numpy()
    pred_idx = int(prob.argmax())
    return classes[pred_idx], prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", type=str, default="mask.png")
    ap.add_argument("--ckpt", type=str, default="artifacts/models/footprints_resnet18.pt")
    ap.add_argument("--classes", type=str, default="I,L,T,U,Z,O")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--no_crop", action="store_true", help="Disable bbox auto-cropping before resize")
    ap.add_argument("--dump_proc", type=str, default="", help="Optional path to save the preprocessed 128x128 mask used for inference")
    ap.add_argument("--tta_rot90", action="store_true", help="Average predictions over 0/90/180/270 degree rotations")
    args = ap.parse_args()

    classes = [s.strip() for s in args.classes.split(",") if s.strip()]
    pred, prob = predict(args.mask, args.ckpt, classes, size=args.size, no_crop=args.no_crop, dump_proc=(args.dump_proc or None), tta_rot90=args.tta_rot90)
    print(f"pred: {pred}")
    for c, p in zip(classes, prob):
        print(f"{c}: {p:.4f}")


if __name__ == "__main__":
    main()


