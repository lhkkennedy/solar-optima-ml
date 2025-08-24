## ML-10: Wiring segmentation to PBSR and training the building family classifier

### Overview

We connect the roof segmentation module to the Procedural Building Shapes and Roofs (PBSR) pipeline, then add two small classifiers:

- Family classifier (Classifier 1): predicts one of 8 PBSR families {T11, T21, T31, T32, T41, T42, T43, T44} from a binary building mask crop.
- Roof family classifier (Classifier 2): per-rect classification {flat, gable, hip, pyramid} using gray/edge crops.

This document covers the wiring plan and the training recipe for the family classifier; the roof classifier follows the same pattern.

### Data flow and wiring

1) Segmentation → building crop
- Use SegFormer (512) to get roof mask.
- Extract single-building crop (mask + RGB bbox); resize to 256×256; binarize.

2) Family prediction + PBSR
- Call the family classifier on the binarized crop to predict one of 8 families.
- PBSR enumerates rect configurations; restrict the search to the predicted family; re-rank by IoU.
- If classifier confidence < τ (e.g., 0.7), fall back to full-family search.

3) Roof family per rect (optional)
- Crop each rect from gray image; run roof-family classifier; use to steer ridge detection and labels in the output.

4) Downstream
- Ridge detection, polygon assembly, and elevation augmentation proceed as before.

### Serving models

- Enable with `PROC_ROOF_USE_CLASSIFIER=1`.
- ONNX files loaded from `PROC_ROOF_ONNX_DIR`:
  - `family_classifier.onnx`
  - `roof_family.onnx`

### Classifier 1: building family detector (training)

Goal: 8-class classification {T11, T21, T31, T32, T41, T42, T43, T44}.

Data sources:
- Synthetic PBSR masks: use `gen_pbsr_families` with strong variation and affine/morph ops.
- Real crops (optional): distill weak labels by fitting all families and choosing the best IoU; use as additional training data.

Training script:
- Tools: `tools/procedural_roof/train_family.py` (ResNet18→8 classes, 128×128 inputs).
- Improvements to apply:
  - Balanced sampling per class
  - Cosine LR or OneCycle; epochs 15–30; batch 64
  - Augment: flips/rot90/affine; small noise; random resized crop to 128
  - ONNX export for runtime

Export:
- Use `tools/procedural_roof/export_onnx.py --in_channels 3 --num_classes 8 --input_size 128` for ONNX.

Integration checkpoints:
- In `procedural_roof/pipeline.py`, before `pbsr.match`, run the family classifier. Restrict candidate families accordingly. If classifier confidence low, allow full family search as a fallback.

Metrics:
- Family classifier accuracy/F1 on mixed synthetic + real validation set.
- End-to-end: IoU between regularized rect mask and segmentation mask; report improvement vs heuristics-only.

### Commands (example)

Generate synthetic dataset:
```
python tools/procedural_roof/gen_synth_footprints.py --out C:\proc_data\pbsr_labeled --num 200000 --mode pbsr
```

Train family classifier:
```
python tools\procedural_roof\train_family.py --data C:\proc_data\pbsr_labeled --out C:\proc_ckpts\family_resnet18.pt --epochs 20
```

Export ONNX:
```
python tools\procedural_roof\export_onnx.py --pt C:\proc_ckpts\family_resnet18.pt --onnx C:\models\family_classifier.onnx --in_channels 3 --num_classes 8 --input_size 128
```

Enable in API:
```
PROC_ROOF_USE_CLASSIFIER=1
PROC_ROOF_ONNX_DIR=C:\models
```

### Next steps (after family classifier)

- Add classifier gating in `procedural_roof/pipeline.py` and confidence fallback.
- Train roof-family classifier using `tools/procedural_roof/train_roof.py` + ONNX export; wire per-rect predictions to ridge detection.
- Add small A/B test harness to compare heuristics-only vs classifier-assisted PBSR on a held-out set.


