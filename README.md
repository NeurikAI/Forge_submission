# FastSAM Fine-tuning Pipeline

End-to-end pipeline for fine-tuning FastSAM (Fast Segment Anything Model) on polybag/fashion item segmentation. Includes data preparation, multi-phase training, and depth-aware evaluation.

## Directory Structure

```
fastsam_finetuning/
├── train.py                      # Main training script
├── validate_fastsam.py           # Validation with metrics & visualizations
├── preferential_f1_score.py      # Depth-aware evaluation (DepthPro + F1)
├── fashion.yaml                  # YOLO dataset config
└── data_handling/
    ├── sam3_annotate.py          # Auto-annotation using SAM3
    ├── coco_to_yolo.py           # COCO JSON to YOLO format
    ├── rle_to_yolo.py            # RLE masks to YOLO format
    ├── visual_test.py            # Verify COCO annotations
    └── merge_datasets.py          # Merge phase 1 & phase 2 data
```

## Installation

```bash
pip install ultralytics wandb torch torchvision transformers opencv-python pillow pyyaml pandas tqdm
```

## Workflow

### 1. Prepare Data

**From SAM3 (auto-annotation):**
```bash
python data_handling/sam3_annotate.py \
    --img-folder /path/to/images \
    --prompt "clear plastic wrapped clothing package"
```

**From COCO JSON:**
```bash
python data_handling/coco_to_yolo.py \
    --coco-json annotations.json \
    --output-lbls /path/to/labels
```

**From RLE masks (iMaterialist):**
```bash
python data_handling/rle_to_yolo.py --base-dir /path/to/dataset
```

**Verify annotations:**
```bash
python data_handling/visual_test.py \
    --coco-json annotations.json \
    --img-dir /path/to/images \
    --out-dir /path/to/output
```

### 2. Train

```bash
python train.py
```

Configuration in `train.py`:
- `epochs`: 20
- `imgsz`: 640
- `batch`: 16
- `lr0`: 1e-3
- `device`: GPU ID
- `workers`: 8

### 3. Validate

```bash
python validate_fastsam.py \
    --model best.pt \
    --data fashion.yaml \
    --imgsz 1024 \
    --conf 0.3
```

Output: mAP50, mAP50-95, mask visualizations

### 4. Evaluate with Depth

```bash
python preferential_f1_score.py \
    --root-dir /path/to/root/containing/scene_folders \
    --output-dir /path/to/output \
    --weights best.pt \
    --depth-model-id apple/DepthPro-hf \
    --conf 0.30 \
    --imgsz 1024 \
    --iou-threshold 0.50
```

Output: Precision, Recall, F1, TP/FP/FN counts, side-by-side GT vs prediction verification images in TP/ and FN/ folders

## Multi-Phase Training

**Phase 1:** Pre-train on iMaterialist
```bash
python rle_to_yolo.py --base-dir /path/to/imaterialistic
python train.py
```

**Phase 2:** Fine-tune with new data
```bash
python data_handling/sam3_annotate.py --img-folder /new/images
python data_handling/coco_to_yolo.py --coco-json annotations.json --output-lbls labels
python data_handling/merge_datasets.py \
    --p1-base /path/to/phase1 \
    --new-base /path/to/phase2 \
    --new-img /new/images \
    --new-lbl /new/labels
python train.py  # with updated data path
```

## Dataset Configuration

Edit `fashion.yaml`:
```yaml
path: /path/to/datasets/fashion
train: train.txt
val: val.txt
nc: 1
names: ['fashion_item']
```

Expected structure:
```
fashion/
├── images/train/  ├── val/
├── labels/train/  ├── val/
├── train.txt
└── val.txt
```

## Script Reference

| Script | Purpose |
|--------|---------|
| `train.py` | Train FastSAM with W&B tracking |
| `validate_fastsam.py` | Compute mAP50, mAP50-95, visualizations |
| `preferential_f1_score.py` | Depth-aware F1 evaluation: FastSAM-s + DepthPro, selects closest object, generates side-by-side GT vs pred verification images |
| `sam3_annotate.py` | Auto-annotation using SAM3 model |
| `coco_to_yolo.py` | Convert COCO JSON to YOLO format |
| `rle_to_yolo.py` | Convert RLE masks to YOLO format |
| `visual_test.py` | Verify COCO annotations visually |
| `merge_datasets.py` | Merge phase 1 and phase 2 datasets |

## Output

Training outputs saved to:
```
runs/segment/fashion_fastsam/fashion_finetune/
├── weights/
│   ├── best.pt
│   ├── last.pt
│   └── epoch*.pt
├── results.csv
└── plots/
```

Validation outputs saved to:
```
runs/detect/fashion_fastsam/
├── mask_visualizations/
│   ├── TP/   # True Positives (correctly detected preferential objects)
│   └── FN/   # False Negatives (missed preferential objects)
├── results.json
└── scores.csv
```

Depth-aware evaluation outputs (from preferential_f1_score.py):
```
output_dir/
├── TP/   # Side-by-side verification images (GT vs FastSAM+DepthPro) - correct matches
└── FN/   # Side-by-side verification images - incorrect or missed matches
```

## Common Issues

- **CUDA OOM**: Reduce batch size or imgsz
- **Slow loading**: Increase workers or enable cache
- **Low scores**: Check data quality, adjust learning rate
- **SAM3 crash**: Need >20GB GPU memory
- **Missing masks**: Use `visual_test.py` to debug

## References

- FastSAM: https://github.com/CASIA-IVA-Lab/FastSAM
- Ultralytics: https://docs.ultralytics.com/
- DepthPro: https://huggingface.co/apple/DepthPro-hf
- COCO Format: https://cocodataset.org/#format-data

---

