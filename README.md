# FastSAM Fine-tuning Pipeline (Fashion Segmentation)

This provides an end-to-end pipeline to train, validate, and evaluate FastSAM for segmenting fashion items such as clothing packages.

It includes:

* Training FastSAM on custom datasets
* Evaluating segmentation performance using standard metrics
* Depth-aware evaluation for improved object selection
* Full classification metrics (TP, TN, FP, FN)
* Visualization of predictions vs ground truth

---

## What This Pipeline Does

This pipeline allows you to:

* Train FastSAM on custom fashion datasets
* Evaluate segmentation accuracy using mAP and classification metrics
* Perform depth-aware evaluation for selecting correct objects in cluttered scenes
* Compute TP, TN, FP, FN for full performance analysis
* Generate visual outputs for debugging and inspection

---

## Pipeline Structure

```
fastsam_finetuning/
├── train.py                  # Train FastSAM model
├── validate_fastsam.py       # mAP evaluation + segmentation metrics
├── preferential_f1_score.py  # Depth-aware evaluation (FastSAM + DepthPro)
└── fashion.yaml              # Dataset configuration file
```

---

## Installation

```bash
pip install ultralytics wandb torch torchvision transformers \
opencv-python pillow pyyaml pandas tqdm
```

---

## Important: Configuration Setup

**BEFORE running any scripts, you MUST update the paths in these files:**

1. **`fashion.yaml`** - Update the dataset paths:
   ```yaml
      path: fashion #in reference to the Data structure(yolo format)
      train: images/train
      val: images/train
      test: images/test
      nc: 1
      names: [object]
   ```

2. **`train.py`** - Update any hardcoded paths in the script for:
   - Dataset path (if different from fashion.yaml)
   - Output directory for model weights
   - Any other custom paths

**Failing to update these paths will cause the scripts to fail or use incorrect data.**

---

## Quick Start

### 1. Train the Model
Before training, make sure `fashion.yaml` and `train.py` are properly configured with your dataset paths.

```bash
python train.py 
```

**CRITICAL: Configuration (MUST edit in `train.py`):**

**Required Changes:**
* `data='path/to/fashion.yaml'` → **Change to your dataset YAML path** (REQUIRED)
* `device=1` → **Change to your GPU ID** (REQUIRED - e.g., 0, 1, 2)
* `name='...'` → **Change output directory name** (REQUIRED - used for saving weights)
* `project='...'` → **Change W&B project name** (for logging)
* `wandb.init(project="...")` → **Change W&B project** (must match above)

**Optional Tuning:**
* `epochs`: 100 (training iterations)
* `imgsz`: 640 (input image size)
* `batch`: 32 (batch size - reduce if CUDA OOM)
* `lr0`: 1e-3 (learning rate)
* `workers`: 16 (data loading workers)

**Output Locations:**
- Weights: `runs/segment/{project}/{name}/weights/best.pt`
- Logs: Weights & Biases (W&B) dashboard
- Plots: `runs/segment/{project}/{name}/`

---

### 2. Validate Model Performance

```bash
python validate_fastsam.py \
    --model best.pt \
    --data fashion.yaml \
    --imgsz 1024 \
    --conf 0.3
```

#### Outputs

**Standard Metrics**
* Precision
* Recall
* mAP50
* mAP50-95

---

### 3. Depth-Aware Evaluation

This step improves evaluation by using depth to select the most relevant object in cluttered scenes.

```bash
python preferential_f1_score.py \
    --root-dir /path/to/data \
    --output-dir /path/to/output \
    --weights best.pt \
    --depth-model-id apple/DepthPro-hf \
    --conf 0.3 \
    --imgsz 1024 \
    --iou-threshold 0.5
```

**What it does:**
1. Uses your trained FastSAM-s model to detect all fashion items in images
2. Uses DepthPro (AI depth estimation) to determine which object is closest to the camera
3. Selects the closest object as the "preferential" target
4. Compares if the model correctly identified this preferential object
5. Generates side-by-side verification images (Ground Truth vs Predictions)

#### Outputs

**Classification Metrics**
* **TP (True Positives)**: Correct detections with correct depth selection
* **TN (True Negatives)**: Correct absence of object detection
* **FP (False Positives)**: Wrong object selected or hallucinated detection
* **FN (False Negatives)**: Missed target objects

**Evaluation Scores**
* Precision
* Recall
* F1 Score

**Visual Outputs**
* `TP/`: Correct GT vs prediction matches (green = preferential, orange = other objects)
* `TN/`: Correct background (no objects detected)
* `FP/`: Incorrect detections shown side-by-side
* `FN/`: Missed detections highlighted

---

## Understanding Classification Metrics

| Metric | Meaning |
|--------|---------|
| **TP** | Correct detection of fashion item |
| **TN** | Correct rejection of background - areas with no fashion items are correctly identified as empty |
| **FP** | Incorrect detection (false alarm) - background incorrectly marked as fashion item |
| **FN** | Missed fashion item - fashion item present but not detected |

---

## Dataset Configuration

**For Training & Validation (YOLO Format)**

Edit `fashion.yaml`:

```yaml
path: fashion #in reference to the Data structure(yolo format)
train: images/train
val : images/train
test: images/test
nc: 1
names: ['object']
```

**Dataset Structure (YOLO Format)**

```
fashion/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
│
├── labels/
│   ├── train/
│   │   ├── image1.txt      # YOLO format annotations
│   │   ├── image2.txt
│   │   └── ...
│   └── val/
│       ├── image1.txt
│       └── ...
│
├── train.txt               # List of training image paths
└── val.txt                 # List of validation image paths
```

**YOLO Format (.txt files):**
Each `.txt` file contains annotations in the format:
```
<class_id> <x_center> <y_center> <width> <height>
<class_id> <x_center> <y_center> <width> <height>
...
```

Where coordinates are normalized (0-1) relative to image dimensions.

---

## Data Structure for Depth-Aware Evaluation

**For preferential_f1_score.py**

The evaluation expects a hierarchical directory structure with annotated test data:

```
VALIDATED_test_data_annotations/
├── scene_name_1/
│   ├── infer_0/
│   │   ├── image.png                    # Test image
│   │   ├── annotations.json             # Ground truth with "preferential": true/false
│   │   └── depth_pro.npy                # DepthPro depth estimation (optional)
│   ├── infer_1/
│   └── infer_N/
├── scene_name_2/
│   └── infer_*/
└── scene_name_N/
    └── infer_*/
```

**Key files in each `infer_N/` folder:**
- `image.png` or `image.jpg`: Test image
- `annotations.json`: Ground truth annotations with `"preferential": true/false` field marking the closest object
- `depth_pro.npy`: (Optional) Pre-computed depth from DepthPro

**Example annotations.json format:**
```json
{
  "annotations": [
    {
      "id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "bbox": [x, y, width, height],
      "area": 1234,
      "preferential": true
    },
    {
      "id": 2,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "bbox": [x, y, width, height],
      "area": 567,
      "preferential": false
    }
  ]
}
```

---

## Outputs Overview

### Training Output

```
runs/segment/fashion_fastsam/
├── weights/
│   ├── best.pt
│   ├── last.pt
│   └── epoch*.pt
├── results.csv
└── plots/
```

### Validation Output (Standard Metrics)

```
runs/detect/fashion_fastsam/
├── mask_visualizations/
│   ├── TP/   # Correct detections
│   ├── FP/   # False detections
│   ├── FN/   # Missed objects
│   └── TN/   # Correct background (no objects detected)
├── results.json
└── scores.csv
```

### Depth Evaluation Output

```
output_dir/
├── TP/   # Correct depth-aware matches (side-by-side GT vs predictions)
├── FP/   # Wrong detections or depth selection errors
├── FN/   # Missed objects despite detection
└── TN/   # Correct background (no objects detected)
```

---

## Training Strategy


**Phase 1: Fine-Tuning**

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

---


## References

* FastSAM: https://github.com/CASIA-IVA-Lab/FastSAM
* Ultralytics YOLO: https://docs.ultralytics.com/
* DepthPro: https://huggingface.co/apple/DepthPro-hf
* COCO Format: https://cocodataset.org/

---


