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

## Quick Start

### 1. Train the Model

```bash
python train.py
```

Default settings:

* Epochs: 100
* Image size: 640
* Batch size: 16
* Learning rate: 1e-3
* Logging: Weights & Biases (W&B)

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
| **TN** | Correct rejection of background |
| **FP** | Incorrect detection (false alarm) |
| **FN** | Missed fashion item |

---

## Dataset Configuration

**For Training & Validation (YOLO Format)**

Edit `fashion.yaml`:

```yaml
path: /path/to/fashion
train: train.txt
val: val.txt

nc: 1
names: ['fashion_item']
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
├── results.json
└── scores.csv
```

### Depth Evaluation Output

```
output_dir/
├── TP/   # Correct depth-aware matches (side-by-side GT vs predictions)
├── FP/   # Wrong detections or depth selection errors
├── FN/   # Missed objects despite detection
```

---


## References

* FastSAM: https://github.com/CASIA-IVA-Lab/FastSAM
* Ultralytics YOLO: https://docs.ultralytics.com/
* DepthPro: https://huggingface.co/apple/DepthPro-hf
* COCO Format: https://cocodataset.org/

---


