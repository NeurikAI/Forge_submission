import json
import logging
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import FastSAM
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def run_depth(image, processor, model):
    """Run DepthPro, return raw depth numpy array (no files written)."""
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    post    = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )
    depth_t  = post[0]["predicted_depth"]
    depth_np = depth_t.detach().cpu().numpy()
    if depth_np.ndim == 3:
        depth_np = depth_np.squeeze(0)
    return depth_np

def find_source_image(infer_dir):
    return [
        p for p in infer_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
        and "mask"     not in p.name
        and "depth"    not in p.name
        and "verified" not in p.name
        and "preview"  not in p.name
    ]
def run_fastsam_masks(image_path, model, conf, imgsz):
    results = model(str(image_path), device=DEVICE, retina_masks=True,
                    conf=conf, imgsz=imgsz, verbose=False)
    out = []
    if results[0].masks is None:
        return out
    image = Image.open(image_path).convert("RGB")
    h, w  = image.height, image.width
    for m, b in zip(
        results[0].masks.data.cpu().numpy(),
        results[0].boxes.xyxy.cpu().numpy()
    ):
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append(((m_resized > 0).astype(np.uint8), b.tolist()))
    return out

def pick_preferential_by_depth(masks_boxes, depth):
    mean_depths = []
    for mask, _ in masks_boxes:
        mask_f  = mask.astype(np.float32)
        if mask_f.shape != depth.shape:
            mp = Image.fromarray((mask_f * 255).astype("uint8"), mode="L")
            mp = mp.resize((depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST)
            mask_f = (np.array(mp) > 0).astype(np.float32)
        px = mask_f.sum()
        mean_depths.append(
            float((depth * mask_f).sum() / px) if px > 0 else float("inf")
        )
    return int(np.argmin(mean_depths)), mean_depths

def seg_to_mask(segmentation, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def draw_verification(image_path, annotations, flag_lines):
  
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    for ann in annotations:
        is_pref    = ann.get("preferential", False)
        seg_color  = (0, 255, 0)   if is_pref else (255, 165, 0)
        bbox_color = (0, 0, 255)   if is_pref else (200, 100, 0)

        for seg in ann["segmentation"]:
            pts = np.array(seg).reshape(-1, 2).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=seg_color, thickness=2)

        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)

        label = f"ID {ann['id']} | {'preferential' if is_pref else 'non-pref'}"
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        (tw, text_h), baseline = cv2.getTextSize(label, font, fs, th)
        ly = y1 - 6 if y1 - 6 > text_h else y1 + text_h + 6
        cv2.rectangle(image, (x1, ly - text_h - baseline),
                      (x1 + tw, ly + baseline), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, label, (x1, ly), font, fs,
                    (255, 255, 255), th, cv2.LINE_AA)
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    pad, gap = 8, 6
    texts  = [t for t, _ in flag_lines]
    max_w  = max(cv2.getTextSize(t, font, fs, th)[0][0] for t in texts)
    line_h = cv2.getTextSize(texts[0], font, fs, th)[0][1]
    bh     = (line_h + gap) * len(texts) + pad * 2
    ih, iw = image.shape[:2]
    bx1, by1 = iw - max_w - pad * 2 - 10, 10
    bx2, by2 = iw - 10, 10 + bh
    cv2.rectangle(image, (bx1, by1), (bx2, by2), (30, 30, 30),    cv2.FILLED)
    cv2.rectangle(image, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
    for i, (text, color) in enumerate(flag_lines):
        ty = by1 + pad + (line_h + gap) * (i + 1) - gap
        cv2.putText(image, text, (bx1 + pad, ty), font, fs,
                    color, th, cv2.LINE_AA)

    return image


def mask_to_polygons(mask_np):
    mask_uint8 = (mask_np > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return [c.flatten().tolist() for c in contours if c.shape[0] >= 3]


def save_side_by_side(image_path, gt_anns, pred_anns,
                      result_label, best_iou_score, out_path):
    gt_flag_lines = [
        ("GROUND TRUTH",          (100, 255, 100)),
        (f"n_masks: {len(gt_anns)}", (200, 200, 200)),
    ]
    left = draw_verification(image_path, gt_anns, gt_flag_lines)
    result_color = (100, 255, 100) if result_label == "TP" else (100, 100, 255)
    pred_flag_lines = [
        ("FastSAM-s + DepthPro",               (100, 100, 255)),
        (f"result:   {result_label}",           result_color),
        (f"best_iou: {best_iou_score:.3f}",     (200, 200, 200)),
        (f"n_masks:  {len(pred_anns)}",         (200, 200, 200)),
    ]
    right = draw_verification(image_path, pred_anns, pred_flag_lines)
    if left is None or right is None:
        logger.warning("Could not render panels for %s", image_path)
        return
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    max_h  = max(h1, h2)
    if h1 < max_h:
        left  = cv2.copyMakeBorder(left,  0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT)
    if h2 < max_h:
        right = cv2.copyMakeBorder(right, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT)

    combined = np.hstack([left, right])
    cv2.imwrite(str(out_path), combined)
    logger.info("  └─ saved %s", out_path)

def eval_infer_dir(infer_dir, fastsam, depth_processor, depth_model,
                   conf, imgsz, iou_threshold, output_dir):
    ann_path = infer_dir / "annotations.json"
    if not ann_path.exists():
        logger.warning("  no annotations.json in %s", infer_dir)
        return None

    with open(ann_path) as f:
        gt_data = json.load(f)

    gt_anns = gt_data.get("annotations", [])
    if not gt_anns:
        logger.info("  └─ empty annotations, skipping")
        return None

    image_paths = find_source_image(infer_dir)
    if not image_paths:
        logger.warning("  └─ no source image in %s", infer_dir)
        return None
    image_path = image_paths[0]
    image = Image.open(image_path).convert("RGB")
    h, w  = image.height, image.width
    gt_pref_masks = [
        seg_to_mask(a["segmentation"], h, w)
        for a in gt_anns if a.get("preferential", False)
    ]
    if not gt_pref_masks:
        logger.warning("  └─ no preferential=True in GT, skipping")
        return None
    masks_boxes = run_fastsam_masks(image_path, fastsam, conf, imgsz)
    if not masks_boxes:
        logger.info("  └─ FastSAM: no masks")
        return (True, False)
    depth = run_depth(image, depth_processor, depth_model)
    pred_pref_idx, mean_depths = pick_preferential_by_depth(masks_boxes, depth)
    pred_pref_mask = masks_boxes[pred_pref_idx][0]
    best_iou_score = max(iou(gt_mask, pred_pref_mask) for gt_mask in gt_pref_masks)
    matched         = best_iou_score >= iou_threshold
    result_label    = "TP" if matched else "FN"

    logger.info("  └─ depth-chosen pred_idx=%d  IoU=%.3f  → %s",
                pred_pref_idx, best_iou_score, result_label)
    pred_anns = []
    for i, (mask, bbox) in enumerate(masks_boxes):
        x1, y1, x2, y2 = bbox
        pred_anns.append({
            "id":           i + 1,
            "image_id":     1,
            "segmentation": mask_to_polygons(mask),
            "bbox":         [x1, y1, x2 - x1, y2 - y1],
            "area":         float(mask.sum()),
            "preferential": (i == pred_pref_idx),
            "mask_path":    "",
        })
    scene     = infer_dir.parent.name
    infer     = infer_dir.name
    out_subdir = output_dir / result_label
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path  = out_subdir / f"{scene}__{infer}__verification.png"
    save_side_by_side(
        image_path     = image_path,
        gt_anns        = gt_anns,
        pred_anns      = pred_anns,
        result_label   = result_label,
        best_iou_score = best_iou_score,
        out_path       = out_path,
    )

    return (True, matched)
def compute_metrics(results):
    tp = sum(1 for g, p in results if g and p)
    fp = sum(1 for g, p in results if not g and p)
    fn = sum(1 for g, p in results if g and not p)
    tn = sum(1 for g, p in results if not g and not p)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1)

def run_eval(root_dir, output_dir, weights, depth_model_id,
             conf, imgsz, iou_threshold):
    root_dir   = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    infer_dirs = sorted(p for p in root_dir.glob("*/infer_*") if p.is_dir())
    if not infer_dirs:
        logger.error("No infer_*/ folders found under %s", root_dir)
        return

    logger.info("Found %d infer_*/ folders", len(infer_dirs))

    logger.info("Loading FastSAM-s : %s", weights)
    fastsam = FastSAM(weights)

    logger.info("Loading DepthPro  : %s", depth_model_id)
    depth_processor = DepthProImageProcessorFast.from_pretrained(depth_model_id)
    depth_model     = DepthProForDepthEstimation.from_pretrained(
        depth_model_id).to(DEVICE).eval()

    all_results = []
    per_folder  = []

    for idx, infer_dir in enumerate(infer_dirs, 1):
        logger.info("[%d/%d] %s", idx, len(infer_dirs), infer_dir)
        try:
            result = eval_infer_dir(
                infer_dir       = infer_dir,
                fastsam         = fastsam,
                depth_processor = depth_processor,
                depth_model     = depth_model,
                conf            = conf,
                imgsz           = imgsz,
                iou_threshold   = iou_threshold,
                output_dir      = output_dir,
            )
        except Exception as e:
            logger.error("  └─ ERROR: %s", e, exc_info=True)
            continue

        if result is None:
            continue

        all_results.append(result)
        per_folder.append((str(infer_dir.relative_to(root_dir)), result))

    if not all_results:
        logger.error("No valid results collected.")
        return

    m = compute_metrics(all_results)

    print("\n" + "─" * 72)
    print(f"  {'FOLDER':<55}  RESULT")
    print("─" * 72)
    for folder, (g, p) in per_folder:
        tag = "✓ TP" if (g and p) else "✗ FN"
        print(f"  {folder:<55}  {tag}")

    print("\n" + "=" * 72)
    print(f"  GLOBAL  ({len(all_results)} folders evaluated)")
    print("=" * 72)
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print("=" * 72)
    print(f"\n  Verification images → {output_dir}/TP/  and  {output_dir}/FN/")


def parse_args():
    p = argparse.ArgumentParser(
        description="FastSAM-s + DepthPro preferential F1 — read-only eval, "
                    "side-by-side GT vs pred saved to --output-dir"
    )
    p.add_argument("--root-dir",       required=True,
                   help="Root dir containing <scene>/infer_N/ subfolders")
    p.add_argument("--output-dir",     required=True,
                   help="Where to write TP/ and FN/ verification images")
    p.add_argument("--weights",        default="FastSAM-s.pt")
    p.add_argument("--depth-model-id", default="apple/DepthPro-hf")
    p.add_argument("--conf",           type=float, default=0.30)
    p.add_argument("--imgsz",          type=int,   default=1024)
    p.add_argument("--iou-threshold",  type=float, default=0.50)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run_eval(
        root_dir      = args.root_dir,
        output_dir    = args.output_dir,
        weights       = args.weights,
        depth_model_id= args.depth_model_id,
        conf          = args.conf,
        imgsz         = args.imgsz,
        iou_threshold = args.iou_threshold,
    )