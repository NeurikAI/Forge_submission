import json, csv, cv2, argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from ultralytics import FastSAM
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = {"TP": (0, 200, 0), "TN": (120, 120, 120), "FP": (0, 200, 200), "FN": (200, 50, 0)}
def get_args():
    parser = argparse.ArgumentParser(description="FastSAM + DepthPro Evaluation")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to images")
    parser.add_argument("--ann_dir", type=str, required=True, help="Path to annotations.json folder")
    parser.add_argument("--weights", type=str, required=True, help="Path to FastSAM weights (.pt)")
    parser.add_argument("--out", type=str, default="eval10_results", help="Output folder name")
    parser.add_argument("--conf", type=float, default=0.30, help="Detection confidence")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for TP")
    parser.add_argument("--imgsz", type=int, default=1024, help="FastSAM input size")
    return parser.parse_args()
def build_gt_mask(anns, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    if not anns: return mask
    targets = [a for a in anns if a.get("preferential", False)] or anns
    for ann in targets:
        for seg in ann.get("segmentation", []):
            pts = np.array(seg).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask
def select_mask_by_depth(sam_results, depth_map, h, w):
    if sam_results[0].masks is None: return np.zeros((h, w), dtype=np.uint8)
    masks = sam_results[0].masks.data.cpu().numpy()
    best_mask, min_depth = np.zeros((h, w), dtype=np.uint8), float('inf')
    for m in masks:
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        if not m_resized.any(): continue
        
        avg_depth = depth_map[m_resized].mean()
        if avg_depth < min_depth:
            min_depth, best_mask = avg_depth, m_resized.astype(np.uint8)  
    return best_mask
def compute_metrics(results):
    stats = {k: sum(1 for r in results if r["label"] == k) for k in ["TP", "TN", "FP", "FN"]}
    tp, tn, fp, fn = stats["TP"], stats["TN"], stats["FP"], stats["FN"]
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    acc  = (tp + tn) / len(results) if results else 0
    
    return {**stats, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}

def save_vis(img_pil, gt, pred, row, out_dir):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    def overlay(base, mask, color, alpha=0.4):
        res = base.copy()
        if mask.any(): res[mask > 0] = (base[mask > 0] * (1-alpha) + np.array(color) * alpha).astype(np.uint8)
        return res
    left = overlay(img, gt, (0, 255, 0))
    right = overlay(img, pred, (255, 0, 0))
    cv2.putText(left, "GT (Pref)", (10, 30), 0, 0.8, (0, 255, 0), 2)
    cv2.putText(right, f"Pred: {row['label']} IoU: {row['iou']:.2f}", (10, 30), 0, 0.8, COLORS[row['label']], 2)
    combined = np.hstack([left, right])
    p = out_dir / row['label']
    p.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p / f"{row['file_name']}"), combined)

def main():
    args = get_args()
    img_dir, ann_dir = Path(args.img_dir), Path(args.ann_dir)
    out_dir = ann_dir / args.out
    out_dir.mkdir(exist_ok=True)
    print("--> Loading Models...")
    sam = FastSAM(args.weights)
    d_proc = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    d_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(DEVICE).eval()
    with open(ann_dir / "annotations.json") as f:
        data = json.load(f)
    img_map = {i["id"]: i["file_name"] for i in data["images"]}
    ann_map = {i["id"]: [] for i in data["images"]}
    for a in data["annotations"]: ann_map[a["image_id"]].append(a)
    results = []
    print(f"Processing {len(img_map)} images...")
    for img_id, fname in img_map.items():
        path = img_dir / fname
        if not path.exists(): continue
        img_pil = Image.open(path).convert("RGB")
        w, h = img_pil.size
        inputs = d_proc(images=img_pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = d_model(**inputs)
            d_map = d_proc.post_process_depth_estimation(output, target_sizes=[(h, w)])[0]["predicted_depth"].cpu().numpy()
        sam_res = sam(str(path), device=DEVICE, retina_masks=True, conf=args.conf, imgsz=args.imgsz, verbose=False)
        gt_mask = build_gt_mask(ann_map[img_id], (h, w))
        pred_mask = select_mask_by_depth(sam_res, d_map, h, w)
        inter = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        iou = float(inter / union) if union > 0 else (1.0 if not gt_mask.any() and not pred_mask.any() else 0.0)
        gt_any, pred_any = gt_mask.any(), pred_mask.any()
        if not gt_any and not pred_any: label = "TN"
        elif not gt_any and pred_any:  label = "FP"
        elif gt_any and not pred_any:  label = "FN"
        else: label = "TP" if iou >= args.iou_thresh else "FN"
        row = {"file_name": fname, "label": label, "iou": iou, "gt_px": int(gt_mask.sum()), "pred_px": int(pred_mask.sum())}
        results.append(row)
        save_vis(img_pil, gt_mask, pred_mask, row, out_dir / "visuals")
        print(f"[{len(results)}] {fname} -> {label} (IoU: {iou:.3f})")
    m = compute_metrics(results)
    print("\n" + "="*30 + "\nRESULTS\n" + "="*30)
    for k, v in m.items(): print(f"{k:<12}: {v:.4f}" if isinstance(v, float) else f"{k:<12}: {v}")
    with open(out_dir / "scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main()