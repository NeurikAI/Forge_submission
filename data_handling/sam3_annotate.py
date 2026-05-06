#python sam3_annotate.py --img-folder /home/swethas/swetha/finetuning_fastsam/images
#python sam3_annotate.py --img-folder /path/to/images --prompt "clear plastic clothing bag top view"
import argparse
import json
import cv2
import numpy as np
import torch
import os
from pathlib import Path
from datetime import datetime
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import Sam3Processor, Sam3Model
def load_sam3(model_id: str = "facebook/sam3"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM3] Loading {model_id} on {device}...")
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model, device
def get_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0
def get_solidity(mask):
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return 0
    return cv2.contourArea(cnt) / hull_area

def mask_to_bbox(mask):
    pos = np.where(mask)
    if len(pos[0]) == 0: return [0, 0, 0, 0]
    xmin, xmax = np.min(pos[1]), np.max(pos[1])
    ymin, ymax = np.min(pos[0]), np.max(pos[0])
    return [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]

def mask_to_poly(mask):
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) >= 3:
            polys.append(c.flatten().tolist())
    return polys
def annotate_image(img_pil, processor, model, device, text_prompt):
    W, H = img_pil.size
    inputs = processor(img_pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, 
        threshold=0.20, 
        target_sizes=[(H, W)]
    )[0]
    masks = results["masks"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    final_masks = []
    indices = np.argsort(scores)[::-1]
    for i in indices:
        mask = masks[i].astype(bool)
        if np.sum(mask) < 6000: continue
        if get_solidity(mask) < 0.85: continue
        bbox = mask_to_bbox(mask)
        aspect = max(bbox[2]/bbox[3], bbox[3]/bbox[2])
        if aspect > 1.6: continue 
        is_duplicate = False
        for existing in final_masks:
            if get_iou(mask, existing) > 0.30:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_masks.append(mask)    
    return final_masks
def run_pipeline(img_dir, output_json, prompt):
    processor, model, device = load_sam3()
    img_dir = Path(img_dir)
    extensions = [".jpg", ".jpeg", ".png", ".webp"]
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in extensions]
    
    print(f"[INFO] Found {len(images)} images. Starting Annotation...")

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "packaged_article"}]
    }
    
    ann_id = 1
    for img_idx, path in enumerate(tqdm(images)):
        try:
            img_pil = Image.open(path).convert("RGB")
            w, h = img_pil.size
            coco_data["images"].append({
                "id": img_idx,
                "file_name": path.name,
                "height": h,
                "width": w
            })
            masks = annotate_image(img_pil, processor, model, device, prompt)
            for m in masks:
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": 1,
                    "segmentation": mask_to_poly(m),
                    "bbox": mask_to_bbox(m),
                    "area": float(np.sum(m)),
                    "iscrowd": 0
                })
                ann_id += 1
                
        except (UnidentifiedImageError, OSError) as e:
            print(f"\n[SKIP] Corrupt image found: {path.name}")
            continue
        except Exception as e:
            print(f"\n[ERROR] Unexpected error on {path.name}: {e}")
            continue
    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=2)
    print(f"\n[FINISH] Processed {len(coco_data['images'])} images.")
    print(f"[FINISH] Generated {len(coco_data['annotations'])} top-layer annotations.")
    print(f"[FINISH] Saved to: {output_json}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3 auto-annotation pipeline to COCO JSON.")
    parser.add_argument("--img-folder", required=True, help="Path to folder containing input images")
    parser.add_argument("--json-name", default="fashion_coco_annotations_ultimate.json", help="Output JSON filename (default: fashion_coco_annotations_ultimate.json)")
    parser.add_argument("--prompt", default="top-most clear plastic wrapped clothing package, individual rectangular polybag garment, sharp edges, centered", help="Text prompt for SAM3 segmentation")
    args = parser.parse_args()

    run_pipeline(args.img_folder, args.json_name, args.prompt)