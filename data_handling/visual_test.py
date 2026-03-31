'''
python visual_verify.py \
    --coco-json /path/to/fashion_coco_annotations_ultimate.json \
    --img-dir   /path/to/images \
    --out-dir   /path/to/visual_verify_ultimate
'''
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Visually verify COCO annotations on sampled images.")
    parser.add_argument("--coco-json", required=True, help="Path to COCO annotations JSON file")
    parser.add_argument("--img-dir", required=True, help="Path to folder containing input images")
    parser.add_argument("--out-dir", required=True, help="Path to output folder for verification images")
    parser.add_argument("--samples", type=int, default=400, help="Number of images to sample (default: 400)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)

    with open(args.coco_json) as f:
        coco = json.load(f)
    img_map = {img["id"]: img for img in coco["images"]}

    ann_map = defaultdict(list)
    for ann in coco["annotations"]:
        ann_map[ann["image_id"]].append(ann)

    annotated_ids = list(ann_map.keys())
    random.seed(args.seed)
    samples = random.sample(annotated_ids, min(args.samples, len(annotated_ids)))
    colors = [
        (255,  80,  80), (80, 255,  80), ( 80, 80, 255),
        (255, 255,  80), (255,  80, 255), (80, 255, 255),
        (255, 160,  80), (160, 255,  80), (80, 160, 255),
        (255,  80, 160),
    ]
    for img_id in samples:
        info = img_map[img_id]
        img_path = Path(args.img_dir) / info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        overlay = img.copy()

        for i, ann in enumerate(ann_map[img_id]):
            color = colors[i % len(colors)]
            for seg in ann["segmentation"]:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
            x, y, bw, bh = [int(v) for v in ann["bbox"]]
            cv2.rectangle(img, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(img, f"#{i+1} area={int(ann['area'])}",
                        (x, max(y-8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        blended = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        n_ann = len(ann_map[img_id])
        cv2.putText(blended, f"{info['file_name']} - {n_ann} annotations",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        out_path = Path(args.out_dir) / f"verify_{info['file_name']}"
        cv2.imwrite(str(out_path), blended)
        print(f"Saved: {out_path.name}  ({n_ann} annotations)")

    print(f"\nAll verification images saved to: {args.out_dir}")


if __name__ == "__main__":
    main()