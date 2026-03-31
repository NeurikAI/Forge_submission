'''python coco_to_yolo.py \
    --coco-json  /path/to/fashion_coco_annotations_ultimate.json \
    --output-lbls /path/to/labels'''
    
import argparse
import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Convert COCO JSON segmentation annotations to YOLO label files.")
    parser.add_argument("--coco-json", required=True, help="Path to COCO annotations JSON file")
    parser.add_argument("--output-lbls", required=True, help="Path to output folder for YOLO label files")
    args = parser.parse_args()

    os.makedirs(args.output_lbls, exist_ok=True)

    with open(args.coco_json) as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}
    ann_map = defaultdict(list)
    for ann in coco["annotations"]:
        ann_map[ann["image_id"]].append(ann)

    converted = 0
    for img_id, anns in ann_map.items():
        img_info = img_map[img_id]
        W, H = img_info["width"], img_info["height"]
        fname = Path(img_info["file_name"]).stem
        lines = []
        for ann in anns:
            for seg in ann["segmentation"]:
                if len(seg) < 6:
                    continue
                pts = np.array(seg).reshape(-1, 2).astype(float)
                pts[:, 0] /= W
                pts[:, 1] /= H
                pts = np.clip(pts, 0.0, 1.0)
                coords = " ".join([f"{c:.6f}" for c in pts.flatten()])
                lines.append(f"0 {coords}")
        if lines:
            with open(os.path.join(args.output_lbls, fname + ".txt"), "w") as f:
                f.write("\n".join(lines))
            converted += 1

    print(f"Converted {converted} images to YOLO labels")
    print(f"Saved to: {args.output_lbls}")


if __name__ == "__main__":
    main()