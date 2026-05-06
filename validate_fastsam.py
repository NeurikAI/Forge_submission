import argparse
import cv2
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import yaml
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()
def draw_mask(image, mask, color, alpha=0.5):
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask > 0] = color
    mask_indices = mask > 0
    image[mask_indices] = cv2.addWeighted(
        image, 1 - alpha, colored_mask, alpha, 0
    )[mask_indices]
    return image
def main():
    args = get_args()
    model = YOLO(args.model)

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,
    )
    print(
        f"  BOX  | mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}"
    )
    print(
        f"  MASK | mAP50: {metrics.seg.map50:.4f}  mAP50-95: {metrics.seg.map:.4f}"
    )
    with open(args.data) as f:
        cfg = yaml.safe_load(f)
    base_path = Path(cfg.get("path", ""))
    test_rel = cfg.get(args.split, "images/test")
    img_dir = base_path / test_rel
    image_paths = sorted(
        [
            p
            for p in img_dir.glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]
    )

    vis_dir = Path(metrics.save_dir) / "mask_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    palette = np.random.default_rng(42).integers(
        60, 255, (100, 3), dtype=np.uint8
    )
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        results = model.predict(
            str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )
        res = results[0]
        overlay = img_bgr.copy()
        if res.masks is not None:
            h, w = img_bgr.shape[:2]
            masks = res.masks.data.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()

            for i, (m_raw, cls_id) in enumerate(zip(masks, classes)):
                color = palette[cls_id % len(palette)].tolist()
                m_resized = cv2.resize(
                    m_raw, (w, h), interpolation=cv2.INTER_NEAREST
                )
                overlay = draw_mask(overlay, m_resized, color)
                contours, _ = cv2.findContours(
                    m_resized.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(overlay, contours, -1, color, 2)
                M = cv2.moments(m_resized)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"{res.names[cls_id]} {confs[i]:.2f}"
                    cv2.putText(
                        overlay,
                        label,
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
        combined = np.hstack([img_bgr, overlay])
        cv2.imwrite(str(vis_dir / f"{img_path.stem}_masked.jpg"), combined)


if __name__ == "__main__":
    main()