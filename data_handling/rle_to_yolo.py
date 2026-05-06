#python rle_to_yolo.py --base-dir /home/swethas/swetha/imaterialistic
import argparse
import os
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0])).T
def process_image(args, image_dir, output_dir):
    image_id, rows = args
    img_path = os.path.join(image_dir, image_id)
    if not os.path.exists(img_path):
        return 'skipped'
    img = cv2.imread(img_path)
    if img is None:
        return 'skipped'
    h, w = img.shape[:2]
    del img
    yolo_lines = []
    for encoded_pixels in rows:
        mask = rle_decode(encoded_pixels, (h, w))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 5:
                continue
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 4:
                continue
            poly = approx.reshape(-1, 2).astype(float)
            poly[:, 0] /= w
            poly[:, 1] /= h
            poly = np.clip(poly, 0.0, 1.0)

            line = "0 " + " ".join([f"{c:.6f}" for c in poly.flatten()])
            yolo_lines.append(line)
    if yolo_lines:
        label_filename = image_id.replace('.jpg', '.txt')
        with open(os.path.join(output_dir, label_filename), 'w') as f:
            f.write("\n".join(yolo_lines))
        return 'found'
    return 'empty'


def main():
    parser = argparse.ArgumentParser(description="Convert RLE masks to YOLO segmentation labels.")
    parser.add_argument("--base-dir", required=True, help="Root directory of the dataset")
    parser.add_argument("--csv-file", default=None, help="Path to train.csv (default: <base-dir>/train.csv)")
    parser.add_argument("--image-dir", default=None, help="Path to image folder (default: <base-dir>/train)")
    parser.add_argument("--output-dir", default=None, help="Path to output label folder (default: <base-dir>/datasets/fashion/labels/train)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    args = parser.parse_args()
    csv_file   = args.csv_file   or os.path.join(args.base_dir, "train.csv")
    image_dir  = args.image_dir  or os.path.join(args.base_dir, "train")
    output_dir = args.output_dir or os.path.join(args.base_dir, "datasets/fashion/labels/train")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['EncodedPixels'])
    image_groups = df.groupby('ImageId')['EncodedPixels'].apply(list)
    tasks = list(image_groups.items())
    n_workers = max(1, args.workers)
    worker_fn = lambda a: process_image(a, image_dir, output_dir)
    results = {'found': 0, 'skipped': 0, 'empty': 0}
    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(worker_fn, tasks, chunksize=32),
            total=len(tasks)
        ):
            results[result] += 1
    print(f"Labels written : {results['found']}")
    print(f"No mask found  : {results['empty']}")
    print(f"Image missing  : {results['skipped']}")


if __name__ == "__main__":
    main()