'''python prepare_phase2.py \
    --p1-base  /home/swethas/swetha/imaterialistic \
    --new-base /home/swethas/swetha/phase2 \
    --new-img  /home/swethas/swetha/finetuning_fastsam/images \
    --new-lbl  /home/swethas/swetha/finetuning_fastsam/labels'''
import argparse
import os
import random
import shutil
from pathlib import Path
def main():
    parser = argparse.ArgumentParser(description="Merge phase 1 and phase 2 datasets and write combined train/val splits.")
    parser.add_argument("--p1-base",  required=True, help="Phase 1 base directory (iMaterialist)")
    parser.add_argument("--new-base", required=True, help="Phase 2 base directory for outputs")
    parser.add_argument("--new-img",  required=True, help="Path to new (phase 2) images folder")
    parser.add_argument("--new-lbl",  required=True, help="Path to new (phase 2) labels folder")
    parser.add_argument("--best-pt",  default=None,  help="Path to phase 1 best.pt (default: <p1-base>/runs/segment/fashion_fastsam/fashion_finetune15/weights/best.pt)")
    parser.add_argument("--split",    type=float, default=0.9, help="Train split ratio for new data (default: 0.9)")
    parser.add_argument("--seed",     type=int,   default=42,  help="Random seed (default: 42)")
    args = parser.parse_args()

    os.makedirs(f"{args.new_base}/datasets", exist_ok=True)
    os.makedirs(f"{args.new_base}/weights",  exist_ok=True)
    os.makedirs(f"{args.new_base}/results",  exist_ok=True)
    src = args.best_pt or f"{args.p1_base}/runs/segment/fashion_fastsam/fashion_finetune15/weights/best.pt"
    dst = f"{args.new_base}/weights/best_phase1.pt"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied best.pt -> {dst}")
    else:
        print(f"best.pt not found at {src} -- check path")
    p1_train = open(f"{args.p1_base}/datasets/fashion/train.txt").read().splitlines()
    p1_val   = open(f"{args.p1_base}/datasets/fashion/val.txt").read().splitlines()
    new_ids = [f.stem for f in Path(args.new_lbl).glob("*.txt")]
    print(f"\nNew labeled images found: {len(new_ids)}")
    random.seed(args.seed)
    random.shuffle(new_ids)
    split = int(len(new_ids) * args.split)
    def resolve_img(stem):
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            p = os.path.join(args.new_img, stem + ext)
            if os.path.exists(p):
                return p
        return None
    new_train = [p for p in [resolve_img(s) for s in new_ids[:split]] if p]
    new_val   = [p for p in [resolve_img(s) for s in new_ids[split:]] if p]
    all_train = p1_train + new_train
    all_val   = p1_val   + new_val
    with open(f"{args.new_base}/datasets/train.txt", "w") as f:
        f.write("\n".join(all_train))
    with open(f"{args.new_base}/datasets/val.txt", "w") as f:
        f.write("\n".join(all_val))
    print(f"\ntrain.txt: {len(all_train)}  ({len(p1_train)} iMaterialist + {len(new_train)} new)")
    print(f"val.txt  : {len(all_val)}  ({len(p1_val)} iMaterialist + {len(new_val)} new)")
    sample_p1 = p1_train[0]
    lbl_p1 = sample_p1.replace("/images/", "/labels/").replace(".jpg", ".txt").replace(".png", ".txt")
    print(f"\n[iMaterialist] sample : {sample_p1}")
    print(f"[iMaterialist] label  : {lbl_p1}")
    print(f"[iMaterialist] exists : {os.path.exists(lbl_p1)}")
    if new_train:
        sample_new = new_train[0]
        lbl_new = sample_new.replace("/images/", "/labels/").replace(".png", ".txt").replace(".jpg", ".txt")
        print(f"\n[New data] sample : {sample_new}")
        print(f"[New data] label  : {lbl_new}")
        print(f"[New data] exists : {os.path.exists(lbl_new)}")
if __name__ == "__main__":
    main()