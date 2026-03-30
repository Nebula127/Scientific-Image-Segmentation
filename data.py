import os
import glob
import random

IMAGE_DIR = "processed/images"
MASK_DIR = "processed/masks"

def split_by_filename(val_ratio=0.2):
    img_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(MASK_DIR, "*.png")))

    assert len(img_paths) == len(mask_paths) and len(img_paths) > 0, "No data found! Run prepare_data.py first."

    data = list(zip(img_paths, mask_paths))
    random.seed(42)
    random.shuffle(data)

    split_idx = int(len(data) * (1 - val_ratio))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_img, train_mask = zip(*train_data)
    val_img, val_mask = zip(*val_data)

    return list(train_img), list(train_mask), list(val_img), list(val_mask)