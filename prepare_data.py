import os
from pathlib import Path
import cv2
import numpy as np

RAW_DIR = Path("raw_data")
OUT_IMG_DIR = Path("processed/images")
OUT_MASK_DIR = Path("processed/masks")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256

def main():
    count = 0
    print("Scanning raw_data directory for MRI images and corresponding masks...")

    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue

            full_path = os.path.join(root, file)
            path_obj = Path(full_path)

            if "_mask" in file.lower():
                continue

            mask_candidates = [
                path_obj.with_name(path_obj.stem + "_mask" + path_obj.suffix),
                path_obj.with_name(path_obj.stem + "_mask.tif"),
                path_obj.with_name(path_obj.stem + "_mask.png")
            ]

            mask_path = None
            for cand in mask_candidates:
                if cand.exists():
                    mask_path = cand
                    break

            if not mask_path:
                continue

            img = cv2.imread(str(path_obj), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

            img = img.astype(np.float32) / 255.0
            mask = (mask > 0).astype(np.uint8)

            filename = path_obj.stem + ".png"

            cv2.imwrite(str(OUT_IMG_DIR / filename), (img * 255).astype("uint8"))
            cv2.imwrite(str(OUT_MASK_DIR / filename), (mask * 255).astype("uint8"))

            count += 1
            if count % 500 == 0:
                print(f"Processed {count} pairs...")

    print(f"Total pairs processed: {count}")
    print("Data saved in processed/images and processed/masks directories.")

if __name__ == "__main__":
    main()