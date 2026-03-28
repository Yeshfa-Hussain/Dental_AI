"""
COCO Segmentation → PNG Masks Converter
Dental AI Project | Classes: CARIES, INFECTION, RESTORATION

AUTO-DETECTS base directory — no hardcoding needed!
Works on any machine regardless of username or OS.

Expected folder structure:
Dental-AI/
├── DentAI.v2i.coco-segmentation/
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   └── (images)
│   ├── test/
│   │   ├── _annotations.coco.json
│   │   └── (images)
│   └── valid/
│       ├── _annotations.coco.json
│       └── (images)
├── masks/               ← created automatically
├── coco_to_masks.py     ← this file
└── main.py

Pixel value legend:
  0 = Background
  1 = Caries
  2 = Infection
  3 = Restoration
"""

import json
import os
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

# ─────────────────────────────────────────────
#  AUTO-DETECT BASE DIRECTORY
#  Works for both .py script and Jupyter Notebook
# ─────────────────────────────────────────────
try:
    # When running as a .py script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # When running in Jupyter Notebook
    BASE_DIR = os.path.abspath(os.getcwd())

DATA_DIR   = os.path.join(BASE_DIR, "DentAI.v2i.coco-segmentation")
OUTPUT_DIR = os.path.join(BASE_DIR, "masks")

SPLITS          = ["train", "test", "valid"]
ANNOTATION_FILE = "_annotations.coco.json"   # exact filename in your folders

# Category name → mask pixel value
CLASS_MAP = {
    "objects":     0,   # parent/background — ignored
    "caries":      1,
    "infection":   2,
    "restoration": 3,
}

# ─────────────────────────────────────────────
#  CONVERT RLE or POLYGON → binary numpy mask
# ─────────────────────────────────────────────
def annotation_to_mask(ann, height, width):
    seg = ann.get("segmentation", {})

    # Compressed RLE (brush tool → Roboflow exports this)
    if isinstance(seg, dict):
        rle = {
            "counts": seg["counts"],
            "size":   [height, width]
        }
        return coco_mask.decode(rle).astype(np.uint8)

    # Polygon format
    if isinstance(seg, list) and len(seg) > 0:
        rles    = coco_mask.frPyObjects(seg, height, width)
        combined = coco_mask.merge(rles)
        return coco_mask.decode(combined).astype(np.uint8)

    return np.zeros((height, width), dtype=np.uint8)


# ─────────────────────────────────────────────
#  PROCESS ONE SPLIT
# ─────────────────────────────────────────────
def convert_split(split):
    ann_path = os.path.join(DATA_DIR, split, ANNOTATION_FILE)
    out_dir  = os.path.join(OUTPUT_DIR, split)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ann_path):
        print(f"  [SKIP] '{ANNOTATION_FILE}' not found in: {split}/")
        print(f"         Expected path: {ann_path}")
        return

    print(f"\n{'='*60}")
    print(f"  Processing: {split.upper()}")
    print(f"{'='*60}")

    with open(ann_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Build category_id → pixel value
    cat_id_to_pixel = {}
    print("\n  Categories found:")
    for cat in coco_data.get("categories", []):
        name    = cat["name"].lower().strip()
        cat_id  = cat["id"]
        pixel   = CLASS_MAP.get(name, 0)
        cat_id_to_pixel[cat_id] = pixel
        print(f"    ID {cat_id}: '{cat['name']}' → pixel value {pixel}")

    # Build image_id → image info
    img_id_to_info = {img["id"]: img for img in coco_data.get("images", [])}

    # Group annotations by image_id
    anns_by_image = {}
    for ann in coco_data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    total = len(img_id_to_info)
    saved = 0

    print(f"\n  Total images     : {total}")
    print(f"  Total annotations: {len(coco_data.get('annotations', []))}")
    print(f"\n  Converting...\n")

    for idx, (img_id, img_info) in enumerate(img_id_to_info.items()):
        height = img_info["height"]
        width  = img_info["width"]

        # Blank mask — 0 = background
        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in anns_by_image.get(img_id, []):
            cat_id    = ann["category_id"]
            pixel_val = cat_id_to_pixel.get(cat_id, 0)

            if pixel_val == 0:
                continue  # skip background/objects category

            try:
                binary_mask = annotation_to_mask(ann, height, width)
                mask[binary_mask == 1] = pixel_val
            except Exception as e:
                print(f"    [WARN] Annotation ID {ann['id']} failed: {e}")

        # Save as PNG with _mask suffix
        base_name = os.path.splitext(img_info["file_name"])[0]
        mask_path = os.path.join(out_dir, f"{base_name}_mask.png")
        Image.fromarray(mask).save(mask_path)
        saved += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  [{idx+1}/{total}] {base_name}_mask.png")

    print(f"\n  Saved {saved} masks → {out_dir}")


# ─────────────────────────────────────────────
#  VERIFY — sanity check on saved masks
# ─────────────────────────────────────────────
def verify_masks(split):
    out_dir = os.path.join(OUTPUT_DIR, split)
    if not os.path.exists(out_dir):
        return

    masks = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    if not masks:
        print(f"  [!] No masks in {out_dir}")
        return

    CLASS_LABELS = {0: "Background", 1: "Caries", 2: "Infection", 3: "Restoration"}

    print(f"\n  {split.upper()} — {len(masks)} masks saved")
    print(f"  Sample check (first 3):")
    for m in masks[:3]:
        arr         = np.array(Image.open(os.path.join(out_dir, m)))
        unique_vals = np.unique(arr)
        classes     = [CLASS_LABELS.get(int(v), f"unknown({v})") for v in unique_vals]
        print(f"    {m}")
        print(f"      shape={arr.shape} | pixel values={unique_vals} | classes={classes}")


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\nDental AI — COCO → PNG Mask Converter")
    print(f"  Base Dir  : {BASE_DIR}")
    print(f"  Data Dir  : {DATA_DIR}")
    print(f"  Output Dir: {OUTPUT_DIR}")

    # Sanity check — make sure data dir exists
    if not os.path.exists(DATA_DIR):
        print(f"\n[ERROR] Data directory not found:\n  {DATA_DIR}")
        print("  Make sure 'DentAI.v2i.coco-segmentation' folder is inside your project root.")
        exit(1)

    for split in SPLITS:
        convert_split(split)

    print("\n" + "="*60)
    print("  VERIFICATION")
    print("="*60)
    for split in SPLITS:
        verify_masks(split)

    print("\nAll done! Masks are ready for U-Net training.")
    print("\nPixel legend: 0=Background | 1=Caries | 2=Infection | 3=Restoration")
