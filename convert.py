import os
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

# 🔹 Change this to your full project folder path
# Example: "C:/Users/ALIZA/Dental_Project"
base_dir = r"C:\Users\Haroon Traders\Documents\GitHub\Dental_AI\DentAI.v2i.coco-segmentation"

datasets = ["train", "val", "test"]

print("Starting COCO → PNG mask conversion...")

for d in datasets:
    ann_file = os.path.join(base_dir, d, "_annotations.coco.json")
    output_dir = os.path.join(base_dir, d, "masks")
    
    print(f"\nProcessing dataset: {d}")
    print(f"Looking for annotation file: {ann_file}")
    
    if not os.path.exists(ann_file):
        print(f"❌ Annotation file not found for {d}, skipping...")
        continue

    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Masks folder created: {output_dir}")

    coco = COCO(ann_file)
    
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            mask[coco.annToMask(ann) > 0] = ann["category_id"]

        mask_filename = os.path.splitext(img_info["file_name"])[0] + ".png"
        mask_path = os.path.join(output_dir, mask_filename)

        Image.fromarray(mask).save(mask_path)

    print(f"✅ All masks created for {d}")

print("\n🎉 Conversion complete! Check masks folders inside train/val/test")