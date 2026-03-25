import json
import os

BASE_DIR = r"C:\Users\YESHFA-HUSSAIN\Documents\GitHub\Dental_AI"  # <-- your actual path
DATA_DIR = os.path.join(BASE_DIR, "DentAI.v2i.coco-segmentation")
ann_path = os.path.join(DATA_DIR, "train", "_annotationS.coco.json")

# Check file exists
print("JSON file exists:", os.path.exists(ann_path))
print("JSON file size (bytes):", os.path.getsize(ann_path))

with open(ann_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

print("Keys in JSON:", list(coco_data.keys()))
print("Total images:", len(coco_data.get("images", [])))
print("Total annotations:", len(coco_data.get("annotations", [])))
print("Categories:", coco_data.get("categories", []))

if coco_data.get("images"):
    print("\nFirst image entry:", coco_data["images"][0])

if coco_data.get("annotations"):
    ann = coco_data["annotations"][0]
    print("\nFirst annotation keys:", list(ann.keys()))
    print("Segmentation type:", type(ann["segmentation"]))
    print("Segmentation value:", ann["segmentation"])