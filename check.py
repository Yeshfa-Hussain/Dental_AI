import json
import numpy as np
from PIL import Image, ImageDraw
import os

# Class mapping — match your Roboflow class names exactly
CLASS_MAP = {
    "caries": 1,
    "infection": 2,
    "restoration": 3
}

def coco_to_multiclass_masks(json_path, output_dir):
    with open(json_path) as f:
        coco = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Build image id -> info map
    images = {img['id']: img for img in coco['images']}
    categories = {cat['id']: cat['name'] for cat in coco['categories']}

    # Group annotations by image
    from collections import defaultdict
    ann_by_image = defaultdict(list)
    for ann in coco['annotations']:
        ann_by_image[ann['image_id']].append(ann)

    for img_id, img_info in images.items():
        h, w = img_info['height'], img_info['width']
        mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
        draw = ImageDraw.Draw(mask)

        for ann in ann_by_image[img_id]:
            cat_name = categories[ann['category_id']].lower()
            class_val = CLASS_MAP.get(cat_name, 0)

        segmentation = ann['segmentation']

        # Handle RLE format (dict) vs polygon format (list)
        if isinstance(segmentation, dict):
            # RLE format — used by brush tool (caries & infection)
            import pycocotools.mask as mask_util
            rle_mask = mask_util.decode(segmentation)  # returns H x W numpy array
            current = np.array(mask)
            current[rle_mask == 1] = class_val
            mask = Image.fromarray(current.astype(np.uint8))
            draw = ImageDraw.Draw(mask)  # refresh draw object
        else:
            # Polygon format — used by polygon tool (restoration)
            for seg in segmentation:
                try:
                    coords = [float(x) for x in seg if isinstance(x, (int, float))]
                    if len(coords) >= 6:
                        poly = [(int(coords[i]), int(coords[i+1])) for i in range(0, len(coords), 2)]
                        draw.polygon(poly, fill=class_val)
                except Exception as e:
                    print(f"Skipping annotation due to error: {e}")

        # Save mask with same name as image
        mask_name = os.path.splitext(img_info['file_name'])[0] + '.png'
        mask.save(os.path.join(output_dir, mask_name))
        print(f"Saved: {mask_name}")

# Run for each split
for split in ['train', 'test', 'valid']:
    coco_to_multiclass_masks(
        json_path=f'DentAI.v2i.coco-segmentation/{split}/_annotations.coco.json',
        output_dir=f'DentAI.v2i.coco-segmentation/{split}/masks_multiclass'
    )