import json

with open(r'DentAI.v2i.coco-segmentation\train\_annotations.coco.json') as f:
    coco = json.load(f)

categories = {cat['id']: cat['name'] for cat in coco['categories']}

for ann in coco['annotations'][:20]:
    seg = ann['segmentation']
    cat = categories[ann['category_id']]
    if isinstance(seg, dict):
        print(f"RLE format → class: {cat}")
    elif isinstance(seg, list):
        print(f"Polygon format → class: {cat}")