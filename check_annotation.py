import json
from collections import Counter

for split in ['train', 'test', 'valid']:
    path = rf'C:\Users\Haroon Traders\Documents\GitHub\Dental_AI\DentAI.v2i.coco-segmentation\{split}\_annotations.coco.json'
    with open(path) as f:
        coco = json.load(f)
    
    cat_ids_used = Counter([ann['category_id'] for ann in coco['annotations']])
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    
    print(f"\n{split} annotations:")
    for cat_id, count in sorted(cat_ids_used.items()):
        print(f"  category_id={cat_id} ({categories.get(cat_id, 'unknown')}) → {count} annotations")