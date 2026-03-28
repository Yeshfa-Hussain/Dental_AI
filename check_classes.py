import json

for split in ['train', 'test', 'valid']:
    path = rf'C:\Users\Haroon Traders\Documents\GitHub\Dental_AI\DentAI.v2i.coco-segmentation\{split}\_annotations.coco.json'
    with open(path) as f:
        coco = json.load(f)
    print(f"\n{split} categories:")
    for cat in coco['categories']:
        print(f"  id={cat['id']}, name='{cat['name']}'")