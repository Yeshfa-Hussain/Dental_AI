import numpy as np
from PIL import Image
import os

all_values = set()

for split in ['train', 'test', 'valid']:
    mask_path = rf'C:\Users\Haroon Traders\Documents\GitHub\Dental_AI\DentAI.v2i.coco-segmentation\{split}\masks_multiclass'
    
    split_values = set()
    count = 0
    for fname in os.listdir(mask_path):
        mask = np.array(Image.open(os.path.join(mask_path, fname)))
        unique = np.unique(mask)
        split_values.update(unique.tolist())
        count += 1
    
    all_values.update(split_values)
    print(f"{split}: {count} masks, classes found: {sorted(split_values)}")

print(f"\nAll classes across dataset: {sorted(all_values)}")
print(f"Class mapping: 0=background, 1=caries, 2=infection, 3=restoration")