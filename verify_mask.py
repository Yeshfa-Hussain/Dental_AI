import numpy as np
from PIL import Image
import os

mask_path = r'C:\Users\Haroon Traders\Documents\GitHub\Dental_AI\DentAI.v2i.coco-segmentation\test\masks_multiclass'

for fname in os.listdir(mask_path)[:5]:  # check first 5 masks
    mask = np.array(Image.open(os.path.join(mask_path, fname)))
    unique = np.unique(mask)
    print(f"{fname[:40]} → unique values: {unique}")