import numpy as np
from PIL import Image
import os

# Point to any one mask file
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
mask_path = os.path.join(BASE_DIR, "masks", "train")

# Pick first mask
mask_files = [f for f in os.listdir(mask_path) if f.endswith(".png")]
first_mask = os.path.join(mask_path, mask_files[0])

mask = np.array(Image.open(first_mask))

print("Mask shape      :", mask.shape)
print("Unique values   :", np.unique(mask))
print("Max pixel value :", mask.max())
print("Non-zero pixels :", np.count_nonzero(mask))