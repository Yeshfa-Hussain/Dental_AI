import numpy as np
from PIL import Image
import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
mask_dir      = os.path.join(BASE_DIR, "masks", "train")
colorized_dir = os.path.join(BASE_DIR, "masks_colorized", "train")
os.makedirs(colorized_dir, exist_ok=True)

# Color map: Background=black, Caries=red, Infection=yellow, Restoration=green
COLOR_MAP = {
    0: (0,   0,   0),    # Background  → black
    1: (255, 0,   0),    # Caries      → red
    2: (255, 255, 0),    # Infection   → yellow
    3: (0,   255, 0),    # Restoration → green
}

mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

for fname in mask_files[:20]:   # colorize first 20 to check
    mask = np.array(Image.open(os.path.join(mask_dir, fname)))
    color_img = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for pixel_val, color in COLOR_MAP.items():
        color_img[mask == pixel_val] = color

    Image.fromarray(color_img).save(os.path.join(colorized_dir, fname))

print(f"Saved colorized masks to: {colorized_dir}")
print("Open them to visually verify your annotations!")