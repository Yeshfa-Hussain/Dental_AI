"""
dataset.py — Custom PyTorch Dataset for Dental AI U-Net
Loads X-ray images + corresponding PNG masks
No augmentation — dataset already augmented by Roboflow
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch


class DentalDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        """
        Args:
            image_dir : path to folder containing X-ray images
            mask_dir  : path to folder containing PNG masks
        """
        self.image_dir = image_dir
        self.mask_dir  = mask_dir

        # Get all image files
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Match masks to images
        self.pairs = []
        for img_file in self.images:
            base      = os.path.splitext(img_file)[0]
            mask_file = f"{base}_mask.png"
            mask_path = os.path.join(mask_dir, mask_file)
            if os.path.exists(mask_path):
                self.pairs.append((img_file, mask_file))
            else:
                print(f"  [WARN] No mask found for: {img_file} — skipping")

        print(f"  Dataset loaded: {len(self.pairs)} image-mask pairs from {image_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_file, mask_file = self.pairs[idx]

        # Load image as grayscale (X-rays are grayscale)
        image = Image.open(os.path.join(self.image_dir, img_file)).convert("L")

        # Load mask (keep as-is — pixel values 0,1,2,3)
        mask  = Image.open(os.path.join(self.mask_dir, mask_file))

        # Resize both to 512x512
        image = image.resize((512, 512), Image.BILINEAR)
        mask  = mask.resize((512, 512),  Image.NEAREST)   # NEAREST preserves class values

        # Convert to tensors
        image = TF.to_tensor(image)                        # shape: (1, 512, 512), range [0,1]
        mask  = torch.from_numpy(np.array(mask)).long()    # shape: (512, 512), values 0-3

        return image, mask


# ─────────────────────────────────────────────
#  QUICK TEST — run this file directly to verify
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os

    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.path.abspath(os.getcwd())

    DATA_ROOT = os.path.join(BASE_DIR, "DentAI.v2i.coco-segmentation")
    MASK_ROOT = os.path.join(BASE_DIR, "masks")

    print("Testing DentalDataset...\n")

    dataset = DentalDataset(
        image_dir = os.path.join(DATA_ROOT, "train"),
        mask_dir  = os.path.join(MASK_ROOT, "train"),
    )

    img, mask = dataset[0]

    print(f"\n  Total pairs    : {len(dataset)}")
    print(f"  Image shape    : {img.shape}")       # should be (1, 512, 512)
    print(f"  Image range    : [{img.min():.2f}, {img.max():.2f}]")
    print(f"  Mask shape     : {mask.shape}")      # should be (512, 512)
    print(f"  Mask unique    : {mask.unique()}")   # should contain values from [0,1,2,3]
    print(f"\n  Dataset looks correct!")
