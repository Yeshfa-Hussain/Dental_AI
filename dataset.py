"""
dataset.py — Dental AI
Custom PyTorch Dataset for loading X-ray images + PNG masks
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random
import torch


class DentalDataset(Dataset):
    """
    Loads paired (image, mask) for U-Net training.

    Folder structure expected:
    Dental-AI/
    ├── DentAI.v2i.coco-segmentation/
    │   ├── train/  (original X-ray images)
    │   ├── test/
    │   └── valid/
    └── masks/
        ├── train/  (PNG masks from coco_to_masks.py)
        ├── test/
        └── valid/
    """

    def __init__(self, images_dir, masks_dir, split="train", img_size=512):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.split      = split
        self.img_size   = img_size
        self.augment    = (split == "train")   # only augment training data

        # Get all image filenames
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Match each image to its mask
        self.pairs = []
        for img_name in self.image_files:
            base      = os.path.splitext(img_name)[0]
            mask_name = f"{base}_mask.png"
            mask_path = os.path.join(masks_dir, mask_name)

            if os.path.exists(mask_path):
                self.pairs.append((
                    os.path.join(images_dir, img_name),
                    mask_path
                ))

        print(f"  [{split}] Found {len(self.pairs)} image-mask pairs "
              f"(out of {len(self.image_files)} images)")

    def __len__(self):
        return len(self.pairs)

    def augment_pair(self, image, mask):
        """Apply identical random augmentations to both image and mask."""

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.3:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # Random rotation (-20 to +20 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-20, 20)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask, angle)

        # Random brightness/contrast (image only, NOT mask)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))

        return image, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Load image as RGB
        image = Image.open(img_path).convert("RGB")
        # Load mask as grayscale (values 0,1,2,3)
        mask  = Image.open(mask_path).convert("L")

        # Resize to model input size
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask  = mask.resize((self.img_size, self.img_size), Image.NEAREST)  # NEAREST preserves class values

        # Augment training data
        if self.augment:
            image, mask = self.augment_pair(image, mask)

        # Convert image to tensor and normalize
        image = TF.to_tensor(image)                          # [3, H, W], range [0,1]
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])     # ImageNet stats

        # Convert mask to long tensor (class indices)
        mask = torch.from_numpy(np.array(mask)).long()       # [H, W], values 0-3

        return image, mask
