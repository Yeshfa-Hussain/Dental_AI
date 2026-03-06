import os
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

def coco_to_png_masks(annotation_file, image_dir, output_mask_dir):
    """
    Converts COCO instance segmentation annotations to PNG mask images.
    
    Args:
        annotation_file (str): Path to the COCO JSON annotation file.
        image_dir (str): Directory containing the original images.
        output_mask_dir (str): Directory to save the output PNG masks.
    """
    # Initialize COCO API
    coco = COCO(annotation_file)

    # Create output directory if it doesn't exist
    os.makedirs(output_mask_dir, exist_ok=True)

    # Get all image IDs
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        # Load image info
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']

        # Get all annotation IDs for the current image
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        # Load annotations
        anns = coco.loadAnns(ann_ids)

        # Create a blank mask image (numpy array)
        # For class segmentation, each pixel value corresponds to the class ID.
        # For binary masks, you might use 0/1, but here we use category_id
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # Iterate over annotations and draw masks
        for ann in anns:
            # Generate the binary mask for the current annotation
            # annToMask returns a 0/1 mask
            binary_mask = coco.annToMask(ann)
            
            # Combine masks, using the category ID as the pixel value
            # This handles overlapping masks by simply using the max (or the last one in the list)
            mask[binary_mask > 0] = ann['category_id']

        # Convert the numpy array mask to a PIL image and save as PNG
        mask_image = Image.fromarray(mask, mode='L') # 'L' mode is for grayscale
        # Ensure the output filename has a .png extension
        mask_filename = os.path.splitext(file_name)[0] + '.png'
        mask_path = os.path.join(output_mask_dir, mask_filename)
        mask_image.save(mask_path)

        print(f"Saved mask for {file_name} to {mask_path}")

# Example usage (replace with your paths):
# ANNOTATION_FILE = 'path/to/your/coco_annotations.json'
# IMAGE_DIR = 'path/to/your/original/images'
# OUTPUT_MASK_DIR = 'path/to/save/png_masks'
# coco_to_png_masks(ANNOTATION_FILE, IMAGE_DIR, OUTPUT_MASK_DIR)