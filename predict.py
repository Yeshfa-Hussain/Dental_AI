"""
predict.py — Dental AI
Run trained U-Net on new X-ray images → colorized segmentation output
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from model import UNet

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.getcwd())

MODEL_PATH  = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
INPUT_DIR   = os.path.join(BASE_DIR, "DentAI.v2i.coco-segmentation", "test")
OUTPUT_DIR  = os.path.join(BASE_DIR, "predictions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE    = 512
NUM_CLASSES = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Colorized output legend
COLOR_MAP = {
    0: (0,   0,   0),     # Background  → black
    1: (255, 0,   0),     # Caries      → red
    2: (255, 255, 0),     # Infection   → yellow
    3: (0,   200, 0),     # Restoration → green
}

CLASS_NAMES = {0: "Background", 1: "Caries", 2: "Infection", 3: "Restoration"}


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
def load_model(model_path):
    model = UNet(in_channels=3, num_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Handle both full checkpoint and state_dict only
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Val loss: {checkpoint.get('val_loss', '?'):.4f}")
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model


# ─────────────────────────────────────────────
#  PREPROCESS IMAGE
# ─────────────────────────────────────────────
def preprocess(img_path):
    image = Image.open(img_path).convert("RGB")
    orig_size = image.size   # (W, H) — save for resizing output back

    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor,
                          mean=[0.485, 0.456, 0.406],
                          std= [0.229, 0.224, 0.225])
    return tensor.unsqueeze(0), orig_size   # [1, 3, H, W]


# ─────────────────────────────────────────────
#  COLORIZE MASK
# ─────────────────────────────────────────────
def colorize_mask(mask_array):
    """Convert (H, W) class index mask → (H, W, 3) RGB color image."""
    color_img = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        color_img[mask_array == class_id] = color
    return color_img


# ─────────────────────────────────────────────
#  OVERLAY ON ORIGINAL IMAGE
# ─────────────────────────────────────────────
def overlay_mask(original_img, color_mask, alpha=0.5):
    """Blend original X-ray with colorized mask."""
    original = np.array(original_img.convert("RGB"), dtype=np.float32)
    overlay  = color_mask.astype(np.float32)

    # Only blend where mask is non-zero (has a detection)
    has_detection = (color_mask.sum(axis=2) > 0)
    blended = original.copy()
    blended[has_detection] = (
        alpha * overlay[has_detection] +
        (1 - alpha) * original[has_detection]
    )
    return Image.fromarray(blended.astype(np.uint8))


# ─────────────────────────────────────────────
#  PREDICT ONE IMAGE
# ─────────────────────────────────────────────
def predict_image(model, img_path, save_prefix):
    tensor, orig_size = preprocess(img_path)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        output = model(tensor)                    # [1, 4, H, W]
        pred   = torch.argmax(output, dim=1)      # [1, H, W]
        pred   = pred.squeeze(0).cpu().numpy()    # [H, W]

    # Classes found in this prediction
    unique_classes = np.unique(pred)
    found = [CLASS_NAMES[c] for c in unique_classes if c != 0]
    print(f"  Detected: {found if found else ['Nothing detected']}")

    # Colorized mask
    color_mask = colorize_mask(pred)
    color_img  = Image.fromarray(color_mask).resize(orig_size, Image.NEAREST)

    # Overlay on original
    original   = Image.open(img_path).convert("RGB")
    overlay    = overlay_mask(original, np.array(color_mask.copy()))
    overlay    = overlay.resize(orig_size, Image.BILINEAR)

    # Save outputs
    color_img.save(os.path.join(OUTPUT_DIR, f"{save_prefix}_mask_colored.png"))
    overlay.save(os.path.join(OUTPUT_DIR,   f"{save_prefix}_overlay.png"))

    return pred, found


# ─────────────────────────────────────────────
#  RUN ON ALL TEST IMAGES
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Dental AI — Prediction")
    print("="*60)
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Input  : {INPUT_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Device : {DEVICE}")

    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model not found: {MODEL_PATH}")
        print("  Train the model first using: python train.py")
        return

    print("\n  Loading model...")
    model = load_model(MODEL_PATH)

    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print(f"[ERROR] No images found in {INPUT_DIR}")
        return

    print(f"\n  Running predictions on {len(image_files)} images...\n")

    for i, fname in enumerate(image_files):
        img_path    = os.path.join(INPUT_DIR, fname)
        save_prefix = os.path.splitext(fname)[0]

        print(f"  [{i+1}/{len(image_files)}] {fname}")
        predict_image(model, img_path, save_prefix)

    print(f"\n  All predictions saved to: {OUTPUT_DIR}")
    print("\n  Output files per image:")
    print("    *_mask_colored.png  → colorized segmentation mask")
    print("    *_overlay.png       → mask overlaid on original X-ray")
    print("\n  Color legend:")
    for cls_id, name in CLASS_NAMES.items():
        color = COLOR_MAP[cls_id]
        print(f"    {name:12s} → RGB{color}")


if __name__ == "__main__":
    main()
