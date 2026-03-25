"""
train.py — Dental AI
Full training loop for U-Net dental segmentation model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json

from dataset import DentalDataset
from model   import UNet

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath(os.getcwd())

DATA_DIR   = os.path.join(BASE_DIR, "DentAI.v2i.coco-segmentation")
MASKS_DIR  = os.path.join(BASE_DIR, "masks")
SAVE_DIR   = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Hyperparameters (tweak as needed) ──
IMG_SIZE    = 512
NUM_CLASSES = 4       # Background, Caries, Infection, Restoration
BATCH_SIZE  = 4       # reduce to 2 if GPU memory is low
NUM_EPOCHS  = 50
LR          = 1e-4
NUM_WORKERS = 0       # set to 2-4 if on Linux/Mac

# Class weights — give more weight to rare classes (caries/infection)
# Background(0), Caries(1), Infection(2), Restoration(3)
CLASS_WEIGHTS = torch.tensor([0.3, 2.5, 3.0, 2.0], dtype=torch.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {0: "Background", 1: "Caries", 2: "Infection", 3: "Restoration"}


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def compute_iou(preds, masks, num_classes=4):
    """Compute per-class IoU (Intersection over Union)."""
    iou_per_class = []
    preds = preds.cpu().numpy()
    masks = masks.cpu().numpy()

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        true_cls = (masks == cls)

        intersection = (pred_cls & true_cls).sum()
        union        = (pred_cls | true_cls).sum()

        if union == 0:
            iou_per_class.append(float("nan"))   # class not present
        else:
            iou_per_class.append(intersection / union)

    return iou_per_class


def compute_dice(preds, masks, num_classes=4):
    """Compute per-class Dice coefficient."""
    dice_per_class = []
    preds = preds.cpu().numpy()
    masks = masks.cpu().numpy()

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        true_cls = (masks == cls)

        intersection = (pred_cls & true_cls).sum()
        total        = pred_cls.sum() + true_cls.sum()

        if total == 0:
            dice_per_class.append(float("nan"))
        else:
            dice_per_class.append(2 * intersection / total)

    return dice_per_class


# ─────────────────────────────────────────────
#  TRAIN ONE EPOCH
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc="  Training", leave=False)
    for images, masks in loop:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)              # [B, 4, H, W]
        loss    = criterion(outputs, masks)  # [B, H, W]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


# ─────────────────────────────────────────────
#  VALIDATE ONE EPOCH
# ─────────────────────────────────────────────

def val_epoch(model, loader, criterion, device, num_classes=4):
    model.eval()
    total_loss = 0
    all_iou    = []
    all_dice   = []

    with torch.no_grad():
        loop = tqdm(loader, desc="  Validating", leave=False)
        for images, masks in loop:
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)
            loss    = criterion(outputs, masks)
            total_loss += loss.item()

            # Get predicted class per pixel
            preds = torch.argmax(outputs, dim=1)   # [B, H, W]

            iou  = compute_iou(preds,  masks, num_classes)
            dice = compute_dice(preds, masks, num_classes)

            all_iou.append(iou)
            all_dice.append(dice)

    # Average metrics ignoring NaN (classes not present in batch)
    avg_iou  = np.nanmean(all_iou,  axis=0)
    avg_dice = np.nanmean(all_dice, axis=0)

    return total_loss / len(loader), avg_iou, avg_dice


# ─────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Dental AI — U-Net Training")
    print("="*60)
    print(f"  Device     : {DEVICE}")
    print(f"  Image size : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Epochs     : {NUM_EPOCHS}")
    print(f"  LR         : {LR}")
    print(f"  Classes    : {list(CLASS_NAMES.values())}")

    # ── Datasets ──
    print("\n  Loading datasets...")
    train_dataset = DentalDataset(
        image_dir = os.path.join(DATA_DIR, "train"),
        mask_dir  = os.path.join(MASKS_DIR, "train"),
        split      = "train",
        img_size   = IMG_SIZE
    )
    val_dataset = DentalDataset(
        image_dir = os.path.join(DATA_DIR, "valid"),
        mask_dir  = os.path.join(MASKS_DIR, "valid"),
        split      = "valid",
        img_size   = IMG_SIZE
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ── Model ──
    print("\n  Building U-Net model...")
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ── Loss, Optimizer, Scheduler ──
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )

    # ── Training ──
    best_val_loss = float("inf")
    history       = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

    print(f"\n  Starting training for {NUM_EPOCHS} epochs...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")

        train_loss             = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_iou, val_dice = val_epoch(model, val_loader, criterion, DEVICE, NUM_CLASSES)

        scheduler.step(val_loss)

        # Per-class metrics
        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Val   Loss : {val_loss:.4f}")
        print(f"  Val IoU    : ", end="")
        for cls_id, iou in enumerate(val_iou):
            print(f"{CLASS_NAMES[cls_id]}={iou:.3f}", end="  ")
        print()
        print(f"  Val Dice   : ", end="")
        for cls_id, dice in enumerate(val_dice):
            print(f"{CLASS_NAMES[cls_id]}={dice:.3f}", end="  ")
        print()
        print(f"  Mean IoU   : {np.nanmean(val_iou[1:]):.4f}")   # exclude background
        print(f"  Mean Dice  : {np.nanmean(val_dice[1:]):.4f}")  # exclude background
        print()

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou.tolist())
        history["val_dice"].append(val_dice.tolist())

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path     = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save({
                "epoch":      epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":   val_loss,
                "val_iou":    val_iou.tolist(),
                "val_dice":   val_dice.tolist(),
            }, best_path)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})\n")

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"checkpoint_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  💾 Checkpoint saved: checkpoint_epoch{epoch}.pth\n")

    # Save training history
    history_path = os.path.join(SAVE_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("="*60)
    print(f"  Training complete!")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Best model    : {os.path.join(SAVE_DIR, 'best_model.pth')}")
    print(f"  History saved : {history_path}")
    print("="*60)


if __name__ == "__main__":
    main()
