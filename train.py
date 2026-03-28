import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import DentalDataset, get_transforms

# Paths — Colab paths
TRAIN_IMG  = '/content/dataset/train'
TRAIN_MASK = '/content/dataset/train/masks_multiclass'
VALID_IMG  = '/content/dataset/valid'
VALID_MASK = '/content/dataset/valid/masks_multiclass'

# Settings
EPOCHS      = 25
BATCH_SIZE  = 8
LR          = 1e-4
NUM_CLASSES = 4
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {DEVICE}")

# Datasets
train_ds = DentalDataset(TRAIN_IMG, TRAIN_MASK, transform=get_transforms(train=True))
valid_ds = DentalDataset(VALID_IMG, VALID_MASK, transform=get_transforms(train=False))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train samples: {len(train_ds)} | Valid samples: {len(valid_ds)}")



# dropout in model (line ~28)
# 1. Model first
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
    decoder_dropout=0.3
).to(DEVICE)

# 2. Loss & Optimizer after
class_weights = torch.tensor([0.1, 2.0, 2.0, 2.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

# Training Loop
best_loss = float('inf')
no_improve = 0

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs).to(DEVICE)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in valid_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss   /= len(valid_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} → Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), 
            f'/content/drive/MyDrive/best_model_ep{epoch+1}_val{val_loss:.4f}.pth')
        print(f"  ✅ Best model saved! Val Loss: {val_loss:.4f}")
    else:
        no_improve += 1
        print(f"  ⚠️ No improvement {no_improve}/5")
        if no_improve >= 5:
            print(f"⛔ Early stopping! Best Val Loss: {best_loss:.4f}")
            break
print("🎉 Training complete!")
