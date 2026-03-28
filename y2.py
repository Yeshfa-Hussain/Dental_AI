import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# ✅ CORS — website se connect karne ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# SETTINGS
# ============================
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 4
MODEL_PATH  = 'Model File/best_model_ep8_val0.4036.pth'

# Colors for each class
COLORS = {
    0: (0,   0,   0),    # background - black
    1: (255, 0,   0),    # caries     - red
    2: (0,   255, 0),    # infection  - green
    3: (0,   0,   255),  # restoration - blue
}

CLASS_NAMES = {
    0: "Background",
    1: "Caries",
    2: "Infection",
    3: "Restoration",
}

# ============================
# LOAD MODEL
# ============================
print(f"Loading model on {DEVICE}...")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
    decoder_dropout=0.3
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded!")

# ============================
# TRANSFORM
# ============================
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ============================
# PREDICT FUNCTION
# ============================
def predict_image(image_bytes):
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    orig_h, orig_w = img.shape[:2]

    tensor = transform(image=img)['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Resize mask back to original size
    pred_mask_resized = np.array(
        Image.fromarray(pred_mask.astype(np.uint8)).resize(
            (orig_w, orig_h), Image.NEAREST
        )
    )

    # Colorize mask
    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        color_mask[pred_mask_resized == cls] = color

    # Overlay
    overlay = (img * 0.6 + color_mask * 0.4).astype(np.uint8)

    # Detected classes
    detected = []
    for cls_id, name in CLASS_NAMES.items():
        if cls_id == 0:
            continue
        if (pred_mask_resized == cls_id).sum() > 100:
            detected.append(name)

    # Convert images to base64
    def to_base64(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()

    return {
        "detected": detected,
        "color_mask": to_base64(color_mask),
        "overlay":    to_base64(overlay),
        "original":   to_base64(img),
    }

# ============================
# ROUTES
# ============================
@app.get("/")
def home():
    return {"message": "✅ Dental AI API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = predict_image(image_bytes)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)