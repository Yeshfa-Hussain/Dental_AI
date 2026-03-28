import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import base64
import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq

# Load .env
load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# GROQ CLIENT
# ============================
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================
# SETTINGS
# ============================
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 4
MODEL_PATH  = 'Model File/best_model_ep8_val0.4036.pth'

COLORS = {
    0: (0,   0,   0),
    1: (255, 0,   0),
    2: (0,   255, 0),
    3: (0,   0,   255),
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

    pred_mask_resized = np.array(
        Image.fromarray(pred_mask.astype(np.uint8)).resize(
            (orig_w, orig_h), Image.NEAREST
        )
    )

    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        color_mask[pred_mask_resized == cls] = color

    overlay = (img * 0.6 + color_mask * 0.4).astype(np.uint8)

    detected = []
    for cls_id, name in CLASS_NAMES.items():
        if cls_id == 0:
            continue
        if (pred_mask_resized == cls_id).sum() > 100:
            detected.append(name)

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
# GROQ CHAT FUNCTION
# ============================
def chat_with_groq(messages):
    system_prompt = """You are a professional bilingual dental AI assistant. 
You speak both Urdu and English fluently. 
Detect the language the user is writing in and respond in the SAME language.

Follow this conversation flow:
1. When user says hello/salam → greet warmly and ask their name
2. After name → ask their age  
3. After age → ask their dental complaint
4. After complaint → give specific dental advice based on these conditions:
   - Bleeding gums → oral hygiene tips (brush twice, floss, mouthwash)
   - Cold/hot sensitivity → possible caries/enamel erosion, suggest sensitivity toothpaste
   - Pain after RCT/filling → explain healing vs complication signs, recommend crown after RCT
   - Toothache → possible causes (caries, pulpitis, infection), temporary relief tips
   - Swelling → URGENT warning, could be abscess, recommend immediate visit
   - Bad breath → hygiene tips, tongue cleaner, water intake
   - Broken tooth → avoid hard foods, keep piece, visit dentist

Always end advice with recommendation to book an appointment.
Keep responses friendly, professional and concise.
Never diagnose definitively — always recommend seeing a dentist.
Use patient's name in responses to make it personal."""

    response = groq_client.chat.completions.create(
       model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}] + messages,
        max_tokens=500,
        temperature=0.7,
    )
    return response.choices[0].message.content

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

@app.post("/chat")
async def chat(request: dict):
    try:
        messages = request.get("messages", [])
        reply = chat_with_groq(messages)
        return JSONResponse({"reply": reply})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
