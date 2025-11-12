from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
import requests

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embedding(text: str):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs.to(device))
    return emb / emb.norm(p=2, dim=-1, keepdim=True)

def get_image_embedding(image_file: UploadFile = None, image_url: str = None):
    try:
        if image_file is not None:
            img = Image.open(image_file.file).convert("RGB")
        elif image_url:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return torch.zeros(1, 512)
        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs.to(device))
        return emb / emb.norm(p=2, dim=-1, keepdim=True)
    except Exception as e:
        print("⚠️ Failed to load image:", e)
        return torch.zeros(1, 512)

def detect_fraud(title: str, description: str, image_file: UploadFile = None, image_url: str = None, threshold=0.25):
    title_emb = get_text_embedding(title)
    desc_emb = get_text_embedding(description)
    img_emb = get_image_embedding(image_file, image_url)

    sim_title = (title_emb @ img_emb.T).item()
    sim_desc = (desc_emb @ img_emb.T).item()
    sim = min(sim_title, sim_desc)

    fraud = sim < threshold
    return {"fraud": fraud, "similarity": sim}

app = FastAPI(title="Tool Pre-Upload Fraud Check API")

@app.post("/precheck/")
async def precheck_tool(
    title: str = Form(...),
    description: str = Form(...),
    image_file: UploadFile = File(None),
    image_url: str = Form(None),
    threshold: float = Form(0.25)
):
    """
    Pre-check a tool before saving it:
    - Upload image or provide URL
    - Provide title + description
    - Returns fraud status and similarity
    """
    result = detect_fraud(title, description, image_file, image_url, threshold)
    can_save = not result["fraud"]
    can_edit = not result["fraud"]
    
    if result["fraud"]:
        # Do not save listing, return warning
        return JSONResponse(
            status_code=400,
            content={
                "message": "❌ Image does not match title/description. Listing cannot be saved.",
                "similarity": result["similarity"],
                "threshold": float(threshold),
                "can_save": can_save,
                "can_edit": can_edit
            }
        )
    else:
        # Safe to save
        return JSONResponse(
            status_code=200,
            content={
                "message": "✅ Image matches listing. Safe to save.",
                "similarity": result["similarity"],
                "threshold": float(threshold),
                "can_save": can_save,
                "can_edit": can_edit
            }
        )
