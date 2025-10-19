# Multimodal Fraud Detector for Tool Listings
# Works on text + image data (image_url)
# Requires: torch, torchvision, transformers, pandas, scikit-learn, PIL, requests

import pandas as pd
import torch
from PIL import Image
from io import BytesIO
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from transformers import CLIPProcessor, CLIPModel
import joblib

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("tool_listings_with_images_100.csv")

# Combine title + description
df['text'] = df['title'] + " " + df['description']

# -------------------------
# Load CLIP Model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# -------------------------
# Encode Text and Images
# -------------------------
def get_image_embedding(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs.to(device))
        return emb.cpu().numpy().flatten()
    except:
        return None

def get_text_embedding(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs.to(device))
    return emb.cpu().numpy().flatten()

print("Encoding text and images...")
text_embeddings = []
image_embeddings = []

for idx, row in df.iterrows():
    text_emb = get_text_embedding(row['text'])
    text_embeddings.append(text_emb)
    
    img_emb = get_image_embedding(row['image_url'])
    if img_emb is None:
        img_emb = torch.zeros(512).numpy()  # fallback if image fails
    image_embeddings.append(img_emb)

X_text = torch.tensor(text_embeddings).numpy()
X_image = torch.tensor(image_embeddings).numpy()

# Concatenate text + image features
import numpy as np
X = np.concatenate([X_text, X_image], axis=1)
y = df['label'].values

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# Train Classifier
# -------------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# -------------------------
# Save Model
# -------------------------
joblib.dump(clf, "multimodal_fraud_model.pkl")
print("✅ Model saved as multimodal_fraud_model.pkl")
