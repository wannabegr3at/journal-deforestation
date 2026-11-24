import io
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Urutan alfabetis standar LabelEncoder (scikit-learn)
CLASS_NAMES = [
    "Grassland shrubland",
    "Other",
    "Plantation",
    "Smallholder agriculture",
]

MODEL_PATH = Path("checkpoints/best_proposed_unfrozen_3topo.pt")
SCALER_PATH = Path("checkpoints/aux_scaler.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultimodalSuperConClassifier(nn.Module):
    """
    Struktur persis mengikuti best_proposed_unfrozen_3topo.pt:
    - encoder: nn.Sequential(effnet_b2.features, effnet_b2.avgpool) -> encoder.0.0.0.weight
    - classifier: BN(1414) -> Linear(512) -> ReLU -> Dropout(0.3) -> Linear(128) -> ReLU -> Linear(4)
    """
    def __init__(self, num_aux: int = 6, num_classes: int = 4):
        super().__init__()
        base = models.efficientnet_b2(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        in_features = 1408 + num_aux  # 1414

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),  # 0
            nn.Linear(in_features, 512),  # 1
            nn.ReLU(inplace=True),        # 2
            nn.Dropout(0.3),              # 3
            nn.Linear(512, 128),          # 4
            nn.ReLU(inplace=True),        # 5
            nn.Linear(128, num_classes),  # 6
        )

    def forward(self, image: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        rep = self.encoder(image)
        rep = torch.flatten(rep, 1)
        fused = torch.cat([rep, aux], dim=1)
        logits = self.classifier(fused)
        return logits


app = FastAPI(title="ForestNet Deforestation Driver Tester")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

image_transform = T.Compose([
    T.Resize((332, 332)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model_loaded_from_checkpoint = False
aux_scaler = None

if SCALER_PATH.exists():
    with open(SCALER_PATH, "rb") as f:
        aux_scaler = pickle.load(f)

model = MultimodalSuperConClassifier(num_aux=6, num_classes=4)

if MODEL_PATH.exists():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    load_info = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Checkpoint dimuat: {MODEL_PATH}")
    print(f"[INFO] Missing keys ({len(load_info.missing_keys)}): {load_info.missing_keys[:5]}")
    print(f"[INFO] Unexpected keys ({len(load_info.unexpected_keys)}): {load_info.unexpected_keys[:5]}")
    model_loaded_from_checkpoint = len(load_info.missing_keys) == 0
else:
    print(f"[WARNING] File checkpoint tidak ditemukan di {MODEL_PATH}")

model.to(DEVICE)
model.eval()


@app.get("/")
def serve_home():
    return FileResponse("static/index.html")


@app.get("/api/status")
def get_status() -> Dict:
    return {
        "status": "ready",
        "device": str(DEVICE),
        "checkpoint_loaded": model_loaded_from_checkpoint,
        "checkpoint_path": str(MODEL_PATH),
    }


@app.post("/api/predict")
async def predict_driver(
    file: UploadFile = File(...),
    slope_mean: float = Form(...),
    slope_std: float = Form(...),
    altitude_mean: float = Form(...),
    altitude_std: float = Form(...),
    aspect_mean: float = Form(...),
    aspect_std: float = Form(...),
):
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = image_transform(pil_image).unsqueeze(0).to(DEVICE)

    aux_array = np.array(
        [[slope_mean, slope_std, altitude_mean, altitude_std, aspect_mean, aspect_std]],
        dtype=np.float32,
    )

    if aux_scaler is not None:
        aux_array = aux_scaler.transform(aux_array)

    aux_tensor = torch.tensor(aux_array, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor, aux_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    predictions = [
        {"class_name": CLASS_NAMES[i], "probability": round(float(probs[i]) * 100, 2)}
        for i in range(len(CLASS_NAMES))
    ]
    predictions.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "all_probabilities": predictions,
        "checkpoint_mode": "Trained Checkpoint (.pt)" if model_loaded_from_checkpoint else "Partial/Untrained",
        "input_summary": {
            "image_size": f"{pil_image.width}x{pil_image.height} -> 332x332 px",
            "topography": {
                "slope": f"{slope_mean} ± {slope_std}°",
                "altitude": f"{altitude_mean} ± {altitude_std} m",
                "aspect": f"{aspect_mean} ± {aspect_std}°",
            },
        },
    }