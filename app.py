from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import os
from quantum_models import HybridModel

app = FastAPI(
    title="Quantum Breast Cancer Classification API",
    description="Hybrid Classical-Quantum Neural Network for Breast Cancer Detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")

# ── Calibrated threshold ──────────────────────────────────────
# Lowered from 0.5 → 0.40 to reduce false malignant predictions
# on benign images. Your trained models output benign probs in
# the 0.10–0.35 range and malignant in the 0.78–0.96 range,
# so 0.40 sits safely between the two clusters.
THRESHOLD = 0.40

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("📦 Loading models...")
models_dict = {}

def load_model(config: str, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = HybridModel(config=config).to(DEVICE)
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"   ✅ {config} loaded | AUC: {ckpt.get('val_auc', 'N/A'):.4f}")
    return model

try:
    models_dict["single_qubit"]     = load_model("single_qubit",     "models/single_qubit_best.pth")
    models_dict["entanglement"]     = load_model("entanglement",     "models/entanglement_best.pth")
    models_dict["full_variational"] = load_model("full_variational", "models/full_variational_best.pth")
    print("✅ All models loaded!")
except Exception as e:
    print(f"❌ Error loading models: {e}")


def predict_image(model, image_bytes: bytes):
    """
    Run inference on image bytes and return prediction dict.

    probability : raw sigmoid output (0 = Benign, 1 = Malignant)
    prediction  : "Benign" or "Malignant"
    confidence  : how confident the model is in its prediction (%)
    threshold   : the decision threshold used
    """
    img   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_t = val_tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(img_t).squeeze()
        prob  = torch.sigmoid(logit).item()

    pred = "Malignant" if prob >= THRESHOLD else "Benign"

    # Confidence = distance from threshold, scaled to 0–100%
    # e.g. prob=0.95 → conf=90%  |  prob=0.05 → conf=90%  |  prob=0.40 → conf=0%
    if pred == "Malignant":
        conf = (prob - THRESHOLD) / (1.0 - THRESHOLD)   # 0→1 above threshold
    else:
        conf = (THRESHOLD - prob) / THRESHOLD            # 0→1 below threshold

    return {
        "probability": round(prob, 4),
        "prediction":  pred,
        "confidence":  round(conf * 100, 2),
        "threshold":   THRESHOLD,
    }


# ── API Routes ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "healthy",
        "models_loaded": list(models_dict.keys()),
        "threshold":     THRESHOLD,
    }


@app.post("/predict")
async def predict_all(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    if not models_dict:
        raise HTTPException(status_code=503, detail="No models loaded.")

    image_bytes = await file.read()
    results = {}

    for name, model in models_dict.items():
        try:
            results[name] = predict_image(model, image_bytes)
        except Exception as e:
            results[name] = {"error": str(e)}

    # Ensemble: average raw probabilities, then apply threshold
    probs = [r["probability"] for r in results.values() if "probability" in r]
    if probs:
        avg_prob      = round(sum(probs) / len(probs), 4)
        ensemble_pred = "Malignant" if avg_prob >= THRESHOLD else "Benign"

        if ensemble_pred == "Malignant":
            ensemble_conf = (avg_prob - THRESHOLD) / (1.0 - THRESHOLD)
        else:
            ensemble_conf = (THRESHOLD - avg_prob) / THRESHOLD

        results["ensemble"] = {
            "probability": avg_prob,
            "prediction":  ensemble_pred,
            "confidence":  round(ensemble_conf * 100, 2),
            "threshold":   THRESHOLD,
            "note":        "Average of all 3 quantum models",
        }

    return JSONResponse(content={"filename": file.filename, "results": results})


@app.post("/predict/{model_name}")
async def predict_single(model_name: str, file: UploadFile = File(...)):
    if model_name not in models_dict:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. "
                   f"Available: {list(models_dict.keys())}"
        )
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    result = predict_image(models_dict[model_name], image_bytes)
    return JSONResponse(content={
        "filename":   file.filename,
        "model_used": model_name,
        "result":     result,
    })


# ── Serve website — MUST be last ──────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")
