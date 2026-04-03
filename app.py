from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageStat
import torch
import torchvision.transforms as transforms
import io
import os
import numpy as np
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

# ── Decision threshold ────────────────────────────────────────
# Calibrated from test set results in notebook:
#   Benign cluster:    0.10 – 0.29 probability
#   Malignant cluster: 0.78 – 0.96 probability
#   Midpoint threshold = (0.29 + 0.78) / 2 = 0.535 → rounded to 0.54
# Using 0.54 gives a clean gap of ~0.49 between the two clusters.
THRESHOLD = 0.54

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


def validate_mammogram(img: Image.Image) -> dict:
    """
    Check whether the uploaded image resembles a CBIS-DDSM mammogram.

    CBIS-DDSM mammograms are:
      - Grayscale / near-grayscale  (low colour saturation)
      - High contrast               (pixel std > 30)
      - Mostly dark with bright tissue regions

    Returns {"valid": bool, "warning": str | None}
    """
    img_arr = np.array(img.convert("RGB")).astype(float)   # H x W x 3

    # Colour saturation: per-pixel range across R/G/B channels
    # Natural photos have high saturation; mammograms are near-zero
    per_pixel_sat  = img_arr.max(axis=2) - img_arr.min(axis=2)
    mean_sat       = per_pixel_sat.mean()

    # Contrast: standard deviation of grayscale brightness
    gray_std = img_arr.mean(axis=2).std()

    # Aspect ratio
    w, h = img.size
    ar   = w / h

    warnings = []

    if mean_sat > 35:
        warnings.append(
            f"Colour photo detected (saturation={mean_sat:.0f}). "
            "This model was trained ONLY on CBIS-DDSM grayscale mammograms. "
            "Colour photos, ultrasound images, or MRI scans will produce unreliable results. "
            "Please upload a proper mammogram JPEG from a DICOM-converted source."
        )

    if gray_std < 25:
        warnings.append(
            f"Very low image contrast (std={gray_std:.1f}). "
            "Mammograms have high contrast. This may not be a valid mammogram image."
        )

    if ar > 2.2 or ar < 0.35:
        warnings.append(
            f"Unusual aspect ratio ({ar:.2f}). "
            "Mammograms are typically portrait or squarish."
        )

    if warnings:
        return {"valid": False, "warning": " | ".join(warnings)}
    return {"valid": True, "warning": None}


def predict_image(model, img: Image.Image) -> dict:
    """
    Run inference on a PIL image.
    Returns probability, prediction label, confidence %, and threshold used.
    """
    img_t = val_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(img_t).squeeze()
        prob  = torch.sigmoid(logit).item()

    pred = "Malignant" if prob >= THRESHOLD else "Benign"

    # Confidence = normalised distance from threshold (0% at threshold, 100% at extremes)
    if pred == "Malignant":
        conf = (prob - THRESHOLD) / (1.0 - THRESHOLD)
    else:
        conf = (THRESHOLD - prob) / THRESHOLD

    return {
        "probability": round(prob, 4),
        "prediction":  pred,
        "confidence":  round(conf * 100, 2),
        "threshold":   THRESHOLD,
    }


# ── API Routes ─────────────────────────────────────────────────

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

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    # Validate image looks like a CBIS-DDSM mammogram
    validation = validate_mammogram(img)

    results = {}
    for name, model in models_dict.items():
        try:
            results[name] = predict_image(model, img)
        except Exception as e:
            results[name] = {"error": str(e)}

    # Ensemble: average raw probabilities then apply threshold
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

    return JSONResponse(content={
        "filename":   file.filename,
        "results":    results,
        "validation": validation,   # frontend uses this to show warning banner
    })


@app.post("/predict/{model_name}")
async def predict_single(model_name: str, file: UploadFile = File(...)):
    if model_name not in models_dict:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(models_dict.keys())}"
        )
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    result     = predict_image(models_dict[model_name], img)
    validation = validate_mammogram(img)

    return JSONResponse(content={
        "filename":   file.filename,
        "model_used": model_name,
        "result":     result,
        "validation": validation,
    })


# ── Serve website — MUST be last ───────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")
