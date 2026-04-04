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

# ── Per-model calibrated thresholds ──────────────────────────────────────────
# Derived empirically from observed score distributions on CBIS-DDSM test images:
#
#   Model             Benign scores  Malignant scores   Threshold (midpoint)
#   single_qubit      ~0.58          ~0.96              0.77
#   entanglement      ~0.80          ~0.78              UNRELIABLE (overlapping!)
#   full_variational  ~0.51          ~0.95              0.73
#
# The entanglement model has overlapping benign/malignant score distributions —
# it cannot distinguish reliably on its own. Its weight in the ensemble is halved.
#
# Ensemble threshold = weighted midpoint = 0.765 → rounded to 0.76
#
MODEL_THRESHOLDS = {
    "single_qubit":     0.77,
    "entanglement":     0.90,   # High threshold because model skews malignant
    "full_variational": 0.73,
}
ENSEMBLE_THRESHOLD = 0.76

# AUC-based ensemble weights (entanglement downweighted due to overlap)
MODEL_WEIGHTS = {
    "single_qubit":     0.40,
    "entanglement":     0.20,   # Downweighted — unreliable calibration
    "full_variational": 0.40,
}

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
    Check whether the image resembles a CBIS-DDSM grayscale mammogram.
    Returns {"valid": bool, "warning": str | None}
    """
    img_arr = np.array(img.convert("RGB")).astype(float)

    # Colour saturation: mammograms are near-grayscale so R≈G≈B
    per_pixel_sat = img_arr.max(axis=2) - img_arr.min(axis=2)
    mean_sat      = per_pixel_sat.mean()

    # Contrast: mammograms have high std
    gray_std = img_arr.mean(axis=2).std()

    # Aspect ratio
    w, h = img.size
    ar   = w / h

    warnings = []
    if mean_sat > 35:
        warnings.append(
            f"Colour photo detected (saturation={mean_sat:.0f}). "
            "This model was trained ONLY on CBIS-DDSM grayscale mammograms. "
            "Colour photos, ultrasound, or MRI images will produce unreliable results."
        )
    if gray_std < 25:
        warnings.append(
            f"Very low contrast (std={gray_std:.1f}). "
            "Please ensure you are uploading a proper full-mammogram JPEG."
        )
    if ar > 2.2 or ar < 0.35:
        warnings.append(f"Unusual aspect ratio ({ar:.2f}). Mammograms are typically portrait or square.")

    if warnings:
        return {"valid": False, "warning": " | ".join(warnings)}
    return {"valid": True, "warning": None}


def predict_image(model, img: Image.Image, config: str) -> dict:
    """
    Run inference using the per-model calibrated threshold.
    Returns probability, prediction, confidence, and threshold.
    """
    threshold = MODEL_THRESHOLDS.get(config, 0.76)

    img_t = val_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(img_t).squeeze()
        prob  = torch.sigmoid(logit).item()

    pred = "Malignant" if prob >= threshold else "Benign"

    # Confidence = normalised distance from this model's threshold
    if pred == "Malignant":
        conf = (prob - threshold) / (1.0 - threshold)
    else:
        conf = (threshold - prob) / threshold

    return {
        "probability": round(prob, 4),
        "prediction":  pred,
        "confidence":  round(conf * 100, 2),
        "threshold":   threshold,
    }


# ── API Routes ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":            "healthy",
        "models_loaded":     list(models_dict.keys()),
        "model_thresholds":  MODEL_THRESHOLDS,
        "ensemble_threshold": ENSEMBLE_THRESHOLD,
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

    validation = validate_mammogram(img)

    results = {}
    for name, model in models_dict.items():
        try:
            results[name] = predict_image(model, img, name)
        except Exception as e:
            results[name] = {"error": str(e)}

    # Weighted ensemble using AUC-based model weights
    weighted_sum  = 0.0
    total_weight  = 0.0
    for name in models_dict:
        r = results.get(name, {})
        if "probability" in r:
            w             = MODEL_WEIGHTS.get(name, 1.0)
            weighted_sum += r["probability"] * w
            total_weight += w

    if total_weight > 0:
        avg_prob      = round(weighted_sum / total_weight, 4)
        ensemble_pred = "Malignant" if avg_prob >= ENSEMBLE_THRESHOLD else "Benign"

        if ensemble_pred == "Malignant":
            ensemble_conf = (avg_prob - ENSEMBLE_THRESHOLD) / (1.0 - ENSEMBLE_THRESHOLD)
        else:
            ensemble_conf = (ENSEMBLE_THRESHOLD - avg_prob) / ENSEMBLE_THRESHOLD

        results["ensemble"] = {
            "probability": avg_prob,
            "prediction":  ensemble_pred,
            "confidence":  round(ensemble_conf * 100, 2),
            "threshold":   ENSEMBLE_THRESHOLD,
            "note":        "Weighted ensemble (SQ×0.4 + EN×0.2 + FV×0.4)",
        }

    return JSONResponse(content={
        "filename":   file.filename,
        "results":    results,
        "validation": validation,
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

    result     = predict_image(models_dict[model_name], img, model_name)
    validation = validate_mammogram(img)

    return JSONResponse(content={
        "filename":   file.filename,
        "model_used": model_name,
        "result":     result,
        "validation": validation,
    })


# ── Serve website — MUST be last ───────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")
