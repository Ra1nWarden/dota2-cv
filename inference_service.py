"""FastAPI inference service for Dota 2 icon classification.

Loads two ONNX models (hero + item classifiers) and exposes a /predict
endpoint that accepts a screenshot, crops HUD regions, and returns
per-slot predictions.

Usage:
    uvicorn inference_service:app --host 0.0.0.0 --port 8080 --workers 1
"""

import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile
from PIL import Image

# Config
WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
IMAGE_SIZE = 128

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Globals populated at startup
hero_session: ort.InferenceSession | None = None
item_session: ort.InferenceSession | None = None
hero_classes: list[str] = []
item_classes: list[str] = []
crop_config: dict = {}

app = FastAPI(title="Dota 2 Icon Classifier")


@app.on_event("startup")
def load_models():
    global hero_session, item_session, hero_classes, item_classes, crop_config

    # Load crop config
    with open(WORKSPACE / "configs" / "crop_config.json") as f:
        crop_config = json.load(f)

    # Load class mappings
    with open(WORKSPACE / "configs" / "heroes_classes.json") as f:
        hero_map = json.load(f)
    hero_classes = [hero_map[str(i)] for i in range(len(hero_map))]

    with open(WORKSPACE / "configs" / "items_classes.json") as f:
        item_map = json.load(f)
    item_classes = [item_map[str(i)] for i in range(len(item_map))]

    # Load ONNX models
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    hero_session = ort.InferenceSession(
        str(WORKSPACE / "models" / "hero_classifier.onnx"), providers=providers
    )
    item_session = ort.InferenceSession(
        str(WORKSPACE / "models" / "item_classifier.onnx"), providers=providers
    )

    print(f"Loaded hero model: {len(hero_classes)} classes")
    print(f"Loaded item model: {len(item_classes)} classes")
    print(f"Crop config: {len(crop_config['regions'])} regions")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")


def preprocess_crop(crop: Image.Image) -> np.ndarray:
    """Preprocess a crop to match training transforms.

    Pad to square, resize to IMAGE_SIZE, normalize with ImageNet stats.
    Returns CHW float32 array.
    """
    # Pad to square
    w, h = crop.size
    if w != h:
        size = max(w, h)
        padded = Image.new("RGB", (size, size), (0, 0, 0))
        padded.paste(crop, ((size - w) // 2, (size - h) // 2))
        crop = padded

    # Resize
    crop = crop.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    # HWC uint8 → CHW float32 [0, 1]
    arr = np.array(crop, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW

    # Normalize
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    return arr


def run_inference(
    session: ort.InferenceSession,
    batch: np.ndarray,
    class_names: list[str],
) -> list[dict]:
    """Run ONNX inference on a batch, return list of {class, confidence}."""
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: batch})[0]

    # Softmax
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    results = []
    for prob in probs:
        idx = int(prob.argmax())
        conf = float(prob[idx])
        name = class_names[idx] if conf >= CONFIDENCE_THRESHOLD else "unknown"
        results.append({"class": name, "confidence": round(conf, 4)})

    return results


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile):
    # Decode image
    data = await file.read()
    image = Image.open(BytesIO(data)).convert("RGB")
    img_w, img_h = image.size

    # Scale factors from reference resolution
    ref_w, ref_h = crop_config["reference_resolution"]
    scale_x = img_w / ref_w
    scale_y = img_h / ref_h

    regions = crop_config["regions"]

    # Crop and preprocess
    hero_crops = []
    hero_names = []
    item_crops = []
    item_names = []

    for name, coords in regions.items():
        x = int(coords["x"] * scale_x)
        y = int(coords["y"] * scale_y)
        w = int(coords["w"] * scale_x)
        h = int(coords["h"] * scale_y)

        crop = image.crop((x, y, x + w, y + h))
        processed = preprocess_crop(crop)

        if name.startswith(("radiant_hero", "dire_hero")):
            hero_crops.append(processed)
            hero_names.append(name)
        else:
            item_crops.append(processed)
            item_names.append(name)

    # Batch inference
    response = {"heroes": {}, "items": {}}

    if hero_crops:
        hero_batch = np.stack(hero_crops).astype(np.float32)
        hero_results = run_inference(hero_session, hero_batch, hero_classes)
        for name, result in zip(hero_names, hero_results):
            response["heroes"][name] = result

    if item_crops:
        item_batch = np.stack(item_crops).astype(np.float32)
        item_results = run_inference(item_session, item_batch, item_classes)
        for name, result in zip(item_names, item_results):
            response["items"][name] = result

    return response
