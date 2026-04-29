"""FastAPI inference service for Dota 2 icon classification.

Loads two ONNX models (hero + item classifiers) and exposes a /predict
endpoint that accepts a screenshot, crops HUD regions, and returns
per-slot predictions.

Item crops use anchor-relative cropping when an anchor template + offsets
are configured (configs/anchor_offsets.json + the referenced template PNG).
This compensates for skill-bar width drift that pushes the item grid off
its fixed position. If the anchor isn't found in a frame (low match score),
we fall back to fixed crops from crop_config.json. Hero crops always use
fixed crops.

Usage:
    Mounted as a router by main.py; not run as a standalone service.
"""

import json
import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, UploadFile
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
anchor_config: dict | None = None
anchor_template: np.ndarray | None = None  # uint8 grayscale Canny edges

talent_anchor_config: dict | None = None
talent_anchor_template: np.ndarray | None = None
ocr_reader = None                     # easyocr.Reader, initialized at startup
hero_display_names: dict[str, str] = {}  # display_name → GSI internal name
last_known_hero: str | None = None    # cached across frames (single worker)

router = APIRouter()


def startup() -> None:
    """Load ONNX models, anchor templates, OCR reader, and class maps into module globals.

    Called once by ``main.py``'s lifespan before any router handler runs.
    """
    global hero_session, item_session, hero_classes, item_classes, crop_config
    global anchor_config, anchor_template
    global talent_anchor_config, talent_anchor_template, ocr_reader, hero_display_names

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

    # Optional anchor configuration for item crops
    anchor_config, anchor_template = load_anchor_assets(WORKSPACE)
    if anchor_template is not None:
        th, tw = anchor_template.shape
        print(
            f"Anchor enabled: {anchor_config['anchor']!r} template {tw}x{th}, "
            f"threshold={anchor_config['match_threshold']}"
        )
    else:
        print("Anchor not configured; using fixed item crops")

    # Talent indicator anchor for hero name OCR
    talent_anchor_config, talent_anchor_template = load_anchor_assets(
        WORKSPACE, config_filename="talent_anchor_offsets.json"
    )
    if talent_anchor_template is not None:
        th, tw = talent_anchor_template.shape
        print(
            f"Talent anchor enabled: template {tw}x{th}, "
            f"threshold={talent_anchor_config['match_threshold']}"
        )
    else:
        print("Talent anchor not configured; hero_name will not be returned")

    # Hero display name lookup table (EN + ZH → GSI internal name)
    dn_path = WORKSPACE / "configs" / "hero_display_names.json"
    if dn_path.exists():
        with open(dn_path) as f:
            hero_display_names = json.load(f)
        print(f"Loaded {len(hero_display_names)} hero display name entries")

    # EasyOCR reader (weights pre-baked in Docker image)
    import easyocr
    ocr_reader = easyocr.Reader(["en", "ch_sim"], gpu=True)
    print("EasyOCR reader initialized")

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


def compute_canny_edges(
    image_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    canny_low: int = 80,
    canny_high: int = 160,
) -> np.ndarray:
    """Crop image to bbox (x,y,w,h) and return Canny edges as uint8 array."""
    x, y, w, h = bbox
    crop = image_rgb[y : y + h, x : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, canny_low, canny_high)


def compute_item_offsets(
    crop_config: dict, anchor_x: int, anchor_y: int
) -> dict[str, dict]:
    """Derive (dx,dy,w,h) for each item slot relative to (anchor_x, anchor_y)."""
    out: dict[str, dict] = {}
    for name, region in crop_config["regions"].items():
        if not name.startswith("item_slot"):
            continue
        out[name] = {
            "dx": region["x"] - anchor_x,
            "dy": region["y"] - anchor_y,
            "w": region["w"],
            "h": region["h"],
        }
    return out


def save_anchor_assets(
    workspace: Path,
    edges: np.ndarray,
    anchor_bbox: tuple[int, int, int, int],
    item_offsets: dict,
    anchor_name: str = "scepter",
    match_threshold: float = 0.5,
    canny_low: int = 80,
    canny_high: int = 160,
    reference_resolution: tuple[int, int] | None = None,
) -> tuple[Path, Path]:
    """Write template PNG + offsets JSON. Returns (template_path, offsets_path)."""
    template_path = workspace / "configs" / "anchors" / f"{anchor_name}_edges.png"
    offsets_path = workspace / "configs" / "anchor_offsets.json"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(template_path), edges)
    x, y, w, h = anchor_bbox
    payload = {
        "anchor": anchor_name,
        "template_path": str(template_path.relative_to(workspace).as_posix()),
        "reference_resolution": list(reference_resolution) if reference_resolution else None,
        "anchor_bbox": {"x": x, "y": y, "w": w, "h": h},
        "match_threshold": match_threshold,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "item_offsets": item_offsets,
    }
    offsets_path.write_text(json.dumps(payload, indent=2))
    return template_path, offsets_path


def load_anchor_assets(
    workspace: Path,
    config_filename: str = "anchor_offsets.json",
) -> tuple[dict | None, np.ndarray | None]:
    """Load anchor config + template PNG from workspace, or (None, None)."""
    cfg_path = workspace / "configs" / config_filename
    if not cfg_path.exists():
        return None, None
    with open(cfg_path) as f:
        cfg = json.load(f)
    tmpl = cv2.imread(str(workspace / cfg["template_path"]), cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        return None, None
    return cfg, tmpl


def find_anchor(
    image_rgb: np.ndarray,
    anchor_cfg: dict,
    template: np.ndarray,
) -> tuple[float, int, int]:
    """Locate the anchor via Canny + matchTemplate. Returns (score, x, y)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, anchor_cfg["canny_low"], anchor_cfg["canny_high"])
    res = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return float(score), int(loc[0]), int(loc[1])


def compute_item_boxes(
    image_rgb: np.ndarray,
    anchor_cfg: dict | None,
    template: np.ndarray | None,
    scale_x: float,
    scale_y: float,
) -> tuple[dict[str, tuple[int, int, int, int]], dict]:
    """Try anchor matching; return (item_boxes, meta).

    item_boxes is empty dict on fallback (caller should use fixed coords).
    meta = {"used": bool, "score": float|None, "anchor_xy": (x,y)|None}.
    """
    meta: dict = {"used": False, "score": None, "anchor_xy": None}
    if anchor_cfg is None or template is None:
        return {}, meta
    score, ax, ay = find_anchor(image_rgb, anchor_cfg, template)
    meta["score"] = round(score, 4)
    meta["anchor_xy"] = (ax, ay)
    if score < anchor_cfg["match_threshold"]:
        return {}, meta
    img_h, img_w = image_rgb.shape[:2]
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for name, off in anchor_cfg["item_offsets"].items():
        x = ax + int(off["dx"] * scale_x)
        y = ay + int(off["dy"] * scale_y)
        w = int(off["w"] * scale_x)
        h = int(off["h"] * scale_y)
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))
        boxes[name] = (x, y, w, h)
    meta["used"] = True
    return boxes, meta


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


@router.get("/health")
def health():
    return {"status": "ok"}


def preprocess_for_ocr(crop_rgb: np.ndarray) -> np.ndarray:
    """Upscale 3× and apply CLAHE to sharpen hero name text before OCR."""
    h, w = crop_rgb.shape[:2]
    crop = cv2.resize(crop_rgb, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def resolve_hero_name(ocr_text: str) -> str | None:
    """Map OCR output (EN or ZH display name) to GSI internal hero name."""
    from difflib import get_close_matches
    if not ocr_text or not hero_display_names:
        return None
    if ocr_text in hero_display_names:
        return hero_display_names[ocr_text]
    candidates = get_close_matches(ocr_text, hero_display_names.keys(), n=1, cutoff=0.75)
    if candidates:
        return hero_display_names[candidates[0]]
    return None


def read_focused_hero(image_rgb: np.ndarray, scale_x: float, scale_y: float) -> str | None:
    """Identify the focused hero via talent-anchor OCR; return last known on failure."""
    global last_known_hero
    if talent_anchor_config is None or talent_anchor_template is None or ocr_reader is None:
        return last_known_hero
    score, ax, ay = find_anchor(image_rgb, talent_anchor_config, talent_anchor_template)
    if score < talent_anchor_config["match_threshold"]:
        return last_known_hero
    r = talent_anchor_config["hero_name_region"]
    x = max(0, ax + int(r["dx"] * scale_x))
    y = max(0, ay + int(r["dy"] * scale_y))
    w = max(1, int(r["w"] * scale_x))
    h = max(1, int(r["h"] * scale_y))
    img_h, img_w = image_rgb.shape[:2]
    x = min(x, img_w - 1); y = min(y, img_h - 1)
    w = min(w, img_w - x); h = min(h, img_h - y)
    crop = preprocess_for_ocr(image_rgb[y:y + h, x:x + w])
    results = ocr_reader.readtext(crop, detail=0, paragraph=False)
    if results:
        resolved = resolve_hero_name(results[0].strip())
        if resolved:
            last_known_hero = resolved
    return last_known_hero


@router.post("/predict")
async def predict(file: UploadFile) -> dict:
    """HTTP wrapper around :func:`predict_bytes`; reads the upload and delegates."""
    data = await file.read()
    return await predict_bytes(data)


async def predict_bytes(image_bytes: bytes) -> dict:
    """Run hero/item classification + hero-name OCR on a screenshot.

    Args:
        image_bytes: Raw bytes of a PNG/JPEG screenshot.

    Returns:
        ``{"heroes": {slot: {class, confidence}}, "items": {slot: {class, confidence}},
        "hero_name": str | None, "anchor": {used, score, anchor_xy}}``.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = image.size

    # Scale factors from reference resolution
    ref_w, ref_h = crop_config["reference_resolution"]
    scale_x = img_w / ref_w
    scale_y = img_h / ref_h

    regions = crop_config["regions"]

    # Anchor-relative item boxes (if anchor configured + matched)
    img_np = np.array(image)
    item_boxes, anchor_meta = compute_item_boxes(
        img_np, anchor_config, anchor_template, scale_x, scale_y
    )
    if anchor_meta["score"] is not None and not anchor_meta["used"]:
        print(
            f"anchor match below threshold ({anchor_meta['score']:.3f} < "
            f"{anchor_config['match_threshold']}); falling back to fixed item crops"
        )

    # Crop and preprocess
    hero_crops = []
    hero_names = []
    item_crops = []
    item_names = []

    for name, coords in regions.items():
        if name in item_boxes:
            x, y, w, h = item_boxes[name]
        else:
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
    response: dict = {"heroes": {}, "items": {}, "hero_name": None, "anchor": anchor_meta}

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

    response["hero_name"] = read_focused_hero(img_np, scale_x, scale_y)

    return response
