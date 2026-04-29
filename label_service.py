"""FastAPI labeling service for Dota 2 evaluator ground truth.

Serves a browser-based labeling UI on a single port. Reuses the
inference-service preprocessing so model suggestions match production.
Auto-saves labels.json to disk on every change.

Mounted as a router at /labeler by main.py; not run as a standalone service.
"""

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image
from pydantic import BaseModel

import inference_service
from inference_service import (
    compute_canny_edges,
    compute_item_boxes,
    compute_item_offsets,
    load_anchor_assets,
    preprocess_crop,
    save_anchor_assets,
)


WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
SCREENSHOTS_DIR = Path(os.environ.get(
    "SCREENSHOTS_DIR", WORKSPACE / "data" / "test_screenshots"))
LABELS_PATH = Path(os.environ.get(
    "LABELS_PATH", WORKSPACE / "data" / "test_screenshots" / "labels.json"))
TOPK = int(os.environ.get("TOPK", "3"))

REFERENCE_SCREENSHOT_PATH = Path(os.environ.get(
    "REFERENCE_SCREENSHOT", WORKSPACE / "data" / "reference_screenshot.png"))

WEB_DIR = Path(__file__).parent / "web" / "labeler"

HERO_PREFIXES = ("radiant_hero", "dire_hero")
IMG_EXTS = (".png", ".jpg", ".jpeg")


def is_hero(slot: str) -> bool:
    return slot.startswith(HERO_PREFIXES)


def crop_screenshot(image: Image.Image, crop_config: dict,
                    anchor_cfg=None, anchor_template=None):
    """Yield (slot_name, crop, anchor_meta) per region. Uses anchor for items
    when configured + matched; falls back to fixed coords otherwise."""
    img_w, img_h = image.size
    ref_w, ref_h = crop_config["reference_resolution"]
    sx = img_w / ref_w
    sy = img_h / ref_h
    img_np = np.array(image)
    item_boxes, anchor_meta = compute_item_boxes(
        img_np, anchor_cfg, anchor_template, sx, sy)
    for name, c in crop_config["regions"].items():
        if name in item_boxes:
            x, y, w, h = item_boxes[name]
        else:
            x = int(c["x"] * sx)
            y = int(c["y"] * sy)
            w = int(c["w"] * sx)
            h = int(c["h"] * sy)
        yield name, image.crop((x, y, x + w, y + h)), anchor_meta


def topk_predict(session, crops, class_names, k=3):
    if not crops:
        return []
    batch = np.stack(crops).astype(np.float32)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: batch})[0]
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    out = []
    for prob in probs:
        idx = prob.argsort()[-k:][::-1]
        out.append([(class_names[int(i)], float(prob[int(i)])) for i in idx])
    return out


def crop_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Cached service state
state: dict = {}
file_lock = Lock()

router = APIRouter()


def _build_data_payload() -> dict:
    """Crop every screenshot, run inference, build the JSON payload."""
    # Refresh anchor refs so calibrations applied via /api/calibrate are picked up on the next page load.
    state["anchor_cfg"] = inference_service.anchor_config
    state["anchor_template"] = inference_service.anchor_template
    if state["anchor_template"] is not None:
        print(f"Anchor enabled: {state['anchor_cfg']['anchor']!r} "
              f"(threshold {state['anchor_cfg']['match_threshold']})")
    else:
        print("Anchor not configured; labeler will show fixed-coord crops")

    crop_config = state["crop_config"]
    region_names = state["region_names"]
    hero_classes = state["hero_classes"]
    item_classes = state["item_classes"]

    if not SCREENSHOTS_DIR.is_dir():
        raise FileNotFoundError(f"Screenshots dir not found: {SCREENSHOTS_DIR}")

    screenshots = sorted(
        f.name for f in SCREENSHOTS_DIR.iterdir()
        if f.suffix.lower() in IMG_EXTS
    )

    if LABELS_PATH.exists():
        labels = json.loads(LABELS_PATH.read_text())
    else:
        labels = {}
    for fname in screenshots:
        labels.setdefault(fname, {})
        for slot in region_names:
            labels[fname].setdefault(slot, "")

    hero_slots = [s for s in region_names if is_hero(s)]
    item_slots = [s for s in region_names if not is_hero(s)]

    crops_b64 = {}
    preds = {}
    anchor_per_screenshot: dict[str, dict] = {}

    print(f"Processing {len(screenshots)} screenshots...")
    for fname in screenshots:
        path = SCREENSHOTS_DIR / fname
        image = Image.open(path).convert("RGB")

        slot_to_crop = {}
        slot_to_processed = {}
        anchor_meta_for_file: dict | None = None
        for slot, crop_img, anchor_meta in crop_screenshot(
                image, crop_config,
                state.get("anchor_cfg"), state.get("anchor_template")):
            slot_to_crop[slot] = crop_img
            slot_to_processed[slot] = preprocess_crop(crop_img)
            anchor_meta_for_file = anchor_meta
        anchor_per_screenshot[fname] = anchor_meta_for_file or {
            "used": False, "score": None, "anchor_xy": None}

        ht = topk_predict(state["hero_session"],
                          [slot_to_processed[s] for s in hero_slots],
                          hero_classes, k=TOPK)
        it = topk_predict(state["item_session"],
                          [slot_to_processed[s] for s in item_slots],
                          item_classes, k=TOPK)

        slot_preds = {}
        for s, t in zip(hero_slots, ht):
            slot_preds[s] = t
        for s, t in zip(item_slots, it):
            slot_preds[s] = t
        preds[fname] = slot_preds

        crops_b64[fname] = {s: crop_to_b64(slot_to_crop[s]) for s in region_names}
        print(f"  {fname}")

    return {
        "filenames": screenshots,
        "slot_order": region_names,
        "hero_classes": hero_classes,
        "item_classes": item_classes,
        "crops": crops_b64,
        "preds": preds,
        "initial_labels": labels,
        "anchor_meta": anchor_per_screenshot,
        "anchor_name": (state["anchor_cfg"]["anchor"]
                        if state.get("anchor_cfg") else None),
    }


def startup() -> None:
    """Build the cached labeling payload. Called once by main.py's lifespan after inference_service.startup."""
    print(f"Workspace: {WORKSPACE}")
    print(f"Screenshots: {SCREENSHOTS_DIR}")
    print(f"Labels file: {LABELS_PATH}")
    state["crop_config"] = inference_service.crop_config
    state["region_names"] = list(inference_service.crop_config["regions"].keys())
    state["hero_classes"] = inference_service.hero_classes
    state["item_classes"] = inference_service.item_classes
    state["hero_session"] = inference_service.hero_session
    state["item_session"] = inference_service.item_session
    state["data_payload"] = _build_data_payload()
    print(f"Ready. {len(state['data_payload']['filenames'])} screenshots loaded.")


@router.get("/health")
def health():
    return {
        "status": "ok",
        "screenshots_dir": str(SCREENSHOTS_DIR),
        "labels_path": str(LABELS_PATH),
        "n_screenshots": len(state.get("data_payload", {}).get("filenames", [])),
    }


@router.get("/api/data")
def get_data():
    return state["data_payload"]


@router.post("/api/reload")
def reload_screenshots():
    """Re-scan screenshots dir and rebuild the cached payload."""
    state["data_payload"] = _build_data_payload()
    return {"status": "ok",
            "n_screenshots": len(state["data_payload"]["filenames"])}


class LabelsPayload(BaseModel):
    labels: dict


@router.post("/api/labels")
def save_labels(payload: LabelsPayload):
    """Atomic write labels.json (temp file + rename)."""
    with file_lock:
        LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = LABELS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload.labels, indent=2))
        tmp.replace(LABELS_PATH)
    # Keep cached initial_labels in sync so a reload of the page
    # mid-session still shows the latest values.
    state["data_payload"]["initial_labels"] = payload.labels
    return {"status": "ok", "path": str(LABELS_PATH)}


def _read_reference_rgb() -> np.ndarray:
    if not REFERENCE_SCREENSHOT_PATH.exists():
        raise HTTPException(404, f"reference screenshot not found at {REFERENCE_SCREENSHOT_PATH}")
    img = cv2.imread(str(REFERENCE_SCREENSHOT_PATH), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, f"failed to read {REFERENCE_SCREENSHOT_PATH}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _png_response(arr: np.ndarray) -> Response:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise HTTPException(500, "failed to encode PNG")
    return Response(content=buf.tobytes(), media_type="image/png")


@router.get("/calibrate", include_in_schema=False, response_class=HTMLResponse)
def calibrate_page():
    return FileResponse(WEB_DIR / "calibrate.html")


@router.get("/calibrate/reference.png", include_in_schema=False)
def calibrate_reference():
    if not REFERENCE_SCREENSHOT_PATH.exists():
        raise HTTPException(404, f"reference screenshot not found at {REFERENCE_SCREENSHOT_PATH}")
    return FileResponse(REFERENCE_SCREENSHOT_PATH, media_type="image/png")


@router.get("/api/calibrate/state")
def calibrate_state():
    cfg, _ = load_anchor_assets(WORKSPACE)
    img_h, img_w = (None, None)
    if REFERENCE_SCREENSHOT_PATH.exists():
        img = cv2.imread(str(REFERENCE_SCREENSHOT_PATH))
        if img is not None:
            img_h, img_w = img.shape[:2]
    return {
        "current": cfg,
        "reference_size": [img_w, img_h] if img_w else None,
        "reference_path": str(REFERENCE_SCREENSHOT_PATH),
    }


@router.get("/api/calibrate/preview")
def calibrate_preview(x: int, y: int, w: int, h: int,
                      canny_low: int = 80, canny_high: int = 160,
                      kind: str = "edges"):
    """kind=edges → Canny PNG; kind=crop → raw RGB crop PNG."""
    if w <= 0 or h <= 0:
        raise HTTPException(400, "w and h must be positive")
    img_rgb = _read_reference_rgb()
    img_h, img_w = img_rgb.shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        raise HTTPException(400, f"bbox ({x},{y},{w},{h}) out of bounds for {img_w}x{img_h}")
    if kind == "crop":
        bgr = cv2.cvtColor(img_rgb[y : y + h, x : x + w], cv2.COLOR_RGB2BGR)
        return _png_response(bgr)
    edges = compute_canny_edges(img_rgb, (x, y, w, h), canny_low, canny_high)
    return _png_response(edges)


class CalibratePayload(BaseModel):
    x: int
    y: int
    w: int
    h: int
    canny_low: int = 80
    canny_high: int = 160
    match_threshold: float = 0.5
    anchor_name: str = "scepter"


@router.post("/api/calibrate")
def calibrate_save(payload: CalibratePayload):
    if payload.w <= 0 or payload.h <= 0:
        raise HTTPException(400, "w and h must be positive")
    img_rgb = _read_reference_rgb()
    img_h, img_w = img_rgb.shape[:2]
    if (payload.x < 0 or payload.y < 0
            or payload.x + payload.w > img_w
            or payload.y + payload.h > img_h):
        raise HTTPException(
            400, f"bbox out of bounds for {img_w}x{img_h}")
    edges = compute_canny_edges(
        img_rgb, (payload.x, payload.y, payload.w, payload.h),
        payload.canny_low, payload.canny_high,
    )
    edge_density = float((edges > 0).sum()) / edges.size
    item_offsets = compute_item_offsets(state["crop_config"], payload.x, payload.y)
    if not item_offsets:
        raise HTTPException(500, "crop_config has no item_slot_* regions")
    template_path, offsets_path = save_anchor_assets(
        WORKSPACE, edges,
        (payload.x, payload.y, payload.w, payload.h),
        item_offsets,
        anchor_name=payload.anchor_name,
        match_threshold=payload.match_threshold,
        canny_low=payload.canny_low, canny_high=payload.canny_high,
        reference_resolution=(img_w, img_h),
    )
    # Refresh inference_service's anchor globals so /inference/predict
    # immediately picks up the new calibration.
    inference_service.anchor_config, inference_service.anchor_template = \
        inference_service.load_anchor_assets(WORKSPACE)
    # Rebuild the labeling payload so the labeler shows crops via the
    # new anchor on its next page load (no manual reload needed).
    state["data_payload"] = _build_data_payload()
    return {
        "status": "ok",
        "template_path": str(template_path),
        "offsets_path": str(offsets_path),
        "n_item_offsets": len(item_offsets),
        "edge_density": round(edge_density, 4),
        "note": "calibration applied; new anchor live for /inference/predict",
    }


def save_talent_anchor_assets(
    workspace: Path,
    edges: np.ndarray,
    anchor_bbox: tuple[int, int, int, int],
    name_bbox: tuple[int, int, int, int],
    match_threshold: float = 0.5,
    canny_low: int = 80,
    canny_high: int = 160,
    reference_resolution: tuple[int, int] | None = None,
) -> tuple[Path, Path]:
    """Write talent anchor template PNG + talent_anchor_offsets.json."""
    template_path = workspace / "configs" / "anchors" / "talent_edges.png"
    offsets_path = workspace / "configs" / "talent_anchor_offsets.json"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(template_path), edges)
    ax, ay, aw, ah = anchor_bbox
    nx, ny, nw, nh = name_bbox
    payload = {
        "anchor": "talent",
        "template_path": str(template_path.relative_to(workspace).as_posix()),
        "reference_resolution": list(reference_resolution) if reference_resolution else None,
        "anchor_bbox": {"x": ax, "y": ay, "w": aw, "h": ah},
        "match_threshold": match_threshold,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "hero_name_region": {
            "dx": nx - ax,
            "dy": ny - ay,
            "w": nw,
            "h": nh,
        },
    }
    offsets_path.write_text(json.dumps(payload, indent=2))
    return template_path, offsets_path


class TalentCalibratePayload(BaseModel):
    anchor_x: int
    anchor_y: int
    anchor_w: int
    anchor_h: int
    name_x: int
    name_y: int
    name_w: int
    name_h: int
    canny_low: int = 80
    canny_high: int = 160
    match_threshold: float = 0.5


@router.get("/talent", include_in_schema=False, response_class=HTMLResponse)
def calibrate_talent_page():
    return FileResponse(WEB_DIR / "talent.html")


@router.get("/api/calibrate/talent/state")
def calibrate_talent_state():
    cfg_path = WORKSPACE / "configs" / "talent_anchor_offsets.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    img_h, img_w = (None, None)
    if REFERENCE_SCREENSHOT_PATH.exists():
        img = cv2.imread(str(REFERENCE_SCREENSHOT_PATH))
        if img is not None:
            img_h, img_w = img.shape[:2]
    return {
        "current": cfg,
        "reference_size": [img_w, img_h] if img_w else None,
    }


@router.post("/api/calibrate/talent")
def calibrate_talent_save(payload: TalentCalibratePayload):
    for name, x, y, w, h in [
        ("anchor", payload.anchor_x, payload.anchor_y, payload.anchor_w, payload.anchor_h),
        ("name",   payload.name_x,   payload.name_y,   payload.name_w,   payload.name_h),
    ]:
        if w <= 0 or h <= 0:
            raise HTTPException(400, f"{name} bbox: w and h must be positive")
    img_rgb = _read_reference_rgb()
    img_h, img_w = img_rgb.shape[:2]
    for name, x, y, w, h in [
        ("anchor", payload.anchor_x, payload.anchor_y, payload.anchor_w, payload.anchor_h),
        ("name",   payload.name_x,   payload.name_y,   payload.name_w,   payload.name_h),
    ]:
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            raise HTTPException(400, f"{name} bbox out of bounds for {img_w}x{img_h}")
    edges = compute_canny_edges(
        img_rgb,
        (payload.anchor_x, payload.anchor_y, payload.anchor_w, payload.anchor_h),
        payload.canny_low, payload.canny_high,
    )
    edge_density = float((edges > 0).sum()) / edges.size
    template_path, offsets_path = save_talent_anchor_assets(
        WORKSPACE, edges,
        (payload.anchor_x, payload.anchor_y, payload.anchor_w, payload.anchor_h),
        (payload.name_x,   payload.name_y,   payload.name_w,   payload.name_h),
        match_threshold=payload.match_threshold,
        canny_low=payload.canny_low,
        canny_high=payload.canny_high,
        reference_resolution=(img_w, img_h),
    )
    # Refresh inference_service's talent anchor globals so /inference/predict
    # immediately picks up the new calibration.
    inference_service.talent_anchor_config, inference_service.talent_anchor_template = \
        inference_service.load_anchor_assets(WORKSPACE, config_filename="talent_anchor_offsets.json")
    return {
        "status": "ok",
        "template_path": str(template_path),
        "offsets_path": str(offsets_path),
        "edge_density": round(edge_density, 4),
        "hero_name_offset": {
            "dx": payload.name_x - payload.anchor_x,
            "dy": payload.name_y - payload.anchor_y,
            "w": payload.name_w,
            "h": payload.name_h,
        },
        "note": "calibration applied; new talent anchor live for /inference/predict",
    }


@router.get("/", include_in_schema=False, response_class=HTMLResponse)
def index():
    return FileResponse(WEB_DIR / "index.html")
