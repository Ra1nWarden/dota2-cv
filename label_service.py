"""FastAPI labeling service for Dota 2 evaluator ground truth.

Serves a browser-based labeling UI on a single port. Reuses the
inference-service preprocessing so model suggestions match production.
Auto-saves labels.json to disk on every change.

Usage:
    uvicorn label_service:app --host 0.0.0.0 --port 8081 --workers 1
"""

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image
from pydantic import BaseModel

from inference_service import (
    compute_canny_edges,
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
CROP_CONFIG_PATH = WORKSPACE / "configs" / "crop_config.json"
HERO_MODEL_PATH = WORKSPACE / "models" / "hero_classifier.onnx"
ITEM_MODEL_PATH = WORKSPACE / "models" / "item_classifier.onnx"
HERO_CLASSES_PATH = WORKSPACE / "configs" / "heroes_classes.json"
ITEM_CLASSES_PATH = WORKSPACE / "configs" / "items_classes.json"

HERO_PREFIXES = ("radiant_hero", "dire_hero")
IMG_EXTS = (".png", ".jpg", ".jpeg")


def is_hero(slot: str) -> bool:
    return slot.startswith(HERO_PREFIXES)


def load_class_list(path: Path) -> list[str]:
    raw = json.loads(path.read_text())
    return [raw[str(i)] for i in range(len(raw))]


def crop_screenshot(image: Image.Image, crop_config: dict):
    img_w, img_h = image.size
    ref_w, ref_h = crop_config["reference_resolution"]
    sx = img_w / ref_w
    sy = img_h / ref_h
    for name, c in crop_config["regions"].items():
        x = int(c["x"] * sx)
        y = int(c["y"] * sy)
        w = int(c["w"] * sx)
        h = int(c["h"] * sy)
        yield name, image.crop((x, y, x + w, y + h))


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

app = FastAPI(title="Dota 2 Labeler")


def _build_data_payload() -> dict:
    """Crop every screenshot, run inference, build the JSON payload."""
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

    print(f"Processing {len(screenshots)} screenshots...")
    for fname in screenshots:
        path = SCREENSHOTS_DIR / fname
        image = Image.open(path).convert("RGB")

        slot_to_crop = {}
        slot_to_processed = {}
        for slot, crop_img in crop_screenshot(image, crop_config):
            slot_to_crop[slot] = crop_img
            slot_to_processed[slot] = preprocess_crop(crop_img)

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
    }


@app.on_event("startup")
def startup():
    print(f"Workspace: {WORKSPACE}")
    print(f"Screenshots: {SCREENSHOTS_DIR}")
    print(f"Labels file: {LABELS_PATH}")

    state["crop_config"] = json.loads(CROP_CONFIG_PATH.read_text())
    state["region_names"] = list(state["crop_config"]["regions"].keys())
    state["hero_classes"] = load_class_list(HERO_CLASSES_PATH)
    state["item_classes"] = load_class_list(ITEM_CLASSES_PATH)

    print("Loading ONNX models...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    state["hero_session"] = ort.InferenceSession(
        str(HERO_MODEL_PATH), providers=providers)
    state["item_session"] = ort.InferenceSession(
        str(ITEM_MODEL_PATH), providers=providers)

    state["data_payload"] = _build_data_payload()
    print(f"Ready. {len(state['data_payload']['filenames'])} screenshots loaded.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "screenshots_dir": str(SCREENSHOTS_DIR),
        "labels_path": str(LABELS_PATH),
        "n_screenshots": len(state.get("data_payload", {}).get("filenames", [])),
    }


@app.get("/api/data")
def get_data():
    return state["data_payload"]


@app.post("/api/reload")
def reload_screenshots():
    """Re-scan screenshots dir and rebuild the cached payload."""
    state["data_payload"] = _build_data_payload()
    return {"status": "ok",
            "n_screenshots": len(state["data_payload"]["filenames"])}


class LabelsPayload(BaseModel):
    labels: dict


@app.post("/api/labels")
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


@app.get("/calibrate", response_class=HTMLResponse)
def calibrate_page():
    return CALIBRATE_HTML


@app.get("/calibrate/reference.png")
def calibrate_reference():
    if not REFERENCE_SCREENSHOT_PATH.exists():
        raise HTTPException(404, f"reference screenshot not found at {REFERENCE_SCREENSHOT_PATH}")
    return FileResponse(REFERENCE_SCREENSHOT_PATH, media_type="image/png")


@app.get("/api/calibrate/state")
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


@app.get("/api/calibrate/preview")
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


@app.post("/api/calibrate")
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
    return {
        "status": "ok",
        "template_path": str(template_path),
        "offsets_path": str(offsets_path),
        "n_item_offsets": len(item_offsets),
        "edge_density": round(edge_density, 4),
        "note": "restart the inference service to pick up the new anchor",
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dota 2 Labeler</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 16px; }
  header { position: sticky; top: 0; background: #1a1a1a; padding: 10px 0; z-index: 10; border-bottom: 1px solid #333; margin-bottom: 16px; }
  header h1 { margin: 0 0 6px; font-size: 16px; font-weight: 600; }
  .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  #progress { font-size: 13px; color: #aaa; }
  #status { font-size: 12px; color: #888; padding: 3px 8px; border-radius: 3px; min-width: 70px; text-align: center; }
  #status.saving { background: #553; color: #ffd; }
  #status.saved { background: #353; color: #cfc; }
  #status.error { background: #533; color: #fcc; }
  button.secondary { background: #555; border: 0; color: white; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; }
  button.secondary:hover { background: #666; }
  details { margin-bottom: 12px; background: #222; border-radius: 6px; padding: 6px 12px; }
  details > summary { font-weight: bold; cursor: pointer; padding: 6px 0; user-select: none; font-size: 14px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; padding: 8px 0; }
  @media (max-width: 1100px) { .grid { grid-template-columns: repeat(2, 1fr); } }
  .slot { background: #2a2a2a; padding: 8px; border-radius: 4px; border: 2px solid #2a2a2a; }
  .slot.filled { border-color: #4a8; }
  .slot.empty-set { border-color: #888; }
  .slot-name { font-size: 11px; color: #aaa; margin-bottom: 4px; font-family: monospace; }
  .slot img { display: block; margin: 0 auto 6px; max-width: 100%; max-height: 120px; image-rendering: auto; background: #000; border-radius: 3px; }
  .preds { display: flex; flex-direction: column; gap: 3px; margin-bottom: 6px; }
  .pred { background: #2d4a4a; border: 0; color: white; padding: 3px 6px; border-radius: 3px; cursor: pointer; font-size: 11px; text-align: left; display: flex; justify-content: space-between; align-items: center; font-family: monospace; }
  .pred:hover { background: #3e6868; }
  .pred .conf { color: #aef; font-size: 10px; margin-left: 6px; }
  .input-row { display: flex; gap: 3px; align-items: center; }
  .input-row input { flex: 1; min-width: 0; background: #1a1a1a; border: 1px solid #444; color: #eee; padding: 4px 6px; border-radius: 3px; font-size: 12px; font-family: monospace; }
  .empty-btn { background: #644; border: 0; color: white; padding: 4px 7px; border-radius: 3px; cursor: pointer; font-size: 11px; }
  .empty-btn:hover { background: #855; }
  .clear-btn { background: #444; border: 0; color: #ccc; padding: 4px 6px; border-radius: 3px; cursor: pointer; font-size: 11px; }
  .clear-btn:hover { background: #555; }
  #loading { color: #aaa; padding: 20px; text-align: center; font-size: 14px; }
</style>
</head>
<body>
<header>
  <h1>Dota 2 Labeler</h1>
  <div class="controls">
    <span id="progress">loading…</span>
    <span id="status">idle</span>
    <button class="secondary" id="reloadBtn" title="Re-scan screenshots dir">Reload screenshots</button>
  </div>
</header>

<div id="loading">Loading data and running inference…</div>
<div id="screenshots"></div>

<datalist id="hero_classes"></datalist>
<datalist id="item_classes"></datalist>

<script>
let DATA = null;
let state = {};
let saveTimer = null;

const HERO_PREFIXES = ["radiant_hero", "dire_hero"];
function isHero(slot) { return HERO_PREFIXES.some(p => slot.startsWith(p)); }

function setStatus(text, cls) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = cls || "";
}

function updateProgress() {
  let total = 0, filled = 0;
  for (const fname of DATA.filenames) {
    for (const slot of DATA.slot_order) {
      total++;
      if (state[fname][slot] !== "") filled++;
    }
  }
  document.getElementById("progress").textContent =
    `${filled}/${total} slots labeled (${(100*filled/total).toFixed(0)}%)`;
}

function updateSummary(fname, sumEl) {
  const filled = DATA.slot_order.filter(s => state[fname][s] !== "").length;
  sumEl.textContent = `${fname}  —  ${filled}/16 labeled`;
}

function scheduleSave() {
  setStatus("saving…", "saving");
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(doSave, 400);
}

async function doSave() {
  try {
    const res = await fetch("/api/labels", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({labels: state}),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    setStatus("saved", "saved");
  } catch (err) {
    setStatus("save error: " + err.message, "error");
  }
}

function populateDatalists() {
  const hero = document.getElementById("hero_classes");
  const item = document.getElementById("item_classes");
  hero.innerHTML = DATA.hero_classes.map(c => `<option value="${c}">`).join("");
  item.innerHTML = DATA.item_classes.map(c => `<option value="${c}">`).join("");
}

function render() {
  const root = document.getElementById("screenshots");
  root.innerHTML = "";
  for (const fname of DATA.filenames) {
    const det = document.createElement("details");
    const filled = DATA.slot_order.filter(s => state[fname][s] !== "").length;
    det.open = filled < 16;

    const sum = document.createElement("summary");
    det.appendChild(sum);
    updateSummary(fname, sum);

    const grid = document.createElement("div");
    grid.className = "grid";

    for (const slot of DATA.slot_order) {
      const card = document.createElement("div");
      card.className = "slot";
      const cur = state[fname][slot];
      if (cur === "empty") card.classList.add("empty-set");
      else if (cur !== "") card.classList.add("filled");

      const datalistId = isHero(slot) ? "hero_classes" : "item_classes";
      const predHtml = (DATA.preds[fname][slot] || []).map(([c, p]) =>
        `<button class="pred" data-cls="${c}">` +
          `<span>${c}</span><span class="conf">${(p*100).toFixed(0)}%</span>` +
        `</button>`
      ).join("");

      card.innerHTML =
        `<div class="slot-name">${slot}</div>` +
        `<img src="data:image/png;base64,${DATA.crops[fname][slot]}" alt="${slot}">` +
        `<div class="preds">${predHtml}</div>` +
        `<div class="input-row">` +
          `<input type="text" list="${datalistId}" value="${cur.replace(/"/g, "&quot;")}" placeholder="class name" spellcheck="false" autocomplete="off">` +
          `<button class="empty-btn" title="Set to 'empty'">empty</button>` +
          `<button class="clear-btn" title="Clear">×</button>` +
        `</div>`;

      const inp = card.querySelector("input");
      const setVal = (val) => {
        state[fname][slot] = val;
        inp.value = val;
        card.classList.remove("filled", "empty-set");
        if (val === "empty") card.classList.add("empty-set");
        else if (val !== "") card.classList.add("filled");
        updateSummary(fname, sum);
        updateProgress();
        scheduleSave();
      };

      inp.addEventListener("change", () => setVal(inp.value.trim()));
      inp.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { setVal(inp.value.trim()); inp.blur(); }
      });
      card.querySelector(".empty-btn").addEventListener("click", () => setVal("empty"));
      card.querySelector(".clear-btn").addEventListener("click", () => setVal(""));
      card.querySelectorAll(".pred").forEach(btn => {
        btn.addEventListener("click", () => setVal(btn.dataset.cls));
      });

      grid.appendChild(card);
    }

    det.appendChild(grid);
    root.appendChild(det);
  }
  updateProgress();
  setStatus("idle");
}

async function loadData() {
  setStatus("loading", "saving");
  const res = await fetch("/api/data");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  DATA = await res.json();
  state = JSON.parse(JSON.stringify(DATA.initial_labels));
  for (const fname of DATA.filenames) {
    if (!state[fname]) state[fname] = {};
    for (const slot of DATA.slot_order) {
      if (!(slot in state[fname])) state[fname][slot] = "";
    }
  }
  populateDatalists();
  document.getElementById("loading").style.display = "none";
  render();
}

document.getElementById("reloadBtn").addEventListener("click", async () => {
  if (!confirm("Re-scan screenshots dir? Current unsaved changes will be flushed first.")) return;
  if (saveTimer) { clearTimeout(saveTimer); await doSave(); }
  setStatus("reloading", "saving");
  try {
    const r = await fetch("/api/reload", {method: "POST"});
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    await loadData();
  } catch (err) {
    setStatus("reload error: " + err.message, "error");
  }
});

loadData().catch(err => {
  document.getElementById("loading").textContent = "Failed to load: " + err.message;
});
</script>
</body>
</html>"""


CALIBRATE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Anchor Calibration</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 12px; }
  header { padding-bottom: 8px; border-bottom: 1px solid #333; margin-bottom: 12px; }
  header h1 { margin: 0 0 4px; font-size: 16px; }
  header p { margin: 0; font-size: 12px; color: #aaa; }
  header a { color: #6cf; }
  .layout { display: flex; gap: 16px; align-items: flex-start; }
  .canvas-wrap { position: relative; flex: 1; min-width: 0; border: 1px solid #333; background: #000; }
  .canvas-wrap img { display: block; width: 100%; height: auto; user-select: none; -webkit-user-drag: none; cursor: crosshair; }
  #bbox { position: absolute; border: 2px solid #4f4; box-shadow: 0 0 0 9999px rgba(0,0,0,0.35); pointer-events: none; display: none; }
  .panel { width: 320px; flex-shrink: 0; background: #222; padding: 12px; border-radius: 6px; font-size: 13px; }
  .panel h2 { margin: 0 0 8px; font-size: 13px; text-transform: uppercase; color: #aaa; letter-spacing: 0.05em; }
  .panel section { margin-bottom: 14px; }
  .row { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
  .row label { width: 90px; font-family: monospace; color: #ccc; }
  .row input[type=number] { flex: 1; min-width: 0; background: #1a1a1a; border: 1px solid #444; color: #eee; padding: 4px 6px; border-radius: 3px; font-family: monospace; font-size: 12px; }
  .row input[type=range] { flex: 1; min-width: 0; }
  .row .val { font-family: monospace; width: 40px; text-align: right; color: #aef; }
  button { background: #355; border: 0; color: white; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 13px; margin-right: 6px; }
  button.primary { background: #485; }
  button:hover { filter: brightness(1.15); }
  button:disabled { background: #333; color: #666; cursor: not-allowed; }
  .preview { display: flex; gap: 8px; align-items: flex-start; }
  .preview > div { flex: 1; text-align: center; }
  .preview img { display: block; max-width: 100%; image-rendering: pixelated; background: #111; border: 1px solid #333; }
  .preview .lbl { font-size: 11px; color: #888; margin-bottom: 4px; }
  #status { font-size: 12px; color: #aaa; min-height: 16px; }
  #status.ok { color: #6f6; }
  #status.err { color: #f66; }
  #current { font-family: monospace; font-size: 11px; color: #aaa; white-space: pre-wrap; }
  .hint { font-size: 11px; color: #888; margin-top: 4px; }
</style>
</head>
<body>
<header>
  <h1>Anchor Calibration</h1>
  <p>Click and drag a box around the scepter+shard region between the skill bar and items. Arrow keys nudge the box by 1 px (Shift = 10). <a href="/">← back to labeler</a></p>
</header>
<div class="layout">
  <div class="canvas-wrap" id="canvasWrap">
    <img id="ref" src="/calibrate/reference.png" draggable="false">
    <div id="bbox"></div>
  </div>
  <div class="panel">
    <section>
      <h2>Bbox (image px)</h2>
      <div class="row"><label>x</label><input type="number" id="x" value="0" min="0"></div>
      <div class="row"><label>y</label><input type="number" id="y" value="0" min="0"></div>
      <div class="row"><label>w</label><input type="number" id="w" value="0" min="1"></div>
      <div class="row"><label>h</label><input type="number" id="h" value="0" min="1"></div>
      <div class="hint">Reference: <span id="refSize">…</span></div>
    </section>
    <section>
      <h2>Canny thresholds</h2>
      <div class="row"><label>low</label><input type="range" id="cannyLow" min="10" max="250" value="80"><span class="val" id="cannyLowVal">80</span></div>
      <div class="row"><label>high</label><input type="range" id="cannyHigh" min="20" max="400" value="160"><span class="val" id="cannyHighVal">160</span></div>
      <div class="row"><label>match thr</label><input type="range" id="matchThr" min="0.1" max="0.95" step="0.05" value="0.5"><span class="val" id="matchThrVal">0.50</span></div>
      <div class="row"><label>name</label><input type="text" id="anchorName" value="scepter" style="flex:1; min-width:0; background:#1a1a1a; border:1px solid #444; color:#eee; padding:4px 6px; border-radius:3px; font-family:monospace; font-size:12px;"></div>
    </section>
    <section>
      <h2>Preview</h2>
      <div class="preview">
        <div><div class="lbl">crop</div><img id="cropPrev" alt="crop"></div>
        <div><div class="lbl">canny edges</div><img id="edgesPrev" alt="edges"></div>
      </div>
      <div class="hint" id="previewHint"></div>
    </section>
    <section>
      <button id="previewBtn">Preview</button>
      <button id="saveBtn" class="primary" disabled>Save</button>
      <div id="status"></div>
    </section>
    <section>
      <h2>Currently saved</h2>
      <div id="current">(loading…)</div>
    </section>
  </div>
</div>
<script>
const img = document.getElementById("ref");
const wrap = document.getElementById("canvasWrap");
const bboxEl = document.getElementById("bbox");
const xIn = document.getElementById("x"), yIn = document.getElementById("y");
const wIn = document.getElementById("w"), hIn = document.getElementById("h");
const cannyLow = document.getElementById("cannyLow");
const cannyHigh = document.getElementById("cannyHigh");
const cannyLowVal = document.getElementById("cannyLowVal");
const cannyHighVal = document.getElementById("cannyHighVal");
const matchThr = document.getElementById("matchThr");
const matchThrVal = document.getElementById("matchThrVal");
const anchorName = document.getElementById("anchorName");
const cropPrev = document.getElementById("cropPrev");
const edgesPrev = document.getElementById("edgesPrev");
const previewHint = document.getElementById("previewHint");
const status = document.getElementById("status");
const saveBtn = document.getElementById("saveBtn");
const refSize = document.getElementById("refSize");
const currentBox = document.getElementById("current");

let bbox = null;  // {x, y, w, h} in image-natural pixels
let dragStart = null;
let displayScale = 1;

function imgToDisplay(v) { return v * displayScale; }
function displayToImg(v) { return Math.round(v / displayScale); }

function recomputeScale() {
  if (!img.naturalWidth) return;
  displayScale = img.clientWidth / img.naturalWidth;
}

function clampBbox(b) {
  const W = img.naturalWidth, H = img.naturalHeight;
  let x = Math.max(0, Math.min(b.x, W - 1));
  let y = Math.max(0, Math.min(b.y, H - 1));
  let w = Math.max(1, Math.min(b.w, W - x));
  let h = Math.max(1, Math.min(b.h, H - y));
  return {x, y, w, h};
}

function setBbox(b, fromInput) {
  bbox = clampBbox(b);
  if (!fromInput) {
    xIn.value = bbox.x; yIn.value = bbox.y;
    wIn.value = bbox.w; hIn.value = bbox.h;
  }
  recomputeScale();
  bboxEl.style.display = "block";
  bboxEl.style.left = imgToDisplay(bbox.x) + "px";
  bboxEl.style.top = imgToDisplay(bbox.y) + "px";
  bboxEl.style.width = imgToDisplay(bbox.w) + "px";
  bboxEl.style.height = imgToDisplay(bbox.h) + "px";
  saveBtn.disabled = false;
}

function eventToImageCoords(ev) {
  const rect = img.getBoundingClientRect();
  const dx = ev.clientX - rect.left;
  const dy = ev.clientY - rect.top;
  return {x: displayToImg(dx), y: displayToImg(dy)};
}

img.addEventListener("mousedown", (ev) => {
  ev.preventDefault();
  recomputeScale();
  dragStart = eventToImageCoords(ev);
});

window.addEventListener("mousemove", (ev) => {
  if (!dragStart) return;
  const cur = eventToImageCoords(ev);
  const x = Math.min(dragStart.x, cur.x);
  const y = Math.min(dragStart.y, cur.y);
  const w = Math.abs(cur.x - dragStart.x);
  const h = Math.abs(cur.y - dragStart.y);
  if (w > 0 && h > 0) setBbox({x, y, w, h});
});

window.addEventListener("mouseup", () => { dragStart = null; });

window.addEventListener("keydown", (ev) => {
  if (!bbox) return;
  if (document.activeElement && document.activeElement.tagName === "INPUT") return;
  const step = ev.shiftKey ? 10 : 1;
  let {x, y, w, h} = bbox;
  if (ev.key === "ArrowLeft")  x -= step;
  else if (ev.key === "ArrowRight") x += step;
  else if (ev.key === "ArrowUp")    y -= step;
  else if (ev.key === "ArrowDown")  y += step;
  else return;
  ev.preventDefault();
  setBbox({x, y, w, h});
});

[xIn, yIn, wIn, hIn].forEach(el => el.addEventListener("input", () => {
  setBbox({
    x: parseInt(xIn.value) || 0,
    y: parseInt(yIn.value) || 0,
    w: parseInt(wIn.value) || 1,
    h: parseInt(hIn.value) || 1,
  }, true);
}));

cannyLow.addEventListener("input", () => cannyLowVal.textContent = cannyLow.value);
cannyHigh.addEventListener("input", () => cannyHighVal.textContent = cannyHigh.value);
matchThr.addEventListener("input", () => matchThrVal.textContent = parseFloat(matchThr.value).toFixed(2));

document.getElementById("previewBtn").addEventListener("click", async () => {
  if (!bbox) { status.textContent = "draw a box first"; status.className = "err"; return; }
  status.textContent = "rendering preview…"; status.className = "";
  const qs = new URLSearchParams({
    x: bbox.x, y: bbox.y, w: bbox.w, h: bbox.h,
    canny_low: cannyLow.value, canny_high: cannyHigh.value,
  });
  const cropUrl = "/api/calibrate/preview?" + qs + "&kind=crop&t=" + Date.now();
  const edgesUrl = "/api/calibrate/preview?" + qs + "&kind=edges&t=" + Date.now();
  cropPrev.src = cropUrl;
  edgesPrev.src = edgesUrl;
  edgesPrev.onload = () => {
    status.textContent = "preview ready"; status.className = "ok";
    previewHint.textContent = "Edges should trace the scepter/shard silhouette clearly. If too sparse, lower thresholds. If noisy, raise them.";
  };
  edgesPrev.onerror = () => { status.textContent = "preview failed"; status.className = "err"; };
});

saveBtn.addEventListener("click", async () => {
  if (!bbox) return;
  status.textContent = "saving…"; status.className = "";
  try {
    const res = await fetch("/api/calibrate", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({
        x: bbox.x, y: bbox.y, w: bbox.w, h: bbox.h,
        canny_low: parseInt(cannyLow.value),
        canny_high: parseInt(cannyHigh.value),
        match_threshold: parseFloat(matchThr.value),
        anchor_name: anchorName.value || "scepter",
      }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    status.textContent = `saved ${data.n_item_offsets} item offsets · edge density ${(data.edge_density*100).toFixed(1)}%. Restart inference to pick up.`;
    status.className = "ok";
    loadState();
  } catch (err) {
    status.textContent = "save failed: " + err.message;
    status.className = "err";
  }
});

async function loadState() {
  const res = await fetch("/api/calibrate/state");
  const data = await res.json();
  if (data.reference_size && data.reference_size[0]) {
    refSize.textContent = data.reference_size[0] + " × " + data.reference_size[1];
  }
  if (data.current) {
    const c = data.current;
    const b = c.anchor_bbox;
    currentBox.textContent =
      `anchor: ${c.anchor}\nbbox:   x=${b.x} y=${b.y} w=${b.w} h=${b.h}\n` +
      `canny:  low=${c.canny_low} high=${c.canny_high}\n` +
      `thresh: ${c.match_threshold}\n` +
      `items:  ${Object.keys(c.item_offsets).length}`;
    if (!bbox) {
      cannyLow.value = c.canny_low; cannyLowVal.textContent = c.canny_low;
      cannyHigh.value = c.canny_high; cannyHighVal.textContent = c.canny_high;
      matchThr.value = c.match_threshold; matchThrVal.textContent = c.match_threshold.toFixed(2);
      anchorName.value = c.anchor;
      img.addEventListener("load", () => { recomputeScale(); setBbox(b); }, {once: true});
      if (img.complete && img.naturalWidth) { recomputeScale(); setBbox(b); }
    }
  } else {
    currentBox.textContent = "(none — no anchor configured yet)";
  }
}

img.addEventListener("load", recomputeScale);
window.addEventListener("resize", () => { recomputeScale(); if (bbox) setBbox(bbox); });
loadState();
</script>
</body>
</html>"""
