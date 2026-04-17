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

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

from inference_service import preprocess_crop


WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
SCREENSHOTS_DIR = Path(os.environ.get(
    "SCREENSHOTS_DIR", WORKSPACE / "data" / "test_screenshots"))
LABELS_PATH = Path(os.environ.get(
    "LABELS_PATH", WORKSPACE / "data" / "test_screenshots" / "labels.json"))
TOPK = int(os.environ.get("TOPK", "3"))

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
