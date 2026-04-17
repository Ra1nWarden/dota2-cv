#!/usr/bin/env python3
"""Build a self-contained HTML labeling tool for the Dota 2 evaluator.

Reads all screenshots in --screenshots, crops each via crop_config.json,
runs both ONNX models to get top-3 predictions per crop, and writes a
single labeling.html with everything embedded (base64 crops, model
suggestions as quick-click buttons, searchable dropdown of all classes
per slot type, "empty" button, localStorage auto-save).

Open the file in any browser, label, then "Download labels.json" or
"Copy JSON to clipboard" — replace the existing labels.json with it.

Usage:
    python scripts/build_label_ui.py \\
        --screenshots /workspace/data/test_screenshots \\
        --labels /workspace/data/test_screenshots/labels.json \\
        --output /workspace/data/test_screenshots/labeling.html
"""

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference_service import preprocess_crop  # noqa: E402

HERO_PREFIXES = ("radiant_hero", "dire_hero")
IMG_EXTS = (".png", ".jpg", ".jpeg")


def is_hero(slot: str) -> bool:
    return slot.startswith(HERO_PREFIXES)


def load_class_list(path: str) -> list[str]:
    raw = json.loads(Path(path).read_text())
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


def topk(session, crops, class_names, k=3):
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


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dota 2 Labeler</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 16px; }
  header { position: sticky; top: 0; background: #1a1a1a; padding: 10px 0; z-index: 10; border-bottom: 1px solid #333; margin-bottom: 16px; }
  header h1 { margin: 0 0 6px; font-size: 16px; font-weight: 600; }
  .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  #progress { font-size: 13px; color: #aaa; flex: 1; min-width: 200px; }
  button.primary { background: #4a8; border: 0; color: white; padding: 7px 14px; border-radius: 4px; cursor: pointer; font-size: 13px; }
  button.primary:hover { background: #5b9; }
  button.secondary { background: #555; border: 0; color: white; padding: 7px 14px; border-radius: 4px; cursor: pointer; font-size: 13px; }
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
  .copied-flash { background: #4a8 !important; }
</style>
</head>
<body>
<header>
  <h1>Dota 2 Labeler — __SCREENSHOT_COUNT__ screenshots × 16 slots</h1>
  <div class="controls">
    <span id="progress">…</span>
    <button class="primary" id="saveBtn">Download labels.json</button>
    <button class="secondary" id="copyBtn">Copy JSON</button>
    <button class="secondary" id="resetBtn" title="Forget local progress and reload from baked-in labels">Reset local</button>
  </div>
</header>

<div id="screenshots"></div>

<datalist id="hero_classes">
__HERO_OPTIONS__
</datalist>
<datalist id="item_classes">
__ITEM_OPTIONS__
</datalist>

<script>
const DATA = __DATA_JSON__;
const STORAGE_KEY = "dota2_labels_state_v1";

let state = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null") || JSON.parse(JSON.stringify(DATA.initial_labels));

// Make sure every screenshot/slot exists in state (handles new screenshots added since last session)
for (const fname of DATA.filenames) {
  if (!state[fname]) state[fname] = {};
  for (const slot of DATA.slot_order) {
    if (!(slot in state[fname])) state[fname][slot] = "";
  }
}

function isHero(slot) { return slot.startsWith("radiant_hero") || slot.startsWith("dire_hero"); }

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  updateProgress();
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
        saveState();
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
}

function downloadJson() {
  const blob = new Blob([JSON.stringify(state, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "labels.json";
  a.click();
  URL.revokeObjectURL(url);
}

async function copyJson() {
  await navigator.clipboard.writeText(JSON.stringify(state, null, 2));
  const btn = document.getElementById("copyBtn");
  const orig = btn.textContent;
  btn.textContent = "Copied!";
  btn.classList.add("copied-flash");
  setTimeout(() => { btn.textContent = orig; btn.classList.remove("copied-flash"); }, 1500);
}

function resetLocal() {
  if (!confirm("Discard local progress and reload baked-in labels?")) return;
  localStorage.removeItem(STORAGE_KEY);
  state = JSON.parse(JSON.stringify(DATA.initial_labels));
  for (const fname of DATA.filenames) {
    if (!state[fname]) state[fname] = {};
    for (const slot of DATA.slot_order) {
      if (!(slot in state[fname])) state[fname][slot] = "";
    }
  }
  render();
}

document.getElementById("saveBtn").addEventListener("click", downloadJson);
document.getElementById("copyBtn").addEventListener("click", copyJson);
document.getElementById("resetBtn").addEventListener("click", resetLocal);
render();
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--screenshots", required=True)
    p.add_argument("--crop-config", default="configs/crop_config.json")
    p.add_argument("--hero-model", default="models/hero_classifier.onnx")
    p.add_argument("--item-model", default="models/item_classifier.onnx")
    p.add_argument("--hero-classes", default="configs/heroes_classes.json")
    p.add_argument("--item-classes", default="configs/items_classes.json")
    p.add_argument("--labels", default=None,
                   help="Existing labels.json to pre-fill from (optional)")
    p.add_argument("--output", required=True, help="Path to write labeling.html")
    p.add_argument("--topk", type=int, default=3,
                   help="How many model predictions to show per crop (default 3)")
    args = p.parse_args()

    crop_config = json.loads(Path(args.crop_config).read_text())
    region_names = list(crop_config["regions"].keys())
    hero_classes = load_class_list(args.hero_classes)
    item_classes = load_class_list(args.item_classes)

    screenshot_dir = Path(args.screenshots)
    screenshots = sorted(
        f.name for f in screenshot_dir.iterdir()
        if f.suffix.lower() in IMG_EXTS
    )
    if not screenshots:
        print(f"No screenshots in {screenshot_dir}", file=sys.stderr)
        sys.exit(1)

    initial_labels = {}
    if args.labels and Path(args.labels).exists():
        initial_labels = json.loads(Path(args.labels).read_text())
    for fname in screenshots:
        initial_labels.setdefault(fname, {})
        for slot in region_names:
            initial_labels[fname].setdefault(slot, "")

    print("Loading ONNX models...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    hero_session = ort.InferenceSession(args.hero_model, providers=providers)
    item_session = ort.InferenceSession(args.item_model, providers=providers)

    hero_slots = [s for s in region_names if is_hero(s)]
    item_slots = [s for s in region_names if not is_hero(s)]

    crops_b64 = {}
    preds = {}

    print(f"Processing {len(screenshots)} screenshots...")
    for fname in screenshots:
        path = screenshot_dir / fname
        image = Image.open(path).convert("RGB")

        slot_to_crop = {}
        slot_to_processed = {}
        for slot, crop_img in crop_screenshot(image, crop_config):
            slot_to_crop[slot] = crop_img
            slot_to_processed[slot] = preprocess_crop(crop_img)

        hero_topk = topk(hero_session,
                         [slot_to_processed[s] for s in hero_slots],
                         hero_classes, k=args.topk)
        item_topk = topk(item_session,
                         [slot_to_processed[s] for s in item_slots],
                         item_classes, k=args.topk)

        slot_preds = {}
        for s, t in zip(hero_slots, hero_topk):
            slot_preds[s] = t
        for s, t in zip(item_slots, item_topk):
            slot_preds[s] = t
        preds[fname] = slot_preds

        crops_b64[fname] = {s: crop_to_b64(slot_to_crop[s]) for s in region_names}
        print(f"  {fname}")

    data_payload = {
        "filenames": screenshots,
        "slot_order": region_names,
        "crops": crops_b64,
        "preds": preds,
        "initial_labels": initial_labels,
    }

    hero_options = "\n".join(f'<option value="{c}">' for c in hero_classes)
    item_options = "\n".join(f'<option value="{c}">' for c in item_classes)

    # Use json.dumps then escape </script for safe embedding inside <script> tag.
    data_json = json.dumps(data_payload).replace("</", "<\\/")

    html = (HTML_TEMPLATE
            .replace("__SCREENSHOT_COUNT__", str(len(screenshots)))
            .replace("__HERO_OPTIONS__", hero_options)
            .replace("__ITEM_OPTIONS__", item_options)
            .replace("__DATA_JSON__", data_json))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nWrote {out_path} ({size_mb:.1f} MB)")
    print("Open it in a browser, label, then click 'Download labels.json' "
          "and replace the existing labels.json with the downloaded file.")


if __name__ == "__main__":
    main()
