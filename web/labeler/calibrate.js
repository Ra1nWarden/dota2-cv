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
  const cropUrl = "api/calibrate/preview?" + qs + "&kind=crop&t=" + Date.now();
  const edgesUrl = "api/calibrate/preview?" + qs + "&kind=edges&t=" + Date.now();
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
    const res = await fetch("api/calibrate", {
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
    status.textContent = `saved ${data.n_item_offsets} item offsets · edge density ${(data.edge_density*100).toFixed(1)}%. Calibration is live.`;
    status.className = "ok";
    loadState();
  } catch (err) {
    status.textContent = "save failed: " + err.message;
    status.className = "err";
  }
});

async function loadState() {
  const res = await fetch("api/calibrate/state");
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
