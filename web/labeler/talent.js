const img = document.getElementById("ref");
const ax = document.getElementById("ax"), ay = document.getElementById("ay");
const aw = document.getElementById("aw"), ah = document.getElementById("ah");
const nx = document.getElementById("nx"), ny = document.getElementById("ny");
const nw = document.getElementById("nw"), nh = document.getElementById("nh");
const cannyLow = document.getElementById("cannyLow");
const cannyHigh = document.getElementById("cannyHigh");
const matchThr = document.getElementById("matchThr");
const cannyLowVal = document.getElementById("cannyLowVal");
const cannyHighVal = document.getElementById("cannyHighVal");
const matchThrVal = document.getElementById("matchThrVal");
const edgesPrev = document.getElementById("edgesPrev");
const cropPrev = document.getElementById("cropPrev");
const status = document.getElementById("status");
const saveBtn = document.getElementById("saveBtn");
const anchorBox = document.getElementById("anchorBox");
const nameBox = document.getElementById("nameBox");
const currentEl = document.getElementById("current");

let activeStep = 1;  // 1 = drawing anchor, 2 = drawing name label
let dragStart = null;
let displayScale = 1;
let anchorBbox = null, nameBbox = null;

function recomputeScale() {
  if (!img.naturalWidth) return;
  displayScale = img.clientWidth / img.naturalWidth;
}
function imgToDisplay(v) { return v * displayScale; }
function displayToImg(v) { return Math.round(v / displayScale); }

function clamp(b) {
  const W = img.naturalWidth, H = img.naturalHeight;
  let x = Math.max(0, Math.min(b.x, W - 1));
  let y = Math.max(0, Math.min(b.y, H - 1));
  let w = Math.max(1, Math.min(b.w, W - x));
  let h = Math.max(1, Math.min(b.h, H - y));
  return {x, y, w, h};
}

function renderBox(el, b) {
  if (!b) { el.style.display = "none"; return; }
  el.style.display = "block";
  el.style.left = imgToDisplay(b.x) + "px";
  el.style.top  = imgToDisplay(b.y) + "px";
  el.style.width  = imgToDisplay(b.w) + "px";
  el.style.height = imgToDisplay(b.h) + "px";
}

function setAnchor(b, fromInput) {
  anchorBbox = clamp(b);
  if (!fromInput) { ax.value = anchorBbox.x; ay.value = anchorBbox.y; aw.value = anchorBbox.w; ah.value = anchorBbox.h; }
  renderBox(anchorBox, anchorBbox);
  checkSaveReady();
}

function setName(b, fromInput) {
  nameBbox = clamp(b);
  if (!fromInput) { nx.value = nameBbox.x; ny.value = nameBbox.y; nw.value = nameBbox.w; nh.value = nameBbox.h; }
  renderBox(nameBox, nameBbox);
  checkSaveReady();
}

function checkSaveReady() {
  saveBtn.disabled = !(anchorBbox && nameBbox);
}

function eventToImg(ev) {
  const rect = img.getBoundingClientRect();
  return { x: displayToImg(ev.clientX - rect.left), y: displayToImg(ev.clientY - rect.top) };
}

img.addEventListener("mousedown", ev => {
  ev.preventDefault();
  recomputeScale();
  dragStart = eventToImg(ev);
  // Determine active step from which area we're drawing in:
  // if click is near existing anchorBbox area or step toggle — use heuristic: alt key = name step
  activeStep = ev.altKey ? 2 : 1;
});
window.addEventListener("mousemove", ev => {
  if (!dragStart) return;
  const cur = eventToImg(ev);
  const x = Math.min(dragStart.x, cur.x), y = Math.min(dragStart.y, cur.y);
  const w = Math.abs(cur.x - dragStart.x), h = Math.abs(cur.y - dragStart.y);
  if (w < 2 || h < 2) return;
  if (activeStep === 1) setAnchor({x, y, w, h});
  else setName({x, y, w, h});
});
window.addEventListener("mouseup", () => { dragStart = null; });

// Keyboard: 1/2 to switch step, arrows to nudge
window.addEventListener("keydown", ev => {
  if (document.activeElement && document.activeElement.tagName === "INPUT") return;
  if (ev.key === "1") { activeStep = 1; status.textContent = "drawing: talent indicator"; status.className = ""; return; }
  if (ev.key === "2") { activeStep = 2; status.textContent = "drawing: hero name label"; status.className = ""; return; }
  const step = ev.shiftKey ? 10 : 1;
  const b = activeStep === 1 ? anchorBbox : nameBbox;
  if (!b) return;
  let {x, y, w, h} = b;
  if (ev.key === "ArrowLeft")  x -= step;
  else if (ev.key === "ArrowRight") x += step;
  else if (ev.key === "ArrowUp")    y -= step;
  else if (ev.key === "ArrowDown")  y += step;
  else return;
  ev.preventDefault();
  if (activeStep === 1) setAnchor({x, y, w, h});
  else setName({x, y, w, h});
});

[ax, ay, aw, ah].forEach(el => el.addEventListener("input", () => {
  setAnchor({x: +ax.value||0, y: +ay.value||0, w: +aw.value||1, h: +ah.value||1}, true);
}));
[nx, ny, nw, nh].forEach(el => el.addEventListener("input", () => {
  setName({x: +nx.value||0, y: +ny.value||0, w: +nw.value||1, h: +nh.value||1}, true);
}));

cannyLow.addEventListener("input", () => cannyLowVal.textContent = cannyLow.value);
cannyHigh.addEventListener("input", () => cannyHighVal.textContent = cannyHigh.value);
matchThr.addEventListener("input", () => matchThrVal.textContent = parseFloat(matchThr.value).toFixed(2));

document.getElementById("previewBtn").addEventListener("click", async () => {
  if (!anchorBbox) { status.textContent = "draw talent indicator box first (Step 1)"; status.className = "err"; return; }
  status.textContent = "rendering…"; status.className = "";
  const qs = new URLSearchParams({x: anchorBbox.x, y: anchorBbox.y, w: anchorBbox.w, h: anchorBbox.h,
    canny_low: cannyLow.value, canny_high: cannyHigh.value, kind: "edges", t: Date.now()});
  edgesPrev.src = "api/calibrate/preview?" + qs;
  edgesPrev.onload = () => { status.textContent = "edges ready"; status.className = "ok"; };
  edgesPrev.onerror = () => { status.textContent = "preview failed"; status.className = "err"; };
});

document.getElementById("cropPreviewBtn").addEventListener("click", async () => {
  if (!nameBbox) { status.textContent = "draw hero name label box first (Step 2)"; status.className = "err"; return; }
  status.textContent = "rendering…"; status.className = "";
  const qs = new URLSearchParams({x: nameBbox.x, y: nameBbox.y, w: nameBbox.w, h: nameBbox.h,
    kind: "crop", t: Date.now()});
  cropPrev.src = "api/calibrate/preview?" + qs;
  cropPrev.onload = () => { status.textContent = "name crop ready"; status.className = "ok"; };
  cropPrev.onerror = () => { status.textContent = "preview failed"; status.className = "err"; };
});

saveBtn.addEventListener("click", async () => {
  if (!anchorBbox || !nameBbox) return;
  status.textContent = "saving…"; status.className = "";
  try {
    const res = await fetch("api/calibrate/talent", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({
        anchor_x: anchorBbox.x, anchor_y: anchorBbox.y,
        anchor_w: anchorBbox.w, anchor_h: anchorBbox.h,
        name_x: nameBbox.x,   name_y: nameBbox.y,
        name_w: nameBbox.w,   name_h: nameBbox.h,
        canny_low: parseInt(cannyLow.value),
        canny_high: parseInt(cannyHigh.value),
        match_threshold: parseFloat(matchThr.value),
      }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    const off = data.hero_name_offset;
    status.textContent = `saved · edge density ${(data.edge_density*100).toFixed(1)}% · name offset dx=${off.dx} dy=${off.dy}. Calibration is live.`;
    status.className = "ok";
    loadState();
  } catch(err) {
    status.textContent = "save failed: " + err.message;
    status.className = "err";
  }
});

async function loadState() {
  const res = await fetch("api/calibrate/talent/state");
  const data = await res.json();
  if (data.current) {
    const c = data.current, b = c.anchor_bbox, r = c.hero_name_region;
    currentEl.textContent =
      `anchor:  x=${b.x} y=${b.y} w=${b.w} h=${b.h}\n` +
      `name:    dx=${r.dx} dy=${r.dy} w=${r.w} h=${r.h}\n` +
      `canny:   low=${c.canny_low} high=${c.canny_high}\n` +
      `thresh:  ${c.match_threshold}`;
    if (!anchorBbox) {
      cannyLow.value = c.canny_low; cannyLowVal.textContent = c.canny_low;
      cannyHigh.value = c.canny_high; cannyHighVal.textContent = c.canny_high;
      matchThr.value = c.match_threshold; matchThrVal.textContent = c.match_threshold.toFixed(2);
      const restore = () => {
        recomputeScale();
        setAnchor(b);
        setName({x: b.x + r.dx, y: b.y + r.dy, w: r.w, h: r.h});
      };
      img.complete && img.naturalWidth ? restore() : img.addEventListener("load", restore, {once: true});
    }
  } else {
    currentEl.textContent = "(none — not calibrated yet)\n\nInstructions:\n  Draw talent box → press 2 → draw name box → Save";
  }
}

img.addEventListener("load", recomputeScale);
window.addEventListener("resize", () => { recomputeScale(); renderBox(anchorBox, anchorBbox); renderBox(nameBox, nameBbox); });
status.textContent = "Press 1 to draw talent indicator, 2 to draw name label (or hold Alt while dragging for name)";
loadState();
