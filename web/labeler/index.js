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
  const a = (DATA.anchor_meta || {})[fname];
  let tag = "";
  if (a) {
    if (a.used) tag = ` · anchor ${a.score.toFixed(2)}`;
    else if (a.score !== null) tag = ` · anchor FB ${a.score.toFixed(2)}`;
  }
  sumEl.textContent = `${fname}  —  ${filled}/16 labeled${tag}`;
}

function updateAnchorStatus() {
  const el = document.getElementById("anchorStatus");
  if (!DATA.anchor_name) { el.textContent = "anchor: off"; return; }
  const all = Object.values(DATA.anchor_meta || {});
  const used = all.filter(a => a && a.used).length;
  el.textContent = `anchor: ${DATA.anchor_name} (${used}/${all.length} matched)`;
}

function scheduleSave() {
  setStatus("saving…", "saving");
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(doSave, 400);
}

async function doSave() {
  try {
    const res = await fetch("api/labels", {
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
  const res = await fetch("api/data");
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
  updateAnchorStatus();
  document.getElementById("loading").style.display = "none";
  render();
}

document.getElementById("reloadBtn").addEventListener("click", async () => {
  if (!confirm("Re-scan screenshots dir? Current unsaved changes will be flushed first.")) return;
  if (saveTimer) { clearTimeout(saveTimer); await doSave(); }
  setStatus("reloading", "saving");
  try {
    const r = await fetch("api/reload", {method: "POST"});
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    await loadData();
  } catch (err) {
    setStatus("reload error: " + err.message, "error");
  }
});

loadData().catch(err => {
  document.getElementById("loading").textContent = "Failed to load: " + err.message;
});
