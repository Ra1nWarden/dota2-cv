# dota2-cv

Real-time Dota 2 computer vision pipeline. Captures in-game screenshots,
classifies item icons via ONNX inference, identifies the focused hero by
OCR-ing the HUD name label, and ingests live game state from Valve's Game
State Integration (GSI) API — building the foundation for a real-time
coaching service.

---

## Architecture

```
Dota 2 client
  │
  ├─ GSI HTTP push (~10 Hz) ──────────────► combined service (host 18080 / container 8080)
  │                                         POST /fuser/gsi
  │                                         snapshots stored in fuser.db,
  │                                         fused with CV inference output
  │
  └─ Screenshots ─────────────────────────► combined service (host 18080 / container 8080)
                                            POST /inference/predict
                                            item classifier + hero OCR
                                            returns {items, hero_name}

combined service mounts
  ├─ /inference/*  — ONNX item classifier + hero name OCR
  ├─ /labeler/*    — labeling UI + anchor calibration
  │                  → configs/anchor_offsets.json
  │                  → configs/talent_anchor_offsets.json
  └─ /fuser/*      — GSI ingest + CV/GSI state fusion
```

All three services run in one FastAPI process; the fuser invokes
inference in-process. The container mounts the repo as `/workspace`
and is defined in `docker-compose.yml`.

---

## Services

### Inference (`/inference`)
ONNX-based icon classifier + hero name OCR.

- `POST /inference/predict` — accepts a screenshot, returns:
  - `hero_name` — GSI internal name of the currently focused hero,
    identified by OCR-ing the hero name label in the HUD (anchored to
    the talent indicator)
  - `items` — item classifier results for the 6 HUD item slots
    (`{slot: {class, confidence}}`)
  - `heroes` — hero portrait classifier results for all 10 top-bar slots
  - `anchor` — scepter anchor match metadata
- `GET /inference/health` — liveness check

Item crops use anchor-relative positioning (robust to HUD drift).
Hero name OCR uses EasyOCR with English + Simplified Chinese models;
the display name is resolved to a GSI internal name via
`configs/hero_display_names.json`.

### Labeler (`/labeler`)
Browser-based labeling and calibration UI.

- `GET /labeler/` — screenshot labeling interface; shows model top-3
  suggestions per crop, lets you confirm or correct ground-truth labels
- `GET /labeler/calibrate` — scepter/shard anchor calibration; draw a
  bounding box around the Aghanim's indicator to set item slot offsets
- `GET /labeler/talent` — talent indicator anchor calibration;
  draw two boxes (talent indicator + hero name label) to configure hero OCR
- `/labeler/api/...` — backing JSON endpoints used by the UI

The labeler's HTML/CSS/JS are served as static files from `web/labeler/`.

### Fuser (`/fuser`)
Valve GSI event ingestor + CV/GSI state fuser.

- `POST /fuser/gsi` — receives live game state payloads from the Dota 2
  client, validates the auth token, and persists snapshots to `fuser.db`
  (SQLite)
- `POST /fuser/screenshot` — accepts a screenshot, runs inference
  in-process, and stores the result in `fuser.db` keyed by hero
- `GET /fuser/health` — current matchid, snapshot count, and clock offset
- `GET /fuser/state` / `GET /fuser/state/latest` — current/latest GSI
  game state
- `GET /fuser/sessions`, `GET /fuser/sessions/{matchid}` — recorded matches
- `GET /fuser/fused`, `GET /fuser/fused/{game_id}` — fused GSI + CV state
  for the current or a historical match

Configure the Dota 2 client to push to this service by placing
`configs/gamestate_integration_coaching.cfg` in the game's
`cfg/` directory.

Auto-generated OpenAPI docs at `/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## Models

| File | Description |
|---|---|
| `models/hero_classifier.onnx` | EfficientNet-B0 trained on hero portrait icons (10 top-bar slots) |
| `models/item_classifier.onnx` | EfficientNet-B0 trained on item icons (6 inventory slots) |

Both models are exported from PyTorch checkpoints (`.pt`) via `scripts/export_onnx.py`.

---

## Configuration

| File | Description |
|---|---|
| `configs/crop_config.json` | Fixed pixel coordinates for all 16 HUD crop regions (10 hero + 6 item) at 3840×2160 reference resolution |
| `configs/anchor_offsets.json` | Scepter/shard anchor template + per-item-slot dx/dy offsets for anchor-relative item cropping |
| `configs/talent_anchor_offsets.json` | Talent indicator anchor template + dx/dy offset to hero name label for OCR |
| `configs/heroes_classes.json` | Integer → hero class name mapping for the hero classifier |
| `configs/items_classes.json` | Integer → item class name mapping for the item classifier |
| `configs/hero_display_names.json` | HUD display name (EN + ZH) → GSI internal name lookup table for OCR resolution |
| `configs/gamestate_integration_coaching.cfg` | Dota 2 GSI config; place in `<dota2>/game/dota/cfg/` |

---

## Scripts

| Script | Description |
|---|---|
| `scripts/train.py` | Train EfficientNet-B0 on synthetic icon data; supports 3-phase progressive unfreezing, mixed precision, and TensorBoard logging |
| `scripts/export_onnx.py` | Export a trained `.pt` checkpoint to `.onnx` for deployment |
| `scripts/evaluate.py` | Run accuracy evaluation against ground-truth labels; emits a per-class report |
| `scripts/generate_synthetic_data.py` | Augment raw icon PNGs into a train/val split for classifier training |
| `scripts/calibrate_crops.py` | Interactive CLI tool to mark HUD crop regions on a reference screenshot and write `crop_config.json` |
| `scripts/calibrate_anchor.py` | CLI alternative to the labeler scepter anchor calibration UI |
| `scripts/build_hero_display_names.py` | Parse Dota 2 localization files to regenerate `hero_display_names.json` after patches |
| `scripts/test_ocr.py` | Smoke-test script; runs `/inference/predict` over all test screenshots and prints `hero_name` + item results |

---

## Setup

### 1. Configure GSI
Copy the GSI config to your Dota 2 installation and update the server URL:
```
configs/gamestate_integration_coaching.cfg → <dota2>/game/dota/cfg/
```
The GSI URL now points to `/fuser/gsi` on port `18080`. The config file
in this repo has been updated, but any previously deployed copy in your
Dota 2 `cfg/` directory must also be updated to match.

### 2. Build and start the service
```bash
docker compose build
docker compose up -d
```
This brings up a single combined FastAPI service on host port `18080`
with `/inference`, `/labeler`, and `/fuser` mounted under one process.

### 3. Calibrate (first run only)
- **Item anchor**: open `http://<host>:18080/labeler/calibrate`, draw a
  box around the Aghanim's Scepter/Shard indicator, save.
- **Talent anchor**: open `http://<host>:18080/labeler/talent`,
  draw a box around the talent indicator (Step 1) then the hero name
  label (Step 2), save.

Calibration takes effect immediately — saving refreshes the inference
globals in-process, so no service restart is required.

### 4. Regenerate hero display names (after patches)
```bash
python scripts/build_hero_display_names.py \
  --game-path /path/to/dota2/game/dota \
  --out configs/hero_display_names.json
```

### 5. Test inference
The inference endpoint is now `/inference/predict`. The `test_ocr.py`
script defaults to that path, so the `--host` flag stays the same:
```bash
python scripts/test_ocr.py --host http://<host>:18080
```
