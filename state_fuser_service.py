import json
import os
import sqlite3
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

WORKSPACE = os.getenv("WORKSPACE", "/workspace")
GSI_AUTH_TOKEN = os.getenv("GSI_AUTH_TOKEN", "dota2_coaching_secret")
INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://inference:8080")
CV_ITEM_CONF_THRESHOLD = float(os.getenv("CV_ITEM_CONF_THRESHOLD", "0.70"))
CV_HERO_CONF_THRESHOLD = float(os.getenv("CV_HERO_CONF_THRESHOLD", "0.50"))
DB_PATH = str(Path(WORKSPACE) / "fuser.db")

app = FastAPI(title="Dota 2 State Fuser")

db: sqlite3.Connection
game_state: dict = {}
snapshots: dict[int, dict] = {}
session: dict = {"matchid": None, "clock_offset": None, "count": 0}
cv_state: dict[str, dict] = {}


@app.on_event("startup")
def startup() -> None:
    global db, cv_state
    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            matchid TEXT NOT NULL,
            clock_time INTEGER NOT NULL,
            received_at REAL NOT NULL,
            payload TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_match_clock
            ON snapshots(matchid, clock_time)
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS cv_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            matchid     TEXT    NOT NULL,
            hero        TEXT    NOT NULL,
            clock_time  INTEGER NOT NULL,
            received_at REAL    NOT NULL,
            inference   TEXT    NOT NULL
        )
    """)
    db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cv_match_hero_clock
            ON cv_snapshots(matchid, hero, clock_time)
    """)
    db.commit()



@app.post("/gsi")
async def receive_gsi(request: Request) -> dict:
    global game_state, snapshots, session
    body = await request.json()
    token = body.get("auth", {}).get("token", "")
    if token != GSI_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="invalid token")
    received_at = time.time()
    map_block = body.get("map", {})
    matchid = map_block.get("matchid")
    clock_time = map_block.get("clock_time")
    if matchid is None or clock_time is None:
        game_state = body
        return {"status": "ok"}
    if matchid != session["matchid"]:
        session = {"matchid": matchid, "clock_offset": None, "count": 0}
        snapshots = {}
        cv_state.clear()
    if session["clock_offset"] is None:
        game_phase = map_block.get("game_state", "")
        if game_phase == "DOTA_GAMERULES_STATE_GAME_IN_PROGRESS":
            provider_ts = body.get("provider", {}).get("timestamp")
            if provider_ts is not None:
                session["clock_offset"] = provider_ts - clock_time
    db.execute(
        "INSERT OR REPLACE INTO snapshots (matchid, clock_time, received_at, payload) VALUES (?, ?, ?, ?)",
        (matchid, clock_time, received_at, json.dumps(body)),
    )
    db.commit()
    snapshots[clock_time] = {"clock_time": clock_time, "received_at": received_at, "payload": body}
    session["count"] += 1
    game_state = body
    return {"status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "matchid": session["matchid"], "snapshot_count": session["count"], "clock_offset": session["clock_offset"]}


@app.get("/state/latest")
def state_latest() -> dict:
    return game_state


@app.get("/state")
def state_at(t: int) -> dict:
    row = snapshots.get(t)
    if row is not None:
        return row
    cur = db.execute(
        "SELECT clock_time, received_at, payload FROM snapshots WHERE matchid = ? AND clock_time = ?",
        (session["matchid"], t),
    )
    result = cur.fetchone()
    if result is None:
        raise HTTPException(status_code=404, detail=f"no snapshot at clock_time={t}")
    return {"clock_time": result["clock_time"], "received_at": result["received_at"], "payload": json.loads(result["payload"])}


@app.get("/sessions")
def sessions() -> list:
    cur = db.execute("""
        SELECT matchid, COUNT(*) AS snapshot_count, MIN(received_at) AS started_at,
               MAX(received_at) AS last_seen_at, MIN(clock_time) AS first_clock, MAX(clock_time) AS last_clock
        FROM snapshots GROUP BY matchid ORDER BY started_at DESC
    """)
    return [dict(row) for row in cur.fetchall()]


@app.get("/sessions/{matchid}")
def session_payloads(matchid: str) -> StreamingResponse:
    def generate():
        cur = db.execute(
            "SELECT clock_time, received_at, payload FROM snapshots WHERE matchid = ? ORDER BY clock_time",
            (matchid,),
        )
        yield "["
        first = True
        for row in cur:
            entry = json.dumps({"clock_time": row["clock_time"], "received_at": row["received_at"], "payload": json.loads(row["payload"])})
            yield ("" if first else ",") + entry
            first = False
        yield "]"
    return StreamingResponse(generate(), media_type="application/json")


@app.post("/screenshot")
async def receive_screenshot(file: UploadFile) -> dict:
    global cv_state
    image_bytes = await file.read()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{INFERENCE_SERVICE_URL}/predict",
                files={"file": (file.filename or "screenshot.png", image_bytes, file.content_type or "image/png")},
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"inference_service error: {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"inference_service unreachable: {exc}") from exc

    hero = result.get("hero_name")
    if not hero:
        return {"status": "skipped", "reason": "hero_name not detected"}

    cv_state[hero] = result

    map_block = game_state.get("map", {})
    matchid = map_block.get("matchid")
    clock_time = map_block.get("clock_time")
    if matchid is None or clock_time is None:
        return {"status": "ok", "hero": hero, "clock_time": clock_time, "inference": result}

    received_at = time.time()
    db.execute(
        "INSERT OR REPLACE INTO cv_snapshots (matchid, hero, clock_time, received_at, inference) VALUES (?, ?, ?, ?, ?)",
        (matchid, hero, clock_time, received_at, json.dumps(result)),
    )
    db.commit()
    return {"status": "ok", "hero": hero, "clock_time": clock_time, "inference": result}


def _strip_hero_prefix(name: str) -> str:
    if name and name.startswith("npc_dota_hero_"):
        return name[len("npc_dota_hero_"):]
    return name or ""


def _strip_item_prefix(name: str) -> str:
    if name and name.startswith("item_"):
        return name[len("item_"):]
    return name or ""


def _heroes_from_minimap() -> tuple[list[str], list[str]]:
    """Return (radiant_heroes, dire_heroes) deduplicated from the minimap block."""
    radiant: list[str] = []
    dire: list[str] = []
    seen: set[str] = set()
    for entry in game_state.get("minimap", {}).values():
        if not isinstance(entry, dict):
            continue
        unitname = entry.get("unitname", "")
        if not unitname.startswith("npc_dota_hero_") or unitname in seen:
            continue
        seen.add(unitname)
        hero = _strip_hero_prefix(unitname)
        if entry.get("team") == 2:
            radiant.append(hero)
        elif entry.get("team") == 3:
            dire.append(hero)
    return radiant, dire


def _build_slot_hero_map() -> dict[int, str]:
    slot_hero: dict[int, str] = {}

    # Pass 1: spectator team blocks (team2/team3) — populated in spectator mode
    for team_key, slot_base in (("team2", 0), ("team3", 5)):
        team_block = game_state.get(team_key, {})
        for i, player_obj in enumerate(team_block.values()):
            if not isinstance(player_obj, dict):
                continue
            hero_raw = player_obj.get("hero", "")
            if hero_raw:
                slot_hero[slot_base + i] = _strip_hero_prefix(hero_raw)

    # Pass 2: CV scoreboard detections
    cv_slot_names = {
        0: "radiant_hero_1", 1: "radiant_hero_2", 2: "radiant_hero_3",
        3: "radiant_hero_4", 4: "radiant_hero_5",
        5: "dire_hero_1", 6: "dire_hero_2", 7: "dire_hero_3",
        8: "dire_hero_4", 9: "dire_hero_5",
    }
    for slot in range(10):
        if slot in slot_hero:
            continue
        cv_key = cv_slot_names[slot]
        for hero_name, inference in cv_state.items():
            heroes_block = inference.get("heroes", {})
            entry = heroes_block.get(cv_key, {})
            confidence = entry.get("confidence", 0.0)
            class_name = entry.get("class", "")
            if class_name and confidence >= CV_HERO_CONF_THRESHOLD:
                slot_hero[slot] = _strip_hero_prefix(class_name)
                break

    # Pass 3: minimap hero entries — fills any remaining unknown slots
    assigned = set(slot_hero.values())
    radiant_mm, dire_mm = _heroes_from_minimap()
    unassigned_radiant = [h for h in radiant_mm if h not in assigned]
    unassigned_dire = [h for h in dire_mm if h not in assigned]
    ri = di = 0
    for slot in range(10):
        if slot in slot_hero:
            continue
        if slot < 5 and ri < len(unassigned_radiant):
            slot_hero[slot] = unassigned_radiant[ri]
            ri += 1
        elif slot >= 5 and di < len(unassigned_dire):
            slot_hero[slot] = unassigned_dire[di]
            di += 1

    return slot_hero


def _cv_items_for_hero(hero: str) -> list[str]:
    inference = cv_state.get(hero, {})
    items: list[str] = []
    for val in inference.get("items", {}).values():
        if not isinstance(val, dict):
            continue
        cls = _strip_item_prefix(val.get("class", ""))
        if cls and cls != "empty" and val.get("confidence", 0.0) >= CV_ITEM_CONF_THRESHOLD:
            items.append(cls)
    return items


def _gsi_own_items() -> list[str]:
    items_block = game_state.get("items", {})
    result: list[str] = []
    for i in range(6):
        raw = items_block.get(f"slot{i}", {})
        name = raw.get("name", "") if isinstance(raw, dict) else str(raw)
        stripped = _strip_item_prefix(name)
        if stripped and stripped != "empty":
            result.append(stripped)
    return result


@app.get("/fused")
def fused() -> dict:
    map_block = game_state.get("map", {})
    clock_time = map_block.get("clock_time")

    player_block = game_state.get("player", {})
    player_team = player_block.get("team_name", "")
    player_team_slot = player_block.get("team_slot")

    player_slot: int | None = None
    if player_team == "radiant" and player_team_slot is not None:
        player_slot = int(player_team_slot)
    elif player_team == "dire" and player_team_slot is not None:
        player_slot = 5 + int(player_team_slot)

    slot_hero = _build_slot_hero_map()

    def build_entry(slot: int) -> dict:
        hero = slot_hero.get(slot, "")
        is_player = slot == player_slot
        items = _gsi_own_items() if is_player else _cv_items_for_hero(hero)
        return {"hero": hero, "is_player": is_player, "items": items}

    return {
        "game_time_s": clock_time,
        "radiant": [build_entry(s) for s in range(5)],
        "dire": [build_entry(s) for s in range(5, 10)],
    }
