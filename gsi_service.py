import json
import os
import sqlite3
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

WORKSPACE = os.getenv("WORKSPACE", "/workspace")
GSI_AUTH_TOKEN = os.getenv("GSI_AUTH_TOKEN", "dota2_coaching_secret")
DB_PATH = str(Path(WORKSPACE) / "gsi.db")

app = FastAPI(title="Dota 2 GSI Ingestor")

db: sqlite3.Connection

game_state: dict = {}
snapshots: dict[int, dict] = {}
session: dict = {"matchid": None, "clock_offset": None, "count": 0}


@app.on_event("startup")
def startup() -> None:
    global db
    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            matchid     TEXT    NOT NULL,
            clock_time  INTEGER NOT NULL,
            received_at REAL    NOT NULL,
            payload     TEXT    NOT NULL
        )
    """)
    db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_match_clock
            ON snapshots(matchid, clock_time)
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

    # Nothing useful to persist without a match or clock position
    if matchid is None or clock_time is None:
        game_state = body
        return {"status": "ok"}

    # New game detected — reset in-memory state
    if matchid != session["matchid"]:
        session = {"matchid": matchid, "clock_offset": None, "count": 0}
        snapshots = {}

    # Compute clock offset on first in-progress packet
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
    return {
        "status": "ok",
        "matchid": session["matchid"],
        "snapshot_count": session["count"],
        "clock_offset": session["clock_offset"],
    }


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
        SELECT matchid,
               COUNT(*)         AS snapshot_count,
               MIN(received_at) AS started_at,
               MAX(received_at) AS last_seen_at,
               MIN(clock_time)  AS first_clock,
               MAX(clock_time)  AS last_clock
        FROM snapshots
        GROUP BY matchid
        ORDER BY started_at DESC
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
            entry = json.dumps({
                "clock_time": row["clock_time"],
                "received_at": row["received_at"],
                "payload": json.loads(row["payload"]),
            })
            yield ("" if first else ",") + entry
            first = False
        yield "]"

    return StreamingResponse(generate(), media_type="application/json")
