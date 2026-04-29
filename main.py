"""Combined entrypoint for dota2-cv.

Mounts three sub-services under prefixed paths in a single FastAPI process:
- /inference  — ONNX item/hero classifier + hero-name OCR
- /labeler    — labeling UI + anchor calibration (HTML/CSS/JS in web/labeler/)
- /fuser      — Valve GSI ingest + CV/GSI state fuser

The labeler shares ONNX sessions and anchor state with the inference service,
so its startup must run after inference's. The fuser invokes the inference
predictor in-process via inference_service.predict_bytes (no HTTP loopback).

Auto-generated OpenAPI docs are at /docs (Swagger UI) and /redoc.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import inference_service
import label_service
import state_fuser_service

WEB_DIR = Path(__file__).parent / "web" / "labeler"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Order matters: label_service reads inference_service globals (ONNX
    # sessions, anchor state) populated by inference_service.startup(), so
    # it must run last. state_fuser_service has no startup-time dependency
    # on inference globals; its predict_bytes call happens at request time.
    inference_service.startup()
    state_fuser_service.startup()
    label_service.startup()
    yield


app = FastAPI(
    title="Dota 2 CV Suite",
    description="Combined inference + labeler + state fuser service.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(inference_service.router, prefix="/inference", tags=["Inference"])
app.include_router(label_service.router, prefix="/labeler", tags=["Labeler"])
app.include_router(state_fuser_service.router, prefix="/fuser", tags=["Fuser"])

app.mount(
    "/labeler/static",
    StaticFiles(directory=WEB_DIR),
    name="labeler_static",
)
