"""FastAPI application. Routes for jobs, SSE and reports arrive in milestone 7."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import ROOT_DIR, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="Citation Checker")


@app.get("/api/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "api_key_configured": bool(settings.nebius_api_key),
        "models": {role: bool(model) for role, model in settings.models.items()},
    }


app.mount("/", StaticFiles(directory=ROOT_DIR / "static", html=True), name="static")
