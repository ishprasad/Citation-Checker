"""FastAPI application: job API, server-sent events, reports, static UI."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from app import demo, llm
from app.config import ROOT_DIR, get_settings
from app.pipeline.ingest import InvalidArxivId, normalize_arxiv_id
from app.pipeline.run import FINISHED, JobBusy, get_runner
from app.report import build_report, export_markdown, snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")

SSE_KEEPALIVE_SECONDS = 15


@asynccontextmanager
async def lifespan(app: FastAPI):
    resumed = get_runner().resume_unfinished()
    if resumed:
        log.info("resumed %d unfinished job(s)", resumed)
    yield


app = FastAPI(title="Citation Checker", lifespan=lifespan)


class JobRequest(BaseModel):
    arxiv_id: str
    mode: Literal["quick", "full"] = "quick"


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": message})


def _job_or_404(job_id: str):
    job = get_runner().store.load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No check with this id.")
    return job


def _job_view(job) -> dict[str, Any]:
    return job.model_copy(update={"usage": llm.get_llm().usage_summary(job.id)}).model_dump(mode="json")


@app.get("/api/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "api_key_configured": bool(settings.nebius_api_key),
        "models": {role: bool(model) for role, model in settings.models.items()},
    }


@app.post("/api/jobs", status_code=201)
async def create_job(request: Request):
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        return await _create_pdf_job(request)
    try:
        body = JobRequest.model_validate(await request.json())
    except (ValidationError, json.JSONDecodeError, UnicodeDecodeError):
        return _error(422, 'Send JSON like {"arxiv_id": "2401.12345", "mode": "quick"}, or upload a PDF.')
    try:
        arxiv_id = normalize_arxiv_id(body.arxiv_id)
    except InvalidArxivId as exc:
        return _error(400, str(exc))
    job = get_runner().create_job(arxiv_id, body.mode)
    return {"id": job.id, "job": _job_view(job)}


async def _create_pdf_job(request: Request):
    settings = get_settings()
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return _error(422, "Attach the PDF as a form field named 'file'.")
    mode = form.get("mode") or "quick"
    if mode not in ("quick", "full"):
        return _error(422, "mode must be 'quick' or 'full'.")
    if upload.content_type not in ("application/pdf", "application/x-pdf", "application/octet-stream"):
        return _error(415, "Only PDF files can be uploaded.")
    data = await upload.read(settings.max_pdf_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_pdf_mb * 1024 * 1024:
        return _error(413, f"PDFs are limited to {settings.max_pdf_mb} MB.")
    if not data.startswith(b"%PDF-"):
        return _error(415, "This file does not look like a PDF.")
    # PDF ingestion is milestone 11; validation is in place so the route contract is stable.
    return _error(501, "PDF uploads are not supported yet. Please use an arXiv ID or link for now.")


@app.post("/api/jobs/{job_id}/expand")
async def expand_job(job_id: str):  # async: starting the job needs the event loop
    _job_or_404(job_id)
    try:
        job = get_runner().expand(job_id)
    except JobBusy as exc:
        return _error(409, str(exc))
    return {"id": job.id, "job": _job_view(job)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job_view(_job_or_404(job_id))


@app.get("/api/jobs/{job_id}/report")
def get_report(job_id: str):
    runner = get_runner()
    return build_report(runner.store, _job_or_404(job_id))


@app.get("/api/jobs/{job_id}/export")
def export_report(job_id: str, format: Literal["md", "json"] = Query("md")):
    runner = get_runner()
    job = _job_or_404(job_id)
    report = build_report(runner.store, job)
    name = (report.get("paper") or {}).get("arxiv_id") or job.id
    filename = f"citation-check-{str(name).replace('/', '_')}.{format}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if format == "json":
        return Response(json.dumps(report, indent=2), media_type="application/json", headers=headers)
    return PlainTextResponse(export_markdown(report), media_type="text/markdown; charset=utf-8", headers=headers)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    runner = get_runner()
    _job_or_404(job_id)
    queue = runner.bus.subscribe(job_id)  # subscribe before the snapshot so nothing is missed

    async def stream() -> AsyncIterator[str]:
        try:
            job = runner.store.load_job(job_id)
            yield _sse("snapshot", snapshot(runner.store, job))
            if job.status in FINISHED:
                yield _sse("done", {"status": job.status, "message": job.message})
                return
            while True:
                if await request.is_disconnected():
                    return
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield _sse(item["event"], item["data"])
                if item["event"] == "done":
                    return
        finally:
            runner.bus.unsubscribe(job_id, queue)

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/demo", status_code=201)
async def start_demo(fresh: bool = False):
    """Check the bundled demo paper (planted errors). Reuses the latest finished check unless
    `fresh`; without an API key, shows the bundled snapshot of a recorded check."""
    if not demo.demo_available():
        return _error(501, "The demo paper has not been generated yet (run scripts/plant_errors.py).")
    runner = get_runner()
    if not fresh:
        existing = runner.latest_demo_job()
        if existing is not None:
            return {"id": existing.id, "job": _job_view(existing)}
    if not get_settings().nebius_api_key:
        if not demo.SNAPSHOT_PATH.exists():
            return _error(503, "No model API key is configured and there is no recorded demo to show.")
        job = runner.import_snapshot()
    else:
        job = runner.create_demo_job()
    return {"id": job.id, "job": _job_view(job)}


app.mount("/", StaticFiles(directory=ROOT_DIR / "static", html=True), name="static")
