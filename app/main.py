"""FastAPI application: job API, server-sent events, reports, static UI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from app import demo, llm
from app.config import ROOT_DIR, Settings, get_settings
from app.http import DownloadTooLarge
from app.limits import RateLimiter, client_ip, minutes_text, passcode_ok
from app.models import IngestResult, Job
from app.pipeline.ingest import IngestError, InvalidArxivId, ingest_pdf, normalize_arxiv_id
from app.pipeline.run import FINISHED, JobBusy, estimate_seconds, get_runner
from app.report import build_report, export_markdown, snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")

SSE_KEEPALIVE_SECONDS = 15


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.job_limiter = RateLimiter()
    app.state.preview_limiter = RateLimiter()
    resumed = get_runner().resume_unfinished()
    if resumed:
        log.info("resumed %d unfinished job(s)", resumed)
    yield


app = FastAPI(title="Citation Checker", lifespan=lifespan)


class JobRequest(BaseModel):
    arxiv_id: str
    mode: Literal["quick", "full"] = "quick"
    passcode: Optional[str] = None  # needed for full checks when JUDGE_MODE_PASSWORD is set
    fresh: bool = False  # check again even if this paper version was checked before


def _error(status: int, message: str, code: Optional[str] = None,
           headers: Optional[dict[str, str]] = None) -> JSONResponse:
    content = {"detail": message}
    if code:
        content["code"] = code
    return JSONResponse(status_code=status, content=content, headers=headers)


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
        "limits": {
            "quick_check_max_citations": settings.quick_check_max_citations,
            "max_references": settings.max_references,
            "max_pdf_mb": settings.max_pdf_mb,
            "passcode_required": bool(settings.judge_mode_password),
        },
    }


# ----------------------------------------------------------------------------
# Limits (brief section 10.4)
# ----------------------------------------------------------------------------


def _passcode_error(settings: Settings, given: Optional[str]) -> Optional[JSONResponse]:
    if passcode_ok(settings, given):
        return None
    if given:
        return _error(403, "That passcode does not match. Quick checks are open to everyone.", code="passcode")
    return _error(403, "Full checks need the passcode from the Citation Checker team. "
                       "Quick checks are open to everyone.", code="passcode")


def _too_many_references(settings: Settings, references: int) -> Optional[JSONResponse]:
    if references <= settings.max_references:
        return None
    return _error(
        422,
        f"This paper has {references} references, more than the {settings.max_references} a full check "
        f"covers here. A quick check of its first {settings.quick_check_max_citations} citing sentences works.",
        code="too_many_references",
    )


def _admit(request: Request, settings: Settings) -> Optional[JSONResponse]:
    """Capacity and per-IP limits for starting new work (reusing an earlier result is always free)."""
    active = get_runner().active_jobs()
    if active >= settings.max_concurrent_jobs:
        return _error(429, f"The checker is busy with {active} other papers right now. "
                           "Please try again in a few minutes.", code="busy", headers={"Retry-After": "60"})
    wait = request.app.state.job_limiter.retry_after(client_ip(request, settings), settings.max_jobs_per_ip_per_hour)
    if wait:
        return _error(429, f"You have started {settings.max_jobs_per_ip_per_hour} checks in the last hour, "
                           f"the limit for now. You can start another in {minutes_text(wait)}.",
                      code="rate_limited", headers={"Retry-After": str(wait)})
    return None


async def _prepare(request: Request, arxiv_id: str) -> Union[IngestResult, JSONResponse]:
    """Download and parse the paper (cached per version). Downloads count toward a per-IP limit."""
    settings = get_settings()
    runner = get_runner()
    cached = runner.cached_paper(arxiv_id)
    if cached is not None:
        return cached
    ip = client_ip(request, settings)
    limiter: RateLimiter = request.app.state.preview_limiter
    wait = limiter.retry_after(ip, settings.max_previews_per_ip_per_hour)
    if wait:
        return _error(429, f"Too many new papers looked up from this address. Try again in {minutes_text(wait)}.",
                      code="rate_limited", headers={"Retry-After": str(wait)})
    limiter.record(ip)
    try:
        result, _ = await runner.prepare_arxiv(arxiv_id)
    except InvalidArxivId as exc:
        return _error(404, str(exc), code="not_found")
    except IngestError as exc:
        return _error(422, str(exc), code="ingest_failed")
    except (httpx.HTTPError, DownloadTooLarge) as exc:
        log.warning("arXiv %s: download failed: %s", arxiv_id, exc)
        return _error(502, "arXiv could not be reached just now. Please try again in a minute.", code="upstream")
    return result


def _reuse_view(job: Optional[Job]) -> Optional[dict]:
    return {"id": job.id, "status": job.status, "mode": job.mode} if job else None


@app.get("/api/preview")
async def preview(request: Request, arxiv_id: str = Query(...)):
    """What a check of this paper involves, shown before it starts: size, estimated time, limits,
    and whether an earlier check can be reopened instantly."""
    settings = get_settings()
    try:
        arxiv_id = normalize_arxiv_id(arxiv_id)
    except InvalidArxivId as exc:
        return _error(400, str(exc), code="invalid_id")
    prepared = await _prepare(request, arxiv_id)
    if isinstance(prepared, JSONResponse):
        return prepared
    paper, sentences = prepared.paper, prepared.sentences
    runner = get_runner()
    quick = sentences[: settings.quick_check_max_citations]
    too_many = _too_many_references(settings, len(paper.references))
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors,
        "references": len(paper.references),
        "citing_sentences": len(sentences),
        "quick": {
            "sentences": len(quick),
            "estimate_seconds": estimate_seconds(quick),
            "existing": _reuse_view(runner.reusable_job(paper.id, "quick")),
        },
        "full": {
            "sentences": len(sentences),
            "estimate_seconds": estimate_seconds(sentences),
            "existing": _reuse_view(runner.reusable_job(paper.id, "full")),
            "allowed": too_many is None,
            "note": json.loads(too_many.body)["detail"] if too_many else None,
        },
        "passcode_required": bool(settings.judge_mode_password),
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
        return _error(400, str(exc), code="invalid_id")
    settings = get_settings()
    runner = get_runner()
    prepared = await _prepare(request, arxiv_id)
    if isinstance(prepared, JSONResponse):
        return prepared
    if not body.fresh:
        existing = runner.reusable_job(prepared.paper.id, body.mode)
        if existing is not None:
            return JSONResponse({"id": existing.id, "job": _job_view(existing), "cached": True})
    if body.mode == "full":
        refused = _passcode_error(settings, body.passcode) or _too_many_references(settings, len(prepared.paper.references))
        if refused:
            return refused
    refused = _admit(request, settings)
    if refused:
        return refused
    job = runner.create_job(arxiv_id, body.mode, ingest=prepared)
    request.app.state.job_limiter.record(client_ip(request, settings))
    return {"id": job.id, "job": _job_view(job), "cached": False}


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
    if mode == "full":
        refused = _passcode_error(settings, form.get("passcode"))
        if refused:
            return refused
    refused = _admit(request, settings)
    if refused:
        return refused
    try:
        result = await ingest_pdf(data)
    except IngestError as exc:
        return _error(422, str(exc), code="ingest_failed")
    if mode == "full":
        refused = _too_many_references(settings, len(result.paper.references))
        if refused:
            return refused
    paper = result.paper
    paper.id = "pdf:" + hashlib.sha256(data).hexdigest()[:16]
    if not paper.title:
        paper.title = Path(getattr(upload, "filename", "") or "Uploaded paper").stem
    # The PDF itself is not stored; only the parsed text needed for the report.
    job = get_runner().create_job(None, mode, ingest=result, source="pdf")
    request.app.state.job_limiter.record(client_ip(request, settings))
    return {"id": job.id, "job": _job_view(job), "cached": False}


@app.post("/api/jobs/{job_id}/expand")
async def expand_job(job_id: str, request: Request):  # async: starting the job needs the event loop
    """Upgrade a finished quick check to a full check. Body (optional): {"passcode": "..."}."""
    job = _job_or_404(job_id)
    if job.mode == "full":
        return {"id": job.id, "job": _job_view(job)}
    if job.status not in FINISHED:
        return _error(409, "This check is still running.", code="busy")
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        data = {}
    settings = get_settings()
    refused = (
        _passcode_error(settings, data.get("passcode") if isinstance(data, dict) else None)
        or _too_many_references(settings, job.references_total)
        or _admit(request, settings)
    )
    if refused:
        return refused
    try:
        job = get_runner().expand(job_id)
    except JobBusy as exc:
        return _error(409, str(exc), code="busy")
    request.app.state.job_limiter.record(client_ip(request, settings))
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
    """The bundled demo paper (planted errors) as checked by the calibrated pipeline: the recorded
    check in demo/snapshot.json, whose numbers the README reports. `fresh` checks it again with the
    current pipeline (needs an API key). Without a recording, the latest check by the current
    pipeline is reused, or a new one runs."""
    if not demo.demo_available():
        return _error(501, "The demo paper has not been generated yet (run scripts/plant_errors.py).")
    runner = get_runner()
    has_key = bool(get_settings().nebius_api_key)
    if fresh and has_key:
        job = runner.create_demo_job()
    elif demo.SNAPSHOT_PATH.exists():
        job = runner.snapshot_demo_job()
    elif (existing := runner.latest_demo_job()) is not None:
        job = existing
    elif has_key:
        job = runner.create_demo_job()
    else:
        return _error(503, "No model API key is configured and there is no recorded demo to show.")
    return {"id": job.id, "job": _job_view(job)}


app.mount("/", StaticFiles(directory=ROOT_DIR / "static", html=True), name="static")
