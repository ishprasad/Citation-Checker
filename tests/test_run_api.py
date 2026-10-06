"""Orchestration and API tests with fake pipeline stages (no network, no models)."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.pipeline.run as run_mod
from app import llm
from app.config import get_settings
from app.db import Database, LLMCache, set_database
from app.models import Claim, Evidence, IngestResult, Passage, Reference, Verdict
from app.pipeline.ingest import IngestError, parse_latex_project
from app.pipeline.run import EventBus, JobRunner, JobStore, pair_id, set_runner

FIXTURE = Path(__file__).parent / "fixtures" / "latex_numeric"


def fixture_ingest() -> IngestResult:
    files = {p.relative_to(FIXTURE).as_posix(): p.read_text(encoding="utf-8") for p in FIXTURE.rglob("*") if p.is_file()}
    paper, sentences = parse_latex_project(files)
    paper = paper.model_copy(update={"id": "arxiv:2101.00001v1", "arxiv_id": "2101.00001v1", "title": "Fixture Paper"})
    return IngestResult(paper=paper, sentences=sentences)


class Fakes:
    """Records calls to each fake stage."""

    def __init__(self):
        self.extracted: list[str] = []
        self.judged: list[str] = []
        self.escalated: list[str] = []
        self.resolved: list[str] = []
        self.fail_pair: str | None = None
        self.budget_on_judge = False
        self.ingest_error: str | None = None


@pytest.fixture
def env(monkeypatch, tmp_path):
    settings = replace(get_settings(), quick_check_max_citations=3, max_concurrent_jobs=2,
                       db_path=tmp_path / "app.sqlite", cache_dir=tmp_path,
                       models={**get_settings().models, "judge": "super", "escalate": "ultra"})
    monkeypatch.setattr(run_mod, "get_settings", lambda: settings)
    db = Database(tmp_path / "app.sqlite")
    set_database(db)
    llm.set_llm(llm.LLMClient(settings, client=object(), db=db, cache=LLMCache(tmp_path / "c.sqlite")))
    fakes = Fakes()

    async def ingest(arxiv_id):
        if fakes.ingest_error:
            raise IngestError(fakes.ingest_error)
        return fixture_ingest()

    class Resolver:
        async def resolve_all(self, refs, on_done=None):
            out = []
            for r in refs:
                fakes.resolved.append(r.key)
                resolved = r.model_copy(update={"fulltext_status": "abstract_only", "abstract": f"About {r.key}."})
                await on_done(resolved)
                out.append(resolved)
            return out

    async def extract(sentences, paper, refs_by_key=None, on_done=None):
        claims_all = []
        for s in sentences:
            fakes.extracted.append(s.id)
            claims = [Claim(id=f"{s.id}.c1", citing_sentence_id=s.id, claim_text=f"claim of {s.id}",
                            reference_keys=list(s.reference_keys))]
            await on_done(s, claims)
            claims_all += claims
        return claims_all

    class Retriever:
        async def query_vectors(self, claims):
            return {c.id: np.ones(3) for c in claims}

        async def retrieve(self, claim, ref, query_vec=None):
            return Evidence(claim_id=claim.id, reference_key=ref.key,
                            passages=[Passage(text=ref.abstract or "", chunk_index=-1, score=0.5)])

        async def abstract_scores(self, claims, refs):
            return {c.id: {} for c in claims}

    async def judge_pair(claim, ref, evidence, ctx):
        pid = pair_id(claim.id, ref.key)
        if fakes.budget_on_judge:
            raise llm.TokenBudgetExceeded("budget")
        if pid == fakes.fail_pair:
            raise ValueError("boom")
        fakes.judged.append(pid)
        label = "not_found" if claim.citing_sentence_id == "s1" else "supported"
        return Verdict(claim_id=claim.id, reference_key=ref.key, label=label, confidence=0.9,
                       attributed_claim=claim.claim_text, model_used="super")

    async def escalate_pair(claim, ref, first, evidence, ctx):
        fakes.escalated.append(pair_id(claim.id, ref.key))
        return first.model_copy(update={"label": "supported", "escalated": True, "original_label": first.label})

    def needs_escalation(verdict, claim, ref):
        return verdict.label == "not_found"

    monkeypatch.setattr(run_mod, "ingest_arxiv", ingest)
    monkeypatch.setattr(run_mod, "ReferenceResolver", Resolver)
    monkeypatch.setattr(run_mod, "extract_claims", extract)
    monkeypatch.setattr(run_mod, "EvidenceRetriever", Retriever)
    monkeypatch.setattr(run_mod, "judge_pair", judge_pair)
    monkeypatch.setattr(run_mod, "escalate_pair", escalate_pair)
    monkeypatch.setattr(run_mod, "needs_escalation", needs_escalation)
    monkeypatch.setattr(run_mod, "read_reference_sidecar", lambda ref: {})
    yield fakes, JobStore(db)
    set_runner(None)
    llm.set_llm(None)
    set_database(None)


async def _run_job(store: JobStore, mode="quick", job_id=None):
    runner = JobRunner(store=store, bus=EventBus())
    events = []
    if job_id is None:
        job = run_mod.Job(id="job1", mode=mode, arxiv_id="2101.00001")
        store.save_job(job)
        job_id = job.id
    queue = runner.bus.subscribe(job_id)
    runner.start(job_id)
    job = await runner.wait(job_id)
    while not queue.empty():
        events.append(queue.get_nowait())
    return runner, job, events


# --------------------------------------------------------------------------- runner


def test_quick_job_streams_and_finishes(env):
    fakes, store = env
    runner, job, events = asyncio.run(_run_job(store))
    assert job.status == "done" and job.title == "Fixture Paper"
    assert (job.sentences_checked, job.sentences_total) == (3, len(fixture_ingest().sentences))
    assert fakes.extracted == ["s1", "s2", "s3"]  # quick check: first 3 citing sentences
    progress = {p.stage: (p.done, p.total, p.state) for p in job.progress}
    assert progress["judge"] == (4, 4, "done") and progress["escalate"] == (2, 2, "done")  # s1 cites two works
    verdicts = store.all(job.id, "verdict", Verdict)
    assert len(verdicts) == 4 and all(v.label == "supported" for v in verdicts.values())
    assert sum(v.escalated for v in verdicts.values()) == 2
    kinds = [e["event"] for e in events]
    assert kinds[0] == "status" and kinds[-1] == "done"
    assert {"paper", "reference", "claims", "verdict", "escalating", "coverage", "progress"} <= set(kinds)


def test_expand_reuses_finished_work(env):
    fakes, store = env

    async def scenario():
        runner, job, _ = await _run_job(store)
        fakes.extracted.clear()
        fakes.judged.clear()
        fakes.resolved.clear()
        runner.expand(job.id)
        return await runner.wait(job.id)

    job = asyncio.run(scenario())
    total = len(fixture_ingest().sentences)
    assert job.mode == "full" and job.status == "done" and job.sentences_checked == total
    assert fakes.extracted == [f"s{i}" for i in range(4, total + 1)]  # only the new sentences
    assert all(not pid.startswith(("s1.", "s2.", "s3.")) for pid in fakes.judged)
    progress = {p.stage: (p.done, p.total) for p in job.progress}
    assert progress["judge"][0] == progress["judge"][1] == len(store.all(job.id, "verdict", Verdict))


def test_resume_skips_stored_results(env):
    fakes, store = env
    asyncio.run(_run_job(store))
    job = store.load_job("job1")
    job.status = "running"  # as if the server died
    store.save_job(job)
    fakes.extracted.clear()
    fakes.judged.clear()
    fakes.resolved.clear()
    _, job, _ = asyncio.run(_run_job(store, job_id="job1"))
    assert job.status == "done"
    assert fakes.extracted == [] and fakes.judged == [] and fakes.resolved == []


def test_failing_pair_does_not_fail_job(env):
    fakes, store = env
    fakes.fail_pair = "s2.c1|" + fixture_ingest().sentences[1].reference_keys[0]
    _, job, _ = asyncio.run(_run_job(store))
    assert job.status == "done"
    v = store.all(job.id, "verdict", Verdict)[fakes.fail_pair]
    assert v.label == "unverifiable" and v.flags == ["pipeline_error"]


def test_token_budget_gives_partial_job(env):
    fakes, store = env
    fakes.budget_on_judge = True
    _, job, events = asyncio.run(_run_job(store))
    assert job.status == "partial" and "token budget" in job.message
    assert events[-1] == {"event": "done", "data": {"status": "partial", "message": job.message}}


def test_ingest_error_fails_job_with_message(env):
    fakes, store = env
    fakes.ingest_error = "arXiv has no paper with ID 2101.00001."
    _, job, _ = asyncio.run(_run_job(store))
    assert job.status == "failed" and job.message == "arXiv has no paper with ID 2101.00001."


# ------------------------------------------------------------------------------ API


@pytest.fixture
def client(env):
    fakes, store = env
    set_runner(JobRunner(store=store, bus=EventBus()))
    from app.main import app

    with TestClient(app) as c:
        yield c, fakes


def _wait_done(c, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "partial", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_api_job_lifecycle(client):
    c, _ = client
    r = c.post("/api/jobs", json={"arxiv_id": " https://arxiv.org/abs/2101.00001v1 ", "mode": "quick"})
    assert r.status_code == 201
    job_id = r.json()["id"]
    assert r.json()["job"]["arxiv_id"] == "2101.00001v1"
    job = _wait_done(c, job_id)
    assert job["status"] == "done" and "per_model" in job["usage"]

    report = c.get(f"/api/jobs/{job_id}/report").json()
    assert report["paper"]["title"] == "Fixture Paper" and report["paper"]["text"]
    assert len(report["verdicts"]) == 4 and report["summary"]["quick"] is True
    assert report["summary"]["labels"]["supported"] == 4
    assert sum(s["checked"] for s in report["sentences"]) == 3
    assert report["summary"]["coverage"]["line"].startswith("Full text found for 0 of")

    md = c.get(f"/api/jobs/{job_id}/export?format=md")
    assert md.status_code == 200 and "attachment" in md.headers["content-disposition"]
    assert md.text.startswith("# Citation check: Fixture Paper") and "quick check" in md.text
    js = c.get(f"/api/jobs/{job_id}/export?format=json")
    assert json.loads(js.text)["job"]["id"] == job_id

    with c.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        body = "".join(stream.iter_text())
    assert body.startswith("event: snapshot") and "event: done" in body
    snapshot = json.loads(body.split("data: ", 1)[1].split("\n", 1)[0])
    assert len(snapshot["verdicts"]) == 4 and "text" not in snapshot["paper"]

    r = c.post(f"/api/jobs/{job_id}/expand")
    assert r.status_code == 200 and r.json()["job"]["mode"] == "full"
    assert _wait_done(c, job_id)["sentences_checked"] == len(fixture_ingest().sentences)


@pytest.mark.parametrize(
    "kwargs, status, fragment",
    [
        (dict(json={"arxiv_id": "hello"}), 400, "doesn't look like an arXiv ID"),
        (dict(json={"mode": "quick"}), 422, "arxiv_id"),
        (dict(json={"arxiv_id": "2101.00001", "mode": "slow"}), 422, "arxiv_id"),
        (dict(files={"file": ("a.pdf", b"hello", "application/pdf")}), 415, "does not look like a PDF"),
        (dict(files={"file": ("a.txt", b"%PDF-1.4", "text/plain")}), 415, "Only PDF"),
        (dict(files={"file": ("a.pdf", b"%PDF-1.4 x", "application/pdf")}), 501, "not supported yet"),
    ],
)
def test_api_input_validation(client, kwargs, status, fragment):
    c, _ = client
    r = c.post("/api/jobs", **kwargs)
    assert r.status_code == status and fragment in r.json()["detail"]


def test_api_pdf_size_limit(client, monkeypatch):
    c, _ = client
    import app.main as main_mod

    small = replace(get_settings(), max_pdf_mb=1)
    monkeypatch.setattr(main_mod, "get_settings", lambda: small)
    r = c.post("/api/jobs", files={"file": ("a.pdf", b"%PDF-" + b"x" * (1024 * 1024 + 10), "application/pdf")})
    assert r.status_code == 413


def test_api_unknown_job(client):
    c, _ = client
    for path in ("/api/jobs/nope", "/api/jobs/nope/report", "/api/jobs/nope/events"):
        assert c.get(path).status_code == 404
    assert c.post("/api/jobs/nope/expand").status_code == 404
