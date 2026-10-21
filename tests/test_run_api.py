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
        self.ingest_exc: Exception | None = None
        self.ingested: list[str] = []


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
        fakes.ingested.append(arxiv_id)
        if fakes.ingest_exc:
            raise fakes.ingest_exc
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


API_SETTINGS = dict(quick_check_max_citations=3, max_references=150, max_jobs_per_ip_per_hour=50,
                    max_previews_per_ip_per_hour=50, max_concurrent_jobs=3, judge_mode_password="",
                    trust_proxy_headers=False)


def api_settings(**overrides):
    return replace(get_settings(), **{**API_SETTINGS, **overrides})


@pytest.fixture
def client(env, monkeypatch):
    fakes, store = env
    set_runner(JobRunner(store=store, bus=EventBus()))
    import app.main as main_mod
    from app.main import app

    settings = api_settings()  # independent of the developer's .env
    monkeypatch.setattr(main_mod, "get_settings", lambda: settings)

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
        (dict(files={"file": ("a.pdf", b"%PDF-1.4 x", "application/pdf")}), 422, "PDF"),
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


def test_ui_files_are_revalidated(client):
    c, _ = client
    for path in ("/", "/app.js", "/styles.css"):
        r = c.get(path)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
        # An unchanged file comes back as a short 304, still marked no-cache.
        again = c.get(path, headers={"If-None-Match": r.headers["etag"]})
        assert again.status_code == 304 and again.headers["cache-control"] == "no-cache"


# ----------------------------------------------------------------------------- demo


@pytest.fixture
def demo_files(monkeypatch, tmp_path):
    import app.demo as demo_mod

    ingest = fixture_ingest()
    ingest.paper.id = "demo:2101.00001v1:abc"
    s1 = ingest.sentences[0]
    truth = demo_mod.GroundTruth(arxiv_id="2101.00001v1", title="Fixture Paper", planted=[
        demo_mod.PlantedError(id="p1", sentence_id=s1.id, reference_keys=[s1.reference_keys[0]], mutation="number_change",
                              description='Changed "12%" to "19%"', original=s1.text, mutated=s1.text,
                              expected_labels=["overstated", "contradicted", "not_found"]),
    ])
    paths = {"PAPER_PATH": tmp_path / "paper.json", "GROUND_TRUTH_PATH": tmp_path / "ground_truth.json",
             "SNAPSHOT_PATH": tmp_path / "snapshot.json"}
    for name, path in paths.items():
        monkeypatch.setattr(demo_mod, name, path)
    paths["PAPER_PATH"].write_text(ingest.model_dump_json(), encoding="utf-8")
    paths["GROUND_TRUTH_PATH"].write_text(truth.model_dump_json(), encoding="utf-8")
    return paths


def test_api_demo_runs_scores_and_reuses(client, demo_files, monkeypatch):
    c, fakes = client
    r = c.post("/api/demo")
    assert r.status_code == 201
    job_id = r.json()["id"]
    job = _wait_done(c, job_id)
    assert job["source"] == "demo" and job["mode"] == "full"
    report = c.get(f"/api/jobs/{job_id}/report").json()
    demo_info = report["demo"]
    # The fake judge flags s1 (escalation turns it into supported), so the planted error is missed.
    assert demo_info["planted"] == 1 and demo_info["eval"]["planted"] == 1
    assert set(demo_info["planted_by_sentence"]) == {"s1"}
    assert demo_info["eval"]["summary_line"].startswith("Caught 0 of 1 planted errors")

    # A second click reuses the finished check; fresh=1 starts a new one.
    assert c.post("/api/demo").json()["id"] == job_id
    fresh = c.post("/api/demo?fresh=1").json()["id"]
    assert fresh != job_id
    _wait_done(c, fresh)

    # Once a check is recorded, the demo shows the recording (here the first check), not the newer one.
    run_mod.get_runner().export_snapshot(job_id)
    assert c.post("/api/demo").json()["id"] == job_id

    # A recording made elsewhere is imported once and reused; no API key is needed, and without one
    # fresh=1 shows the recording too.
    data = json.loads(demo_files["SNAPSHOT_PATH"].read_text(encoding="utf-8"))
    data["job"]["created_at"] = "2026-01-01T00:00:00Z"
    demo_files["SNAPSHOT_PATH"].write_text(json.dumps(data), encoding="utf-8")
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "get_settings", lambda: replace(get_settings(), nebius_api_key=""))
    r = c.post("/api/demo?fresh=1")
    snap_id = r.json()["id"]
    assert snap_id not in (job_id, fresh) and r.json()["job"]["status"] == "done"
    assert c.post("/api/demo").json()["id"] == snap_id
    snap = c.get(f"/api/jobs/{snap_id}/report").json()
    assert len(snap["verdicts"]) == len(report["verdicts"]) and snap["demo"]["eval"] is not None


def test_api_demo_fresh_run_needs_the_passcode_and_counts_toward_limits(client, demo_files, monkeypatch):
    # A new demo run is a paid full check: on a public server, ?fresh=1 must not bypass the passcode
    # or the per-IP limit. Showing the existing check stays open to everyone.
    c, _ = client
    first = c.post("/api/demo").json()["id"]
    _wait_done(c, first)
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "get_settings", lambda: api_settings(judge_mode_password="s3cret"))
    assert c.post("/api/demo").json()["id"] == first  # no passcode needed to see the demo
    r = c.post("/api/demo?fresh=1")
    assert r.status_code == 403 and r.json()["code"] == "passcode"
    r = c.post("/api/demo?fresh=1", json={"passcode": "wrong"})
    assert r.status_code == 403
    r = c.post("/api/demo?fresh=1", json={"passcode": "s3cret"})
    assert r.status_code == 201 and r.json()["id"] != first
    _wait_done(c, r.json()["id"])
    monkeypatch.setattr(main_mod, "get_settings",
                        lambda: api_settings(judge_mode_password="s3cret", max_jobs_per_ip_per_hour=1))
    r = c.post("/api/demo?fresh=1", json={"passcode": "s3cret"})
    assert r.status_code == 429 and r.json()["code"] == "rate_limited"


def test_api_demo_without_a_recording_reuses_only_the_current_pipeline(client, demo_files, monkeypatch):
    c, _ = client
    old = c.post("/api/demo").json()["id"]
    assert _wait_done(c, old)["pipeline_version"] == run_mod.PIPELINE_VERSION

    # After a pipeline change, the demo runs again instead of showing the older check.
    monkeypatch.setattr(run_mod, "PIPELINE_VERSION", run_mod.PIPELINE_VERSION + 1)
    new = c.post("/api/demo").json()["id"]
    assert new != old
    _wait_done(c, new)
    assert c.post("/api/demo").json()["id"] == new

    # Without an API key there is nothing current to show or run.
    monkeypatch.setattr(run_mod, "PIPELINE_VERSION", run_mod.PIPELINE_VERSION + 1)
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "get_settings", lambda: replace(get_settings(), nebius_api_key=""))
    assert c.post("/api/demo").status_code == 503


def test_api_demo_unavailable(client, monkeypatch, tmp_path):
    import app.demo as demo_mod

    monkeypatch.setattr(demo_mod, "PAPER_PATH", tmp_path / "missing.json")
    c, _ = client
    r = c.post("/api/demo")
    assert r.status_code == 501 and "plant_errors" in r.json()["detail"]


# --------------------------------------------------------------- live judging (section 10)


def test_estimate_seconds_grows_with_citation_pairs():
    sentences = fixture_ingest().sentences
    assert run_mod.estimate_seconds([]) == 0
    one, all_ = run_mod.estimate_seconds(sentences[:1]), run_mod.estimate_seconds(sentences)
    assert one == round(run_mod.ESTIMATE_BASE_SECONDS + 2 * run_mod.ESTIMATE_SECONDS_PER_PAIR)
    assert all_ > one


def test_rate_limiter_sliding_window():
    from app.limits import RateLimiter

    now = [1000.0]
    limiter = RateLimiter(clock=lambda: now[0])
    for _ in range(2):
        assert limiter.retry_after("ip", 2) is None
        limiter.record("ip")
    wait = limiter.retry_after("ip", 2)
    assert wait is not None and 3500 < wait <= 3601
    assert limiter.retry_after("other", 2) is None and limiter.retry_after("ip", 0) is None
    now[0] += 3601
    assert limiter.retry_after("ip", 2) is None


def test_passcode_and_client_ip():
    from types import SimpleNamespace

    from app.limits import client_ip, minutes_text, passcode_ok

    assert passcode_ok(api_settings(), None)
    locked = api_settings(judge_mode_password="s3cret")
    assert passcode_ok(locked, " s3cret ") and not passcode_ok(locked, "nope") and not passcode_ok(locked, None)
    request = SimpleNamespace(headers={"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, client=SimpleNamespace(host="10.0.0.1"))
    assert client_ip(request, api_settings()) == "10.0.0.1"
    # Two proxies: the outer one appended the client (1.2.3.4), the inner one the outer proxy (10.0.0.1).
    assert client_ip(request, api_settings(trust_proxy_headers=True, trusted_proxy_hops=2)) == "1.2.3.4"
    assert client_ip(request, api_settings(trust_proxy_headers=True)) == "10.0.0.1"  # one hop: the last entry
    # A client cannot dodge the limits by sending its own X-Forwarded-For: only the end is trusted.
    forged = SimpleNamespace(headers={"x-forwarded-for": "6.6.6.6, 1.2.3.4"}, client=SimpleNamespace(host="10.0.0.9"))
    assert client_ip(forged, api_settings(trust_proxy_headers=True)) == "1.2.3.4"
    assert client_ip(forged, api_settings(trust_proxy_headers=True, trusted_proxy_hops=5)) == "6.6.6.6"  # capped
    assert (minutes_text(40), minutes_text(300)) == ("about a minute", "about 5 minutes")


def test_prepare_caches_parsed_paper_by_version(env):
    fakes, store = env

    async def scenario():
        runner = JobRunner(store=store, bus=EventBus())
        first, cached1 = await runner.prepare_arxiv("2101.00001")
        again, cached2 = await runner.prepare_arxiv("2101.00001")  # alias -> v1
        versioned, cached3 = await runner.prepare_arxiv("2101.00001v1")
        return cached1, cached2, cached3, again.paper.arxiv_id

    assert asyncio.run(scenario()) == (False, True, True, "2101.00001v1")
    assert fakes.ingested == ["2101.00001"]


def test_api_preview(client):
    c, fakes = client
    r = c.get("/api/preview", params={"arxiv_id": "arXiv:2101.00001"})
    assert r.status_code == 200
    body = r.json()
    assert body["arxiv_id"] == "2101.00001v1" and body["title"] == "Fixture Paper"
    assert (body["references"], body["citing_sentences"]) == (3, 7)
    assert body["quick"]["sentences"] == 3 and body["full"]["sentences"] == 7
    assert 0 < body["quick"]["estimate_seconds"] < body["full"]["estimate_seconds"]
    assert body["full"]["allowed"] and body["quick"]["existing"] is None and not body["passcode_required"]
    assert c.get("/api/preview", params={"arxiv_id": "hello"}).status_code == 400
    assert fakes.ingested == ["2101.00001"]

    # Starting a check afterwards reuses the parsed paper.
    r = c.post("/api/jobs", json={"arxiv_id": "2101.00001", "mode": "quick"})
    assert r.status_code == 201 and fakes.ingested == ["2101.00001"]
    job = r.json()["job"]
    assert job["arxiv_id"] == "2101.00001v1" and job["references_total"] == 3
    assert job["estimate_seconds"] == body["quick"]["estimate_seconds"]
    assert job["rest_estimate_seconds"] == run_mod.estimate_seconds(fixture_ingest().sentences[3:])
    _wait_done(c, r.json()["id"])
    existing = c.get("/api/preview", params={"arxiv_id": "2101.00001"}).json()["quick"]["existing"]
    assert existing == {"id": r.json()["id"], "status": "done", "mode": "quick"}


def test_api_reuses_finished_checks(client):
    c, fakes = client
    quick = c.post("/api/jobs", json={"arxiv_id": "2101.00001v1", "mode": "quick"}).json()
    _wait_done(c, quick["id"])
    fakes.extracted.clear()

    again = c.post("/api/jobs", json={"arxiv_id": "https://arxiv.org/abs/2101.00001v1"})
    assert again.status_code == 200 and again.json()["cached"] and again.json()["id"] == quick["id"]
    assert fakes.extracted == []

    full = c.post("/api/jobs", json={"arxiv_id": "2101.00001v1", "mode": "full"})
    assert full.status_code == 201 and full.json()["id"] != quick["id"]  # a quick check does not answer a full one
    _wait_done(c, full.json()["id"])
    # Now a full check exists, and it answers quick requests too.
    assert c.post("/api/jobs", json={"arxiv_id": "2101.00001v1"}).json()["id"] == full.json()["id"]
    fresh = c.post("/api/jobs", json={"arxiv_id": "2101.00001v1", "fresh": True})
    assert fresh.status_code == 201 and fresh.json()["id"] not in (quick["id"], full.json()["id"])
    _wait_done(c, fresh.json()["id"])


def test_api_passcode_gates_full_checks(client, monkeypatch):
    c, _ = client
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "get_settings", lambda: api_settings(judge_mode_password="s3cret"))
    assert c.get("/api/health").json()["limits"]["passcode_required"] is True
    r = c.post("/api/jobs", json={"arxiv_id": "2101.00001v1", "mode": "full"})
    assert r.status_code == 403 and r.json()["code"] == "passcode" and "Quick checks are open" in r.json()["detail"]
    r = c.post("/api/jobs", json={"arxiv_id": "2101.00001v1", "mode": "full", "passcode": "wrong"})
    assert r.status_code == 403 and "does not match" in r.json()["detail"]

    quick = c.post("/api/jobs", json={"arxiv_id": "2101.00001v1", "mode": "quick"})
    assert quick.status_code == 201
    job_id = quick.json()["id"]
    _wait_done(c, job_id)
    assert c.post(f"/api/jobs/{job_id}/expand").status_code == 403
    r = c.post(f"/api/jobs/{job_id}/expand", json={"passcode": "s3cret"})
    assert r.status_code == 200 and r.json()["job"]["mode"] == "full"
    assert r.json()["job"]["estimate_seconds"] == r.json()["job"]["rest_estimate_seconds"]
    _wait_done(c, job_id)


def test_api_too_many_references(client, monkeypatch):
    c, _ = client
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "get_settings", lambda: api_settings(max_references=2))
    preview = c.get("/api/preview", params={"arxiv_id": "2101.00001v1"}).json()
    assert preview["full"]["allowed"] is False and "3 references" in preview["full"]["note"]
    r = c.post("/api/jobs", json={"arxiv_id": "2101.00001v1", "mode": "full"})
    assert r.status_code == 422 and r.json()["code"] == "too_many_references" and "quick check" in r.json()["detail"]
    quick = c.post("/api/jobs", json={"arxiv_id": "2101.00001v1", "mode": "quick"})
    assert quick.status_code == 201
    _wait_done(c, quick.json()["id"])
    assert c.post(f"/api/jobs/{quick.json()['id']}/expand").json()["code"] == "too_many_references"


def test_api_rate_limits(client, monkeypatch):
    c, _ = client
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "get_settings", lambda: api_settings(max_jobs_per_ip_per_hour=1))
    first = c.post("/api/jobs", json={"arxiv_id": "2101.00001v1"})
    assert first.status_code == 201
    _wait_done(c, first.json()["id"])
    # Reopening a cached result is free; new work is not.
    assert c.post("/api/jobs", json={"arxiv_id": "2101.00001v1"}).status_code == 200
    r = c.post("/api/jobs", json={"arxiv_id": "2101.00001v1", "fresh": True})
    assert r.status_code == 429 and r.json()["code"] == "rate_limited" and int(r.headers["retry-after"]) > 0
    assert "in the last hour" in r.json()["detail"]

    monkeypatch.setattr(main_mod, "get_settings", lambda: api_settings())
    monkeypatch.setattr(run_mod.get_runner(), "active_jobs", lambda: 3)
    r = c.post("/api/jobs", json={"arxiv_id": "2101.00001v1", "fresh": True})
    assert r.status_code == 429 and r.json()["code"] == "busy" and "busy with 3 other papers" in r.json()["detail"]


def test_api_preview_rate_limit_counts_only_downloads(client, monkeypatch):
    c, fakes = client
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "get_settings", lambda: api_settings(max_previews_per_ip_per_hour=1))
    for _ in range(3):  # one download, then cache hits
        assert c.get("/api/preview", params={"arxiv_id": "2101.00001v1"}).status_code == 200
    r = c.get("/api/preview", params={"arxiv_id": "2101.00002v1"})
    assert r.status_code == 429 and fakes.ingested == ["2101.00001v1"]


@pytest.mark.parametrize(
    "exc, status, code",
    [
        ("unusable", 422, "ingest_failed"),
        ("missing", 404, "not_found"),
        ("network", 502, "upstream"),
    ],
)
def test_api_ingest_errors(client, exc, status, code):
    import httpx

    from app.pipeline.ingest import InvalidArxivId

    c, fakes = client
    fakes.ingest_exc = {
        "unusable": IngestError("This PDF has almost no extractable text. It may be a scanned document."),
        "missing": InvalidArxivId("arXiv has no paper with ID 2101.00001."),
        "network": httpx.ConnectError("down"),
    }[exc]
    r = c.post("/api/jobs", json={"arxiv_id": "2101.00001"})
    assert r.status_code == status and r.json()["code"] == code
    assert c.get("/api/preview", params={"arxiv_id": "2101.00001"}).status_code == status


def test_api_quick_check_covering_everything_answers_full_requests(client, monkeypatch):
    c, _ = client
    import app.main as main_mod

    monkeypatch.setattr(run_mod, "get_settings", lambda: api_settings(quick_check_max_citations=50))
    monkeypatch.setattr(main_mod, "get_settings", lambda: api_settings(quick_check_max_citations=50))
    quick = c.post("/api/jobs", json={"arxiv_id": "2101.00001v1", "mode": "quick"}).json()
    assert quick["job"]["sentences_checked"] == quick["job"]["sentences_total"] == 7
    _wait_done(c, quick["id"])
    full = c.post("/api/jobs", json={"arxiv_id": "2101.00001v1", "mode": "full"})
    assert full.status_code == 200 and full.json()["id"] == quick["id"]


def test_api_pdf_upload_runs_a_check(client, monkeypatch):
    from .test_pdf import author_year_pdf

    c, fakes = client
    r = c.post("/api/jobs", files={"file": ("widgets.pdf", author_year_pdf(), "application/pdf")}, data={"mode": "quick"})
    assert r.status_code == 201, r.text
    job = r.json()["job"]
    assert job["source"] == "pdf" and job["arxiv_id"] is None and job["title"] == "Sparse Widgets Improve Gadget Throughput"
    done = _wait_done(c, r.json()["id"])
    assert done["status"] == "done" and fakes.ingested == []  # parsed in the request, not by the arXiv path
    report = c.get(f"/api/jobs/{r.json()['id']}/report").json()
    assert report["paper"]["parsed_from"] == "pdf" and report["paper"]["id"].startswith("pdf:")
    assert len(report["verdicts"]) == 4

    # Full checks of uploads need the passcode too.
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "get_settings", lambda: api_settings(judge_mode_password="s3cret"))
    r = c.post("/api/jobs", files={"file": ("w.pdf", author_year_pdf(), "application/pdf")}, data={"mode": "full"})
    assert r.status_code == 403
    r = c.post("/api/jobs", files={"file": ("w.pdf", author_year_pdf(), "application/pdf")},
               data={"mode": "full", "passcode": "s3cret"})
    assert r.status_code == 201
    _wait_done(c, r.json()["id"])
