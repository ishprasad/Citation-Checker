"""Job orchestration.

Stages overlap so results stream as early as possible:

    parse ──┬── resolve references ─────────────────┐
            └── extract claims (per sentence) ──┐   │
                                                └───┴─> per (claim, reference) pair:
                                                        retrieve -> judge -> escalate -> detectors

Each (claim, reference) pair starts as soon as both its claim and its reference are
ready. Every intermediate result is persisted to SQLite, so a crashed or restarted job
resumes where it stopped (and every model call is cached as well). A final pass
re-runs the detectors with paper-level context (the off-topic baseline needs all
citations) and publishes any verdicts that changed.

Events are published to an in-memory bus that the SSE endpoint relays to the UI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from app import demo, llm
from app.config import get_settings, portable_path
from app.db import Database, get_database
from app.models import (
    CitingSentence,
    Claim,
    Evidence,
    IngestResult,
    Job,
    Reference,
    Verdict,
)
from app.pipeline.claims import citation_clause, extract_claims, mark_duplicate_verdicts
from app.pipeline.detectors import apply_detectors, detect
from app.pipeline.evidence import EvidenceRetriever
from app.pipeline.ingest import INGEST_VERSION, IngestError, ingest_arxiv
from app.pipeline.judge import (
    JUDGE_FAILED_EXPLANATION,
    JudgeContext,
    escalate_pair,
    judge_pair,
    needs_escalation,
)
from app.pipeline.references import ReferenceResolver, read_reference_sidecar

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)
FINISHED = ("done", "partial", "failed")

# Bump when judging changes enough that earlier results should not be served from the cache.
PIPELINE_VERSION = 3
# Parsed arXiv papers are shared across checks under this pseudo job id in the artifacts table.
PAPER_CACHE = "_papers"

# Duration of an uncached check, fitted on real runs (three papers, 49-141 citation pairs,
# 140-310 s): a fixed part (downloads, start-up) plus a cost per (citing sentence, reference) pair.
ESTIMATE_BASE_SECONDS = 80
ESTIMATE_SECONDS_PER_PAIR = 1.6


def estimate_seconds(sentences: list[CitingSentence]) -> int:
    pairs = sum(len(s.reference_keys) for s in sentences)
    return round(ESTIMATE_BASE_SECONDS + ESTIMATE_SECONDS_PER_PAIR * pairs) if pairs else 0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobBusy(Exception):
    """The job is still running, so it cannot be expanded yet."""


def pair_id(claim_id: str, key: str) -> str:
    return f"{claim_id}|{key}"


# ----------------------------------------------------------------------------
# Storage and events
# ----------------------------------------------------------------------------


class JobStore:
    """Typed access to jobs and their intermediate results."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()

    def save_job(self, job: Job) -> None:
        self.db.save_job(job.id, job.model_dump_json(), job.status, job.created_at.timestamp())

    def load_job(self, job_id: str) -> Optional[Job]:
        data = self.db.load_job(job_id)
        return Job.model_validate_json(data) if data else None

    def unfinished_jobs(self) -> list[Job]:
        return [Job.model_validate_json(d) for d in self.db.jobs_with_status(("queued", "running"))]

    def put(self, job_id: str, kind: str, item_id: str, value: Any) -> None:
        if isinstance(value, BaseModel):
            data = value.model_dump_json()
        else:
            data = json.dumps([v.model_dump(mode="json") for v in value])
        self.db.put_artifact(job_id, kind, item_id, data)

    def get(self, job_id: str, kind: str, item_id: str, model: type[M]) -> Optional[M]:
        data = self.db.get_artifact(job_id, kind, item_id)
        return model.model_validate_json(data) if data else None

    def all(self, job_id: str, kind: str, model: type[M]) -> dict[str, M]:
        return {k: model.model_validate_json(v) for k, v in self.db.artifacts(job_id, kind).items()}

    def all_lists(self, job_id: str, kind: str, model: type[M]) -> dict[str, list[M]]:
        return {k: [model.model_validate(x) for x in json.loads(v)] for k, v in self.db.artifacts(job_id, kind).items()}


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[job_id].add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        self._subscribers[job_id].discard(queue)
        if not self._subscribers[job_id]:
            self._subscribers.pop(job_id, None)

    def publish(self, job_id: str, event: str, data: Any) -> None:
        for queue in list(self._subscribers.get(job_id, ())):
            queue.put_nowait({"event": event, "data": data})


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------


class JobRunner:
    def __init__(self, store: Optional[JobStore] = None, bus: Optional[EventBus] = None):
        settings = get_settings()
        self.store = store or JobStore()
        self.bus = bus or EventBus()
        self.slots = asyncio.Semaphore(settings.max_concurrent_jobs)
        self.tasks: dict[str, asyncio.Task] = {}
        self._prepare_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # -- public API ---------------------------------------------------------------

    async def prepare_arxiv(self, arxiv_id: str) -> tuple[IngestResult, bool]:
        """Download and parse an arXiv paper once; later checks of the same version reuse it.

        `arxiv_id` must be normalized. Returns (result, whether it came from the cache).
        """
        async with self._prepare_locks[arxiv_id]:
            cached = self.cached_paper(arxiv_id)
            if cached is not None:
                return cached, True
            result = await ingest_arxiv(arxiv_id)
            versioned = result.paper.arxiv_id or arxiv_id
            self.store.put(PAPER_CACHE, "ingest", _paper_key(versioned), result)
            if versioned != arxiv_id:
                self.store.db.put_artifact(PAPER_CACHE, "alias", arxiv_id, json.dumps(versioned))
            return result, False

    def cached_paper(self, arxiv_id: str) -> Optional[IngestResult]:
        """The parsed paper if it is cached. An unversioned ID maps to the version fetched last time."""
        versioned = arxiv_id if re.search(r"v\d+$", arxiv_id) else self._alias(arxiv_id)
        if not versioned:
            return None
        return self.store.get(PAPER_CACHE, "ingest", _paper_key(versioned), IngestResult)

    def _alias(self, arxiv_id: str) -> Optional[str]:
        data = self.store.db.get_artifact(PAPER_CACHE, "alias", arxiv_id)
        return json.loads(data) if data else None

    def reusable_job(self, paper_id: str, mode: str) -> Optional[Job]:
        """An earlier check of the same paper version that already answers a request for `mode`.

        A full check answers a quick request too, and so does a quick check of a paper short
        enough that it covered every citation. Finished checks win over running ones, so a
        second visitor gets the report instantly, or joins the live check of the first.
        """
        jobs = [Job.model_validate_json(d) for d in self.store.db.jobs_with_status(("done", "running", "queued"))]

        def complete(j: Job) -> bool:
            return j.mode == "full" or (j.sentences_total > 0 and j.sentences_checked >= j.sentences_total)

        covering = [
            j for j in jobs
            if j.source == "arxiv" and j.paper_id == paper_id and j.pipeline_version == PIPELINE_VERSION
            and (mode == "quick" or complete(j))
        ]
        finished = [j for j in covering if j.status == "done"]
        return max(finished or covering, key=lambda j: (complete(j), j.created_at), default=None)

    def active_jobs(self) -> int:
        return sum(not task.done() for task in self.tasks.values())

    def create_job(self, arxiv_id: Optional[str], mode: str = "quick", ingest: Optional[IngestResult] = None,
                   source: str = "arxiv") -> Job:
        """Start a check. Uploaded PDFs (`source="pdf"`) must come with their parsed `ingest`."""
        job = Job(id=uuid.uuid4().hex[:12], mode=mode, arxiv_id=arxiv_id, source=source,
                  pipeline_version=PIPELINE_VERSION)
        if ingest is not None:
            self._describe(job, ingest)
            self.store.put(job.id, "ingest", "paper", ingest)
        self.store.save_job(job)
        self.start(job.id)
        return job

    # -- demo -----------------------------------------------------------------------

    def create_demo_job(self, start: bool = True) -> Job:
        """A full check of the bundled demo paper (model calls are cached after the first run)."""
        paper = demo.load_demo_paper()
        if paper is None:
            raise FileNotFoundError("demo paper missing")
        job = Job(id=uuid.uuid4().hex[:12], mode="full", source="demo", arxiv_id=paper.paper.arxiv_id,
                  paper_id=paper.paper.id, title=paper.paper.title, pipeline_version=PIPELINE_VERSION)
        self.store.save_job(job)
        if start:
            self.start(job.id)
        return job

    def latest_demo_job(self) -> Optional[Job]:
        """The newest finished check of the current demo paper made by the current pipeline, if any."""
        paper = demo.load_demo_paper()
        if paper is None:
            return None
        candidates = [
            Job.model_validate_json(d) for d in self.store.db.jobs_with_status(("done",))
        ]
        candidates = [j for j in candidates if j.source == "demo" and j.paper_id == paper.paper.id
                      and j.pipeline_version == PIPELINE_VERSION]
        return max(candidates, key=lambda j: j.created_at, default=None)

    def snapshot_demo_job(self, path=None) -> Job:
        """The recorded demo check (demo/snapshot.json), imported on first use and reused after that.
        An import keeps the recorded check's creation time, which identifies it (and the original)."""
        data = json.loads((path or demo.SNAPSHOT_PATH).read_text(encoding="utf-8"))
        recorded = Job.model_validate(data["job"])
        for d in self.store.db.jobs_with_status(("done",)):
            j = Job.model_validate_json(d)
            if j.source == "demo" and j.paper_id == recorded.paper_id and j.created_at == recorded.created_at:
                return j
        return self.import_snapshot(data=data)

    def export_snapshot(self, job_id: str, path=None) -> None:
        """Save a finished job's results so the demo can be shown without network or API key."""
        job = self.store.load_job(job_id)
        db = self.store.db
        artifacts = {kind: db.artifacts(job_id, kind) for kind in ("ingest", "reference", "claims", "judged", "verdict")}
        # The snapshot is committed: cited-text paths are stored relative to the project.
        for key, data in artifacts["reference"].items():
            ref = Reference.model_validate_json(data)
            artifacts["reference"][key] = ref.model_copy(update={"text_path": portable_path(ref.text_path)}).model_dump_json()
        demo.write_json(path or demo.SNAPSHOT_PATH, {"job": json.loads(job.model_dump_json()), "artifacts": artifacts})

    def import_snapshot(self, path=None, data: Optional[dict] = None) -> Job:
        data = data or json.loads((path or demo.SNAPSHOT_PATH).read_text(encoding="utf-8"))
        job = Job.model_validate(data["job"]).model_copy(update={"id": uuid.uuid4().hex[:12]})
        for kind, items in data["artifacts"].items():
            for item_id, value in items.items():
                self.store.db.put_artifact(job.id, kind, item_id, value)
        job.message = None
        self.store.save_job(job)
        return job

    def expand(self, job_id: str) -> Job:
        """Upgrade a finished quick check to a full check; completed work is reused."""
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.mode == "full":
            return job
        if job.status not in FINISHED:
            raise JobBusy("This check is still running.")
        job.mode = "full"
        job.status = "queued"
        job.message = None
        job.estimate_seconds = job.rest_estimate_seconds
        self.store.save_job(job)
        self.start(job.id)
        return job

    def start(self, job_id: str) -> None:
        if job_id in self.tasks and not self.tasks[job_id].done():
            return
        self.tasks[job_id] = asyncio.create_task(self._run(job_id), name=f"job-{job_id}")

    def resume_unfinished(self) -> int:
        jobs = self.store.unfinished_jobs()
        for job in jobs:
            log.info("job=%s resuming after restart", job.id)
            self.start(job.id)
        return len(jobs)

    async def wait(self, job_id: str) -> Job:
        if job_id in self.tasks:
            await asyncio.shield(self.tasks[job_id])
        return self.store.load_job(job_id)

    # -- helpers ------------------------------------------------------------------

    def _save(self, job: Job) -> None:
        self.store.save_job(job)

    def _progress(self, job: Job, stage: str, *, done: Optional[int] = None, total: Optional[int] = None,
                  add_done: int = 0, add_total: int = 0, state: Optional[str] = None) -> None:
        p = next(p for p in job.progress if p.stage == stage)
        if total is not None:
            p.total = total
        if done is not None:
            p.done = done
        p.total += add_total
        p.done += add_done
        if state is not None:
            p.state = state
        elif p.state == "pending" and (p.total or p.done):
            p.state = "running"
        self._save(job)
        self.bus.publish(job.id, "progress", p.model_dump())

    def _status(self, job: Job, status: str, message: Optional[str] = None) -> None:
        job.status = status
        job.message = message
        if status == "running":
            job.started_at, job.finished_at = _utcnow(), None
        elif status in FINISHED:
            job.finished_at = _utcnow()
        job.usage = llm.get_llm().usage_summary(job.id)
        self._save(job)
        self.bus.publish(job.id, "status", {"status": status, "message": message})

    # -- the job ------------------------------------------------------------------

    async def _run(self, job_id: str) -> None:
        job = self.store.load_job(job_id)
        if job is None:
            return
        async with self.slots:
            with llm.job_context(job.id):
                pending: list[asyncio.Task] = []
                try:
                    self._status(job, "running")
                    await self._pipeline(job, pending)
                    self._status(job, "done")
                except llm.TokenBudgetExceeded:
                    log.warning("job=%s token budget exceeded; keeping partial results", job.id)
                    await _cancel(pending)
                    await self._finalize(job)
                    self._status(job, "partial", "This check reached its token budget. Results so far are shown.")
                except IngestError as exc:
                    log.info("job=%s ingest failed: %s", job.id, exc)
                    self._status(job, "failed", str(exc))
                except asyncio.CancelledError:
                    await _cancel(pending)
                    raise
                except Exception:  # noqa: BLE001 - report, never crash the server
                    log.exception("job=%s failed", job.id)
                    await _cancel(pending)
                    self._status(job, "failed", "Something went wrong while checking this paper.")
                self.bus.publish(job.id, "done", {"status": job.status, "message": job.message})

    async def _ingest(self, job: Job) -> IngestResult:
        cached = self.store.get(job.id, "ingest", "paper", IngestResult)
        if cached is not None:
            return cached
        self._progress(job, "parse", total=1)
        log.info("job=%s stage=parse source=%s arxiv=%s", job.id, job.source, job.arxiv_id)
        if job.source == "demo":
            result = demo.load_demo_paper()
            if result is None:
                raise IngestError("The demo paper is not available. Run scripts/plant_errors.py to create it.")
        elif job.source == "pdf":  # uploads are parsed before the job exists; the file itself is not kept
            raise IngestError("The uploaded PDF is no longer available. Please upload it again.")
        else:
            result, _ = await self.prepare_arxiv(job.arxiv_id)
        self.store.put(job.id, "ingest", "paper", result)
        return result

    def _describe(self, job: Job, result: IngestResult) -> None:
        """Fill in what is known about the check once the paper is parsed."""
        settings = get_settings()
        paper = result.paper
        checked = result.sentences[: settings.quick_check_max_citations] if job.mode == "quick" else result.sentences
        if job.source == "arxiv" and paper.arxiv_id:
            job.arxiv_id = paper.arxiv_id
        job.paper_id, job.title = paper.id, paper.title
        job.references_total = len(paper.references)
        job.sentences_total, job.sentences_checked = len(result.sentences), len(checked)
        if job.estimate_seconds is None:
            job.estimate_seconds = estimate_seconds(checked)
            if job.mode == "quick":
                job.rest_estimate_seconds = estimate_seconds(result.sentences[len(checked):])

    async def _pipeline(self, job: Job, pending: list[asyncio.Task]) -> None:
        settings = get_settings()
        result = await self._ingest(job)
        paper = result.paper
        sentences = result.sentences
        if job.mode == "quick":
            sentences = sentences[: settings.quick_check_max_citations]
        self._describe(job, result)
        self._progress(job, "parse", done=1, total=1, state="done")
        self.bus.publish(job.id, "paper", {"title": paper.title, "paper_id": paper.id, "notes": paper.notes,
                                           "sentences_total": job.sentences_total,
                                           "sentences_checked": job.sentences_checked})

        sentences_by_id = {s.id: s for s in sentences}
        needed = list(dict.fromkeys(k for s in sentences for k in s.reference_keys))
        by_key = {r.key: r for r in paper.references}
        loop = asyncio.get_running_loop()
        ref_futures: dict[str, asyncio.Future] = {k: loop.create_future() for k in needed}
        resolved: dict[str, Reference] = {}
        retriever = EvidenceRetriever()
        ctx = JudgeContext(paper=paper, sentences=sentences_by_id, references=resolved)

        # Restore what a previous run already finished.
        stored_refs = self.store.all(job.id, "reference", Reference)
        stored_claims = self.store.all_lists(job.id, "claims", Claim)
        stored_evidence = self.store.all(job.id, "evidence", Evidence)
        stored_judged = self.store.all(job.id, "judged", Verdict)

        # Pair counters are rebuilt from scratch on every (re)run; stored pairs count as done.
        for stage in ("retrieve", "judge", "escalate"):
            self._progress(job, stage, done=0, total=0, state="pending")
        escalated_before = sum(v.escalated for v in stored_judged.values())
        if escalated_before:
            self._progress(job, "escalate", done=escalated_before, total=escalated_before)

        # Stage: resolve references (in the background).
        todo_refs = [by_key[k] for k in needed if k not in stored_refs]
        self._progress(job, "resolve", total=len(needed), done=len(needed) - len(todo_refs))

        def reference_done(ref: Reference) -> None:
            resolved[ref.key] = ref
            if not ref_futures[ref.key].done():
                ref_futures[ref.key].set_result(ref)

        for key in needed:
            if key in stored_refs:
                reference_done(stored_refs[key])

        async def on_resolved(ref: Reference) -> None:
            self.store.put(job.id, "reference", ref.key, ref)
            reference_done(ref)
            self._progress(job, "resolve", add_done=1)
            self.bus.publish(job.id, "reference", _reference_summary(ref))

        async def resolve() -> None:
            log.info("job=%s stage=resolve references=%d", job.id, len(todo_refs))
            try:
                if todo_refs:
                    await ReferenceResolver().resolve_all(todo_refs, on_done=on_resolved)
            finally:
                # Never leave a pair waiting on a reference that will not come.
                for key in needed:
                    if not ref_futures[key].done():
                        reference_done(by_key[key].model_copy(update={
                            "fulltext_status": "unavailable", "status_reason": "Looking up this reference failed."}))
            self._progress(job, "resolve", state="done")
            self.bus.publish(job.id, "coverage", _coverage([resolved[k] for k in needed if k in resolved]))

        pending.append(asyncio.create_task(resolve()))

        # Per-pair work: retrieve -> judge -> escalate -> detectors.
        async def run_pair(claim: Claim, key: str, query_vec) -> None:
            pid = pair_id(claim.id, key)
            ref: Reference = await ref_futures[key]
            verdict = stored_judged.get(pid)
            if verdict is None:
                try:
                    evidence = stored_evidence.get(pid)
                    if evidence is None:
                        evidence = await retriever.retrieve(claim, ref, query_vec)
                        self.store.put(job.id, "evidence", pid, evidence)
                    self._progress(job, "retrieve", add_done=1)
                    verdict = await judge_pair(claim, ref, evidence, ctx)
                    self._progress(job, "judge", add_done=1)
                    if needs_escalation(verdict, claim, ref):
                        self._progress(job, "escalate", add_total=1)
                        self.bus.publish(job.id, "escalating", {"pair_id": pid})
                        verdict = await escalate_pair(claim, ref, verdict, evidence, ctx)
                        self._progress(job, "escalate", add_done=1)
                except llm.TokenBudgetExceeded:
                    raise
                except Exception:  # noqa: BLE001 - one bad pair never fails the job
                    log.exception("job=%s pair=%s failed", job.id, pid)
                    verdict = Verdict(claim_id=claim.id, reference_key=key, label="unverifiable", confidence=0.0,
                                      attributed_claim=claim.claim_text, explanation=JUDGE_FAILED_EXPLANATION,
                                      flags=["pipeline_error"])
                self.store.put(job.id, "judged", pid, verdict)
            else:
                self._progress(job, "retrieve", add_done=1)
                self._progress(job, "judge", add_done=1)
            # Pair-level detectors now; paper-level ones (off-topic baseline) in the final pass.
            clause = citation_clause(sentences_by_id[claim.citing_sentence_id], key, paper.text)
            final = detect(verdict, claim, ref, read_reference_sidecar, clause=clause)
            self.store.put(job.id, "verdict", pid, final)
            self.bus.publish(job.id, "verdict", {"verdict": final.model_dump(mode="json"),
                                                 "claim": claim.model_dump(mode="json")})

        async def start_pairs(sentence: CitingSentence, claims: list[Claim]) -> None:
            pairs = [(c, k) for c in claims for k in c.reference_keys if k in ref_futures]
            self._progress(job, "retrieve", add_total=len(pairs))
            self._progress(job, "judge", add_total=len(pairs))
            query_vecs = await retriever.query_vectors(claims) if pairs else {}
            for claim, key in pairs:
                pending.append(asyncio.create_task(run_pair(claim, key, query_vecs[claim.id])))

        # Stage: extract claims (overlaps with reference resolution).
        todo_sentences = [s for s in sentences if s.id not in stored_claims]
        self._progress(job, "extract", total=len(sentences), done=len(sentences) - len(todo_sentences))
        for sentence in sentences:
            if sentence.id in stored_claims:
                await start_pairs(sentence, stored_claims[sentence.id])

        async def on_extracted(sentence: CitingSentence, claims: list[Claim]) -> None:
            self.store.put(job.id, "claims", sentence.id, claims)
            self._progress(job, "extract", add_done=1)
            self.bus.publish(job.id, "claims", {"sentence_id": sentence.id,
                                                "claims": [c.model_dump(mode="json") for c in claims]})
            await start_pairs(sentence, claims)

        log.info("job=%s stage=extract sentences=%d", job.id, len(todo_sentences))
        await extract_claims(todo_sentences, paper, by_key, on_done=on_extracted)
        self._progress(job, "extract", state="done")

        # Wait for every pair (the list grows while extraction runs, so loop until stable).
        while True:
            unfinished = [t for t in pending if not t.done()]
            if not unfinished:
                break
            await asyncio.gather(*unfinished)
        for task in pending:  # surface the first error (e.g. token budget)
            if task.exception() is not None:
                raise task.exception()

        await self._finalize(job, retriever)

    async def _finalize(self, job: Job, retriever: Optional[EvidenceRetriever] = None) -> None:
        """Re-run detectors with paper-level context and publish verdicts whose tags changed."""
        claims = {c.id: c for cs in self.store.all_lists(job.id, "claims", Claim).values() for c in cs}
        references = self.store.all(job.id, "reference", Reference)
        judged = self.store.all(job.id, "judged", Verdict)
        evidence = list(self.store.all(job.id, "evidence", Evidence).values())
        current = self.store.all(job.id, "verdict", Verdict)
        verdicts = [v for v in judged.values() if v.claim_id in claims]
        ingest = self.store.get(job.id, "ingest", "paper", IngestResult)
        sentences = {s.id: s for s in ingest.sentences} if ingest else {}
        clauses = {
            (c.citing_sentence_id, key): citation_clause(sentences[c.citing_sentence_id], key, ingest.paper.text)
            for c in claims.values() if c.citing_sentence_id in sentences for key in c.reference_keys
        }
        try:
            scores = await (retriever or EvidenceRetriever()).abstract_scores(list(claims.values()), references)
        except llm.TokenBudgetExceeded:
            scores = None
        final = apply_detectors(verdicts, claims, references, evidence, abstract_scores=scores, clauses=clauses)
        final = mark_duplicate_verdicts(final, claims)
        for v in final:
            pid = pair_id(v.claim_id, v.reference_key)
            before = current.get(pid)
            if before is None or before.model_dump() != v.model_dump():
                self.store.put(job.id, "verdict", pid, v)
                self.bus.publish(job.id, "verdict", {"verdict": v.model_dump(mode="json"),
                                                     "claim": claims[v.claim_id].model_dump(mode="json")})
        for stage in ("retrieve", "judge", "escalate"):
            self._progress(job, stage, state="done")


async def _cancel(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _paper_key(versioned_arxiv_id: str) -> str:
    return f"{versioned_arxiv_id}|parser{INGEST_VERSION}"


def _reference_summary(ref: Reference) -> dict:
    return ref.model_dump(mode="json", include={
        "key", "label", "title", "year", "doi", "arxiv_id", "fulltext_status", "status_reason", "retracted",
    })


def _coverage(refs: list[Reference]) -> dict:
    from app.models import Coverage

    coverage = Coverage.of(refs)
    return {**coverage.model_dump(), "line": coverage.line()}


_runner: Optional[JobRunner] = None


def get_runner() -> JobRunner:
    global _runner
    if _runner is None:
        _runner = JobRunner()
    return _runner


def set_runner(runner: Optional[JobRunner]) -> None:
    global _runner
    _runner = runner
