"""Held-out evaluation: papers from several fields with planted errors, scored without tuning.

    python scripts/heldout.py screen              sample candidates per stratum and screen them (no model cost)
    python scripts/heldout.py plant [SLUG ...]    baseline full check of each accepted paper, then plant errors
    python scripts/heldout.py run [SLUG ...]      full check of each mutated paper, scored against the plan
    python scripts/heldout.py run-old [SLUG ...]  the same mutated papers checked by round 1's pipeline (paired comparison)
    python scripts/heldout.py review SLUG         review sheet for flags on untouched citations
    python scripts/heldout.py report              aggregate everything into eval/<round>/README.md

Every command takes --round N (default: the latest round).

Selection is fixed in advance (STRATA, CRITERIA): an OpenAlex random sample with a fixed seed
per stratum, screened in sample order, taking the first papers that pass. Papers used while
building or calibrating the checker are excluded. The pipeline is not changed between
planting and scoring; flags on untouched citations are reviewed by hand (annotations.json).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import demo, llm  # noqa: E402
from app.config import ROOT_DIR, get_settings  # noqa: E402
from app.http import fetch_cached  # noqa: E402
from app.models import Claim, Coverage, Evidence, IngestResult, Job, Reference, Verdict  # noqa: E402
from app.pipeline.ingest import IngestError, ingest_pdf, normalize_arxiv_id  # noqa: E402
from app.pipeline.judge import escalation_stats  # noqa: E402
from app.pipeline.references import OPENALEX_WORKS, ReferenceResolver, _get_json, _polite  # noqa: E402
from app.pipeline.run import PIPELINE_VERSION, JobRunner  # noqa: E402
from app.planting import NEUTRAL_SCOPE_PHRASES, plant  # noqa: E402

ARXIV = "S4306400194"
SAMPLE_SIZE, PLANT_SEED = 25, 7
MAX_SAMPLES = 3  # seeds s, s+1, s+2, screened in that order

# Each round is a fresh sample, fixed before screening. Round 1 measured the checker as built for
# the demo; its papers then became development data for the fixes measured by round 2.
ROUNDS = {
    1: {"dir": "heldout", "seed": 42, "per_stratum": 1},
    2: {"dir": "heldout-2", "seed": 101, "per_stratum": 2},
}
ROUND = max(ROUNDS)
EVAL_DIR = ROOT_DIR / "eval" / ROUNDS[ROUND]["dir"]
MANIFEST = EVAL_DIR / "manifest.json"
WORK_DIR = get_settings().cache_dir / ROUNDS[ROUND]["dir"]  # parsed source papers (not committed)
SAMPLE_SEED = ROUNDS[ROUND]["seed"]


def use_round(n: int) -> None:
    global ROUND, EVAL_DIR, MANIFEST, WORK_DIR, SAMPLE_SEED
    if n not in ROUNDS:
        raise SystemExit(f"Unknown round {n}; rounds: {sorted(ROUNDS)}")
    ROUND, SAMPLE_SEED = n, ROUNDS[n]["seed"]
    EVAL_DIR = ROOT_DIR / "eval" / ROUNDS[n]["dir"]
    MANIFEST = EVAL_DIR / "manifest.json"
    WORK_DIR = get_settings().cache_dir / ROUNDS[n]["dir"]
SWAP_OPTIONS = 5  # citation swaps may use any of the 5 least related references (each once)
COMMON = "best_oa_location.license:cc-by,type:article"  # CC BY: the mutated texts can be redistributed

STRATA = [
    {"name": "ml", "field": "Computer science: machine learning", "kind": "arxiv",
     "filter": f"locations.source.id:{ARXIV},primary_topic.subfield.id:1702,publication_year:2022-2023,{COMMON}"},
    {"name": "cv", "field": "Computer science: computer vision", "kind": "arxiv",
     "filter": f"locations.source.id:{ARXIV},primary_topic.subfield.id:1707,publication_year:2022-2023,{COMMON}"},
    {"name": "astro", "field": "Physics: astronomy and astrophysics", "kind": "arxiv",
     "filter": f"locations.source.id:{ARXIV},primary_topic.subfield.id:3103,publication_year:2022-2023,{COMMON}"},
    {"name": "medicine", "field": "Medicine", "kind": "pdf",
     "filter": f"primary_location.source.id:S202381698,primary_topic.field.id:27,publication_year:2021-2023,{COMMON}"},
    {"name": "biology", "field": "Biology / neuroscience", "kind": "pdf",
     "filter": f"primary_location.source.id:S1336409049,publication_year:2021-2023,{COMMON}"},
    {"name": "psychology", "field": "Psychology", "kind": "pdf",
     "filter": f"primary_location.source.id:S9692511,publication_year:2021-2023,{COMMON}"},
    {"name": "psych-plos", "field": "Psychology", "kind": "pdf",
     "filter": f"primary_location.source.id:S202381698,primary_topic.field.id:32,publication_year:2021-2023,{COMMON}"},
]
CRITERIA = {
    "citing_sentences": [25, 90],
    "min_citations_mapped": 0.5,  # share of in-text citations matched to a reference entry
    "min_fulltext_coverage": 0.5,  # share of cited references whose full text is openly available
}
# Used while building, debugging or calibrating the checker.
DEVELOPMENT_SET = {
    "2201.11903", "1706.03762", "2106.09685", "2310.06825",
    "10.1371/journal.pone.0262090", "10.1038/s41467-022-35648-w", "10.3389/fpsyg.2021.689563",
}
# Round 1's papers, used to develop the fixes that round 2 measures.
ROUND1_PAPERS = {
    "2208.01063", "2106.05735", "2304.13057",
    "10.1371/journal.pone.0267015", "10.7554/elife.87318", "10.1371/journal.pone.0288609",
}


def development_set() -> set[str]:
    return DEVELOPMENT_SET | (ROUND1_PAPERS if ROUND >= 2 else set())

log = logging.getLogger("heldout")


# ----------------------------------------------------------------------------
# Manifest and files
# ----------------------------------------------------------------------------


def load_manifest() -> dict[str, Any]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"round": ROUND, "strata": STRATA, "per_stratum": ROUNDS[ROUND]["per_stratum"], "criteria": CRITERIA,
            "sample": {"size": SAMPLE_SIZE, "seeds": list(range(SAMPLE_SEED, SAMPLE_SEED + MAX_SAMPLES))},
            "development_set": sorted(development_set()), "candidates": {}, "papers": []}


def save_manifest(m: dict[str, Any]) -> None:
    demo.write_json(MANIFEST, m)


def paper_dir(slug: str) -> Path:
    return EVAL_DIR / slug


def entry(m: dict[str, Any], slug: str) -> dict[str, Any]:
    for p in m["papers"]:
        if p["slug"] == slug:
            return p
    raise SystemExit(f"No accepted paper {slug!r}. Accepted: {[p['slug'] for p in m['papers']]}")


def selected(m: dict[str, Any], slugs: list[str]) -> list[dict[str, Any]]:
    return [entry(m, s) for s in slugs] if slugs else list(m["papers"])


# ----------------------------------------------------------------------------
# Screening (no model calls)
# ----------------------------------------------------------------------------


def _arxiv_id(work: dict) -> Optional[str]:
    for loc in work.get("locations") or []:
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:v\d+)?(?:\.pdf)?$", loc.get("landing_page_url") or "")
        if m:
            return m.group(1)
    return None


def _pdf_urls(work: dict) -> list[str]:
    urls = []
    for loc in [work.get("best_oa_location") or {}] + list(work.get("locations") or []):
        if loc.get("pdf_url") and loc["pdf_url"] not in urls:
            urls.append(loc["pdf_url"])
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    if doi.startswith("10.1371/"):
        urls.append(f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable")
    elif doi.startswith("10.3389/"):
        urls.append(f"https://www.frontiersin.org/articles/{doi}/pdf")
    elif doi.lower().startswith("10.7554/elife."):
        urls.append(f"https://elifesciences.org/articles/{doi.split('.')[-1]}.pdf")
    return urls


def _slug(stratum: str, work: dict) -> str:
    ident = _arxiv_id(work) or (work.get("doi") or work["id"]).rsplit("/", 1)[-1]
    return f"{stratum}-{re.sub(r'[^A-Za-z0-9.]+', '-', ident).strip('-').lower()}"


class Inconclusive(Exception):
    """A lookup failed (rate limit, outage): the candidate is neither accepted nor rejected, and
    screening of its stratum stops so that a later candidate cannot overtake it."""


async def sample(stratum: dict, seed: int = SAMPLE_SEED) -> list[dict]:
    data = await _get_json(OPENALEX_WORKS, "openalex_heldout", _polite({
        "filter": stratum["filter"], "sample": str(SAMPLE_SIZE), "seed": str(seed),
        "per-page": str(SAMPLE_SIZE),
        "select": "id,doi,title,publication_year,locations,best_oa_location,primary_location,referenced_works",
    }))
    if data is None:
        raise Inconclusive(f"the OpenAlex sample (seed {seed}) could not be fetched")
    return data.get("results", [])


async def openalex_text_share(work: dict) -> Optional[float]:
    """Share of the referenced works with an open full text according to OpenAlex (None if unknown).

    A quick pre-screen: the real check below resolves and downloads every reference."""
    ids = [w.rsplit("/", 1)[-1] for w in work.get("referenced_works") or []]
    if len(ids) < 10:
        return None  # arXiv records often carry no reference list
    open_text = 0
    for start in range(0, len(ids), 50):
        batch = ids[start : start + 50]
        data = await _get_json(OPENALEX_WORKS, "openalex_heldout", _polite({
            "filter": "openalex:" + "|".join(batch), "per-page": "50", "select": "id,locations"}))
        if data is None:
            raise Inconclusive("the OpenAlex reference lookup failed")
        for w in data.get("results", []):
            if any(loc.get("pdf_url") or "arxiv.org" in (loc.get("landing_page_url") or "")
                   for loc in w.get("locations") or []):
                open_text += 1
    return open_text / len(ids)


async def get_paper(runner: JobRunner, stratum: dict, work: dict) -> IngestResult:
    if stratum["kind"] == "arxiv":
        arxiv_id = _arxiv_id(work)
        if not arxiv_id:
            raise IngestError("no arXiv version")
        result, _ = await runner.prepare_arxiv(normalize_arxiv_id(arxiv_id))
        return result
    for url in _pdf_urls(work):
        try:
            data = await fetch_cached(url, "heldout_pdf", suffix=".pdf", allow_missing=True)
        except Exception as exc:  # noqa: BLE001 - try the next location
            log.info("download failed %s: %s", url, exc)
            continue
        if data and data.startswith(b"%PDF-"):
            result = await ingest_pdf(data)
            result.paper.id = "pdf:" + hashlib.sha256(data).hexdigest()[:16]
            result.paper.title = work.get("title") or result.paper.title
            return result
    raise IngestError("no downloadable PDF")


async def screen_one(runner: JobRunner, stratum: dict, work: dict) -> tuple[bool, str, dict, Optional[IngestResult]]:
    ident = _arxiv_id(work) or (work.get("doi") or "").replace("https://doi.org/", "")
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    excluded = {x.lower() for x in development_set()}
    if ident.lower() in excluded or doi.lower() in excluded:
        return False, "used during development", {}, None
    share = await openalex_text_share(work)
    if share is not None and share < CRITERIA["min_fulltext_coverage"]:
        return False, f"OpenAlex lists open full text for {share:.0%} of the references (need 50%)", {}, None
    try:
        result = await get_paper(runner, stratum, work)
    except IngestError as exc:
        return False, f"text not available ({exc})", {}, None
    except Exception as exc:  # noqa: BLE001 - network failures: try again later
        raise Inconclusive(f"the paper could not be downloaded ({type(exc).__name__}: {exc})") from exc
    paper, sentences = result.paper, result.sentences
    stats: dict[str, Any] = {"citing_sentences": len(sentences), "parsed_from": paper.parsed_from,
                             "citations_mapped": f"{paper.citation_keys_mapped}/{paper.citation_keys_total}"}
    lo, hi = CRITERIA["citing_sentences"]
    if not lo <= len(sentences) <= hi:
        return False, f"{len(sentences)} citing sentences (need {lo}-{hi})", stats, None
    if paper.citation_keys_total and paper.citation_keys_mapped < CRITERIA["min_citations_mapped"] * paper.citation_keys_total:
        return False, "fewer than half of the in-text citations matched a reference", stats, None
    cited = {k for s in sentences for k in s.reference_keys}
    refs = [r for r in paper.references if r.key in cited]
    resolver = ReferenceResolver()
    resolved = await resolver.resolve_all(refs)
    coverage = Coverage.of(resolved)
    stats["coverage"] = coverage.line()
    share = coverage.full / coverage.total if coverage.total else 0.0
    unknown = coverage.total if resolver.prefetch_degraded else resolver.incomplete
    best_case = (coverage.full + unknown) / coverage.total if coverage.total else 0.0
    if share < CRITERIA["min_fulltext_coverage"] <= best_case:
        # Failed lookups only lower coverage: an acceptance stands, and a rejection stands when even
        # success on every failed lookup would leave coverage below the bar. Otherwise, try again.
        raise Inconclusive(f"full text for {share:.0%} of cited references, but {resolver.incomplete} "
                           "reference lookups failed")
    if share < CRITERIA["min_fulltext_coverage"]:
        return False, f"full text for {share:.0%} of cited references (need 50%)", stats, None
    return True, "accepted", stats, result


async def cmd_screen(args) -> int:
    m = load_manifest()
    m["sample"] = {"size": SAMPLE_SIZE, "seeds": list(range(SAMPLE_SEED, SAMPLE_SEED + MAX_SAMPLES))}
    runner = JobRunner()
    accepted = {p["slug"] for p in m["papers"]}
    for stratum in STRATA:
        if args.only and stratum["name"] not in args.only:
            continue
        have = sum(p["stratum"] == stratum["name"] for p in m["papers"])
        want = ROUNDS[ROUND]["per_stratum"]
        log_rows = m["candidates"].setdefault(stratum["name"], [])
        done = {row["openalex_id"] for row in log_rows}
        # Fixed before screening medicine: if a sample has no eligible paper, draw the next seeded
        # sample (skipping decided candidates), up to MAX_SAMPLES; then report the stratum as empty.
        works: list[dict] = []
        try:
            for seed in range(SAMPLE_SEED, SAMPLE_SEED + MAX_SAMPLES):
                works += [w for w in await sample(stratum, seed) if w["id"] not in {x["id"] for x in works}]
        except Inconclusive as exc:
            print(f"\n== {stratum['name']}: INCONCLUSIVE ({exc}); run `screen` again later")
            continue
        print(f"\n== {stratum['name']} ({stratum['field']}): {len(works)} sampled, {have}/{want} accepted")
        for rank, work in enumerate(works, start=1):
            if have >= want:
                break
            if work["id"] in done:
                continue
            try:
                ok, reason, stats, result = await screen_one(runner, stratum, work)
            except Inconclusive as exc:
                print(f"  {rank:>2}. INCONCLUSIVE  {(work.get('title') or '')[:70]}  ({exc}); "
                      "stopping this stratum, run `screen` again later")
                break
            row = {"rank": rank, "openalex_id": work["id"], "doi": work.get("doi"), "arxiv_id": _arxiv_id(work),
                   "title": work.get("title"), "year": work.get("publication_year"),
                   "decision": "accepted" if ok else "rejected", "reason": reason, **stats}
            log_rows.append(row)
            print(f"  {rank:>2}. {'ACCEPT' if ok else 'reject'}  {(work.get('title') or '')[:70]}  ({reason}) {stats}")
            if ok and result is not None:
                slug = _slug(stratum["name"], work)
                if slug not in accepted:
                    WORK_DIR.mkdir(parents=True, exist_ok=True)
                    (WORK_DIR / f"{slug}.ingest.json").write_text(result.model_dump_json(), encoding="utf-8")
                    m["papers"].append({
                        "slug": slug, "stratum": stratum["name"], "field": stratum["field"], "kind": stratum["kind"],
                        "title": result.paper.title, "arxiv_id": result.paper.arxiv_id,
                        "doi": (work.get("doi") or "").replace("https://doi.org/", "") or None,
                        "openalex_id": work["id"], "sample_rank": rank, **stats,
                    })
                    accepted.add(slug)
                have += 1
            save_manifest(m)
    print(f"\nAccepted papers: {[p['slug'] for p in m['papers']]}")
    return 0


# ----------------------------------------------------------------------------
# Planting and running (model calls)
# ----------------------------------------------------------------------------


def load_source(slug: str) -> IngestResult:
    path = WORK_DIR / f"{slug}.ingest.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `screen` again.")
    return IngestResult.model_validate_json(path.read_text(encoding="utf-8"))


def job_outputs(runner: JobRunner, job_id: str):
    store = runner.store
    claims = {c.id: c for cs in store.all_lists(job_id, "claims", Claim).values() for c in cs}
    verdicts = [v for v in store.all(job_id, "verdict", Verdict).values() if not v.duplicate_of]
    references = store.all(job_id, "reference", Reference)
    return claims, verdicts, references


OLD_PIPELINE_FILE = "old_pipeline.json"  # written by scripts/heldout_old.py


def old_outputs(slug: str):
    """(raw data, claims, verdicts, references, evidence) of the old-pipeline check, or None."""
    path = paper_dir(slug) / OLD_PIPELINE_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    claims = {c["id"]: Claim.model_validate(c) for c in data["claims"]}
    verdicts = [Verdict.model_validate(v) for v in data["verdicts"]]
    references = {k: Reference.model_validate(r) for k, r in data["references"].items()}
    evidence = {k: Evidence.model_validate(e) for k, e in data["evidence"].items()}
    return data, claims, verdicts, references, evidence


async def full_check(runner: JobRunner, ingest: IngestResult, kind: str, job_id: Optional[str],
                     references: Optional[dict[str, Reference]] = None) -> str:
    """A full check; with `references`, the job uses exactly those resolved references (the
    baseline's), so the evaluation differs from the baseline only by the planted edits."""
    if job_id:
        job = runner.store.load_job(job_id)
        if job and job.status == "done":
            return job_id
    job = Job(id=uuid.uuid4().hex[:12], mode="full", source=kind, pipeline_version=PIPELINE_VERSION,
              arxiv_id=ingest.paper.arxiv_id if kind == "arxiv" else None)
    runner._describe(job, ingest)
    runner.store.put(job.id, "ingest", "paper", ingest)
    for key, ref in (references or {}).items():
        runner.store.put(job.id, "reference", key, ref)
    runner.store.save_job(job)
    runner.start(job.id)
    job = await runner.wait(job.id)
    if job.status != "done":
        raise RuntimeError(f"job {job.id} ended {job.status}: {job.message}")
    return job.id


async def plant_one(runner: JobRunner, m: dict, p: dict) -> None:
    ingest = load_source(p["slug"])
    print(f"[{p['slug']}] baseline full check ({len(ingest.sentences)} citing sentences)...")
    t0 = time.perf_counter()
    p["baseline_job"] = await full_check(runner, ingest, p["kind"], p.get("baseline_job"))
    save_manifest(m)
    claims, verdicts, references = job_outputs(runner, p["baseline_job"])
    result = await plant(ingest, claims, verdicts, references, seed=PLANT_SEED, id_prefix="heldout",
                         field_name=p["field"], source=p.get("doi") or f"arXiv:{p.get('arxiv_id')}",
                         swap_options=SWAP_OPTIONS, scope_phrases=NEUTRAL_SCOPE_PHRASES)
    d = paper_dir(p["slug"])
    demo.write_json(d / "paper.json", result.mutated.model_dump(mode="json"))
    demo.write_json(d / "ground_truth.json", result.truth.model_dump(mode="json"))
    usage = llm.get_llm().usage_summary(p["baseline_job"])
    p.update(planted=len(result.truth.planted), plant_candidates=result.candidates,
             baseline_flags=len(result.truth.baseline_flags),
             baseline_cost_usd=round(usage.total_estimated_cost_usd, 3), pipeline_version=PIPELINE_VERSION)
    save_manifest(m)
    print(f"[{p['slug']}] planted {len(result.truth.planted)} errors (candidates {result.candidates}) "
          f"in {time.perf_counter() - t0:.0f}s, baseline cost ${usage.total_estimated_cost_usd:.2f}")


async def run_one(runner: JobRunner, m: dict, p: dict) -> None:
    d = paper_dir(p["slug"])
    mutated = IngestResult.model_validate_json((d / "paper.json").read_text(encoding="utf-8"))
    print(f"[{p['slug']}] full check of the mutated paper...")
    t0 = time.perf_counter()
    _, _, baseline_refs = job_outputs(runner, p["baseline_job"])
    p["eval_job"] = await full_check(runner, mutated, p["kind"], p.get("eval_job"), references=baseline_refs)
    p["eval_wall_seconds"] = round(time.perf_counter() - t0, 1)
    save_manifest(m)
    score_one(runner, p)


def score_one(runner: JobRunner, p: dict) -> dict:
    d = paper_dir(p["slug"])
    truth = demo.GroundTruth.model_validate_json((d / "ground_truth.json").read_text(encoding="utf-8"))
    claims, verdicts, references = job_outputs(runner, p["eval_job"])
    result = demo.score(truth, verdicts, claims, annotations=demo.load_annotations(d / "annotations.json"))
    esc = escalation_stats(verdicts)
    coverage = Coverage.of(list(references.values()))
    usage = llm.get_llm().usage_summary(p["eval_job"])
    out = {
        "slug": p["slug"], "field": p["field"], "title": p["title"], "job_id": p["eval_job"],
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{k: v for k, v in result.items() if k != "details"}, "details": result["details"],
        "verdicts": len(verdicts), "escalation": esc.model_dump(),
        "coverage": {**coverage.model_dump(), "line": coverage.line()},
        "usage": usage.model_dump(), "baseline_cost_usd": p.get("baseline_cost_usd"),
    }
    old = old_outputs(p["slug"])
    if old is not None:
        data, old_claims, old_verdicts, _, _ = old
        r = demo.score(truth, old_verdicts, old_claims, annotations=demo.load_annotations(d / "annotations.json"))
        out["old_pipeline"] = {
            "commit": data["commit"], "job_id": data["job_id"], "cost_usd": data["usage"]["total_estimated_cost_usd"],
            **{k: v for k, v in r.items() if k not in ("details", "summary_line")},
            "caught_ids": [x["id"] for x in r["details"] if x["caught"]],
        }
    out["caught_ids"] = [x["id"] for x in result["details"] if x["caught"]]
    demo.write_json(d / "results.json", out)
    print(f"[{p['slug']}] {result['summary_line']} · flagged untouched citations: {result['flagged_citations']} "
          f"(reviewed: {result['review']})")
    return out


async def cmd_plant(args) -> int:
    m = load_manifest()
    runner = JobRunner()
    papers = [p for p in selected(m, args.slugs) if args.force or not p.get("planted")]
    with _quiet():
        await asyncio.gather(*(plant_one(runner, m, p) for p in papers))
    return 0


async def cmd_run(args) -> int:
    m = load_manifest()
    runner = JobRunner()
    papers = [p for p in selected(m, args.slugs) if p.get("planted")]
    with _quiet():
        await asyncio.gather(*(run_one(runner, m, p) for p in papers))
    return 0


def cmd_run_old(args) -> int:
    """Paired comparison: the same edited papers, checked by round 1's pipeline (scripts/heldout_old.py)."""
    m = load_manifest()
    slugs = [p["slug"] for p in selected(m, args.slugs) if p.get("planted")]
    if not slugs:
        print("Nothing planted yet.")
        return 1
    script = Path(__file__).resolve().parent / "heldout_old.py"
    return subprocess.run([sys.executable, "-u", str(script), "--round", str(ROUND), *slugs]).returncode


# ----------------------------------------------------------------------------
# Review sheets
# ----------------------------------------------------------------------------


def cmd_review(args) -> int:
    m = load_manifest()
    runner = JobRunner()
    p = entry(m, args.slug)
    d = paper_dir(p["slug"])
    truth = demo.GroundTruth.model_validate_json((d / "ground_truth.json").read_text(encoding="utf-8"))
    mutated = IngestResult.model_validate_json((d / "paper.json").read_text(encoding="utf-8"))
    sentences = {s.id: s for s in mutated.sentences}
    planted = {x.sentence_id for x in truth.planted}
    annotations_path = d / "annotations.json"
    existing = demo.load_annotations(annotations_path)

    # Flags on untouched citations from the current pipeline and, for the paired comparison, the
    # old one. A citation is reviewed once, whichever pipeline flagged it.
    runs = [("current", *job_outputs(runner, p["eval_job"]), runner.store.all(p["eval_job"], "evidence", Evidence))]
    old = old_outputs(p["slug"])
    if old is not None:
        runs.append((f"old pipeline ({old[0]['commit']})", *old[1:]))
    flagged: dict[tuple[str, str], dict[str, list[Verdict]]] = {}
    for name, claims, verdicts, _, _ in runs:
        for v in verdicts:
            sid = claims[v.claim_id].citing_sentence_id
            if sid not in planted and demo.is_flag(v):
                flagged.setdefault((sid, v.reference_key), {}).setdefault(name, []).append(v)

    lines = [f"# Review: {p['title']}", "", f"Field: {p['field']} · job `{p['eval_job']}` · "
             f"{len(flagged)} flagged citations on untouched sentences"
             + (" (current and old pipeline)." if old is not None else "."), "",
             "Mark each one in annotations.json as `genuine` (a real citation problem), `false_alarm`, or "
             "`debatable`, with a one-line note.", ""]
    for (sid, key), by_run in sorted(flagged.items(), key=lambda kv: sentences[kv[0][0]].start):
        s = sentences[sid]
        ref = next((refs.get(key) for _, _, _, refs, _ in runs if refs.get(key)), None)
        lines += [f"## {sid} → {ref.label if ref else key}: {ref.title if ref else ''}",
                  f"*Flagged by: {', '.join(by_run)}*", "", f"> {s.text}", ""]
        if s.context_before:
            lines += [f"Before: {s.context_before[:400]}", ""]
        for name, claims, verdicts, refs, evidence in runs:
            run_ref = refs.get(key)
            lines.append(f"**{name}** · reference text: {run_ref.fulltext_status if run_ref else '?'}"
                         + (f" · `{run_ref.text_path}`" if run_ref and run_ref.text_path else ""))
            for v in [v for v in verdicts if v.reference_key == key and claims[v.claim_id].citing_sentence_id == sid]:
                c = claims[v.claim_id]
                lines += [f"- **{v.label}** ({v.confidence:.2f}){' · escalated' if v.escalated else ''}"
                          f"{' · ' + ', '.join(v.failure_modes) if v.failure_modes else ''}",
                          f"  - Claim ({c.claim_type}): {c.claim_text}",
                          f"  - Attributed: {v.attributed_claim}",
                          f"  - Explanation: {v.explanation}",
                          f"  - Quote{' (verified)' if v.quote_verified else ''}: {(v.quote or '-')[:500]}"]
                for note in v.failure_notes or []:
                    lines.append(f"  - Note: {note}")
                ev = evidence.get(f"{v.claim_id}|{key}")
                if ev:
                    for passage in ev.passages[:3]:
                        where = "abstract" if passage.chunk_index < 0 else (passage.heading or "text")
                        lines.append(f"  - Passage ({passage.score:.2f}, {where}): {passage.text[:600]}")
        lines.append("")
    (d / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    skeleton = {"paper": p["slug"], "annotations": [
        existing.get((sid, key)) or {"sentence_id": sid, "reference_key": key, "verdict": "", "note": ""}
        for (sid, key) in sorted(flagged, key=lambda k: sentences[k[0]].start)
    ]}
    demo.write_json(annotations_path, skeleton)
    print(f"Wrote {d / 'review.md'} and {annotations_path} ({len(flagged)} flagged citations)")
    return 0


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs (b: only A right, c: only B right)."""
    n = b + c
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_comparison(scored: list[dict]) -> Optional[dict]:
    """The current and the old pipeline on the same edited papers: planted errors caught by each,
    and flags on untouched citations by review verdict. Papers without an old run are left out."""
    papers = [r for r in scored if "old_pipeline" in r]
    if not papers:
        return None
    caught = {"both": 0, "current_only": 0, "old_only": 0, "neither": 0}
    flags: dict[str, dict[str, int]] = {}
    totals = {"current": {"planted": 0, "caught": 0, "genuine": 0, "false": 0, "debatable": 0, "cost": 0.0},
              "old": {"planted": 0, "caught": 0, "genuine": 0, "false": 0, "debatable": 0, "cost": 0.0}}
    for r in papers:
        old = r["old_pipeline"]
        for d in r["details"]:
            now, before = d["id"] in r["caught_ids"], d["id"] in old["caught_ids"]
            caught["both" if now and before else "current_only" if now else "old_only" if before else "neither"] += 1
        for name, res in (("current", r), ("old", old)):
            t = totals[name]
            t["planted"] += res["planted"]
            t["caught"] += res["caught"]
            t["genuine"] += res["review"]["genuine"]
            t["false"] += res["review"]["false_alarm"] + res["review"]["unreviewed"]
            t["debatable"] += res["review"]["debatable"]
        totals["current"]["cost"] += r["usage"]["total_estimated_cost_usd"]
        totals["old"]["cost"] += old["cost_usd"]
        sets = {name: {verdict: {(x["sentence_id"], x["reference_key"]) for x in items}
                       for verdict, items in res["review_details"].items()}
                for name, res in (("current", r), ("old", old))}
        for verdict in ("genuine", "false_alarm", "debatable", "unreviewed"):
            now, before = sets["current"][verdict], sets["old"][verdict]
            row = flags.setdefault(verdict, {"both": 0, "current_only": 0, "old_only": 0})
            row["both"] += len(now & before)
            row["current_only"] += len(now - before)
            row["old_only"] += len(before - now)
    for t in totals.values():
        good = t["caught"] + t["genuine"]
        t["recall"] = t["caught"] / t["planted"] if t["planted"] else 0.0
        t["precision_reviewed"] = good / (good + t["false"]) if good + t["false"] else 0.0
    return {"papers": len(papers), "old_commit": papers[0]["old_pipeline"]["commit"], "totals": totals,
            "planted": caught, "recall_mcnemar_p": mcnemar_exact(caught["current_only"], caught["old_only"]),
            "flags": flags}


def paired_section(c: dict, table) -> list[str]:
    cur, old = c["totals"]["current"], c["totals"]["old"]
    pl, fl = c["planted"], c["flags"]

    def row(name):
        f = fl.get(name, {"both": 0, "current_only": 0, "old_only": 0})
        return f["both"], f["current_only"], f["old_only"]

    return [
        f"## Paired comparison with round 1's pipeline ({c['old_commit']})",
        "",
        f"The same {c['papers']} edited papers checked by the pipeline round 1 measured "
        f"(`scripts/heldout_old.py`, its own reference resolution), so the difference between the columns is "
        "the effect of the fixes, not of which papers were drawn. Flags from either pipeline are reviewed in the "
        "same annotations.",
        "",
        table(("Metric", "Current pipeline", f"Round 1 pipeline ({c['old_commit']})"), [
            ("Planted errors caught", f"{cur['caught']}/{cur['planted']} = {cur['recall']:.0%}",
             f"{old['caught']}/{old['planted']} = {old['recall']:.0%}"),
            ("Genuine problems found / false alarms / debatable", f"{cur['genuine']} / {cur['false']} / {cur['debatable']}",
             f"{old['genuine']} / {old['false']} / {old['debatable']}"),
            ("Precision with human review", f"{cur['precision_reviewed']:.0%}", f"{old['precision_reviewed']:.0%}"),
            ("Model cost of the evaluation checks", f"${cur['cost']:.2f}", f"${old['cost']:.2f}"),
        ]),
        "",
        table(("", "Both pipelines", "Current only", "Round 1 pipeline only"), [
            ("Planted errors caught", pl["both"], pl["current_only"], pl["old_only"]),
            ("Genuine problems flagged", *row("genuine")),
            ("False alarms", *row("false_alarm")),
            ("Debatable flags", *row("debatable")),
        ]),
        "",
        f"Planted errors caught by neither: {pl['neither']}. Exact McNemar test on the planted errors caught by only "
        f"one pipeline: p = {c['recall_mcnemar_p']:.2f}.",
        "",
    ]


def _metric_cells(r: dict) -> list[str]:
    """The results table's values for one round's summary."""
    return [
        f"{r['papers']} ({r['fields']} fields)" if r.get("fields") else str(r["papers"]),
        f"{r['caught']}/{r['planted']} = {r['recall']:.0%} (95% CI {r['recall_ci95'][0]:.0%}–{r['recall_ci95'][1]:.0%})",
        f"{r['label_accuracy']:.0%}",
        f"{r['expected_mode_rate']:.0%}",
        f"{r['genuine']} / {r['false_alarms']} / {r['debatable']}" + (f" (+{r['unreviewed']} unreviewed)" if r["unreviewed"] else ""),
        f"{r['precision_reviewed']:.0%} (95% CI {r['precision_ci95'][0]:.0%}–{r['precision_ci95'][1]:.0%})",
        f"{r['escalated']} of {r['verdicts']} / {r['escalation_changed']}",
        f"${r['cost_usd']:.2f}",
    ]


def cmd_report(args) -> int:
    m = load_manifest()
    runner = JobRunner()
    rows, totals = [], {"planted": 0, "caught": 0, "label_ok": 0, "mode_ok": 0, "genuine": 0, "false": 0,
                        "debatable": 0, "unreviewed": 0, "verdicts": 0, "escalated": 0, "changed": 0, "cost": 0.0}
    by_mutation: dict[str, list[int]] = {}
    scored = []
    for p in m["papers"]:
        if not p.get("eval_job"):
            continue
        r = score_one(runner, p)
        scored.append(r)
        rv = r["review"]
        totals["planted"] += r["planted"]
        totals["caught"] += r["caught"]
        totals["label_ok"] += round(r["label_accuracy"] * r["planted"])
        totals["mode_ok"] += round(r["expected_mode_rate"] * r["planted"])
        totals["genuine"] += rv["genuine"]
        totals["false"] += rv["false_alarm"]
        totals["debatable"] += rv["debatable"]
        totals["unreviewed"] += rv["unreviewed"]
        totals["verdicts"] += r["verdicts"]
        totals["escalated"] += r["escalation"]["escalated"]
        totals["changed"] += r["escalation"]["label_changed"]
        cost = r["usage"]["total_estimated_cost_usd"] + (r.get("baseline_cost_usd") or 0.0)
        totals["cost"] += cost
        for mut, row in r["by_mutation"].items():
            agg = by_mutation.setdefault(mut, [0, 0])
            agg[0] += row["planted"]
            agg[1] += row["caught"]
        correct = r["caught"] + rv["genuine"]
        wrong = rv["false_alarm"] + rv["unreviewed"]
        rows.append((p["field"], p["title"][:60], p.get("parsed_from", ""), r["coverage"]["line"].split(" · ")[0],
                     f"{r['caught']}/{r['planted']}", f"{rv['genuine']}", f"{rv['false_alarm']}"
                     + (f" (+{rv['unreviewed']} unreviewed)" if rv["unreviewed"] else ""), f"{rv['debatable']}",
                     f"{correct / (correct + wrong):.0%}" if correct + wrong else "-", f"${cost:.2f}"))
    if not rows:
        print("Nothing scored yet.")
        return 1
    t = totals
    correct, wrong = t["caught"] + t["genuine"], t["false"] + t["unreviewed"]
    rlo, rhi = wilson(t["caught"], t["planted"])
    plo, phi = wilson(correct, correct + wrong)
    summary = {
        "papers": len(rows), "planted": t["planted"], "caught": t["caught"],
        "recall": t["caught"] / t["planted"], "recall_ci95": [rlo, rhi],
        "label_accuracy": t["label_ok"] / t["planted"], "expected_mode_rate": t["mode_ok"] / t["planted"],
        "genuine": t["genuine"], "false_alarms": t["false"], "debatable": t["debatable"], "unreviewed": t["unreviewed"],
        "precision_reviewed": correct / (correct + wrong) if correct + wrong else 0.0, "precision_ci95": [plo, phi],
        "verdicts": t["verdicts"], "escalated": t["escalated"], "escalation_changed": t["changed"],
        "cost_usd": round(t["cost"], 2), "by_mutation": by_mutation,
        "pipeline_version": max(p.get("pipeline_version", PIPELINE_VERSION) for p in m["papers"] if p.get("eval_job")),
        "fields": len({p["field"] for p in m["papers"] if p.get("eval_job")}),
    }
    paired = paired_comparison(scored)
    if paired:
        summary["paired_with_old_pipeline"] = paired
    demo.write_json(EVAL_DIR / "results.json", {"summary": summary, "papers": [
        json.loads((paper_dir(p["slug"]) / "results.json").read_text(encoding="utf-8"))
        for p in m["papers"] if p.get("eval_job")]})

    def table(headers, body):
        out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        out += ["| " + " | ".join(str(x) for x in row) + " |" for row in body]
        return "\n".join(out)

    if ROUND == 1:
        compare_title, compare = "Development paper (CoT)", ["1", "9/10", "70%", "80%", "3 / 7 / 3", "63%", "30 of 147 / 13", "-"]
    else:
        previous = json.loads((ROOT_DIR / "eval" / ROUNDS[ROUND - 1]["dir"] / "results.json").read_text(encoding="utf-8"))
        compare_title, compare = f"Round {ROUND - 1} (before the fixes)", _metric_cells(previous["summary"])
    title = "Held-out evaluation" + (f", round {ROUND}" if ROUND > 1 else "")
    intro = ("Papers the checker was never built, debugged or calibrated on, from several fields, each with about "
             "ten planted citation errors. Generated by `python scripts/heldout.py report`; the protocol is in "
             "`scripts/heldout.py` and every selection decision in `manifest.json`.")
    if ROUND > 1:
        intro += (f" A fresh sample (seeds {SAMPLE_SEED}-{SAMPLE_SEED + MAX_SAMPLES - 1}, up to "
                  f"{ROUNDS[ROUND]['per_stratum']} papers per stratum) drawn after the fixes prompted by round "
                  f"{ROUND - 1}, whose papers are excluded.")
    labels = ["Papers", "Planted errors caught (recall)", "Label accuracy on planted errors",
              "Expected failure-mode tag present", "Flags on untouched citations: genuine / false alarm / debatable",
              "Precision with human review (debatable left out)", "Escalated to Ultra / label changed",
              "Model cost (baseline + evaluation checks)"]
    md = [
        f"# {title}",
        "",
        intro,
        "",
        "## Results",
        "",
        table(("Metric", "Held-out set", compare_title), list(zip(labels, _metric_cells(summary), compare))),
        "",
        table(("Planted error", "Planted", "Caught"), [(k, v[0], v[1]) for k, v in sorted(by_mutation.items())]),
        "",
        "## Per paper",
        "",
        table(("Field", "Paper", "Parsed from", "Coverage", "Caught", "Genuine", "False alarms", "Debatable",
               "Precision", "Cost"), rows),
        "",
    ]
    if paired:
        md += paired_section(paired, table)
    readme = EVAL_DIR / "README.md"
    notes = ""
    if readme.exists() and "<!-- notes -->" in readme.read_text(encoding="utf-8"):
        notes = readme.read_text(encoding="utf-8").split("<!-- notes -->", 1)[1]
    readme.write_text("\n".join(md) + "\n<!-- notes -->" + (notes or "\n"), encoding="utf-8")
    print("\n".join(md))
    return 0


class _quiet:
    """Keep library INFO logs out of the progress output."""

    def __enter__(self):
        for name in ("httpx", "app", "app.http", "app.pipeline.references", "app.pipeline.run", "app.llm"):
            logging.getLogger(name).setLevel(logging.WARNING)

    def __exit__(self, *exc):
        return False


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=int, default=max(ROUNDS), help="evaluation round (default: the latest)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--only", nargs="*", help="strata to screen")
    s = sub.add_parser("plant")
    s.add_argument("slugs", nargs="*")
    s.add_argument("--force", action="store_true", help="plant again even if already planted")
    s = sub.add_parser("run")
    s.add_argument("slugs", nargs="*")
    s = sub.add_parser("run-old", help="check the planted papers with round 1's pipeline (paired comparison)")
    s.add_argument("slugs", nargs="*")
    s = sub.add_parser("review")
    s.add_argument("slug")
    sub.add_parser("report")
    args = parser.parse_args()
    use_round(args.round)
    if args.cmd == "review":
        return cmd_review(args)
    if args.cmd == "run-old":
        return cmd_run_old(args)
    if args.cmd == "report":
        return cmd_report(args)
    return asyncio.run({"screen": cmd_screen, "plant": cmd_plant, "run": cmd_run}[args.cmd](args))


if __name__ == "__main__":
    sys.exit(main())
