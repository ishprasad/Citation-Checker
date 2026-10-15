"""Check held-out papers with an earlier version of the pipeline, for a paired comparison.

    python scripts/heldout_old.py --round 2 SLUG [SLUG ...]

Runs the code of commit OLD_COMMIT (the version round 1 measured) from a detached git worktree
in .cache/old-pipeline, on the same edited papers the current pipeline checks. The old code
resolves references itself (its resolver is part of what is compared) and has its own cache
folder: both versions write cited-paper texts under the same names with different contents.
Raw downloads and metadata responses are shared through directory junctions, so nothing is
fetched twice and the OpenAlex budget is spared. API keys come from this checkout's .env; it
is not copied. Each paper's old verdicts are written to eval/<round>/<slug>/old_pipeline.json,
which `heldout.py` scores and reviews alongside the current pipeline's.

Started by `python scripts/heldout.py run-old`; it can also be run directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

MAIN = Path(__file__).resolve().parent.parent
OLD_COMMIT = "da5f9e3"
OLD_CODE = MAIN / ".cache" / "old-pipeline"
OLD_CACHE = MAIN / ".cache" / "old-pipeline-cache"
# Raw HTTP caches (URL -> response) are the same for both versions; derived data is not.
SHARED_NAMESPACES = ("arxiv_src", "arxiv_pdf", "arxiv_meta", "oa_pdf", "europepmc", "europepmc_xml",
                     "openalex", "crossref")
ROUND_DIRS = {1: "heldout", 2: "heldout-2"}


def prepare() -> None:
    if not (OLD_CODE / "app").exists():
        subprocess.run(["git", "-C", str(MAIN), "worktree", "add", "--detach", str(OLD_CODE), OLD_COMMIT], check=True)
    OLD_CACHE.mkdir(parents=True, exist_ok=True)
    for name in SHARED_NAMESPACES:
        link, target = OLD_CACHE / name, MAIN / ".cache" / name
        target.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            if os.name == "nt":  # a junction needs no admin rights
                subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
            else:
                link.symlink_to(target, target_is_directory=True)


def load_old_code() -> None:
    from dotenv import load_dotenv

    load_dotenv(MAIN / ".env")  # keys; the old code's own load_dotenv finds no .env and changes nothing
    os.environ["CACHE_DIR"] = str(OLD_CACHE)
    os.environ.pop("DB_PATH", None)
    sys.path.insert(0, str(OLD_CODE))


REFERENCE_FIELDS = ("key", "label", "raw", "title", "year", "doi", "arxiv_id", "fulltext_status", "fulltext_source",
                    "status_reason", "match_score", "text_path")
PASSAGES_KEPT, PASSAGE_CHARS = 3, 800


def slim_reference(ref: dict) -> dict:
    return {k: ref.get(k) for k in REFERENCE_FIELDS}


def slim_evidence(ev: dict) -> dict:
    passages = [{**p, "text": p["text"][:PASSAGE_CHARS]} for p in ev.get("passages", [])[:PASSAGES_KEPT]]
    return {**ev, "passages": passages}


async def check(runner, slug: str, eval_dir: Path) -> None:
    from app import llm
    from app.models import Claim, Evidence, IngestResult, Job, Reference, Verdict
    from app.pipeline.run import PIPELINE_VERSION

    out_path = eval_dir / slug / "old_pipeline.json"
    if out_path.exists():
        print(f"[{slug}] old pipeline: already done")
        return
    mutated = IngestResult.model_validate_json((eval_dir / slug / "paper.json").read_text(encoding="utf-8"))
    kind = "arxiv" if mutated.paper.parsed_from == "latex" else "pdf"
    job = Job(id=uuid.uuid4().hex[:12], mode="full", source=kind, pipeline_version=PIPELINE_VERSION,
              arxiv_id=mutated.paper.arxiv_id if kind == "arxiv" else None)
    runner._describe(job, mutated)
    runner.store.put(job.id, "ingest", "paper", mutated)
    runner.store.save_job(job)
    print(f"[{slug}] old pipeline ({OLD_COMMIT}): full check, job {job.id}...")
    runner.start(job.id)
    job = await runner.wait(job.id)
    if job.status != "done":
        raise RuntimeError(f"job {job.id} ended {job.status}: {job.message}")
    store = runner.store
    claims = [c for cs in store.all_lists(job.id, "claims", Claim).values() for c in cs]
    verdicts = [v for v in store.all(job.id, "verdict", Verdict).values() if not v.duplicate_of]
    references = store.all(job.id, "reference", Reference)
    evidence = store.all(job.id, "evidence", Evidence)
    usage = llm.get_llm().usage_summary(job.id)
    data = {
        "commit": OLD_COMMIT, "pipeline_version": PIPELINE_VERSION, "job_id": job.id,
        "claims": [c.model_dump(mode="json") for c in claims],
        "verdicts": [v.model_dump(mode="json") for v in verdicts],
        "references": {k: slim_reference(r.model_dump(mode="json")) for k, r in references.items()},
        # The top passages behind each verdict, trimmed, for the review sheet.
        "evidence": {k: slim_evidence(e.model_dump(mode="json")) for k, e in evidence.items()},
        "usage": usage.model_dump(mode="json"),
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{slug}] old pipeline: {len(verdicts)} verdicts, ${usage.total_estimated_cost_usd:.2f}")


async def main_async(slugs: list[str], eval_dir: Path) -> None:
    from app.pipeline.run import JobRunner

    runner = JobRunner()
    await asyncio.gather(*(check(runner, slug, eval_dir) for slug in slugs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=int, default=max(ROUND_DIRS))
    parser.add_argument("slugs", nargs="+")
    args = parser.parse_args()
    prepare()
    load_old_code()
    asyncio.run(main_async(args.slugs, MAIN / "eval" / ROUND_DIRS[args.round]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
