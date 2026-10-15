"""Create the demo paper: a copy of a real paper with ~10 citation errors planted.

    python scripts/plant_errors.py [2201.11903] [--seed 7] [--baseline-job JOB_ID]

1. Checks the unmodified paper (full check; cached after the first run) to learn which
   citations the checker already considers supported. Errors are only planted on those,
   so a "caught" error cannot be a pre-existing problem.
2. Plants four kinds of error (see app/planting.py), editing the paper text and shifting
   every offset.
3. Writes demo/paper.json (the mutated paper) and demo/ground_truth.json.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import demo  # noqa: E402
from app.models import Claim, IngestResult, Reference, Verdict  # noqa: E402
from app.pipeline.ingest import ingest_arxiv, normalize_arxiv_id  # noqa: E402
from app.pipeline.run import JobRunner  # noqa: E402
from app.planting import plant  # noqa: E402


async def baseline(runner: JobRunner, arxiv_id: str, job_id: Optional[str]):
    if job_id is None:
        job = runner.create_job(arxiv_id, "full")
        print(f"Checking the unmodified paper (job {job.id}); this is cached after the first run...")
        job = await runner.wait(job.id)
        if job.status != "done":
            raise SystemExit(f"Baseline check did not finish: {job.status} {job.message}")
        job_id = job.id
    store = runner.store
    ingest = store.get(job_id, "ingest", "paper", IngestResult)
    claims = {c.id: c for cs in store.all_lists(job_id, "claims", Claim).values() for c in cs}
    verdicts = [v for v in store.all(job_id, "verdict", Verdict).values() if not v.duplicate_of]
    references = store.all(job_id, "reference", Reference)
    return job_id, ingest, claims, verdicts, references


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arxiv_id", nargs="?", default="2201.11903")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--baseline-job", help="reuse a finished full check of the unmodified paper")
    args = parser.parse_args()

    runner = JobRunner()
    job_id, ingest, claims, verdicts, references = await baseline(runner, normalize_arxiv_id(args.arxiv_id), args.baseline_job)
    # Re-parse with the current converter (fixes since the baseline ran), as long as every
    # citing sentence is unchanged so the baseline verdicts still apply.
    fresh = await ingest_arxiv(paper_id := ingest.paper.arxiv_id)
    if [(s.id, s.text, s.reference_keys) for s in fresh.sentences] == [(s.id, s.text, s.reference_keys) for s in ingest.sentences]:
        ingest = fresh
    else:
        print(f"Note: re-parsing {paper_id} changed citing sentences; using the baseline's parse.")
    print(f"Baseline job {job_id}: {len(ingest.sentences)} citing sentences, {len(verdicts)} verdicts")

    result = await plant(ingest, claims, verdicts, references, seed=args.seed, id_prefix="demo",
                         source=f"arXiv:{ingest.paper.arxiv_id}")
    print("Candidates:", result.candidates, "| chosen:", len(result.truth.planted))
    demo.write_json(demo.PAPER_PATH, result.mutated.model_dump(mode="json"))
    demo.write_json(demo.GROUND_TRUTH_PATH, result.truth.model_dump(mode="json"))

    planted = result.truth.planted
    print(f"\nWrote {demo.PAPER_PATH.name} and {demo.GROUND_TRUTH_PATH.name}: {len(planted)} planted errors\n")
    for p in planted:
        print(f"  {p.id} {p.sentence_id:<4} {p.mutation:<16} {p.description}")
        print(f"        {p.mutated[:160]}")
    print(f"\nThe unmodified paper already had {len(result.truth.baseline_flags)} flagged citations "
          "(kept in the ground truth).")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
