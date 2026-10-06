"""Check the demo paper and score the result against the planted errors.

    python scripts/evaluate.py [--job JOB_ID] [--snapshot]

Runs a full check of demo/paper.json (or scores an existing job), then reports:
planted errors caught (recall), flags on untouched citations (precision), label accuracy,
how often escalation changed the verdict, reference coverage, and tokens and cost.
Saves demo/eval_results.json; --snapshot also bundles the finished check in
demo/snapshot.json so the demo can be shown without network access or an API key.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import demo, llm  # noqa: E402
from app.models import Claim, Coverage, Reference, Verdict  # noqa: E402
from app.pipeline.judge import escalation_stats  # noqa: E402
from app.pipeline.run import JobRunner  # noqa: E402

RESULTS_PATH = demo.DEMO_DIR / "eval_results.json"


def table(rows: list[tuple], headers: tuple) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths))  # noqa: E731
    return "\n".join([line(headers), "  ".join("-" * w for w in widths), *map(line, rows)])


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", help="score an existing check of the demo paper instead of running one")
    parser.add_argument("--snapshot", action="store_true", help="also write demo/snapshot.json")
    args = parser.parse_args()

    truth = demo.load_ground_truth()
    if truth is None or not demo.demo_available():
        print("No demo paper yet. Run: python scripts/plant_errors.py")
        return 1

    runner = JobRunner()
    t0 = time.perf_counter()
    if args.job:
        job = runner.store.load_job(args.job)
    else:
        job = runner.create_demo_job()
        print(f"Checking the demo paper (job {job.id})...")
        job = await runner.wait(job.id)
    elapsed = time.perf_counter() - t0
    if job is None or job.status not in ("done", "partial"):
        print(f"The check did not finish: {job and job.status} {job and job.message}")
        return 1

    store = runner.store
    claims = {c.id: c for cs in store.all_lists(job.id, "claims", Claim).values() for c in cs}
    verdicts = [v for v in store.all(job.id, "verdict", Verdict).values() if not v.duplicate_of]
    references = list(store.all(job.id, "reference", Reference).values())
    result = demo.score(truth, verdicts, claims)
    esc = escalation_stats(verdicts)
    coverage = Coverage.of(references)
    usage = llm.get_llm().usage_summary(job.id)

    print(f"\n{truth.title} (arXiv:{truth.arxiv_id}), demo job {job.id}\n")
    rows = [(m, r["planted"], r["caught"]) for m, r in result["by_mutation"].items()]
    print(table(rows, ("Planted error", "Planted", "Caught")))
    print()
    print(table([
        ("Recall (planted errors caught)", f"{result['caught']}/{result['planted']} = {result['recall']:.0%}"),
        ("Label accuracy on planted errors", f"{result['label_accuracy']:.0%}"),
        ("Expected failure-mode tag present", f"{result['expected_mode_rate']:.0%}"),
        ("Flags on untouched citations", f"{result['other_flags']} of {result['untouched_verdicts']} verdicts"),
        ("  ...also flagged in the original paper", str(result["other_flags_preexisting"])),
        ("Precision (strict)", f"{result['precision']:.0%}"),
        ("Precision (excluding pre-existing flags)", f"{result['precision_excluding_preexisting']:.0%}"),
        ("Escalated to Nemotron Ultra", f"{esc.escalated} of {esc.verdicts} verdicts"),
        ("Escalation changed the label", f"{esc.label_changed} ({', '.join(f'{k}: {v}' for k, v in esc.transitions.items()) or 'none'})"),
        ("Reference coverage", coverage.line()),
        ("Wall time for this run", f"{elapsed:.0f}s (model calls are cached after the first run)"),
    ], ("Metric", "Value")))
    print()
    print(table([(m.role, m.model, m.calls, m.cached_calls, f"{m.prompt_tokens:,}", f"{m.completion_tokens:,}",
                  f"${m.estimated_cost_usd:.3f}") for m in usage.per_model],
                ("Role", "Model", "Calls", "Cached", "In", "Out", "Cost")))
    print(f"Total tokens: {usage.total_prompt_tokens + usage.total_completion_tokens:,} · "
          f"estimated cost ${usage.total_estimated_cost_usd:.3f}")
    print()
    print(table([(d["id"], d["sentence_id"], d["mutation"], "yes" if d["caught"] else "MISSED", ",".join(d["labels"]) or "-",
                  ",".join(d["failure_modes"]) or "-", "yes" if d["escalated"] else "")
                 for d in result["details"]], ("Id", "Sentence", "Mutation", "Caught", "Labels", "Tags", "Escalated")))
    print(f"\n{result['summary_line']}")

    demo.write_json(RESULTS_PATH, {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paper": {"arxiv_id": truth.arxiv_id, "title": truth.title},
        "job_id": job.id,
        **{k: v for k, v in result.items() if k != "details"},
        "details": result["details"],
        "escalation": esc.model_dump(),
        "coverage": {**coverage.model_dump(), "line": coverage.line()},
        "usage": usage.model_dump(),
        "wall_time_seconds": round(elapsed, 1),
    })
    print(f"Saved {RESULTS_PATH.relative_to(demo.DEMO_DIR.parent)}")
    if args.snapshot:
        runner.export_snapshot(job.id)
        print(f"Saved {demo.SNAPSHOT_PATH.relative_to(demo.DEMO_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
