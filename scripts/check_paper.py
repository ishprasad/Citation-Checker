"""Run the whole pipeline (ingest -> resolve -> extract -> retrieve -> judge -> escalate) from the CLI.

    python scripts/check_paper.py 2310.06825 [--sentences 10] [--start 0] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import textwrap
import time
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402
from app.models import Coverage  # noqa: E402
from app.pipeline.claims import extract_claims  # noqa: E402
from app.pipeline.evidence import EvidenceRetriever  # noqa: E402
from app.pipeline.ingest import ingest_arxiv  # noqa: E402
from app.pipeline.judge import JudgeContext, escalation_stats, judge_all  # noqa: E402
from app.pipeline.references import resolve_references  # noqa: E402

ICON = {"supported": "OK  ", "overstated": "OVER", "contradicted": "CONTR", "not_found": "NOTF", "unverifiable": "??  "}


def wrap(text: str, indent: str) -> str:
    return textwrap.fill(text, width=112, initial_indent=indent, subsequent_indent=indent)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arxiv_id")
    parser.add_argument("--sentences", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--json", type=Path, help="write claims and verdicts here")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    result = await ingest_arxiv(args.arxiv_id)
    paper = result.paper
    sentences = result.sentences[args.start : args.start + args.sentences]
    keys = {k for s in sentences for k in s.reference_keys}
    print(f"{paper.id}: {paper.title}\n{len(sentences)} citing sentences, {len(keys)} references\n")

    job_id = f"cli-{uuid.uuid4().hex[:8]}"
    timings = {}
    with llm.job_context(job_id):
        t = time.perf_counter()
        refs = await resolve_references([r for r in paper.references if r.key in keys])
        refs_by_key = {r.key: r for r in refs}
        timings["resolve"] = time.perf_counter() - t

        t = time.perf_counter()
        claims = await extract_claims(sentences, paper, refs_by_key)
        timings["extract"] = time.perf_counter() - t

        t = time.perf_counter()
        evidence = await EvidenceRetriever().retrieve_all(claims, refs_by_key)
        timings["retrieve"] = time.perf_counter() - t

        t = time.perf_counter()
        ctx = JudgeContext(paper=paper, sentences={s.id: s for s in sentences}, references=refs_by_key)
        verdicts = await judge_all(claims, evidence, ctx)
        timings["judge+escalate"] = time.perf_counter() - t

    claims_by_id = {c.id: c for c in claims}
    for v in verdicts:
        claim = claims_by_id[v.claim_id]
        ref = refs_by_key.get(v.reference_key)
        esc = f" ESCALATED (was {v.original_label} {v.original_confidence:.2f})" if v.escalated else ""
        flags = f" flags={v.flags}" if v.flags else ""
        print(f"[{ICON[v.label]}] {v.claim_id} -> [{ref.label if ref else v.reference_key}] "
              f"{v.label} {v.confidence:.2f}{esc}{flags}")
        print(wrap(f"claim: {claim.claim_text}", "      "))
        if v.attributed_claim and v.attributed_claim.strip(". ") != claim.claim_text.strip(". "):
            print(wrap(f"judged as: {v.attributed_claim}", "      "))
        if v.quote:
            mark = "verbatim" if v.quote_verified else "NOT FOUND IN SOURCE"
            print(wrap(f"quote ({mark}): \"{v.quote[:300]}\"", "      "))
        print(wrap(f"why: {v.explanation}", "      "))
        print()

    stats = escalation_stats(verdicts)
    labels = Counter(v.label for v in verdicts)
    print("=" * 112)
    print("Verdicts: " + ", ".join(f"{k}={labels[k]}" for k in ICON if labels[k]))
    print(f"Escalated {stats.escalated} of {stats.verdicts}; label changed in {stats.label_changed} "
          f"{stats.transitions or ''}")
    print(f"Quotes verified: {sum(v.quote_verified for v in verdicts)} / {sum(bool(v.quote) for v in verdicts)}")
    print(Coverage.of(refs).line())
    print("Timing: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    summary = llm.get_llm().usage_summary(job_id)
    for item in summary.per_model:
        print(f"  {item.role:<9} {item.model:<42} calls={item.calls:<3} (cached {item.cached_calls}) "
              f"in={item.prompt_tokens:>9,} out={item.completion_tokens:>7,}  ${item.estimated_cost_usd:.4f}")
    if args.json:
        import json

        args.json.write_text(json.dumps({
            "claims": [c.model_dump() for c in claims],
            "verdicts": [v.model_dump() for v in verdicts],
            "escalation": stats.model_dump(),
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
