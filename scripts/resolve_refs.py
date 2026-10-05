"""Ingest an arXiv paper, resolve its references, fetch their text, and print coverage.

    python scripts/resolve_refs.py 1706.03762 [--limit 10]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import Coverage  # noqa: E402
from app.pipeline.ingest import ingest_arxiv  # noqa: E402
from app.pipeline.references import resolve_references  # noqa: E402

STATUS = {"full": "FULL", "abstract_only": "ABS ", "unavailable": "----"}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arxiv_id")
    parser.add_argument("--limit", type=int, default=0, help="only the first N references")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    t0 = time.perf_counter()
    result = await ingest_arxiv(args.arxiv_id)
    refs = result.paper.references[: args.limit or None]
    print(f"{result.paper.id}: {result.paper.title}  ({len(refs)} references)\n")

    done = 0

    def progress(ref):
        nonlocal done
        done += 1
        print(f"\r  resolving {done}/{len(refs)}", end="", flush=True)

    t1 = time.perf_counter()
    resolved = await resolve_references(refs, on_done=progress)
    t2 = time.perf_counter()
    print("\n")

    for r in resolved:
        chars = Path(r.text_path).stat().st_size if r.text_path else 0
        ident = f"arXiv:{r.arxiv_id}" if r.arxiv_id else (f"doi:{r.doi}" if r.doi else "")
        flags = " RETRACTED" if r.retracted else ""
        print(
            f"  [{r.label:>4}] {STATUS[r.fulltext_status]} {r.fulltext_source or '':<11} {chars:>7,}ch "
            f"match={r.match_score if r.match_score is not None else '-':<5} retr={str(r.retracted):<5} "
            f"{(r.title or r.raw)[:60]}  {ident}{flags}"
        )

    coverage = Coverage.of(resolved)
    print(f"\n{coverage.line()}  Retracted: {coverage.retracted}.")
    print(f"Ingest {t1 - t0:.1f}s, resolution {t2 - t1:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
