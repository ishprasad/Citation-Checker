"""Sanity-check claim extraction and evidence retrieval on a few citing sentences.

    python scripts/sample_claims.py 2310.06825 [--sentences 6] [--start 0] [--top 3]

Only the references cited by the chosen sentences are resolved, so this is quick.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import textwrap
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402
from app.pipeline.claims import extract_claims, render_sentence  # noqa: E402
from app.pipeline.evidence import EvidenceRetriever  # noqa: E402
from app.pipeline.ingest import ingest_arxiv  # noqa: E402
from app.pipeline.references import resolve_references  # noqa: E402


def wrap(text: str, indent: str, width: int = 110) -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arxiv_id")
    parser.add_argument("--sentences", type=int, default=6)
    parser.add_argument("--start", type=int, default=0, help="skip the first N citing sentences")
    parser.add_argument("--top", type=int, default=3, help="passages to print per claim/reference")
    args = parser.parse_args()

    result = await ingest_arxiv(args.arxiv_id)
    paper = result.paper
    sentences = result.sentences[args.start : args.start + args.sentences]
    keys = {k for s in sentences for k in s.reference_keys}
    refs = [r for r in paper.references if r.key in keys]
    print(f"{paper.id}: {paper.title}\n{len(sentences)} sentences, {len(refs)} references\n")

    job_id = f"sample-{uuid.uuid4().hex[:8]}"
    with llm.job_context(job_id):
        t0 = time.perf_counter()
        resolved = await resolve_references(refs)
        refs_by_key = {r.key: r for r in resolved}
        t1 = time.perf_counter()
        claims = await extract_claims(sentences, paper, refs_by_key)
        t2 = time.perf_counter()
        evidence = await EvidenceRetriever().retrieve_all(claims, refs_by_key)
        t3 = time.perf_counter()

    ev_by_pair = {(e.claim_id, e.reference_key): e for e in evidence}
    for sentence in sentences:
        rendered, id_map = render_sentence(sentence, paper.text)
        print("=" * 110)
        print(wrap(f"{sentence.id} [{sentence.section}] {rendered}", ""))
        for claim in [c for c in claims if c.citing_sentence_id == sentence.id]:
            flag = " (FALLBACK)" if claim.fallback else ""
            meta = f"type={claim.claim_type} numbers={claim.numbers} scope={claim.scope}"
            print(f"\n  CLAIM {claim.id}{flag}: {claim.claim_text}\n    {meta}")
            for key in claim.reference_keys:
                ref = refs_by_key[key]
                print(f"    -> [{ref.label}] {ref.title} ({ref.fulltext_status})")
                ev = ev_by_pair.get((claim.id, key))
                ranked = sorted(ev.passages, key=lambda p: -p.score) if ev else []
                for p in ranked[: args.top]:
                    where = "abstract" if p.chunk_index < 0 else f"chunk {p.chunk_index}, {p.heading[:40] or 'no heading'}"
                    print(wrap(f"{p.score:.3f} ({where}) {p.text[:330]}...", "         "))
        print()

    summary = llm.get_llm().usage_summary(job_id)
    print("=" * 110)
    print(f"{len(claims)} claims from {len(sentences)} sentences; {len(evidence)} evidence sets")
    print(f"Timing: resolve {t1 - t0:.1f}s, extract {t2 - t1:.1f}s, retrieve {t3 - t2:.1f}s")
    for item in summary.per_model:
        print(f"  {item.role:<8} {item.model:<42} calls={item.calls} (cached {item.cached_calls}) "
              f"in={item.prompt_tokens:,} out={item.completion_tokens:,}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
