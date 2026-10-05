"""Run the ingest stage on an arXiv paper and print what it found.

    python scripts/ingest_paper.py 1706.03762 [--sentences 15] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.ingest import ingest_arxiv  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arxiv_id")
    parser.add_argument("--sentences", type=int, default=12, help="citing sentences to print")
    parser.add_argument("--json", type=Path, help="write the IngestResult JSON here")
    args = parser.parse_args()

    result = await ingest_arxiv(args.arxiv_id)
    paper, sentences = result.paper, result.sentences
    refs = {r.key: r for r in paper.references}

    print(f"{paper.id}: {paper.title}")
    print(f"  style={paper.citation_style}  text={len(paper.text):,} chars  sections={len(paper.sections)}")
    print(f"  cited keys: {paper.citation_keys_mapped}/{paper.citation_keys_total} mapped to bibliography")
    print(f"  references: {len(paper.references)}  "
          f"(arXiv IDs: {sum(bool(r.arxiv_id) for r in paper.references)}, "
          f"DOIs: {sum(bool(r.doi) for r in paper.references)}, "
          f"titles: {sum(bool(r.title) for r in paper.references)})")
    print(f"  citing sentences: {len(sentences)}")
    by_section = Counter(s.section for s in sentences)
    print("  by section: " + ", ".join(f"{k or '(none)'}={v}" for k, v in by_section.most_common(8)))
    for note in paper.notes:
        print(f"  NOTE: {note}")

    print("\nSections:")
    for s in paper.sections[:25]:
        print(f"  [{s.start:>6}] {s.title[:70]}")

    print(f"\nFirst {args.sentences} citing sentences:")
    for s in sentences[: args.sentences]:
        print(f"\n  ({s.id}, {s.section[:30]}) {s.text}")
        for key in s.reference_keys:
            r = refs[key]
            ident = f"arXiv:{r.arxiv_id}" if r.arxiv_id else (f"doi:{r.doi}" if r.doi else "")
            print(f"      -> [{r.label}] {(r.title or r.raw)[:90]}  {ident}")

    print("\nSample references:")
    for r in paper.references[:8]:
        print(f"  [{r.label}] key={r.key} year={r.year} arxiv={r.arxiv_id} doi={r.doi}\n      {r.raw[:150]}")

    if args.json:
        args.json.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
