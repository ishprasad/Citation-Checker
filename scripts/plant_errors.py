"""Create the demo paper: a copy of a real paper with ~10 citation errors planted.

    python scripts/plant_errors.py [2201.11903] [--seed 7] [--baseline-job JOB_ID]

1. Checks the unmodified paper (full check; cached after the first run) to learn which
   citations the checker already considers supported. Errors are only planted on those,
   so a "caught" error cannot be a pre-existing problem.
2. Plants four kinds of error, editing the paper text and shifting every offset:
     number_change     a number the cited work actually reports is changed
     scope_inflation   a sweeping scope phrase is added before the citation
     hedge_removal     "may"/"suggests" become "will"/"shows"
     citation_swap     the citation is replaced by the least related paper in the bibliography
3. Writes demo/paper.json (the mutated paper) and demo/ground_truth.json.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import demo  # noqa: E402
from app.models import Claim, CitingSentence, IngestResult, Reference, Verdict  # noqa: E402
from app.pipeline.evidence import EvidenceRetriever  # noqa: E402
from app.pipeline.ingest import ingest_arxiv, normalize_arxiv_id  # noqa: E402
from app.pipeline.run import JobRunner  # noqa: E402
from app.pipeline.textutil import NUMBER_RE, extract_numbers, numbers_match  # noqa: E402

PLAN = {"citation_swap": 3, "number_change": 3, "scope_inflation": 2, "hedge_removal": 2}
EXPECTED = {
    "number_change": (["overstated", "contradicted", "not_found"], ["number_drift"]),
    "scope_inflation": (["overstated"], ["scope_inflation"]),
    "hedge_removal": (["overstated"], ["hypothesis_as_finding"]),
    "citation_swap": (["not_found"], ["off_topic_citation"]),
}
_IDENTIFICATION_RE = re.compile(
    r"\b(introduces?|describes?|presents?|proposes?|is|are)\b.{0,60}\b(benchmark|dataset|corpus|model|method|"
    r"architecture|tool|library|framework|task)s?\b",
    re.IGNORECASE,
)
_NO_SCOPE_AFTER = {"by", "in", "of", "from", "see", "to", "and", "or", "as", "e.g", "i.e", "cf", "with",
                   "following", "using", "like", "via", "on", "for", "the", "a", "an"}
SCOPE_PHRASES = ["across all tasks and model sizes", "in every setting and domain", "universally, in all cases"]
HEDGES = [
    (r"\bsuggests that\b", "shows that"), (r"\bsuggest that\b", "show that"), (r"\bsuggests\b", "shows"),
    (r"\bsuggest\b", "show"), (r"\bmay\b", "will"), (r"\bmight\b", "will"), (r"\bcould\b", "will"),
    (r"\bpossibly\s+", ""), (r"\bpotentially\s+", ""), (r"\blikely\s+", ""),
]


@dataclass
class Edit:
    start: int
    end: int
    new: str
    sentence_id: str
    marker_index: Optional[int] = None  # for citation swaps: the marker being replaced
    new_keys: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Text edits with offset bookkeeping
# ----------------------------------------------------------------------------


def apply_edits(result: IngestResult, edits: list[Edit]) -> IngestResult:
    """Apply edits from the end of the text backwards so earlier offsets stay valid."""
    result = result.model_copy(deep=True)
    paper = result.paper
    original_sentences = {s.id: s.text for s in result.sentences}
    for e in sorted(edits, key=lambda e: e.start, reverse=True):
        delta = len(e.new) - (e.end - e.start)
        paper.text = paper.text[: e.start] + e.new + paper.text[e.end :]

        def shift(pos: int) -> int:
            # Replacement: positions at or after the replaced span move. Insertion: positions
            # at or after the insertion point move (the inserted text goes before them).
            moved = pos >= e.end if e.end > e.start else pos >= e.start
            return pos + delta if moved else pos

        for section in paper.sections:
            section.start, section.end = shift(section.start), shift(section.end)
            section.text = paper.text[section.start : section.end]
        for s in result.sentences:
            if s.end < e.start:
                continue
            s.marker_spans = [
                (e.start, e.start + len(e.new)) if s.id == e.sentence_id and i == e.marker_index else (shift(a), shift(b))
                for i, (a, b) in enumerate(s.marker_spans)
            ]
            s.start, s.end = shift(s.start), shift(s.end)
            s.text = paper.text[s.start : s.end]
            if s.id == e.sentence_id and e.marker_index is not None:
                old_keys = s.marker_keys[e.marker_index]
                s.marker_keys[e.marker_index] = e.new_keys
                still_cited = {k for i, ks in enumerate(s.marker_keys) for k in ks}
                s.reference_keys = [k for k in s.reference_keys if k in still_cited or k not in old_keys]
                s.reference_keys += [k for k in e.new_keys if k not in s.reference_keys]
    # Neighbouring sentences quote the edited sentences as context.
    changed = {sid: s.text for s in result.sentences if (sid := s.id) and s.text != original_sentences[sid]}
    for s in result.sentences:
        for sid, new in changed.items():
            old = original_sentences[sid]
            if old in s.context_before:
                s.context_before = s.context_before.replace(old, new)
            if old in s.context_after:
                s.context_after = s.context_after.replace(old, new)
    return result


# ----------------------------------------------------------------------------
# Candidate mutations
# ----------------------------------------------------------------------------


@dataclass
class Candidate:
    mutation: str
    sentence: CitingSentence
    keys: list[str]
    edit: Edit
    description: str


def _inside_marker(s: CitingSentence, pos: int) -> bool:
    return any(a <= pos < b for a, b in s.marker_spans)


def _bump(raw: str, value: float, unit: str) -> Optional[str]:
    if "^" in raw or "×10" in raw.replace(" ", ""):
        return None
    new = value * 1.6
    if unit == "%" and new > 100:
        new = value * 0.55
    m = re.search(r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?", raw)
    if not m:
        return None
    digits = m.group(0)
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    if not value:
        return None
    # `value` is the number as written ("175" for "175B"), so scale the written digits.
    scaled = float(digits.replace(",", "")) * (new / value)
    text = f"{scaled:,.{decimals}f}" if "," in digits else f"{scaled:.{decimals}f}"
    return raw[: m.start()] + text + raw[m.end() :]


def number_candidates(s: CitingSentence, key: str, marker_index: int, verdict: Verdict, paper_text: str) -> list[Candidate]:
    if not verdict.quote_verified:
        return []
    quote_numbers = extract_numbers(verdict.quote_context or verdict.quote or "")
    # Only the clause this citation is attached to: from the previous marker up to this one.
    clause_start = s.marker_spans[marker_index - 1][1] if marker_index > 0 else s.start
    clause_end = s.marker_spans[marker_index][0]
    out = []
    for m in NUMBER_RE.finditer(paper_text, clause_start, clause_end):
        if _inside_marker(s, m.start()):
            continue
        number = extract_numbers(m.group(0))
        if not number:
            continue
        n = number[0]
        if n.unit == "" and float(n.value).is_integer() and (1900 <= n.value <= 2035 or n.value < 10):
            continue  # a year, or a small count / list enumerator like "(5)"
        if not any(numbers_match(n, q) is True for q in quote_numbers):
            continue  # the number is not attributed to the cited work
        new_raw = _bump(n.raw, n.value, n.unit)
        if not new_raw or numbers_match(n, extract_numbers(new_raw)[0]) is not False:
            continue
        start = m.start() + m.group(0).find(n.raw)
        out.append(Candidate("number_change", s, [key], Edit(start, start + len(n.raw), new_raw, s.id),
                             f'Changed "{n.raw}" to "{new_raw}"'))
    return out


def hedge_candidates(s: CitingSentence, key: str, marker_index: int, paper_text: str) -> list[Candidate]:
    a, _ = s.marker_spans[marker_index]
    clause_start = s.marker_spans[marker_index - 1][1] if marker_index > 0 else s.start
    clause = paper_text[clause_start:a]
    for pattern, replacement in HEDGES:
        m = re.search(pattern, clause, re.IGNORECASE)
        if m:
            start = clause_start + m.start()
            word = m.group(0).strip()
            return [Candidate("hedge_removal", s, [key], Edit(start, clause_start + m.end(), replacement, s.id),
                              f'Removed the hedge: "{word}" became "{replacement.strip() or "(deleted)"}"')]
    return []


def scope_candidate(s: CitingSentence, key: str, marker_index: int, phrase: str, paper_text: str) -> Optional[Candidate]:
    a, _ = s.marker_spans[marker_index]
    if a == s.start or paper_text[a - 1] != " ":
        return None
    b = s.marker_spans[marker_index][1]
    if not paper_text[a:b].startswith(("(", "[")) or not re.match(r"\s*([.,;:]|$)", paper_text[b:s.end]):
        return None  # only a parenthetical citation that closes its clause carries the claim before it
    previous_word = re.findall(r"[A-Za-z.]+", paper_text[max(s.start, a - 30):a])
    if not previous_word or previous_word[-1].lower().rstrip(".") in _NO_SCOPE_AFTER:
        return None  # "as shown by [scope] (Brown et al.)" would be ungrammatical
    return Candidate("scope_inflation", s, [key], Edit(a, a, f"{phrase} ", s.id),
                     f'Added the scope "{phrase}" before the citation')


def swap_candidate(s: CitingSentence, key: str, marker_index: int, paper_text: str, new_ref: Reference,
                   author_year: bool) -> Optional[Candidate]:
    a, b = s.marker_spans[marker_index]
    marker = paper_text[a:b]
    if s.marker_keys[marker_index] != [key] or not marker.startswith(("(", "[")):
        return None
    new_marker = f"({new_ref.label})" if author_year else f"[{new_ref.label}]"
    return Candidate("citation_swap", s, [new_ref.key],
                     Edit(a, b, new_marker, s.id, marker_index=marker_index, new_keys=[new_ref.key]),
                     f'Replaced the citation {marker} with {new_marker} ("{new_ref.title}")')


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


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
    rng = random.Random(args.seed)

    runner = JobRunner()
    job_id, ingest, claims, verdicts, references = await baseline(runner, normalize_arxiv_id(args.arxiv_id), args.baseline_job)
    # Re-parse with the current converter (fixes since the baseline ran), as long as every
    # citing sentence is unchanged so the baseline verdicts still apply.
    fresh = await ingest_arxiv(paper_id := ingest.paper.arxiv_id)
    if [(s.id, s.text, s.reference_keys) for s in fresh.sentences] == [(s.id, s.text, s.reference_keys) for s in ingest.sentences]:
        ingest = fresh
    else:
        print(f"Note: re-parsing {paper_id} changed citing sentences; using the baseline's parse.")
    paper = ingest.paper
    sentences = {s.id: s for s in ingest.sentences}
    author_year = paper.citation_style == "author_year"
    print(f"Baseline job {job_id}: {len(sentences)} citing sentences, {len(verdicts)} verdicts")

    by_pair = {}
    for v in verdicts:
        by_pair.setdefault((claims[v.claim_id].citing_sentence_id, v.reference_key), []).append(v)
    baseline_flags = sorted({(claims[v.claim_id].citing_sentence_id, v.reference_key, v.label)
                             for v in verdicts if demo.is_flag(v)})

    # Abstract-level similarity of every claim to every reference: swaps use the least related one.
    full_refs = {k: r for k, r in references.items() if r.fulltext_status == "full"}
    scores = await EvidenceRetriever().abstract_scores(list(claims.values()), full_refs)

    candidates: dict[str, list[Candidate]] = {m: [] for m in PLAN}
    for (sid, key), vs in by_pair.items():
        s = sentences[sid]
        # Only citations the checker considered clean, so planted errors are the only problem.
        if not all(v.label == "supported" and v.confidence >= 0.8 and not v.failure_modes for v in vs):
            continue
        if any(claims[v.claim_id].claim_type == "background" for v in vs):
            continue
        markers = [i for i, ks in enumerate(s.marker_keys) if key in ks]
        if not markers:
            continue
        i = markers[0]
        finding = any(claims[v.claim_id].claim_type == "finding" for v in vs)
        for v in vs:
            candidates["number_change"] += number_candidates(s, key, i, v, paper.text)
        # Scope and hedge errors only make sense where the citation supports a finding, not
        # where it merely identifies a dataset, model or method ("on the GSM8K benchmark (Cobbe)").
        identification = any(_IDENTIFICATION_RE.search(v.attributed_claim or "") for v in vs)
        group = len(s.marker_keys[i]) > 2  # "a line of work (A; B; C; D)" is judged as examples, not claims
        if finding and not identification and not group:
            candidates["hedge_removal"] += hedge_candidates(s, key, i, paper.text)
            c = scope_candidate(s, key, i, SCOPE_PHRASES[len(candidates["scope_inflation"]) % len(SCOPE_PHRASES)], paper.text)
            if c:
                candidates["scope_inflation"].append(c)
        # Swap target: the reference least related to this citation's claims (max similarity
        # over the claims, so it is unrelated to all of them), and clearly below the
        # similarity of the original reference.
        claim_scores = [scores.get(v.claim_id, {}) for v in vs if scores.get(v.claim_id)]
        if claim_scores:
            original = min(cs.get(key, 0.0) for cs in claim_scores)
            options = sorted((max(cs.get(k, 1.0) for cs in claim_scores), k)
                             for k in full_refs if k not in s.reference_keys)
            if options and options[0][0] <= original - 0.15:
                c = swap_candidate(s, key, i, paper.text, full_refs[options[0][1]], author_year)
                if c:
                    c.description += f" (similarity {options[0][0]:.2f} vs {original:.2f} for the original)"
                    candidates["citation_swap"].append(c)

    # Pick per plan, one error per sentence.
    chosen: list[Candidate] = []
    used: set[int] = set()

    def index(c: Candidate) -> int:
        return int(c.sentence.id.lstrip("s"))

    plan = dict(PLAN)
    if len(candidates["hedge_removal"]) < plan["hedge_removal"]:
        missing = plan["hedge_removal"] - len(candidates["hedge_removal"])
        plan["hedge_removal"] -= missing
        plan["scope_inflation"] += missing
    for mutation, count in plan.items():
        pool = candidates[mutation][:]
        rng.shuffle(pool)
        for c in pool:
            if sum(x.mutation == mutation for x in chosen) >= count:
                break
            if index(c) in used:  # one planted error per sentence
                continue
            if mutation == "citation_swap" and any(x.mutation == mutation and x.keys == c.keys for x in chosen):
                continue  # each unrelated paper is swapped in only once
            chosen.append(c)
            used.add(index(c))
    # Backfill when a mutation type ran short (e.g. few hedged citations in the paper).
    target = sum(PLAN.values())
    for mutation in ("number_change", "citation_swap", "scope_inflation", "hedge_removal"):
        pool = [c for c in candidates[mutation] if c not in chosen]
        rng.shuffle(pool)
        for c in pool:
            if len(chosen) >= target:
                break
            if index(c) in used:  # one planted error per sentence
                continue
            if mutation == "citation_swap" and any(x.mutation == mutation and x.keys == c.keys for x in chosen):
                continue
            chosen.append(c)
            used.add(index(c))
    print("Candidates:", {m: len(v) for m, v in candidates.items()}, "| chosen:", len(chosen))

    mutated = apply_edits(ingest, [c.edit for c in chosen])
    digest = hashlib.sha256(mutated.paper.text.encode("utf-8")).hexdigest()[:8]
    mutated.paper.id = f"demo:{paper.arxiv_id}:{digest}"
    mutated.paper.notes = []
    new_sentences = {s.id: s for s in mutated.sentences}

    planted = []
    for n, c in enumerate(sorted(chosen, key=index), start=1):
        labels, modes = EXPECTED[c.mutation]
        planted.append(demo.PlantedError(
            id=f"p{n}", sentence_id=c.sentence.id, reference_keys=c.keys, mutation=c.mutation,
            description=c.description, original=c.sentence.text, mutated=new_sentences[c.sentence.id].text,
            expected_labels=labels, expected_failure_modes=modes,
        ))
    truth = demo.GroundTruth(arxiv_id=paper.arxiv_id, title=paper.title, planted=planted, baseline_flags=baseline_flags)
    demo.write_json(demo.PAPER_PATH, mutated.model_dump(mode="json"))
    demo.write_json(demo.GROUND_TRUTH_PATH, truth.model_dump(mode="json"))

    print(f"\nWrote {demo.PAPER_PATH.name} and {demo.GROUND_TRUTH_PATH.name}: {len(planted)} planted errors\n")
    for p in planted:
        print(f"  {p.id} {p.sentence_id:<4} {p.mutation:<16} {p.description}")
        print(f"        {p.mutated[:160]}")
    print(f"\nThe unmodified paper already had {len(baseline_flags)} flagged citations (kept in the ground truth).")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
