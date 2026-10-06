"""Claim extraction (MODEL_EXTRACT, Nemotron Nano, reasoning off).

Each citing sentence is shown to the model with its citation markers replaced by short
ids (R1, R2, ...) so the model can tie each atomic claim to exactly the references it
relies on. The output is validated in code: unknown ids are dropped, references the
model forgot still get a claim, and numbers are re-extracted from the claim text.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from typing import Awaitable, Callable, Optional, Union

import openai
from pydantic import BaseModel, Field

from app import llm
from app.models import CitingSentence, Claim, ClaimType, Paper, Reference, Verdict
from app.pipeline.textutil import number_strings
from app.prompts import load_prompt

log = logging.getLogger(__name__)

OnDone = Callable[[CitingSentence, list[Claim]], Union[None, Awaitable[None]]]


class AttributedClaim(BaseModel):
    claim_text: str = Field(min_length=3)
    claim_type: ClaimType
    numbers: list[str] = Field(default_factory=list)
    scope: Optional[str] = None


class CitationClaims(BaseModel):
    reference_ids: list[str]
    claims: list[AttributedClaim]


class ExtractionOutput(BaseModel):
    """One entry per citation marker group: what the sentence attributes to those works."""

    citations: list[CitationClaims]


_TRAILING_CITATION_RE = re.compile(r"\s*[\[(][^\[\]()]*[\])]\s*$")
_REF_ID_RE = re.compile(r"\s*\[(?:R\d+(?:,\s*)?)+\]")


def render_sentence(sentence: CitingSentence, paper_text: str) -> tuple[str, dict[str, str]]:
    """Replace citation markers with [R1, R2] ids. Returns (rendered text, {id: reference key})."""
    id_for: dict[str, str] = {}  # key -> id
    pieces: list[str] = []
    cursor = sentence.start
    spans = sentence.marker_spans
    keys_per_marker = sentence.marker_keys or [sentence.reference_keys] * len(spans)
    for (a, b), keys in zip(spans, keys_per_marker):
        for key in keys:
            id_for.setdefault(key, f"R{len(id_for) + 1}")
        tag = "[" + ", ".join(id_for[k] for k in keys) + "]" if keys else ""
        marker = paper_text[a:b]
        if marker.startswith(("[", "(")) or not tag:
            replacement = tag
        else:  # textual citation such as "Lee et al. [2]" or "Brown et al. (2020)": keep the names
            replacement = f"{_TRAILING_CITATION_RE.sub('', marker)} {tag}"
        pieces.append(paper_text[cursor:a])
        pieces.append(replacement)
        cursor = b
    pieces.append(paper_text[cursor : sentence.end])
    text = re.sub(r"\s+", " ", "".join(pieces))
    text = re.sub(r"\s+([.,;:!?)])", r"\1", text).strip()
    return text, {rid: key for key, rid in id_for.items()}


def strip_markers(sentence: CitingSentence, paper_text: str) -> str:
    pieces, cursor = [], sentence.start
    for a, b in sentence.marker_spans:
        pieces.append(paper_text[cursor:a])
        cursor = b
    pieces.append(paper_text[cursor : sentence.end])
    text = re.sub(r"\s+", " ", "".join(pieces))
    text = re.sub(r"\(\s*\)|\[\s*\]", "", text)
    return re.sub(r"\s+([.,;:!?)])", r"\1", text).strip()


def build_messages(sentence: CitingSentence, paper: Paper, refs_by_key: dict[str, Reference]) -> tuple[list[dict], dict[str, str]]:
    # Reference titles are deliberately not shown: a small model tends to copy them into claims.
    rendered, id_map = render_sentence(sentence, paper.text)
    user = (
        f"Citing paper (its own systems and results are never attributed to cited works): {paper.title or '(unknown)'}\n"
        f"Context before: {sentence.context_before or '(none)'}\n"
        f"Citing sentence: {rendered}\n"
        f"Context after: {sentence.context_after or '(none)'}\n"
        f"Reference ids in this sentence: {', '.join(id_map)}"
    )
    return [{"role": "system", "content": load_prompt("extract")}, {"role": "user", "content": user}], id_map


def _normalize_id(raw: str) -> str:
    return raw.strip().strip("[]").upper()


# Words the prompt asks the model to add ("... is a method described in the cited work")
# or that carry no content; ignored when checking a claim is grounded in its sentence.
_FILLER = {
    "the", "and", "for", "with", "that", "this", "these", "those", "are", "was", "were", "has", "have",
    "been", "from", "into", "its", "their", "which", "also", "can", "cited", "work", "works", "described",
    "describes", "provides", "provided", "introduced", "introduces", "proposed", "proposes", "method",
    "methods", "approach", "model", "models", "paper", "study", "studies", "shows", "show", "showed",
    "based", "using", "used", "uses", "use", "than", "such", "other", "more", "is", "an", "a", "of", "in",
    "on", "to", "by", "as", "it", "be", "or", "at", "dataset", "benchmark", "tool", "system", "technique",
}
_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
MIN_GROUNDING = 0.5


def grounding(claim_text: str, source_text: str) -> float:
    """Share of the claim's content words that appear in the citing sentence."""
    source = set(_WORD_RE.findall(source_text.lower()))
    words = [w for w in _WORD_RE.findall(claim_text.lower()) if w not in _FILLER and len(w) > 1]
    if not words:
        return 1.0
    return sum(w in source for w in words) / len(words)


_LEADING_FILLER_RE = re.compile(
    r"^(?:[\s,;:()\-–—]|and\b|or\b|such as\b|e\.g\.|i\.e\.|including\b|like\b|see\b|cf\.)+", re.IGNORECASE
)


def marker_phrase(sentence: CitingSentence, paper_text: str, marker_index: int) -> str:
    """The phrase a marker is attached to: text since the previous marker or clause break."""
    a, _ = sentence.marker_spans[marker_index]
    seg_start = sentence.marker_spans[marker_index - 1][1] if marker_index > 0 else sentence.start
    segment = paper_text[seg_start:a]
    for sep in (";", ":", ","):
        head, _, tail = segment.rpartition(sep)
        if head and len(tail.split()) >= 1:
            segment = tail
            break
    phrase = _LEADING_FILLER_RE.sub("", re.sub(r"\s+", " ", segment)).strip(" .,;:")
    return phrase if len(phrase) >= 2 else strip_markers(sentence, paper_text)


def fallback_claims(sentence: CitingSentence, paper: Paper, keys: list[str], first_n: int) -> list[Claim]:
    """Lenient 'background' claims for references the model gave no claim for.

    Using the whole sentence as a finding would risk attributing the citing authors'
    own results to the cited work, so we only record what the marker is attached to.
    """
    claims: list[Claim] = []
    remaining = list(keys)
    keys_per_marker = sentence.marker_keys or [sentence.reference_keys] * len(sentence.marker_spans)
    for idx, marker_keys in enumerate(keys_per_marker):
        group = [k for k in marker_keys if k in remaining]
        if not group:
            continue
        remaining = [k for k in remaining if k not in group]
        phrase = marker_phrase(sentence, paper.text, idx)
        claims.append(
            Claim(
                id=f"{sentence.id}.c{first_n + len(claims)}",
                citing_sentence_id=sentence.id,
                claim_text=f"Cited in connection with: {phrase}",
                reference_keys=group,
                claim_type="background",
                numbers=[],
                fallback=True,
            )
        )
    return claims


async def extract_sentence_claims(
    sentence: CitingSentence, paper: Paper, refs_by_key: dict[str, Reference]
) -> list[Claim]:
    messages, id_map = build_messages(sentence, paper, refs_by_key)
    if not id_map:
        return []
    try:
        output = await llm.extract(messages, ExtractionOutput)
    except llm.TokenBudgetExceeded:
        raise
    except (llm.LLMError, openai.OpenAIError) as exc:
        log.warning("sentence %s: claim extraction failed (%s); using background claims", sentence.id, exc)
        return fallback_claims(sentence, paper, list(sentence.reference_keys), 1)

    plain_sentence = strip_markers(sentence, paper.text)
    sentence_numbers = set(number_strings(plain_sentence))
    # Pronoun resolution may borrow words from the previous sentence; author names come from markers.
    grounding_source = " ".join([sentence.text, sentence.context_before])
    claims: list[Claim] = []
    attributed = [(group.reference_ids, item) for group in output.citations for item in group.claims]
    for reference_ids, item in attributed:
        keys: list[str] = []
        for rid in reference_ids:
            key = id_map.get(_normalize_id(rid))
            if key and key not in keys:
                keys.append(key)
        if not keys:
            continue
        text = _REF_ID_RE.sub("", item.claim_text).strip()
        if grounding(text, grounding_source) < MIN_GROUNDING:
            log.info("sentence %s: dropping claim not grounded in the sentence: %r", sentence.id, text)
            continue
        numbers = number_strings(text)
        invented = [n for n in numbers if n not in sentence_numbers]
        if invented:
            log.warning("sentence %s: claim has numbers not in the sentence: %s", sentence.id, invented)
        claims.append(
            Claim(
                id=f"{sentence.id}.c{len(claims) + 1}",
                citing_sentence_id=sentence.id,
                claim_text=text,
                reference_keys=keys,
                claim_type=item.claim_type,
                numbers=numbers,
                scope=(item.scope or "").strip() or None,
            )
        )

    claims = merge_near_duplicates(claims)
    covered = {k for c in claims for k in c.reference_keys}
    uncovered = [k for k in sentence.reference_keys if k not in covered]
    if uncovered:
        log.info("sentence %s: no claim for %s; adding background claims", sentence.id, uncovered)
        claims.extend(fallback_claims(sentence, paper, uncovered, len(claims) + 1))
    return claims


# ----------------------------------------------------------------------------
# Near-duplicate claims
# ----------------------------------------------------------------------------

_SUFFIXES = ("izations", "ization", "isations", "isation", "ations", "ation", "izers", "izer", "ings", "ing",
             "ers", "er", "ies", "ied", "es", "ed", "ly", "s")
_TYPE_RANK = {"finding": 3, "definition": 2, "method": 1, "background": 0}
MERGE_JACCARD = 0.6


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def content_stems(text: str) -> frozenset[str]:
    return frozenset(_stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _FILLER and len(w) > 1)


def _same_statement(a: Claim, b: Claim) -> bool:
    """Same references, and either the same statement (same numbers, near-identical content
    words) or one claim fully subsumes the other (its words and numbers)."""
    if set(a.reference_keys) != set(b.reference_keys):
        return False
    sa, sb = content_stems(a.claim_text), content_stems(b.claim_text)
    if not sa or not sb:
        return False
    na, nb = set(a.numbers), set(b.numbers)
    if (sa <= sb and na <= nb) or (sb <= sa and nb <= na):
        return True
    return na == nb and len(sa & sb) / len(sa | sb) >= MERGE_JACCARD


def merge_near_duplicates(claims: list[Claim]) -> list[Claim]:
    """Keep one claim per statement, so each citation gets one card and one verdict.

    Small models often restate a claim ("Adam is an optimizer" / "Adam is used for model
    optimization"). Of a group of such claims the most specific one is kept: the most
    numbers, the most specific type, then the most content words. Claims stating different
    numbers ("a 350M model" / "a 1.3B model") are never merged, so each number can still
    be checked for drift.
    """
    kept: list[Claim] = []
    for claim in claims:
        match = next((i for i, k in enumerate(kept) if _same_statement(k, claim)), None)
        if match is None:
            kept.append(claim)
            continue
        current = kept[match]
        rank = lambda c: (  # noqa: E731
            len(c.numbers), _TYPE_RANK.get(c.claim_type, 0), len(content_stems(c.claim_text)), len(c.claim_text)
        )
        if rank(claim) > rank(current):
            kept[match] = claim
        log.info("sentence %s: merged near-duplicate claims %r and %r",
                 claim.citing_sentence_id, current.claim_text, claim.claim_text)
    # Renumber so ids stay dense and stable for a given extraction.
    return [c.model_copy(update={"id": f"{c.citing_sentence_id}.c{i}"}) for i, c in enumerate(kept, start=1)]


async def extract_claims(
    sentences: list[CitingSentence],
    paper: Paper,
    refs_by_key: Optional[dict[str, Reference]] = None,
    on_done: Optional[OnDone] = None,
) -> list[Claim]:
    """Extract claims for all sentences concurrently (the LLM semaphore bounds parallelism)."""
    refs_by_key = refs_by_key if refs_by_key is not None else {r.key: r for r in paper.references}

    async def run(sentence: CitingSentence) -> list[Claim]:
        claims = await extract_sentence_claims(sentence, paper, refs_by_key)
        if on_done is not None:
            result = on_done(sentence, claims)
            if inspect.isawaitable(result):
                await result
        return claims

    per_sentence = await asyncio.gather(*(run(s) for s in sentences))
    return [c for claims in per_sentence for c in claims]


# Most severe first: when duplicates disagree, keep the verdict that asks for a human check.
_SEVERITY = ("contradicted", "not_found", "overstated", "unverifiable", "supported")


def mark_duplicate_verdicts(verdicts: list[Verdict], claims: dict[str, Claim]) -> list[Verdict]:
    """Collapse verdicts that restate each other for the same sentence and reference.

    Extraction merges near-duplicate claims, but the judge can also narrow a misattributed
    claim to the same statement as a sibling ("A feedforward dimension of 4x is used [Adam]"
    becomes "Adam is used for model optimization"). The judge's attributed claim is what was
    actually checked, so duplicates are found on that. The most cautious verdict is kept
    (most severe label, then highest confidence); the others get `duplicate_of`.
    """
    out = [v.model_copy(update={"duplicate_of": None}) for v in verdicts]
    groups: dict[tuple[str, str], list[int]] = {}
    for i, v in enumerate(out):
        claim = claims.get(v.claim_id)
        if claim is not None:
            groups.setdefault((claim.citing_sentence_id, v.reference_key), []).append(i)

    def as_claim(v: Verdict) -> Claim:
        text = v.attributed_claim or claims[v.claim_id].claim_text
        return Claim(id=v.claim_id, citing_sentence_id="", claim_text=text, reference_keys=[v.reference_key],
                     numbers=number_strings(text))

    def rank(v: Verdict) -> tuple:
        return (_SEVERITY.index(v.label), -v.confidence, not v.escalated, v.claim_id)

    for members in groups.values():
        kept: list[int] = []
        for i in sorted(members, key=lambda i: rank(out[i])):
            twin = next((k for k in kept if _same_statement(as_claim(out[k]), as_claim(out[i]))), None)
            if twin is None:
                kept.append(i)
            else:
                out[i].duplicate_of = f"{out[twin].claim_id}|{out[twin].reference_key}"
    return out
