"""Verdicts (MODEL_JUDGE, Nemotron Super) and escalation (MODEL_ESCALATE, Nemotron Ultra).

The judge sees the citing sentence, the extracted claim and the retrieved passages.
Hard rules are enforced in code, not only in the prompt:
- supported / overstated / contradicted need a quote that is verbatim in the cited text
  (normalized matching); otherwise confidence is halved and `quote_unverified` is flagged;
- background claims can only be supported or unverifiable;
- "not found" against an abstract alone is downgraded to unverifiable;
- references without any text are unverifiable without calling a model.

Escalation sends the full cited text to the long-context model when the judge's
confidence is below 0.6, the label is not_found / contradicted (these read as
accusations), the quote failed verification, or the label is unverifiable although the
full text is available (usually a retrieval miss). The escalation verdict replaces the
judge's, and the original label is kept for the "escalation changed the label" stats.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Union

import openai
from pydantic import BaseModel, Field

from app import llm
from app.config import get_settings
from app.models import (
    CitingSentence,
    Claim,
    Evidence,
    JudgeSignals,
    Paper,
    Passage,
    Reference,
    Verdict,
    VerdictLabel,
)
from app.pipeline.claims import render_sentence
from app.pipeline.references import read_reference_text
from app.pipeline.textutil import locate_quote
from app.prompts import load_prompt

log = logging.getLogger(__name__)

ESCALATE_BELOW_CONFIDENCE = 0.6
ACCUSATORY_LABELS = ("not_found", "contradicted")
QUOTE_REQUIRED = ("supported", "overstated", "contradicted")
QUOTE_UNVERIFIED_FACTOR = 0.5
PASSAGE_MAX_CHARS = 3000
CONTEXT_CHARS = 280

NO_TEXT_EXPLANATION = "No text of the cited work was available to check this claim."
JUDGE_FAILED_EXPLANATION = "The automatic check could not be completed for this citation."

OnVerdict = Callable[[Verdict], Union[None, Awaitable[None]]]


class JudgeOutput(BaseModel):
    attributed_claim: str
    label: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    quote: Optional[str] = None
    passage_id: Optional[str] = None
    explanation: str
    signals: JudgeSignals = Field(default_factory=JudgeSignals)


@dataclass
class JudgeContext:
    paper: Paper
    sentences: dict[str, CitingSentence]
    references: dict[str, Reference]


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------


def _rules() -> str:
    return load_prompt("judge").split("## Rules", 1)[1].strip()


def judge_system_prompt() -> str:
    return load_prompt("judge")


def escalate_system_prompt() -> str:
    return f"{load_prompt('escalate')}\n\n## Rules\n\n{_rules()}"


def _sentence_block(claim: Claim, sentence: CitingSentence, ref: Reference, paper: Paper) -> str:
    rendered, id_map = render_sentence(sentence, paper.text)
    target = next((rid for rid, key in id_map.items() if key == ref.key), "the cited work")
    year = f", {ref.year}" if ref.year else ""
    available = {"full": "full text", "abstract_only": "abstract only"}.get(ref.fulltext_status, "none")
    return (
        f"Citing paper (its own method and results are never attributed to cited works): {paper.title or '(unknown)'}\n"
        f"Context before: {sentence.context_before or '(none)'}\n"
        f"Citing sentence: {rendered}\n"
        f"Context after: {sentence.context_after or '(none)'}\n"
        f"Reference being checked: {target} = {ref.title or ref.raw[:200]}{year}. Text available: {available}.\n"
        f"Extracted claim (a hint, may attribute too much; type: {claim.claim_type}): {claim.claim_text}"
    )


def _passage_label(p: Passage) -> str:
    return "Abstract" if p.chunk_index < 0 else (f"Section: {p.heading}" if p.heading else "Body")


def ordered_passages(evidence: Evidence) -> list[Passage]:
    """Abstract first, then chunks by descending similarity."""
    abstract = [p for p in evidence.passages if p.chunk_index < 0]
    chunks = sorted((p for p in evidence.passages if p.chunk_index >= 0), key=lambda p: -p.score)
    return abstract + chunks


def build_judge_messages(
    claim: Claim, sentence: CitingSentence, ref: Reference, evidence: Evidence, paper: Paper
) -> list[dict]:
    lines = []
    for i, p in enumerate(ordered_passages(evidence)):
        text = p.text if len(p.text) <= PASSAGE_MAX_CHARS else p.text[:PASSAGE_MAX_CHARS] + " [...]"
        lines.append(f"[P{i}] ({_passage_label(p)})\n{text}")
    user = (
        _sentence_block(claim, sentence, ref, paper)
        + "\n\nPassages from the cited work:\n\n"
        + "\n\n".join(lines)
    )
    return [{"role": "system", "content": judge_system_prompt()}, {"role": "user", "content": user}]


def escalation_text(ref: Reference, evidence: Optional[Evidence], max_chars: int) -> str:
    """Full text for the long-context model; if too long, keep the head plus the retrieved passages."""
    full = read_reference_text(ref) or ""
    abstract = f"Abstract: {ref.abstract}\n\n" if ref.abstract and ref.abstract[:200] not in full else ""
    text = abstract + full
    if len(text) <= max_chars:
        return text
    extra = [p.text for p in (evidence.passages if evidence else []) if p.chunk_index >= 0]
    tail = "\n\n[...]\n\n" + "\n\n[...]\n\n".join(extra)
    return text[: max(0, max_chars - len(tail))] + tail


def build_escalation_messages(
    claim: Claim, sentence: CitingSentence, ref: Reference, first: Verdict, full_text: str, paper: Paper
) -> list[dict]:
    user = (
        _sentence_block(claim, sentence, ref, paper)
        + f"\n\nFirst reviewer (saw excerpts only): label {first.label}, confidence {first.confidence:.2f}. "
        + f"{first.explanation}\n\n"
        + ("Full text of the cited work" if ref.fulltext_status == "full" else "Only the abstract of the cited work is openly available")
        + f":\n<<<\n{full_text}\n>>>"
    )
    return [{"role": "system", "content": escalate_system_prompt()}, {"role": "user", "content": user}]


# ----------------------------------------------------------------------------
# Hard rules
# ----------------------------------------------------------------------------

_TONE = [
    (re.compile(r"\berroneously\b", re.I), "inaccurately"),
    (re.compile(r"\berroneous\b", re.I), "inaccurate"),
    (re.compile(r"\ban error\b", re.I), "a discrepancy"),
    (re.compile(r"\berrors\b", re.I), "discrepancies"),
    (re.compile(r"\berror\b", re.I), "discrepancy"),
    (re.compile(r"\bwrongly\b", re.I), "inaccurately"),
    (re.compile(r"\bis wrong\b", re.I), "is not supported by the cited text"),
    (re.compile(r"\ba wrong\b", re.I), "an inaccurate"),
    (re.compile(r"\bwrong\b", re.I), "inaccurate"),
    (re.compile(r"\bmisconduct\b", re.I), "an issue"),
]


def neutral_tone(text: str) -> str:
    for pattern, repl in _TONE:
        text = pattern.sub(repl, text)
    return text.strip()


def _snap_context(text: str, start: int, end: int) -> tuple[str, int, int]:
    lo = max(0, start - CONTEXT_CHARS)
    hi = min(len(text), end + CONTEXT_CHARS)
    if lo > 0:
        space = text.find(" ", lo, start)
        lo = space + 1 if space != -1 else lo
    if hi < len(text):
        space = text.rfind(" ", end, hi)
        hi = space if space != -1 else hi
    return text[lo:hi], start - lo, end - lo


def apply_rules(out: JudgeOutput, claim: Claim, ref: Reference, model: str, search_texts: list[str]) -> Verdict:
    label: VerdictLabel = out.label
    confidence = out.confidence
    flags: list[str] = []

    if claim.claim_type == "background" and label not in ("supported", "unverifiable"):
        label = "unverifiable"
        flags.append("background_lenient")
    if ref.fulltext_status == "abstract_only" and label == "not_found":
        label = "unverifiable"
        flags.append("abstract_only")

    verdict = Verdict(
        claim_id=claim.id,
        reference_key=ref.key,
        label=label,
        confidence=confidence,
        attributed_claim=(out.attributed_claim or claim.claim_text).strip(),
        explanation=neutral_tone(out.explanation),
        signals=out.signals,
        model_used=model,
    )

    if label in QUOTE_REQUIRED or out.quote:
        for text in search_texts:
            span = locate_quote(out.quote, text)
            if span:
                verdict.quote = text[span[0] : span[1]]  # the source's exact characters
                verdict.quote_verified = True
                verdict.quote_context, verdict.quote_start, verdict.quote_end = _snap_context(text, *span)
                break
        if not verdict.quote_verified and label in QUOTE_REQUIRED:
            verdict.quote = out.quote
            verdict.confidence = round(confidence * QUOTE_UNVERIFIED_FACTOR, 3)
            flags.append("quote_unverified")
    verdict.flags = flags
    return verdict


def needs_escalation(verdict: Verdict, claim: Claim, ref: Reference) -> bool:
    if claim.claim_type == "background" or verdict.escalated or "no_text" in verdict.flags:
        return False
    if ref.fulltext_status == "abstract_only":
        # Nothing more to read, but an accusation from an abstract alone deserves a second look.
        return verdict.label in ACCUSATORY_LABELS or "quote_unverified" in verdict.flags
    if ref.fulltext_status != "full":
        return False
    if "judge_error" in verdict.flags:
        return True
    return (
        verdict.confidence < ESCALATE_BELOW_CONFIDENCE
        or verdict.label in ACCUSATORY_LABELS
        or "quote_unverified" in verdict.flags
        or verdict.label == "unverifiable"  # with full text available, usually a retrieval miss
    )


def _search_texts(ref: Reference, full_text: Optional[str]) -> list[str]:
    return [t for t in (full_text, ref.abstract) if t]


# ----------------------------------------------------------------------------
# Judge and escalate one (claim, reference) pair
# ----------------------------------------------------------------------------


def _no_text_verdict(claim: Claim, ref: Reference) -> Verdict:
    return Verdict(
        claim_id=claim.id,
        reference_key=ref.key,
        label="unverifiable",
        confidence=1.0,
        attributed_claim=claim.claim_text,
        explanation=ref.status_reason or NO_TEXT_EXPLANATION,
        flags=["no_text"],
    )


async def judge_pair(claim: Claim, ref: Reference, evidence: Optional[Evidence], ctx: JudgeContext) -> Verdict:
    if evidence is None or not evidence.passages:
        return _no_text_verdict(claim, ref)
    sentence = ctx.sentences[claim.citing_sentence_id]
    model = get_settings().model_for("judge")
    try:
        out = await llm.judge(build_judge_messages(claim, sentence, ref, evidence, ctx.paper), JudgeOutput)
    except llm.TokenBudgetExceeded:
        raise
    except (llm.LLMError, openai.OpenAIError) as exc:
        log.warning("judge %s / %s failed: %s", claim.id, ref.key, exc)
        return Verdict(
            claim_id=claim.id, reference_key=ref.key, label="unverifiable", confidence=0.0,
            attributed_claim=claim.claim_text, explanation=JUDGE_FAILED_EXPLANATION,
            flags=["judge_error"], model_used=model,
        )
    full_text = read_reference_text(ref) if ref.fulltext_status == "full" else None
    return apply_rules(out, claim, ref, model, _search_texts(ref, full_text))


async def escalate_pair(
    claim: Claim, ref: Reference, first: Verdict, evidence: Optional[Evidence], ctx: JudgeContext
) -> Verdict:
    sentence = ctx.sentences[claim.citing_sentence_id]
    settings = get_settings()
    model = settings.model_for("escalate")
    full_text = escalation_text(ref, evidence, settings.escalate_max_chars)
    try:
        out = await llm.escalate(
            build_escalation_messages(claim, sentence, ref, first, full_text, ctx.paper), JudgeOutput
        )
    except llm.TokenBudgetExceeded:
        raise
    except (llm.LLMError, openai.OpenAIError) as exc:
        log.warning("escalation %s / %s failed: %s", claim.id, ref.key, exc)
        return first.model_copy(update={"flags": [*first.flags, "escalation_failed"]})
    verdict = apply_rules(out, claim, ref, model, _search_texts(ref, read_reference_text(ref)))
    verdict.escalated = True
    verdict.original_label = first.label
    verdict.original_confidence = first.confidence
    return verdict


async def judge_all(
    claims: list[Claim],
    evidence: list[Evidence],
    ctx: JudgeContext,
    on_judged: Optional[OnVerdict] = None,
    on_final: Optional[OnVerdict] = None,
) -> list[Verdict]:
    """Judge every (claim, reference) pair, escalating where the rules say so.

    `on_judged` fires after the first verdict; `on_final` after escalation (or right away
    when no escalation is needed). Failures become unverifiable verdicts, never exceptions.
    """
    by_pair = {(e.claim_id, e.reference_key): e for e in evidence}

    async def run(claim: Claim, key: str) -> Verdict:
        ref = ctx.references.get(key) or Reference(key=key, raw="", status_reason=NO_TEXT_EXPLANATION)
        ev = by_pair.get((claim.id, key))
        verdict = await judge_pair(claim, ref, ev, ctx)
        await _call(on_judged, verdict)
        if needs_escalation(verdict, claim, ref):
            verdict = await escalate_pair(claim, ref, verdict, ev, ctx)
        await _call(on_final, verdict)
        return verdict

    return list(await asyncio.gather(*(run(c, k) for c in claims for k in c.reference_keys)))


async def _call(callback: Optional[OnVerdict], verdict: Verdict) -> None:
    if callback is not None:
        result = callback(verdict)
        if inspect.isawaitable(result):
            await result


class EscalationStats(BaseModel):
    verdicts: int
    escalated: int
    label_changed: int
    transitions: dict[str, int]  # "not_found->supported": 3


def escalation_stats(verdicts: list[Verdict]) -> EscalationStats:
    escalated = [v for v in verdicts if v.escalated]
    changed = [v for v in escalated if v.original_label != v.label]
    transitions = Counter(f"{v.original_label}->{v.label}" for v in changed)
    return EscalationStats(
        verdicts=len(verdicts),
        escalated=len(escalated),
        label_changed=len(changed),
        transitions=dict(transitions.most_common()),
    )
