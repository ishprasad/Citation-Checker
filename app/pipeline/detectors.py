"""Failure-mode detectors (brief section 6).

Each tag needs rule-based evidence on top of the judge's structured signals, so a
single hallucinated signal cannot put a chip on a citation:

- scope_inflation: the judge found the claim's scope broader than the source's
  (population, dataset, setting) on an overstated/contradicted verdict.
- number_drift: the judge paired a claim number with a source number as the same
  quantity; both numbers really occur in the claim and in the verified quote, and they
  differ beyond the claim's rounding. A "supported" verdict with drift becomes
  "overstated".
- hypothesis_as_finding: the source passage hedges (judge signal or hedge words in
  the quote) while the claim states the point as fact.
- secondary_citation: the verified supporting quote itself carries a citation marker,
  i.e. the cited paper credits the point to another work; that work is named from the
  cited paper's own bibliography when possible.
- retracted_source: the reference is marked as retracted.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Optional

from app.models import Claim, Reference, Verdict
from app.pipeline.references import read_reference_sidecar
from app.pipeline.textutil import Number, extract_numbers, numbers_match

HEDGE_RE = re.compile(
    r"\b(may|might|could|possibly|potentially|perhaps|likely|unlikely|suggests?|suggested|suggesting|"
    r"hypothes[ie][sz]e?[sd]?|hypothesis|preliminary|tentative(?:ly)?|speculate[sd]?|appears? to|"
    r"seems? to|we believe|indicates?|plausibl[ey])\b",
    re.IGNORECASE,
)
NUMERIC_MARKER_RE = re.compile(r"\[(\d+)(?:\s?[,–-]\s?\d+)*\]")
AUTHOR_YEAR_MARKER_RE = re.compile(
    r"\(?(?:see\s+)?([A-Z][A-Za-z'\-]+)(?:\s+(?:et al\.?|and [A-Z][A-Za-z'\-]+))?,?\s+\(?((?:19|20)\d{2})[a-z]?\)?"
)

SidecarLoader = Callable[[Reference], dict]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _mostly_numbers(text: str) -> bool:
    """A "scope" that is just a list of values ("350M, 1.3B, 6.7B") is a number issue, not scope."""
    words = text.split()
    return bool(words) and sum(bool(extract_numbers(w)) for w in words) >= len(words) / 2


def _first_number(text: str) -> Optional[Number]:
    found = extract_numbers(text or "")
    return found[0] if found else None


def _occurs(number: Number, texts: list[str]) -> bool:
    return any(numbers_match(number, other) is True for t in texts for other in extract_numbers(t or ""))


def citation_markers(text: str) -> list[str]:
    """Citation markers in a passage: "[12]", "(Smith et al., 2019)", "Smith et al. (2019)"."""
    found = [m.group(0) for m in NUMERIC_MARKER_RE.finditer(text or "")]
    for m in AUTHOR_YEAR_MARKER_RE.finditer(text or ""):
        span = m.group(0)
        if "et al" in span or " and " in span or span.startswith("("):
            found.append(span.strip())
    return found


def lookup_original(marker: str, sidecar: dict) -> Optional[str]:
    """Name the work behind a citation marker using the cited paper's bibliography."""
    numeric = NUMERIC_MARKER_RE.search(marker)
    author = AUTHOR_YEAR_MARKER_RE.search(marker)
    for entry in sidecar.get("references", []):
        label = entry.get("label") or ""
        if numeric and label == numeric.group(1):
            return entry.get("title") or entry.get("raw")
        if author and author.group(1).lower() in label.lower() and author.group(2) in label:
            return entry.get("title") or entry.get("raw")
    refs_text = sidecar.get("references_text") or ""
    if refs_text:
        if numeric:
            m = re.search(rf"(?m)^\s*\[?{numeric.group(1)}\]?\.?\s+(.+)$", refs_text)
            if m:
                return m.group(1).strip()[:200]
        if author:
            for line in refs_text.splitlines():
                if author.group(1) in line and author.group(2) in line:
                    return line.strip()[:200]
    return None


def detect(verdict: Verdict, claim: Claim, ref: Optional[Reference], sidecar_loader: SidecarLoader) -> Verdict:
    v = verdict.model_copy(deep=True)
    modes: list[str] = []
    notes: dict[str, str] = {}
    s = v.signals
    claim_texts = [v.attributed_claim, claim.claim_text]
    source_texts = [v.quote_context or v.quote or ""] if v.quote_verified else []
    is_background = claim.claim_type == "background"

    # Retracted source applies to every claim citing the work.
    if ref is not None and ref.retracted:
        modes.append("retracted_source")
        notes["retracted_source"] = (
            f"The cited work has been retracted (notice: doi:{ref.retraction_notice})."
            if ref.retraction_notice else "The cited work is marked as retracted."
        )

    if not is_background:
        # Scope inflation.
        scopes_differ = bool(
            s.claim_scope and s.source_scope
            and _norm(s.claim_scope) != _norm(s.source_scope)
            and not _mostly_numbers(s.claim_scope)
        )
        if v.label in ("overstated", "contradicted") and s.scope_broader_than_source and scopes_differ:
            modes.append("scope_inflation")
            notes["scope_inflation"] = f"Claim scope: {s.claim_scope}. Source scope: {s.source_scope}."

        # Number drift: judge-confirmed same quantity, numbers grounded in both texts.
        if source_texts:
            for cmp in s.numbers:
                if not cmp.same_quantity:
                    continue
                cn, sn = _first_number(cmp.claim_value), _first_number(cmp.source_value)
                if cn is None or sn is None:
                    continue
                if not _occurs(cn, claim_texts) or not _occurs(sn, source_texts):
                    continue
                if numbers_match(cn, sn) is False:
                    modes.append("number_drift")
                    notes["number_drift"] = f"{cn.raw} in the claim; {sn.raw} in the cited work."
                    break
            if "number_drift" in modes and v.label == "supported":
                v.label = "overstated"
                v.confidence = min(v.confidence, 0.7)
                v.flags.append("label_adjusted_number_drift")

        # Hypothesis stated as finding.
        if claim.claim_type == "finding" and v.label == "overstated":
            hedge = HEDGE_RE.search(v.quote or "") if v.quote_verified else None
            claim_hedged = s.claim_hedged or any(HEDGE_RE.search(t or "") for t in claim_texts)
            if (s.source_hedged or hedge) and not claim_hedged:
                modes.append("hypothesis_as_finding")
                notes["hypothesis_as_finding"] = (
                    f'The cited work hedges this point ("{hedge.group(0)}"); the claim states it as fact.'
                    if hedge else "The cited work presents this tentatively; the claim states it as fact."
                )

        # Secondary citation: the supporting passage itself cites another work.
        # Needs both the judge's reading (the point itself is credited elsewhere, not a
        # passing mention) and a citation marker actually visible in the source text.
        if v.label in ("supported", "overstated") and v.quote_verified and s.support_attributed_to_other_work:
            markers = citation_markers(v.quote or "")
            other = s.other_work if s.other_work and s.other_work in (v.quote_context or "") else None
            marker = other or (markers[0] if markers else None)
            if marker:
                modes.append("secondary_citation")
                original = lookup_original(marker, sidecar_loader(ref)) if ref is not None else None
                notes["secondary_citation"] = (
                    f'The supporting passage credits {marker}: "{original}".' if original
                    else f"The supporting passage credits {marker}."
                )

    v.failure_modes = list(dict.fromkeys(modes))
    v.failure_notes = notes
    return v


def apply_detectors(
    verdicts: list[Verdict],
    claims: dict[str, Claim],
    references: dict[str, Reference],
    sidecar_loader: SidecarLoader = read_reference_sidecar,
) -> list[Verdict]:
    cache: dict[str, dict] = {}

    def load(ref: Reference) -> dict:
        if ref.key not in cache:
            cache[ref.key] = sidecar_loader(ref)
        return cache[ref.key]

    return [detect(v, claims[v.claim_id], references.get(v.reference_key), load) for v in verdicts]


def failure_mode_counts(verdicts: list[Verdict]) -> dict[str, int]:
    return dict(Counter(m for v in verdicts for m in v.failure_modes).most_common())
