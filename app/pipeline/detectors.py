"""Failure-mode detectors (brief section 6).

Each tag needs rule-based evidence on top of the judge's structured signals, so a
single hallucinated signal cannot put a chip on a citation:

- scope_inflation: the judge found the claim's scope broader than the source's
  (population, dataset, setting) on an overstated/contradicted verdict.
- number_drift: the judge paired a claim number with a source number as the same
  quantity; both numbers really occur in the claim and in the verified quote, and they
  differ beyond the claim's rounding or truncation. A "supported" verdict with drift becomes
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
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

from app.models import Claim, Evidence, Reference, Verdict
from app.pipeline.references import read_reference_sidecar
from app.pipeline.textutil import Number, extract_numbers, numbers_match, restore_exponents

HEDGE_RE = re.compile(
    r"\b(may|might|could|possibly|potentially|perhaps|likely|unlikely|suggests?|suggested|suggesting|"
    r"hypothes[ie][sz]e?[sd]?|hypothesis|preliminary|tentative(?:ly)?|speculate[sd]?|appears? to|"
    r"seems? to|we believe|indicates?|plausibl[ey]|postulat(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
# For "supported" verdicts only: a modal hedge in the source (not a reporting verb such as "suggests",
# which papers routinely use for their own results) against an explicitly certain claim.
MODAL_HEDGE_RE = re.compile(
    r"\b(may|might|could|possibly|potentially|perhaps|likely|hypothes[ie][sz]e[sd]?|hypothesis|speculat\w*|"
    r"postulat(?:e|es|ed|ing)|tentative(?:ly)?)\b",
    re.IGNORECASE,
)
CERTAINTY_RE = re.compile(
    r"\b(will|(?:is|are|was|were) known to|(?:is|are) (?:well[- ])?established|prove[sdn]?|demonstrates?|"
    r"clearly|certainly|undoubtedly|definitely|inevitably|must)\b",
    re.IGNORECASE,
)
CAUSAL_RE = re.compile(
    r"\b(caus(?:e|es|ed|ing)|leads? to|led to|results? in|resulted in|drives?|drove|induces?|induced|"
    r"prevents?|prevented|reduces?|reduced|increases?|increased|improves?|improved|decreases?|decreased|"
    r"protects? against|effects? (?:of|on)|because of|due to|responsible for)\b",
    re.IGNORECASE,
)
ASSOCIATIONAL_RE = re.compile(
    r"\b(associat(?:ed|ion|ions)|correlat(?:ed|es|ion|ions)|linked (?:to|with)|link between|"
    r"relationship between|co-?occur\w*|predictive of|odds ratios?|hazard ratios?|risk ratios?|"
    r"observational|cross-sectional)\b",
    re.IGNORECASE,
)
SIGNIFICANCE_CLAIM_RE = re.compile(
    r"\b(statistically significant(?:ly)?|significant(?:ly)?|proven|proves?|conclusive(?:ly)?|definitive(?:ly)?)\b",
    re.IGNORECASE,
)
NOT_SIGNIFICANT_RE = re.compile(
    r"(not (?:statistically )?significant|non-?significant|\bn\.\s?s\.|(?:did not|failed to) reach (?:statistical )?"
    r"significance|no (?:statistically )?significant|\bp\s?[>≥]\s?0?\.0?5\b|\bp\s?=\s?0?\.(?:0[5-9]|[1-9])\d*)",
    re.IGNORECASE,
)
# Sweeping qualifiers: a claim that holds universally needs a source that says so.
SWEEPING_RE = re.compile(
    r"\b(universally|without (?:any )?exceptions?|invariably|always|never|regardless of|"
    r"in (?:all|every|any) (?:cases?|populations?|settings?|contexts?|conditions?|circumstances?|situations?|instances?)|"
    r"(?:for|across|under) all (?:populations|settings|contexts|conditions|circumstances))\b",
    re.IGNORECASE,
)
# Tags that rest on a concrete mismatch rather than on a detail an abstract does not mention.
MISMATCH_MODES = {"scope_inflation", "number_drift", "hypothesis_as_finding", "correlation_as_causation",
                  "significance_inflation"}
CORRECTION_TYPES = {"correction", "erratum", "corrigendum", "partial_retraction"}
CONCERN_TYPES = {"expression_of_concern"}
OFF_TOPIC_Z = -1.5  # robust z-score of the best passage similarity, relative to the paper's citations
OFF_TOPIC_GAP = 0.15  # another reference's abstract fits the claim at least this much better
SUGGEST_GAP = 0.15  # ...name it as the possibly intended reference
MIN_BASELINE_PAIRS = 8

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
    """The number is stated in one of the texts. The judge often drops units ("5" for "5%"), so a
    unitless number also matches the same value with a unit."""
    for other in (o for t in texts for o in extract_numbers(t or "")):
        if numbers_match(number, other) is True:
            return True
        if not number.unit and numbers_match(number, Number(other.raw, other.value, "")) is True:
            return True
    return False


_NUM = r"[-−]?\d[\d,]*(?:\.\d+)?"
# "3–6", "1800 −2000", "between z 3 and z 6.6", "9 M⊕ to 10 M⊕" (a short unit or variable may
# follow either end)
_RANGE_RE = re.compile(
    rf"(?P<a>{_NUM})\s*(?:[A-Za-z%⊕☉]{{1,4}}\s*)?(?:[-–—−]|to\b|and\b)\s*(?:[A-Za-z~∼≈]{{1,2}}\s+|[~∼≈])?(?P<b>{_NUM})")
_PLUS_MINUS_RE = re.compile(rf"(?P<a>{_NUM})\s*(?:±|\+/-|\+-)\s*(?P<e>{_NUM})")
_LOWER_BOUND_RE = re.compile(
    r"(?:>|≥|≳|⩾|at least|more than|greater than|higher than|larger than|over|above|exceeds?|exceeding|in excess of)"
    r"\s*[~∼≈]?\s*$", re.IGNORECASE)
_UPPER_BOUND_RE = re.compile(
    r"(?:<|≤|≲|⩽|up to|out to|less than|lower than|smaller than|below|under|at most|no more than|within)"
    r"\s*[~∼≈]?\s*$", re.IGNORECASE)


def _value(raw: str) -> float:
    return float(raw.replace("−", "-").replace(",", ""))


def _intervals(text: str) -> list[tuple[float, float]]:
    """Value ranges stated in a text: "a–b" (ascending only, so "10-5" stays a subtraction or an
    identifier) and "a ± e"."""
    out = []
    for m in _RANGE_RE.finditer(restore_exponents(text or "")):
        a, b = _value(m.group("a")), abs(_value(m.group("b")))
        if a < b:
            out.append((a, b))
    for m in _PLUS_MINUS_RE.finditer(text or ""):
        a, e = _value(m.group("a")), abs(_value(m.group("e")))
        out.append((a - e, a + e))
    return out


def _bound(number: Number, text: str) -> Optional[str]:
    """'lower' for "at least 100" / "> 0.1", 'upper' for "up to 300" / "< 5", else None."""
    for m in re.finditer(re.escape(number.raw), text or ""):
        before = text[max(0, m.start() - 20) : m.start()]
        if _LOWER_BOUND_RE.search(before):
            return "lower"
        if _UPPER_BOUND_RE.search(before):
            return "upper"
    return None


def _inside(value: float, intervals: list[tuple[float, float]]) -> bool:
    return any(lo - 1e-9 * max(1.0, abs(lo)) <= value <= hi + 1e-9 * max(1.0, abs(hi)) for lo, hi in intervals)


def numbers_consistent(cmp_claim: str, cmp_source: str, claim_text: str, cn: Number, sn: Number) -> bool:
    """A claim number that differs from the source number but does not contradict it:
    - the source value lies within a range the claim states ("between 3 and 6.6" for 6), unless
      the source states a range of its own whose ends differ from the claim's;
    - the claim value lies within a range or ± interval the source states (2000 K for "1800–2000 K");
    - the claim states a bound the source value satisfies ("at least 100" for 125,000);
    - any number of the judge's claim value matches any number of its source value."""
    claim_ranges = _intervals(cmp_claim) + _intervals(claim_text)
    source_ranges = _intervals(cmp_source)
    if _inside(cn.value, source_ranges):
        return True
    if _inside(sn.value, claim_ranges) and not source_ranges:
        return True
    bound = _bound(cn, cmp_claim) or _bound(cn, claim_text)
    if bound == "lower" and sn.value >= cn.value or bound == "upper" and sn.value <= cn.value:
        return True
    claim_numbers, source_numbers = extract_numbers(cmp_claim), extract_numbers(cmp_source)
    return any(numbers_match(c, s) is True for c in claim_numbers for s in source_numbers) and not (
        len(claim_numbers) == len(source_numbers) == 2 and not all(
            numbers_match(c, s) is True for c, s in zip(claim_numbers, source_numbers)))


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


def _notice_note(what: str, updates: list) -> str:
    first = updates[0]
    details = ", ".join(p for p in (f"doi:{first.doi}" if first.doi else "", first.date or "") if p)
    more = f" ({len(updates)} notices)" if len(updates) > 1 else ""
    return f"{what} was published for the cited work{more}" + (f" ({details})." if details else ".")


def _downgrade(v: Verdict, reason: str) -> None:
    """Hard rule-based evidence of an overstatement overrides a 'supported' verdict."""
    if v.label == "supported":
        v.label = "overstated"
        v.confidence = min(v.confidence, 0.7)
        v.flags.append(f"label_adjusted_{reason}")


@dataclass
class PaperBaseline:
    """Distribution of best-passage similarities across a paper's citations."""

    median: float
    mad: float

    @classmethod
    def from_scores(cls, scores: list[float]) -> Optional["PaperBaseline"]:
        if len(scores) < MIN_BASELINE_PAIRS:
            return None
        median = statistics.median(scores)
        mad = statistics.median(abs(x - median) for x in scores)
        return cls(median, mad)

    def z(self, score: float) -> float:
        return (score - self.median) / (1.4826 * self.mad + 1e-6)


def detect(
    verdict: Verdict,
    claim: Claim,
    ref: Optional[Reference],
    sidecar_loader: SidecarLoader,
    best_score: Optional[float] = None,
    baseline: Optional[PaperBaseline] = None,
    alternative: Optional[tuple[str, float]] = None,  # (label of best-fitting other reference, similarity gap)
    clause: Optional[str] = None,  # the part of the citing sentence this citation closes (citation_clause)
) -> Verdict:
    v = verdict.model_copy(deep=True)
    modes: list[str] = []
    notes: dict[str, str] = {}
    s = v.signals
    claim_texts = [v.attributed_claim, claim.claim_text]
    # The sentence's own wording: extraction and the judge's restatement can drop a qualifier or a hedge.
    stated_texts = claim_texts + ([clause] if clause else [])
    source_texts = [v.quote_context or v.quote or ""] if v.quote_verified else []
    is_background = claim.claim_type == "background"

    # Editorial notices apply to every claim citing the work.
    if ref is not None and ref.retracted:
        modes.append("retracted_source")
        notes["retracted_source"] = (
            f"The cited work has been retracted (notice: doi:{ref.retraction_notice})."
            if ref.retraction_notice else "The cited work is marked as retracted."
        )
    if ref is not None:
        corrections = [u for u in ref.updates if u.type in CORRECTION_TYPES]
        concerns = [u for u in ref.updates if u.type in CONCERN_TYPES]
        if corrections:
            modes.append("corrected_source")
            notes["corrected_source"] = _notice_note("A correction", corrections)
        if concerns:
            modes.append("expression_of_concern")
            notes["expression_of_concern"] = _notice_note("An expression of concern", concerns)

    # Off-topic citation: the judge says the work is not about the claim's topic, and the
    # embeddings agree: another reference in this paper's bibliography fits the claim clearly
    # better, or the best passage is an outlier among this paper's citations.
    if (
        not is_background
        and v.label in ("not_found", "unverifiable")
        and "no_text" not in v.flags
        and not s.cited_work_on_topic
    ):
        outlier = baseline is not None and best_score is not None and baseline.z(best_score) <= OFF_TOPIC_Z
        better_fit = alternative is not None and alternative[1] >= OFF_TOPIC_GAP
        if outlier or better_fit or (baseline is None and alternative is None and v.escalated):
            modes.append("off_topic_citation")
            note = "The cited work does not appear to be about this topic. Possibly the wrong reference."
            if alternative is not None and alternative[1] >= SUGGEST_GAP:
                note += f" The claim matches {alternative[0]} in this paper's bibliography more closely."
            notes["off_topic_citation"] = note

    if not is_background and source_texts:
        # Correlation stated as causation.
        if s.claim_relation == "causal" and s.source_relation == "associational":
            causal = next((m for t in claim_texts if (m := CAUSAL_RE.search(t or ""))), None)
            assoc = ASSOCIATIONAL_RE.search(source_texts[0])
            if causal and assoc:
                modes.append("correlation_as_causation")
                notes["correlation_as_causation"] = (
                    f'The cited work reports an association ("{assoc.group(0)}"); '
                    f'the claim states a causal effect ("{causal.group(0)}").'
                )
                _downgrade(v, "correlation_as_causation")

        # Significance inflation: the source says the result is not significant.
        sig = next((m for t in claim_texts if (m := SIGNIFICANCE_CLAIM_RE.search(t or ""))), None)
        nonsig = NOT_SIGNIFICANT_RE.search(source_texts[0])
        if sig and nonsig and s.source_statistics == "not_significant":
            modes.append("significance_inflation")
            notes["significance_inflation"] = (
                f'The cited work reports this as not statistically significant ("{nonsig.group(0)}"); '
                f'the claim presents it as "{sig.group(0)}".'
            )
            _downgrade(v, "significance_inflation")
        elif (
            v.label == "overstated" and s.source_statistics == "not_tested"
            and any(re.search(r"statistically significant", t or "", re.I) for t in claim_texts)
        ):
            modes.append("significance_inflation")
            notes["significance_inflation"] = (
                "The claim says the result is statistically significant; the cited work reports it without a statistical test."
            )

    if not is_background:
        # Scope inflation.
        scopes_differ = bool(
            s.claim_scope and s.source_scope
            and _norm(s.claim_scope) != _norm(s.source_scope)
            and not _mostly_numbers(s.claim_scope)
        )
        # Only for findings and definitions: identification claims ("PIQA is a commonsense
        # benchmark") are filed under the citing paper's own categories.
        if (
            claim.claim_type in ("finding", "definition")
            and v.label in ("overstated", "contradicted") and s.scope_broader_than_source and scopes_differ
        ):
            modes.append("scope_inflation")
            notes["scope_inflation"] = f"Claim scope: {s.claim_scope}. Source scope: {s.source_scope}."
        # A sweeping qualifier ("without exception", "always") that the supporting passage does not share.
        elif (
            claim.claim_type in ("finding", "definition") and v.label in ("supported", "overstated") and source_texts
        ):
            sweep = next((m for t in stated_texts if (m := SWEEPING_RE.search(t or ""))), None)
            if sweep and not SWEEPING_RE.search(source_texts[0]):
                modes.append("scope_inflation")
                notes["scope_inflation"] = (
                    f'The claim says "{sweep.group(0)}"; the supporting passage makes no such general statement.')
                _downgrade(v, "sweeping_qualifier")

        # Number drift: judge-confirmed same quantity, numbers grounded in both texts. The claim
        # number must come from the citing sentence's own claim: the judge's attributed claim may
        # quote a value from a neighbouring sentence that the claim is only compared against.
        if source_texts:
            for cmp in s.numbers:
                if not cmp.same_quantity:
                    continue
                cn, sn = _first_number(cmp.claim_value), _first_number(cmp.source_value)
                if cn is None or sn is None:
                    continue
                if not _occurs(cn, [claim.claim_text]) or not _occurs(sn, source_texts):
                    continue
                if numbers_match(cn, sn) is False and not numbers_consistent(
                        cmp.claim_value, cmp.source_value, claim.claim_text, cn, sn):
                    modes.append("number_drift")
                    notes["number_drift"] = f"{cn.raw} in the claim; {sn.raw} in the cited work."
                    break
            if "number_drift" in modes:
                _downgrade(v, "number_drift")

        # Hypothesis stated as finding. A "supported" verdict is downgraded only when the verified
        # quote hedges with a modal and the citing sentence asserts the point as certain.
        if claim.claim_type == "finding" and v.label in ("overstated", "supported"):
            claim_hedged = s.claim_hedged or any(HEDGE_RE.search(t or "") for t in stated_texts)
            if v.label == "overstated":
                hedge = HEDGE_RE.search(v.quote or "") if v.quote_verified else None
                source_hedged = s.source_hedged or hedge
            else:
                hedge = MODAL_HEDGE_RE.search(v.quote or "") if v.quote_verified else None
                certain = any(CERTAINTY_RE.search(t or "") for t in stated_texts)
                source_hedged = hedge and certain
            if source_hedged and not claim_hedged:
                modes.append("hypothesis_as_finding")
                notes["hypothesis_as_finding"] = (
                    f'The cited work hedges this point ("{hedge.group(0)}"); the claim states it as fact.'
                    if hedge else "The cited work presents this tentatively; the claim states it as fact."
                )
                _downgrade(v, "hedge_removed")

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

    # An abstract leaves out most of a paper: "overstated" against an abstract alone stands only on a
    # concrete mismatch (a narrower scope, a different number, a hedge), not on a missing detail.
    if (ref is not None and ref.fulltext_status == "abstract_only" and v.label == "overstated"
            and not MISMATCH_MODES & set(modes)):
        v.label = "unverifiable"
        v.flags.append("abstract_only_lenient")

    v.failure_modes = list(dict.fromkeys(modes))
    v.failure_notes = notes
    return v


def _better_alternative(
    v: Verdict, claim: Claim, references: dict[str, Reference], abstract_scores: Optional[dict[str, dict[str, float]]]
) -> Optional[tuple[str, float]]:
    scores = (abstract_scores or {}).get(v.claim_id) or {}
    cited = scores.get(v.reference_key)
    others = [(score, key) for key, score in scores.items() if key not in claim.reference_keys]
    if cited is None or not others:
        return None
    score, key = max(others)
    ref = references.get(key)
    return (ref.label if ref and ref.label else key), score - cited


def best_passage_scores(evidence: list[Evidence]) -> dict[tuple[str, str], float]:
    return {(e.claim_id, e.reference_key): max(p.score for p in e.passages) for e in evidence if e.passages}


def apply_detectors(
    verdicts: list[Verdict],
    claims: dict[str, Claim],
    references: dict[str, Reference],
    evidence: Optional[list[Evidence]] = None,
    sidecar_loader: SidecarLoader = read_reference_sidecar,
    abstract_scores: Optional[dict[str, dict[str, float]]] = None,
    clauses: Optional[dict[tuple[str, str], str]] = None,  # (citing sentence id, reference key) -> clause
) -> list[Verdict]:
    """`abstract_scores[claim_id][ref_key]`: claim-to-abstract similarity for every reference
    (EvidenceRetriever.abstract_scores), used to spot citations of the wrong work."""
    cache: dict[str, dict] = {}

    def load(ref: Reference) -> dict:
        if ref.key not in cache:
            cache[ref.key] = sidecar_loader(ref)
        return cache[ref.key]

    best = best_passage_scores(evidence or [])
    # Baseline over full-text citations only; abstracts alone score systematically differently.
    full_scores = [
        score for (_, key), score in best.items()
        if references.get(key) is not None and references[key].fulltext_status == "full"
    ]
    baseline = PaperBaseline.from_scores(full_scores)
    return [
        detect(
            v, claims[v.claim_id], references.get(v.reference_key), load,
            best_score=best.get((v.claim_id, v.reference_key)),
            alternative=_better_alternative(v, claims[v.claim_id], references, abstract_scores),
            baseline=baseline if (ref := references.get(v.reference_key)) and ref.fulltext_status == "full" else None,
            clause=(clauses or {}).get((claims[v.claim_id].citing_sentence_id, v.reference_key)),
        )
        for v in verdicts
    ]


def failure_mode_counts(verdicts: list[Verdict]) -> dict[str, int]:
    return dict(Counter(m for v in verdicts for m in v.failure_modes).most_common())
