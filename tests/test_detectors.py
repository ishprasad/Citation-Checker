"""Failure-mode detector tests (rule-based evidence on top of judge signals)."""

from __future__ import annotations

import pytest

from app.models import Claim, JudgeSignals, NumberComparison, Reference, Verdict
from app.pipeline.detectors import (
    apply_detectors,
    citation_markers,
    detect,
    failure_mode_counts,
    lookup_original,
)
from app.pipeline.textutil import extract_numbers, numbers_match

CLAIM = Claim(id="s1.c1", citing_sentence_id="s1", claim_text="The compound reduces tumor volume by 31% in humans",
              reference_keys=["r"], claim_type="finding")
REF = Reference(key="r", raw="")
NO_SIDECAR = lambda ref: {}  # noqa: E731


def make(label="overstated", quote=None, context=None, verified=True, **signals) -> Verdict:
    return Verdict(
        claim_id="s1.c1", reference_key="r", label=label, confidence=0.9,
        attributed_claim=CLAIM.claim_text, quote=quote, quote_verified=verified and quote is not None,
        quote_context=context or quote, signals=JudgeSignals(**signals),
    )


# --------------------------------------------------------------------------- numbers


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("34B", "33B", False),
        ("175B", "175 billion", True),
        ("31%", "30.9%", True),  # the claim rounds the source
        ("31%", "29%", False),
        ("12%", "0.12", True),  # fraction vs percentage
        ("2x", "2.1x", True),
        ("6.7B", "6B", False),
        ("28.4", "28.41", True),
        ("12%", "12x", None),  # not comparable
    ],
)
def test_numbers_match(a, b, expected):
    assert numbers_match(extract_numbers(a)[0], extract_numbers(b)[0]) is expected


def test_number_drift_with_grounded_numbers():
    v = make(quote="We train three model sizes (1.3B, 6B, and 175B parameters)",
             numbers=[NumberComparison(claim_value="6.7B", source_value="6B", same_quantity=True)])
    claim = CLAIM.model_copy(update={"claim_text": "InstructGPT has models of 1.3B, 6.7B and 175B parameters"})
    v = v.model_copy(update={"attributed_claim": claim.claim_text})
    out = detect(v, claim, REF, NO_SIDECAR)
    assert out.failure_modes == ["number_drift"]
    assert out.failure_notes["number_drift"] == "6.7B in the claim; 6B in the cited work."


def test_number_drift_requires_judge_same_quantity_and_grounding():
    quote = "Tumor volume fell by 21% in mice."
    different_quantity = make(quote=quote, numbers=[NumberComparison(claim_value="31%", source_value="21%", same_quantity=False)])
    assert "number_drift" not in detect(different_quantity, CLAIM, REF, NO_SIDECAR).failure_modes
    hallucinated_source = make(quote=quote, numbers=[NumberComparison(claim_value="31%", source_value="11%", same_quantity=True)])
    assert "number_drift" not in detect(hallucinated_source, CLAIM, REF, NO_SIDECAR).failure_modes
    unverified_quote = make(quote=quote, verified=False,
                            numbers=[NumberComparison(claim_value="31%", source_value="21%", same_quantity=True)])
    assert "number_drift" not in detect(unverified_quote, CLAIM, REF, NO_SIDECAR).failure_modes
    rounding = make(quote="Tumor volume fell by 30.9% in mice.",
                    numbers=[NumberComparison(claim_value="31%", source_value="30.9%", same_quantity=True)])
    assert "number_drift" not in detect(rounding, CLAIM, REF, NO_SIDECAR).failure_modes


def test_number_drift_downgrades_supported():
    v = make(label="supported", quote="Tumor volume fell by 21% in mice.",
             numbers=[NumberComparison(claim_value="31%", source_value="21%", same_quantity=True)])
    out = detect(v, CLAIM, REF, NO_SIDECAR)
    assert out.label == "overstated" and out.confidence == 0.7 and "label_adjusted_number_drift" in out.flags


# ----------------------------------------------------------------------- scope, hedging


def test_scope_inflation():
    v = make(quote="In a pilot study of 24 mice, tumor volume fell by 31%.", scope_broader_than_source=True,
             claim_scope="in humans", source_scope="pilot study of 24 mice")
    out = detect(v, CLAIM, REF, NO_SIDECAR)
    assert "scope_inflation" in out.failure_modes
    assert out.failure_notes["scope_inflation"] == "Claim scope: in humans. Source scope: pilot study of 24 mice."


@pytest.mark.parametrize(
    "label, signals",
    [
        ("supported", dict(scope_broader_than_source=True, claim_scope="in humans", source_scope="in mice")),
        ("overstated", dict(scope_broader_than_source=True, claim_scope=None, source_scope="in mice")),
        ("overstated", dict(scope_broader_than_source=True, claim_scope="in mice", source_scope="In mice.")),
        ("overstated", dict(scope_broader_than_source=True, claim_scope="350M, 1.3B, 6.7B", source_scope="1.3B, 6B")),
        ("overstated", dict(scope_broader_than_source=False, claim_scope="in humans", source_scope="in mice")),
    ],
)
def test_scope_inflation_needs_evidence(label, signals):
    assert "scope_inflation" not in detect(make(label=label, quote="x" * 10, **signals), CLAIM, REF, NO_SIDECAR).failure_modes


def test_hypothesis_as_finding():
    v = make(quote="Our results suggest that the compound may reduce tumor volume by 31%.")
    out = detect(v, CLAIM, REF, NO_SIDECAR)
    assert out.failure_modes == ["hypothesis_as_finding"]
    assert '("suggest")' in out.failure_notes["hypothesis_as_finding"]


def test_hypothesis_as_finding_not_when_claim_hedges_or_supported():
    hedged_claim = CLAIM.model_copy(update={"claim_text": "The compound may reduce tumor volume"})
    v = make(quote="the compound may reduce tumor volume").model_copy(update={"attributed_claim": hedged_claim.claim_text})
    assert detect(v, hedged_claim, REF, NO_SIDECAR).failure_modes == []
    supported = make(label="supported", quote="the compound may reduce tumor volume")
    assert detect(supported, CLAIM, REF, NO_SIDECAR).failure_modes == []
    background = CLAIM.model_copy(update={"claim_type": "background"})
    assert detect(make(quote="this may help"), background, REF, NO_SIDECAR).failure_modes == []


# ------------------------------------------------------------------- secondary citation


def test_citation_markers():
    assert citation_markers("as shown in [12] and [3, 5]") == ["[12]", "[3, 5]"]
    assert citation_markers("as shown by Smith et al. (2019) and (Lee and Kim, 2020)") == [
        "Smith et al. (2019)", "(Lee and Kim, 2020)"]
    assert citation_markers("We ran 2019 experiments in 12 settings.") == []


def test_lookup_original_from_latex_and_pdf_sidecars():
    latex = {"references": [{"label": "12", "title": "Original Paper", "raw": "..."},
                            {"label": "Smith et al., 2019", "title": "Smith's Paper", "raw": "..."}]}
    assert lookup_original("[12]", latex) == "Original Paper"
    assert lookup_original("Smith et al. (2019)", latex) == "Smith's Paper"
    pdf = {"references_text": "[11] Other. 2018.\n[12] A. Author. The Real Source. In ICML, 2017.\nJ. Smith. 2019. Old work."}
    assert lookup_original("[12]", pdf) == "A. Author. The Real Source. In ICML, 2017."
    assert lookup_original("(Smith et al., 2019)", pdf) == "J. Smith. 2019. Old work."
    assert lookup_original("[99]", pdf) is None


def test_secondary_citation_names_original_work():
    v = make(label="supported", quote="Tumor volume falls by 31% in humans [12].",
             support_attributed_to_other_work=True, other_work="[12]")
    out = detect(v, CLAIM, REF, lambda ref: {"references": [{"label": "12", "title": "Original Paper"}]})
    assert out.failure_modes == ["secondary_citation"]
    assert out.failure_notes["secondary_citation"] == 'The supporting passage credits [12]: "Original Paper".'


def test_secondary_citation_needs_judge_signal_and_visible_marker():
    passing_mention = make(label="supported", quote="We fine-tune GPT-3 (Brown et al., 2020) on 31% of humans.")
    assert detect(passing_mention, CLAIM, REF, NO_SIDECAR).failure_modes == []
    no_marker = make(label="supported", quote="Tumor volume falls by 31% in humans.",
                     support_attributed_to_other_work=True, other_work="[12]")
    assert detect(no_marker, CLAIM, REF, NO_SIDECAR).failure_modes == []


# --------------------------------------------------------------------------- retraction


def test_retracted_source_applies_even_to_supported_and_background():
    ref = REF.model_copy(update={"retracted": True, "retraction_notice": "10.1016/s0140-6736(10)60175-4"})
    for claim in (CLAIM, CLAIM.model_copy(update={"claim_type": "background"})):
        out = detect(make(label="supported", quote="x" * 10), claim, ref, NO_SIDECAR)
        assert out.failure_modes == ["retracted_source"]
        assert "10.1016/s0140-6736(10)60175-4" in out.failure_notes["retracted_source"]
    unknown = REF.model_copy(update={"retracted": None})
    assert detect(make(label="supported", quote="x" * 10), CLAIM, unknown, NO_SIDECAR).failure_modes == []


def test_apply_detectors_and_counts():
    verdicts = [make(quote="Our results suggest the compound may reduce tumor volume by 31%."),
                make(label="supported", quote="x" * 10)]
    out = apply_detectors(verdicts, {"s1.c1": CLAIM}, {"r": REF}, NO_SIDECAR)
    assert failure_mode_counts(out) == {"hypothesis_as_finding": 1}
    assert verdicts[0].failure_modes == []  # inputs are not mutated
