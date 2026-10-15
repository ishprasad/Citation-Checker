"""Failure-mode detector tests (rule-based evidence on top of judge signals)."""

from __future__ import annotations

import pytest

from app.models import (
    Claim,
    EditorialUpdate,
    Evidence,
    JudgeSignals,
    NumberComparison,
    Passage,
    Reference,
    Verdict,
)
from app.pipeline.references import crossref_retraction, crossref_updates
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
        ("18", "18.7", True),  # the claim truncates the source ("1-18 kpc" for 1.0-18.7)
        ("10", "10.8", True),
        ("19", "18.7", True),  # rounding
        ("17.5", "15.6", False),
        ("18", "17.2", False),  # truncation only shortens the source value
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


def test_number_drift_ignores_numbers_from_neighbouring_sentences():
    # "High-redshift galaxies have larger ratios than this [39]": the judge's attributed claim
    # quotes the 2-4 ratio of the previous sentence, which is only what [39] is compared against.
    claim = CLAIM.model_copy(update={"claim_text": "High-redshift galaxies observed by R1 have larger ratios than this"})
    v = make(label="supported", quote="the median size ratio is 10.8",
             numbers=[NumberComparison(claim_value="2-4", source_value="10.8", same_quantity=True)])
    v = v.model_copy(update={"attributed_claim": "Galaxies in [39] have ratios larger than the 2-4 found by [66]."})
    out = detect(v, claim, REF, NO_SIDECAR)
    assert "number_drift" not in out.failure_modes and out.label == "supported"


def test_number_drift_downgrades_supported():
    v = make(label="supported", quote="Tumor volume fell by 21% in mice.",
             numbers=[NumberComparison(claim_value="31%", source_value="21%", same_quantity=True)])
    out = detect(v, CLAIM, REF, NO_SIDECAR)
    assert out.label == "overstated" and out.confidence == 0.7 and "label_adjusted_number_drift" in out.flags


def _drift(claim_text, quote, claim_value, source_value):
    claim = CLAIM.model_copy(update={"claim_text": claim_text})
    v = make(label="supported", quote=quote,
             numbers=[NumberComparison(claim_value=claim_value, source_value=source_value, same_quantity=True)])
    return detect(v.model_copy(update={"attributed_claim": claim_text}), claim, REF, NO_SIDECAR)


@pytest.mark.parametrize("claim_text, quote, claim_value, source_value", [
    # The source value lies within the claim's range (co-cited works together span 3 to 6.6).
    ("Neither sees evolution between z 3 and z 6.6", "no significant evolution between redshifts 3 and 6", "6.6", "6"),
    # The claim value lies within the source's range, written with a spaced minus sign.
    ("2000 K sets the upper limit", "the sublimation temperature, Tsub ≈1800 −2000 K", "2000 K", "1800 −2000 K"),
    # A bound the source value satisfies.
    ("studies with at least 100 samples", "compiled data on ~125,000 exomes", "100", "~125,000"),
    ("equivalent widths higher than 0.1 Å", "averaging ≃ 1 Å in Lyα equivalent width", "0.1 Å", "≃ 1 Å"),
    # A superscript lost in PDF text: "10−17" is 10^-17.
    ("Brown obtained a theoretical value of ∼ 10^-17", "a lower bound on the threshold of ε ∼ 10−17", "∼ 10^-17", "∼ 10−17"),
])
def test_number_drift_not_for_consistent_values(claim_text, quote, claim_value, source_value):
    out = _drift(claim_text, quote, claim_value, source_value)
    assert "number_drift" not in out.failure_modes and out.label == "supported"


@pytest.mark.parametrize("claim_text, quote, claim_value, source_value", [
    ("The average multiplicity is 2.03±0.12", "each system has 1.27 ± 0.12 cold giants", "2.03±0.12", "1.27±0.12"),
    ("The fraction of stars with planets is ∼30±5%", "the fraction is 30 ± 3%", "5", "3"),  # unitless judge value
    ("proportions ranging from 35%-57%", "About 75% felt guilty", "35%-57%", "75%"),
    ("isolation mass is always ≳ 9–10 M⊕", "we calculate M crit ∼ 8 M ⊕ at 5 AU", "9", "8"),  # bound not met
    ("scale lengths between 1 and 18 kpc", "scale lengths between 1 and 7 kpc", "18", "1 and 7"),  # both ranges
])
def test_number_drift_for_conflicting_values(claim_text, quote, claim_value, source_value):
    assert "number_drift" in _drift(claim_text, quote, claim_value, source_value).failure_modes


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


def test_hedge_removed_downgrades_supported_only_against_a_certain_claim():
    claim = CLAIM.model_copy(update={"claim_text": "Stigma will fail to protect public health"})
    quote = "stigmas may be important drivers of failures in protecting public health"
    v = make(label="supported", quote=quote, source_hedged=True).model_copy(update={"attributed_claim": claim.claim_text})
    out = detect(v, claim, REF, NO_SIDECAR)
    assert out.label == "overstated" and out.failure_modes == ["hypothesis_as_finding"]
    assert "label_adjusted_hedge_removed" in out.flags
    # A reporting verb is routine phrasing, not a hedge; and a claim without a certainty marker stays.
    reporting = make(label="supported", quote="This study suggests that mental health deteriorated", source_hedged=True)
    assert detect(reporting.model_copy(update={"attributed_claim": claim.claim_text}), claim, REF, NO_SIDECAR).label == "supported"
    plain = CLAIM.model_copy(update={"claim_text": "Stigma fails to protect public health"})
    v = make(label="supported", quote=quote, source_hedged=True).model_copy(update={"attributed_claim": plain.claim_text})
    assert detect(v, plain, REF, NO_SIDECAR).label == "supported"
    # The citing sentence's own hedge counts even when extraction dropped it.
    v = make(label="supported", quote=quote, source_hedged=True).model_copy(update={"attributed_claim": claim.claim_text})
    assert detect(v, claim, REF, NO_SIDECAR, clause="Stigma will possibly fail to protect public health").label == "supported"


def test_sweeping_qualifier_absent_from_the_source():
    claim = CLAIM.model_copy(update={"claim_text": "The compound reduces tumor volume"})
    v = make(label="supported", quote="In mice, the compound reduced tumor volume by 31%.")
    out = detect(v.model_copy(update={"attributed_claim": claim.claim_text}), claim, REF, NO_SIDECAR,
                 clause="The compound reduces tumor volume without exception")
    assert out.label == "overstated" and out.failure_modes == ["scope_inflation"]
    assert '"without exception"' in out.failure_notes["scope_inflation"]
    # The source makes the same general statement: nothing to flag.
    v = make(label="supported", quote="The compound always reduced tumor volume, in all cases studied.")
    out = detect(v.model_copy(update={"attributed_claim": claim.claim_text}), claim, REF, NO_SIDECAR,
                 clause="The compound always reduces tumor volume")
    assert out.label == "supported" and out.failure_modes == []
    # Background and method claims are not checked.
    for kind in ("background", "method"):
        other = claim.model_copy(update={"claim_type": kind})
        v = make(label="supported", quote="In mice, the compound reduced tumor volume by 31%.")
        assert detect(v, other, REF, NO_SIDECAR, clause="It always works").label == "supported"


def test_overstated_against_an_abstract_needs_a_concrete_mismatch():
    abstract_ref = REF.model_copy(update={"fulltext_status": "abstract_only"})
    missing_detail = make(quote="We studied tumor growth in mice.")
    out = detect(missing_detail, CLAIM, abstract_ref, NO_SIDECAR)
    assert out.label == "unverifiable" and "abstract_only_lenient" in out.flags
    narrower = make(quote="In a pilot study of 24 mice, tumor volume fell by 31%.", scope_broader_than_source=True,
                    claim_scope="in humans", source_scope="pilot study of 24 mice")
    assert detect(narrower, CLAIM, abstract_ref, NO_SIDECAR).label == "overstated"
    assert detect(missing_detail, CLAIM, REF, NO_SIDECAR).label == "overstated"  # full text: unchanged


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
    out = apply_detectors(verdicts, {"s1.c1": CLAIM}, {"r": REF}, sidecar_loader=NO_SIDECAR)
    assert failure_mode_counts(out) == {"hypothesis_as_finding": 1}
    assert verdicts[0].failure_modes == []  # inputs are not mutated


# ------------------------------------------------------------- corrections and concerns


def test_corrections_and_expressions_of_concern():
    ref = REF.model_copy(update={"updates": [
        EditorialUpdate(type="correction", doi="10.1/corr", source="retraction-watch", date="2004-03-06"),
        EditorialUpdate(type="expression_of_concern", doi="10.1/eoc", source="publisher", date="2026-05-01"),
        EditorialUpdate(type="addendum", doi="10.1/add"),  # not a problem
    ]})
    out = detect(make(label="supported", quote="x" * 10), CLAIM.model_copy(update={"claim_type": "background"}),
                 ref, NO_SIDECAR)
    assert out.failure_modes == ["corrected_source", "expression_of_concern"]
    assert out.failure_notes["corrected_source"] == "A correction was published for the cited work (doi:10.1/corr, 2004-03-06)."
    assert "doi:10.1/eoc" in out.failure_notes["expression_of_concern"]


def test_crossref_updates_parsing():
    item = {"updated-by": [
        {"DOI": "10.1016/S0140-6736(04)15715-2", "type": "correction", "source": "retraction-watch",
         "updated": {"date-time": "2004-03-06T00:00:00Z"}},
        {"DOI": "10.1/x", "type": "withdrawal", "source": "publisher"},
        {"DOI": "10.1/y", "type": "expression_of_concern", "source": "publisher"},
    ]}
    updates = crossref_updates(item)
    assert [(u.type, u.date) for u in updates] == [
        ("correction", "2004-03-06"), ("withdrawal", None), ("expression_of_concern", None)]
    assert crossref_retraction(item) == (True, "10.1/x")  # withdrawals count as retractions


# ---------------------------------------------------------------- off-topic citation


def _evidence(claim_id: str, key: str, score: float) -> Evidence:
    return Evidence(claim_id=claim_id, reference_key=key, passages=[Passage(text="t", chunk_index=0, score=score)])


def _paper(n: int, odd_score: float, odd_signal_on_topic: bool, odd_label="not_found"):
    claims, refs, verdicts, evidence = {}, {}, [], []
    for i in range(n):
        cid, key = f"s{i}.c1", f"r{i}"
        claims[cid] = Claim(id=cid, citing_sentence_id=f"s{i}", claim_text="X improves Y", reference_keys=[key])
        refs[key] = Reference(key=key, raw="", fulltext_status="full")
        score = odd_score if i == 0 else 0.60 + 0.01 * (i % 5)
        label = odd_label if i == 0 else "supported"
        verdicts.append(Verdict(claim_id=cid, reference_key=key, label=label, confidence=0.9,
                                signals=JudgeSignals(cited_work_on_topic=odd_signal_on_topic if i == 0 else True)))
        evidence.append(_evidence(cid, key, score))
    return verdicts, claims, refs, evidence


def test_off_topic_citation_needs_judge_and_outlier_similarity():
    out = apply_detectors(*_paper(10, 0.30, False), sidecar_loader=NO_SIDECAR)
    assert out[0].failure_modes == ["off_topic_citation"] and "wrong reference" in out[0].failure_notes["off_topic_citation"]
    assert all(v.failure_modes == [] for v in out[1:])
    # Judge thinks it is on topic: no tag even with a weak match.
    assert apply_detectors(*_paper(10, 0.30, True), sidecar_loader=NO_SIDECAR)[0].failure_modes == []
    # Judge says off topic but similarity is typical: no tag.
    assert apply_detectors(*_paper(10, 0.61, False), sidecar_loader=NO_SIDECAR)[0].failure_modes == []
    # Supported verdicts are never off topic.
    assert apply_detectors(*_paper(10, 0.30, False, "supported"), sidecar_loader=NO_SIDECAR)[0].failure_modes == []


def test_off_topic_from_better_fitting_reference_names_it():
    verdicts, claims, refs, evidence = _paper(3, 0.60, False)  # too few pairs for a baseline
    refs["r1"] = refs["r1"].model_copy(update={"label": "Cobbe et al., 2021"})
    scores = {"s0.c1": {"r0": 0.45, "r1": 0.70, "r2": 0.50}}
    out = apply_detectors(verdicts, claims, refs, evidence, NO_SIDECAR, abstract_scores=scores)
    assert out[0].failure_modes == ["off_topic_citation"]
    assert "matches Cobbe et al., 2021 in this paper's bibliography more closely" in out[0].failure_notes["off_topic_citation"]
    # A small gap is not evidence enough (an on-topic citation can trail a closely related paper).
    scores = {"s0.c1": {"r0": 0.567, "r1": 0.697}}
    assert apply_detectors(verdicts, claims, refs, evidence, NO_SIDECAR, abstract_scores=scores)[0].failure_modes == []


def test_off_topic_without_baseline_requires_escalation():
    verdicts, claims, refs, evidence = _paper(3, 0.30, False)
    assert apply_detectors(verdicts, claims, refs, evidence, NO_SIDECAR)[0].failure_modes == []
    verdicts[0] = verdicts[0].model_copy(update={"escalated": True})
    assert apply_detectors(verdicts, claims, refs, evidence, NO_SIDECAR)[0].failure_modes == ["off_topic_citation"]


# -------------------------------------------------------- correlation as causation


CAUSAL_CLAIM = Claim(id="s1.c1", citing_sentence_id="s1", claim_text="Coffee reduces the risk of liver disease",
                     reference_keys=["r"], claim_type="finding")


def _causal(label="overstated", quote="Coffee consumption was associated with a lower risk of liver disease.", **sig):
    signals = dict(claim_relation="causal", source_relation="associational") | sig
    return make(label=label, quote=quote, **signals).model_copy(update={"attributed_claim": CAUSAL_CLAIM.claim_text})


def test_correlation_as_causation():
    out = detect(_causal(), CAUSAL_CLAIM, REF, NO_SIDECAR)
    assert out.failure_modes == ["correlation_as_causation"]
    assert out.failure_notes["correlation_as_causation"] == (
        'The cited work reports an association ("associated"); the claim states a causal effect ("reduces").')


def test_correlation_as_causation_downgrades_supported():
    out = detect(_causal(label="supported"), CAUSAL_CLAIM, REF, NO_SIDECAR)
    assert out.label == "overstated" and "label_adjusted_correlation_as_causation" in out.flags


@pytest.mark.parametrize(
    "verdict",
    [
        _causal(source_relation="causal"),  # judge: source is causal too
        _causal(quote="In a randomized trial, coffee lowered the risk of liver disease."),  # no associational wording
        _causal(claim_relation="associational"),
    ],
)
def test_correlation_as_causation_needs_both_signal_and_wording(verdict):
    assert "correlation_as_causation" not in detect(verdict, CAUSAL_CLAIM, REF, NO_SIDECAR).failure_modes


# ------------------------------------------------------------ significance inflation

SIG_CLAIM = Claim(id="s1.c1", citing_sentence_id="s1", claim_text="The drug significantly lowers blood pressure",
                  reference_keys=["r"], claim_type="finding")


def _sig(label="supported", quote="Blood pressure fell slightly, but the difference was not significant (p = 0.12).", **sig):
    signals = dict(source_statistics="not_significant") | sig
    return make(label=label, quote=quote, **signals).model_copy(update={"attributed_claim": SIG_CLAIM.claim_text})


def test_significance_inflation_from_explicit_non_significance():
    out = detect(_sig(), SIG_CLAIM, REF, NO_SIDECAR)
    assert out.failure_modes == ["significance_inflation"]
    assert out.label == "overstated" and "label_adjusted_significance_inflation" in out.flags
    assert '"not significant"' in out.failure_notes["significance_inflation"]


@pytest.mark.parametrize(
    "verdict, claim",
    [
        (_sig(source_statistics="significant"), SIG_CLAIM),  # judge disagrees
        (_sig(quote="Blood pressure fell by 8 mmHg in the treatment group."), SIG_CLAIM),  # no non-significance text
        (_sig(), SIG_CLAIM.model_copy(update={"claim_text": "The drug lowers blood pressure"})),  # claim makes no significance claim
    ],
)
def test_significance_inflation_needs_evidence(verdict, claim):
    v = verdict.model_copy(update={"attributed_claim": claim.claim_text})
    assert "significance_inflation" not in detect(v, claim, REF, NO_SIDECAR).failure_modes


def test_significance_inflation_untested_needs_explicit_claim_and_overstated():
    claim = SIG_CLAIM.model_copy(update={"claim_text": "The drug lowers blood pressure, a statistically significant effect"})
    v = _sig(label="overstated", quote="Blood pressure fell by 8 mmHg.", source_statistics="not_tested")
    v = v.model_copy(update={"attributed_claim": claim.claim_text})
    assert detect(v, claim, REF, NO_SIDECAR).failure_modes == ["significance_inflation"]
    ml_claim = SIG_CLAIM.model_copy(update={"claim_text": "The method significantly improves accuracy"})
    v = v.model_copy(update={"attributed_claim": ml_claim.claim_text})
    assert detect(v, ml_claim, REF, NO_SIDECAR).failure_modes == []  # "significantly" alone is common wording
