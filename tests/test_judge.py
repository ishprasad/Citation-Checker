"""Judge and escalation tests: quote verification, hard rules, escalation rules (fake LLM)."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

import app.pipeline.judge as judge_mod
from app import llm
from app.config import get_settings
from app.models import CitingSentence, Claim, Evidence, JudgeSignals, Paper, Passage, Reference, Verdict
from app.pipeline.judge import (
    JudgeContext,
    JudgeOutput,
    apply_rules,
    escalation_stats,
    escalation_text,
    judge_all,
    needs_escalation,
    neutral_tone,
)
from app.pipeline.textutil import locate_quote

SOURCE = (
    "We evaluate on 24 mice. Our results suggest that the compound may reduce tumor volume by 31% "
    "in this pilot study [12]. Further work is needed before human trials."
)

# --------------------------------------------------------------------- quote verification


@pytest.mark.parametrize(
    "quote",
    [
        "the compound may reduce tumor volume by 31% in this pilot study",
        "“The compound may reduce tumor volume by 31% in this pilot study.”",  # curly quotes, period
        "Our results suggest that ... tumor volume by 31%",  # elided
        "may reduce tumor volume by 31% in this pilot study.",  # citation marker dropped
        "THE COMPOUND MAY REDUCE   TUMOR\nVOLUME",  # case and whitespace
    ],
)
def test_locate_quote_accepts_verbatim_variants(quote):
    span = locate_quote(quote, SOURCE)
    assert span is not None
    assert "tumor volume" in SOURCE[span[0] : span[1]]


@pytest.mark.parametrize(
    "quote",
    [
        "the compound reduces tumor volume by 31% in humans",  # paraphrase
        "the compound may reduce tumor volume by 13%",  # changed number
        "Further work ... tumor volume",  # elided pieces out of order
        "mice",  # too short to mean anything
        "",
        None,
    ],
)
def test_locate_quote_rejects_non_verbatim(quote):
    assert locate_quote(quote, SOURCE) is None


def test_locate_quote_handles_pdf_artifacts():
    pdf = "The model achieves state‑of‑the‑art re-\nsults on ﬁve bench- marks."
    span = locate_quote("achieves state-of-the-art results on five benchmarks", pdf)
    assert span and pdf[span[0] : span[1]].startswith("achieves") and pdf[span[0] : span[1]].endswith("marks")


# ----------------------------------------------------------------------------- hard rules

FULL_REF = Reference(key="r1", raw="", title="Tumors", fulltext_status="full", abstract="We study mice.")
ABSTRACT_REF = Reference(key="r2", raw="", title="Abs", fulltext_status="abstract_only", abstract="We study mice.")
FINDING = Claim(id="s1.c1", citing_sentence_id="s1", claim_text="The compound reduces tumor volume by 31%",
                reference_keys=["r1"], claim_type="finding")
BACKGROUND = FINDING.model_copy(update={"id": "s1.c2", "claim_type": "background"})


def out(label, confidence=0.9, quote=None, explanation="The cited work reports this for mice.") -> JudgeOutput:
    return JudgeOutput(attributed_claim="X", label=label, confidence=confidence, quote=quote, explanation=explanation)


def test_verified_quote_gets_source_text_and_context():
    v = apply_rules(out("overstated", quote="the compound may reduce tumor volume by 31% in this pilot study"),
                    FINDING, FULL_REF, "super", [SOURCE])
    assert v.quote_verified and "quote_unverified" not in v.flags
    assert v.confidence == 0.9
    assert v.quote_context[v.quote_start : v.quote_end] == v.quote
    assert v.quote in SOURCE  # exact characters from the source, not the model's copy


def test_unverified_quote_halves_confidence_and_flags():
    v = apply_rules(out("supported", 0.9, quote="the compound cures cancer"), FINDING, FULL_REF, "super", [SOURCE])
    assert not v.quote_verified and v.flags == ["quote_unverified"]
    assert v.confidence == pytest.approx(0.45)
    missing = apply_rules(out("contradicted", 0.8, quote=None), FINDING, FULL_REF, "super", [SOURCE])
    assert missing.flags == ["quote_unverified"] and missing.confidence == pytest.approx(0.4)


def test_quote_found_in_abstract():
    v = apply_rules(out("supported", quote="We study mice"), FINDING, FULL_REF, "super", [SOURCE, "We study mice."])
    assert v.quote_verified and v.quote_context == "We study mice."


def test_background_claims_cannot_be_flagged():
    for label in ("overstated", "not_found", "contradicted"):
        v = apply_rules(out(label, quote="We evaluate on 24 mice"), BACKGROUND, FULL_REF, "super", [SOURCE])
        assert v.label == "unverifiable" and "background_lenient" in v.flags
    assert apply_rules(out("supported", quote="We evaluate on 24 mice"), BACKGROUND, FULL_REF, "s", [SOURCE]).label == "supported"


def test_abstract_only_not_found_becomes_unverifiable():
    v = apply_rules(out("not_found"), FINDING, ABSTRACT_REF, "super", ["We study mice."])
    assert v.label == "unverifiable" and v.flags == ["abstract_only"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The claim is wrong; this is an error.", "The claim is not supported by the cited text; this is a discrepancy."),
        ("Two errors and a wrong number.", "Two discrepancies and an inaccurate number."),
        ("It was erroneously cited.", "It was inaccurately cited."),
        ("This suggests misconduct.", "This suggests an issue."),
        ("The cited work reports 31% in mice.", "The cited work reports 31% in mice."),
    ],
)
def test_neutral_tone(text, expected):
    assert neutral_tone(text) == expected


# ------------------------------------------------------------------------ escalation rules


def verdict(label, confidence=0.9, flags=()):
    return Verdict(claim_id="c", reference_key="r", label=label, confidence=confidence, flags=list(flags))


@pytest.mark.parametrize(
    "v, ref, claim, expected",
    [
        (verdict("supported", 0.9), FULL_REF, FINDING, False),
        (verdict("supported", 0.55), FULL_REF, FINDING, True),  # low confidence
        (verdict("overstated", 0.8), FULL_REF, FINDING, False),
        (verdict("not_found", 0.95), FULL_REF, FINDING, True),  # accusation
        (verdict("contradicted", 0.95), FULL_REF, FINDING, True),  # accusation
        (verdict("supported", 0.9, ["quote_unverified"]), FULL_REF, FINDING, True),
        (verdict("unverifiable", 0.9), FULL_REF, FINDING, True),  # full text exists: retrieval miss
        (verdict("not_found", 0.9), FULL_REF, BACKGROUND, False),  # background never escalates
        (verdict("contradicted", 0.9), ABSTRACT_REF, FINDING, True),  # accusation from an abstract
        (verdict("unverifiable", 0.3), ABSTRACT_REF, FINDING, False),  # nothing more to read
        (verdict("unverifiable", 1.0, ["no_text"]), Reference(key="u", raw=""), FINDING, False),
        (verdict("unverifiable", 0.0, ["judge_error"]), FULL_REF, FINDING, True),
    ],
)
def test_needs_escalation(v, ref, claim, expected):
    assert needs_escalation(v, claim, ref) is expected


def test_escalated_verdicts_are_not_escalated_again():
    v = verdict("not_found", 0.9).model_copy(update={"escalated": True})
    assert not needs_escalation(v, FINDING, FULL_REF)


def test_escalation_text_truncates_but_keeps_retrieved_passages(tmp_path):
    path = tmp_path / "ref.txt"
    path.write_text("HEAD " + "x" * 5000 + " NEEDLE IN THE MIDDLE " + "y" * 5000, encoding="utf-8")
    ref = FULL_REF.model_copy(update={"text_path": str(path), "abstract": None})
    ev = Evidence(claim_id="c", reference_key="r1",
                  passages=[Passage(text="NEEDLE IN THE MIDDLE", chunk_index=3, score=0.9)])
    text = escalation_text(ref, ev, max_chars=2000)
    assert len(text) <= 2000 and text.startswith("HEAD") and text.endswith("NEEDLE IN THE MIDDLE")
    assert escalation_text(ref, ev, max_chars=100_000).count("NEEDLE") == 1


# ---------------------------------------------------------------------- judge_all (fake LLM)


@pytest.fixture
def setup(monkeypatch, tmp_path):
    settings = replace(get_settings(), models={**get_settings().models, "judge": "super", "escalate": "ultra"})
    monkeypatch.setattr(judge_mod, "get_settings", lambda: settings)
    path = tmp_path / "ref.txt"
    path.write_text(SOURCE, encoding="utf-8")
    full = FULL_REF.model_copy(update={"text_path": str(path)})
    paper_text = "Prior work. The compound reduces tumor volume by 31% [1]. More."
    start = paper_text.index("The compound")
    end = paper_text.index("[1].") + 4
    marker = paper_text.index("[1]")
    sentence = CitingSentence(id="s1", text=paper_text[start:end], start=start, end=end, reference_keys=["r1"],
                              marker_spans=[(marker, marker + 3)], marker_keys=[["r1"]])
    ctx = JudgeContext(paper=Paper(id="p", source="arxiv", title="Paper", text=paper_text),
                       sentences={"s1": sentence},
                       references={"r1": full, "u": Reference(key="u", raw="", status_reason="Not open.")})
    evidence = [Evidence(claim_id="s1.c1", reference_key="r1", passages=[Passage(text=SOURCE, chunk_index=0, score=0.8)])]
    return ctx, evidence


def _fake(model_outputs: list):
    calls = []

    async def fake(messages, schema):
        calls.append(messages)
        item = model_outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    fake.calls = calls
    return fake


def test_judge_all_escalates_and_records_original_label(monkeypatch, setup):
    ctx, evidence = setup
    judge = _fake([out("not_found", 0.9)])
    escalate = _fake([out("overstated", 0.85, quote="may reduce tumor volume by 31% in this pilot study")])
    monkeypatch.setattr(llm, "judge", judge)
    monkeypatch.setattr(llm, "escalate", escalate)
    judged, final = [], []
    [v] = asyncio.run(judge_all([FINDING], evidence, ctx, on_judged=judged.append, on_final=final.append))

    assert v.label == "overstated" and v.escalated and v.original_label == "not_found"
    assert v.original_confidence == 0.9 and v.model_used == "ultra" and v.quote_verified
    assert [x.label for x in judged] == ["not_found"] and final == [v]
    # The escalation prompt carries the full text and the first verdict.
    user = escalate.calls[0][1]["content"]
    assert "Full text of the cited work" in user and "Further work is needed" in user and "not_found" in user
    # The judge prompt marks the reference and lists passages.
    judge_user = judge.calls[0][1]["content"]
    assert "Citing sentence: The compound reduces tumor volume by 31% [R1]." in judge_user and "[P0]" in judge_user
    stats = escalation_stats([v])
    assert (stats.escalated, stats.label_changed, stats.transitions) == (1, 1, {"not_found->overstated": 1})


def test_confident_supported_verdict_is_not_escalated(monkeypatch, setup):
    ctx, evidence = setup
    monkeypatch.setattr(llm, "judge", _fake([out("supported", 0.9, quote="tumor volume by 31%")]))
    escalate = _fake([])
    monkeypatch.setattr(llm, "escalate", escalate)
    [v] = asyncio.run(judge_all([FINDING], evidence, ctx))
    assert v.label == "supported" and not v.escalated and escalate.calls == []


def test_reference_without_text_skips_the_model(monkeypatch, setup):
    ctx, _ = setup
    judge = _fake([])
    monkeypatch.setattr(llm, "judge", judge)
    claim = FINDING.model_copy(update={"reference_keys": ["u"]})
    [v] = asyncio.run(judge_all([claim], [], ctx))
    assert v.label == "unverifiable" and v.explanation == "Not open." and v.flags == ["no_text"]
    assert judge.calls == []


def test_failures_never_raise(monkeypatch, setup):
    ctx, evidence = setup
    monkeypatch.setattr(llm, "judge", _fake([llm.StructuredOutputError("bad")]))
    monkeypatch.setattr(llm, "escalate", _fake([llm.StructuredOutputError("bad again")]))
    [v] = asyncio.run(judge_all([FINDING], evidence, ctx))
    assert v.label == "unverifiable" and v.flags == ["judge_error", "escalation_failed"]


def test_token_budget_propagates(monkeypatch, setup):
    ctx, evidence = setup
    monkeypatch.setattr(llm, "judge", _fake([llm.TokenBudgetExceeded("budget")]))
    with pytest.raises(llm.TokenBudgetExceeded):
        asyncio.run(judge_all([FINDING], evidence, ctx))


def test_signals_are_kept():
    o = out("overstated", quote="may reduce tumor volume by 31% in this pilot study")
    o.signals = JudgeSignals(scope_broader_than_source=True, source_hedged=True, claim_scope="in general",
                             source_scope="pilot study of 24 mice")
    v = apply_rules(o, FINDING, FULL_REF, "super", [SOURCE])
    assert v.signals.scope_broader_than_source and v.signals.source_scope == "pilot study of 24 mice"


def test_reference_ids_are_replaced_with_citation_labels(monkeypatch, setup):
    ctx, evidence = setup
    ctx.references["r1"] = ctx.references["r1"].model_copy(update={"label": "1"})
    monkeypatch.setattr(llm, "judge", _fake([out(
        "supported", quote="tumor volume by 31%",
        explanation="The marker [R1] is attached to the compound; R1 reports this.")]))
    [v] = asyncio.run(judge_all([FINDING], evidence, ctx))
    assert v.explanation == "The marker [1] is attached to the compound; [1] reports this."
