"""Planted-error edits keep every offset consistent; demo scoring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from app.demo import GroundTruth, PlantedError, score
from app.models import Claim, Verdict
from app.pipeline.ingest import parse_latex_project
from app.models import IngestResult

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("plant_errors", ROOT / "scripts" / "plant_errors.py")
plant = importlib.util.module_from_spec(spec)
sys.modules["plant_errors"] = plant  # dataclasses look the module up by name
spec.loader.exec_module(plant)

FIXTURE = Path(__file__).parent / "fixtures" / "latex_numeric"


def fixture() -> IngestResult:
    files = {p.relative_to(FIXTURE).as_posix(): p.read_text(encoding="utf-8") for p in FIXTURE.rglob("*") if p.is_file()}
    paper, sentences = parse_latex_project(files)
    return IngestResult(paper=paper, sentences=sentences)


def check_offsets(result: IngestResult) -> None:
    text = result.paper.text
    for section in result.paper.sections:
        assert text[section.start : section.end] == section.text
    for s in result.sentences:
        assert text[s.start : s.end] == s.text
        for a, b in s.marker_spans:
            assert s.start <= a < b <= s.end


def test_apply_edits_keeps_offsets_for_all_mutation_kinds():
    base = fixture()
    by_text = {s.text: s for s in base.sentences}
    s_number = by_text["The model reaches 12% higher accuracy than the baseline [3]."]
    s_scope = by_text["However, they are data hungry [2]."]
    s_swap = by_text["Deep networks dominate vision benchmarks [3, 1]."]
    text = base.paper.text
    num_at = text.index("12%", s_number.start)
    scope_at = s_scope.marker_spans[0][0]
    swap_a, swap_b = s_swap.marker_spans[0]
    edits = [
        plant.Edit(num_at, num_at + 3, "19.2%", s_number.id),
        plant.Edit(scope_at, scope_at, "in every setting and domain ", s_scope.id),
        plant.Edit(swap_a, swap_b, "[2]", s_swap.id, marker_index=0, new_keys=["lee2021"]),
    ]
    out = plant.apply_edits(base, edits)
    check_offsets(out)
    new = {s.id: s for s in out.sentences}
    assert new[s_number.id].text == "The model reaches 19.2% higher accuracy than the baseline [3]."
    assert new[s_scope.id].text == "However, they are data hungry in every setting and domain [2]."
    assert new[s_swap.id].text == "Deep networks dominate vision benchmarks [2]."
    assert new[s_swap.id].reference_keys == ["lee2021"] and new[s_swap.id].marker_keys == [["lee2021"]]
    marker = new[s_scope.id].marker_spans[0]
    assert out.paper.text[marker[0] : marker[1]] == "[2]"
    # Neighbours quoting an edited sentence as context see the new text.
    assert any("in every setting and domain" in s.context_before or "in every setting and domain" in s.context_after
               for s in out.sentences)
    check_offsets(base)  # the input is untouched


def test_bump_numbers():
    assert plant._bump("12%", 12.0, "%") == "19%"
    assert plant._bump("8.5K", 8.5, "k") == "13.6K"  # values are as written ("8.5" for "8.5K")
    assert plant._bump("175B", 175.0, "B") == "280B"
    assert plant._bump("80%", 80.0, "%") == "44%"  # stays a valid percentage
    assert plant._bump("1,000", 1000.0, "") == "1,600"
    assert plant._bump("10^-9", 1e-9, "") is None


def test_hedge_candidates():
    base = fixture()
    s = base.sentences[0].model_copy(update={"text": "X may help"})
    text = "Prior. X may help a lot [1]."
    s = s.model_copy(update={"start": 7, "end": len(text), "marker_spans": [(24, 27)], "marker_keys": [["k"]],
                             "text": text[7:], "reference_keys": ["k"]})
    [c] = plant.hedge_candidates(s, "k", 0, text)
    assert text[c.edit.start : c.edit.end] == "may" and c.edit.new == "will"


# ------------------------------------------------------------------------------ scoring


def _claim(cid, sid, key):
    return Claim(id=cid, citing_sentence_id=sid, claim_text="x", reference_keys=[key])


def _verdict(cid, key, label, modes=()):
    return Verdict(claim_id=cid, reference_key=key, label=label, confidence=0.9, failure_modes=list(modes))


def test_score():
    truth = GroundTruth(arxiv_id="x", title="t", planted=[
        PlantedError(id="p1", sentence_id="s1", reference_keys=["a"], mutation="number_change", description="",
                     original="", mutated="", expected_labels=["overstated", "contradicted"], expected_failure_modes=["number_drift"]),
        PlantedError(id="p2", sentence_id="s2", reference_keys=["wrong"], mutation="citation_swap", description="",
                     original="", mutated="", expected_labels=["not_found"], expected_failure_modes=["off_topic_citation"]),
        PlantedError(id="p3", sentence_id="s3", reference_keys=["c"], mutation="scope_inflation", description="",
                     original="", mutated="", expected_labels=["overstated"]),
    ], baseline_flags=[("s5", "e", "overstated")])
    claims = {c.id: c for c in [_claim("s1.c1", "s1", "a"), _claim("s2.c1", "s2", "wrong"), _claim("s3.c1", "s3", "c"),
                                _claim("s4.c1", "s4", "d"), _claim("s5.c1", "s5", "e"), _claim("s6.c1", "s6", "f")]}
    verdicts = [
        _verdict("s1.c1", "a", "contradicted", ["number_drift"]),     # caught, right label, right mode
        _verdict("s2.c1", "wrong", "unverifiable", ["off_topic_citation"]),  # caught via the tag, label off
        _verdict("s3.c1", "c", "supported"),                          # missed
        _verdict("s4.c1", "d", "not_found"),                          # new flag on an untouched citation
        _verdict("s5.c1", "e", "overstated"),                         # pre-existing flag
        _verdict("s6.c1", "f", "supported"),
    ]
    result = score(truth, verdicts, claims, annotations={})
    assert (result["planted"], result["caught"]) == (3, 2)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["label_accuracy"] == pytest.approx(1 / 3)
    assert (result["other_flags"], result["other_flags_preexisting"]) == (2, 1)
    assert result["precision"] == pytest.approx(2 / 4)
    assert result["precision_excluding_preexisting"] == pytest.approx(2 / 3)
    assert result["by_mutation"]["citation_swap"] == {"planted": 1, "caught": 1}
    assert result["summary_line"] == "Caught 2 of 3 planted errors · 2 other citations flagged for review"
    # With a human review: s5 is a real problem, s4 a false alarm.
    reviewed = score(truth, verdicts, claims, annotations={
        ("s5", "e"): {"verdict": "genuine", "note": ""}, ("s4", "d"): {"verdict": "false_alarm", "note": ""}})
    assert reviewed["review"] == {"genuine": 1, "false_alarm": 1, "debatable": 0, "unreviewed": 0}
    assert reviewed["precision_reviewed"] == pytest.approx((2 + 1) / (2 + 1 + 1))
    assert reviewed["summary_line"] == "Caught 2 of 3 planted errors · 1 false alarm · 1 real citation problem found in the original paper"
    unreviewed = score(truth, verdicts, claims, annotations={("s5", "e"): {"verdict": "debatable", "note": ""}})
    assert unreviewed["review"]["unreviewed"] == 1 and unreviewed["precision_reviewed"] == pytest.approx(2 / 3)
