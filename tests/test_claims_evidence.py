"""Claim extraction and evidence retrieval tests (fake LLM and embeddings, no network)."""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import replace

import numpy as np
import pytest

import app.pipeline.evidence as evidence_mod
from app import llm
from app.config import get_settings
from app.models import CitingSentence, Claim, Paper, Reference
from app.pipeline.claims import (
    ExtractionOutput,
    extract_claims,
    extract_sentence_claims,
    grounding,
    marker_phrase,
    render_sentence,
    strip_markers,
)
from app.pipeline.evidence import CHUNK_WORDS, EvidenceRetriever, chunk_text, query_text
from app.pipeline.textutil import extract_numbers, find_normalized, normalize_for_match, number_strings

# ------------------------------------------------------------------------------ numbers


@pytest.mark.parametrize(
    "text, expected",
    [
        ("improves accuracy by 12% and is 3.5x faster", [("12%", 12.0, "%"), ("3.5x", 3.5, "x")]),
        ("1,000 samples and 175B parameters", [("1,000", 1000.0, ""), ("175B", 175.0, "B")]),
        ("epsilon 10^-9, lr 2×10^-4", [("10^-9", 1e-9, ""), ("2×10^-4", 2e-4, "")]),
        ("a 2-fold increase; 28.4 BLEU; 16K tokens", [("2-fold", 2.0, "x"), ("28.4", 28.4, ""), ("16K", 16.0, "k")]),
        ("a drop of -3.5 points", [("-3.5", -3.5, "")]),
        ("12 percent", [("12 percent", 12.0, "%")]),
        # PDF text without superscripts: a minus sign right after 10 is an exponent; dashes are ranges.
        ("a threshold of 10−17 and 2×10−5 s", [("10^-17", 1e-17, ""), ("2×10^-5", 2e-5, "")]),
        ("aged 10–17 years", [("10", 10.0, ""), ("17", 17.0, "")]),
    ],
)
def test_extract_numbers(text, expected):
    got = [(n.raw, n.value, n.unit) for n in extract_numbers(text)]
    assert [(r, u) for r, _, u in got] == [(r, u) for r, _, u in expected]
    assert [v for _, v, _ in got] == pytest.approx([v for _, v, _ in expected])


def test_numbers_inside_identifiers_are_ignored():
    assert number_strings("GPT-3 and β1 and v2 and BERT-base") == []


def test_same_quantity():
    a, b, c = extract_numbers("12% 12.0% 12")
    assert a.same_quantity(b) and not a.same_quantity(c)


def test_normalized_quote_matching():
    source = "The model achieves state‑of‑the‑art results — “surprisingly” — on all bench-\nmarks."
    assert find_normalized('achieves state-of-the-art results - "surprisingly" - on all benchmarks', source) is not None
    assert find_normalized("achieves poor results", source) is None
    assert normalize_for_match("eﬀect  of\nX") == "effect of x"


# ------------------------------------------------------------------- sentence rendering

PAPER_TEXT = (
    "Prior work is broad. Our model is fast. X improves accuracy by 12% [3] and generalizes to clinical data [5, 7]. "
    "Brown et al. (2020) introduced few-shot prompting. It works well."
)


def _sentence(text: str, markers: list[tuple[str, list[str]]], sid: str = "s1") -> CitingSentence:
    start = PAPER_TEXT.index(text)
    spans, keys = [], []
    for marker, mkeys in markers:
        a = PAPER_TEXT.index(marker, start)
        spans.append((a, a + len(marker)))
        keys.append(mkeys)
    all_keys = list(dict.fromkeys(k for ks in keys for k in ks))
    return CitingSentence(
        id=sid, text=text, start=start, end=start + len(text), reference_keys=all_keys,
        marker_spans=spans, marker_keys=keys, context_before="Our model is fast.",
    )


S1 = _sentence("X improves accuracy by 12% [3] and generalizes to clinical data [5, 7].", [("[3]", ["c3"]), ("[5, 7]", ["c5", "c7"])])
S2 = _sentence("Brown et al. (2020) introduced few-shot prompting.", [("Brown et al. (2020)", ["brown"])], "s2")
PAPER = Paper(id="p", title="Our Paper", source="arxiv", text=PAPER_TEXT)


def test_render_sentence_numeric():
    text, ids = render_sentence(S1, PAPER_TEXT)
    assert text == "X improves accuracy by 12% [R1] and generalizes to clinical data [R2, R3]."
    assert ids == {"R1": "c3", "R2": "c5", "R3": "c7"}


def test_render_sentence_textual_keeps_author_names():
    text, ids = render_sentence(S2, PAPER_TEXT)
    assert text == "Brown et al. [R1] introduced few-shot prompting."
    assert ids == {"R1": "brown"}


def test_strip_markers_and_marker_phrase():
    assert strip_markers(S1, PAPER_TEXT) == "X improves accuracy by 12% and generalizes to clinical data."
    assert marker_phrase(S1, PAPER_TEXT, 0) == "X improves accuracy by 12%"
    assert marker_phrase(S1, PAPER_TEXT, 1) == "generalizes to clinical data"


def test_grounding():
    assert grounding("X improves accuracy by 12%", S1.text) == 1.0
    assert grounding("Grouped-query attention is an attention method described in the cited work",
                     "We use grouped-query attention [R1].") == 1.0
    assert grounding("Training generalized multi-query transformer checkpoints", "We use GQA [R1].") == 0.0


# ------------------------------------------------------------------- extraction (fake LLM)


def _fake_extract(payload: dict):
    async def fake(messages, schema):
        assert schema is ExtractionOutput
        fake.messages = messages
        return ExtractionOutput.model_validate(payload)

    return fake


def test_extraction_maps_ids_to_keys(monkeypatch):
    fake = _fake_extract({"citations": [
        {"reference_ids": ["R1"], "claims": [
            {"claim_text": "X improves accuracy by 12% [R1]", "claim_type": "finding", "numbers": ["12"], "scope": None}]},
        {"reference_ids": ["[r2]", "R3", "R9"], "claims": [
            {"claim_text": "X generalizes to clinical data", "claim_type": "finding", "scope": "clinical data"}]},
    ]})
    monkeypatch.setattr(llm, "extract", fake)
    claims = asyncio.run(extract_sentence_claims(S1, PAPER, {}))

    assert [(c.id, c.reference_keys) for c in claims] == [("s1.c1", ["c3"]), ("s1.c2", ["c5", "c7"])]
    assert claims[0].claim_text == "X improves accuracy by 12%"  # stray id removed
    assert claims[0].numbers == ["12%"]  # numbers re-extracted from the text, not trusted from the model
    assert claims[1].scope == "clinical data"
    user = fake.messages[1]["content"]
    assert "Citing sentence: X improves accuracy by 12% [R1] and generalizes to clinical data [R2, R3]." in user
    assert "Our Paper" in user and "Context before: Our model is fast." in user


def test_ungrounded_claims_dropped_and_uncovered_refs_get_background(monkeypatch):
    monkeypatch.setattr(llm, "extract", _fake_extract({"citations": [
        {"reference_ids": ["R1"], "claims": [
            {"claim_text": "X improves accuracy by 12%", "claim_type": "finding"}]},
        {"reference_ids": ["R2", "R3"], "claims": [
            {"claim_text": "Transformers dominate protein folding benchmarks", "claim_type": "finding"}]},
    ]}))
    claims = asyncio.run(extract_sentence_claims(S1, PAPER, {}))
    assert [c.claim_text for c in claims] == [
        "X improves accuracy by 12%",
        "Cited in connection with: generalizes to clinical data",
    ]
    assert claims[1].claim_type == "background" and claims[1].fallback
    assert claims[1].reference_keys == ["c5", "c7"]


def test_extraction_failure_falls_back_to_background(monkeypatch):
    async def failing(messages, schema):
        raise llm.StructuredOutputError("bad json")

    monkeypatch.setattr(llm, "extract", failing)
    claims = asyncio.run(extract_sentence_claims(S1, PAPER, {}))
    assert [(c.claim_type, c.fallback, c.reference_keys) for c in claims] == [
        ("background", True, ["c3"]),
        ("background", True, ["c5", "c7"]),
    ]


def test_token_budget_propagates(monkeypatch):
    async def over_budget(messages, schema):
        raise llm.TokenBudgetExceeded("budget")

    monkeypatch.setattr(llm, "extract", over_budget)
    with pytest.raises(llm.TokenBudgetExceeded):
        asyncio.run(extract_sentence_claims(S1, PAPER, {}))


def test_extract_claims_runs_all_sentences(monkeypatch):
    monkeypatch.setattr(llm, "extract", _fake_extract({"citations": [
        {"reference_ids": ["R1"], "claims": [{"claim_text": "few-shot prompting was introduced", "claim_type": "method"}]},
    ]}))
    done = []
    claims = asyncio.run(extract_claims([S2], PAPER, {}, on_done=lambda s, cs: done.append((s.id, len(cs)))))
    assert done == [("s2", 1)] and claims[0].reference_keys == ["brown"]


# ------------------------------------------------------------------------------ chunking

SECTION = " ".join(f"Sentence {i} talks about topic {i % 7} with some filler words here." for i in range(120))
REF_TEXT = f"Introduction\n\n{SECTION}\n\nResults\n\nThe model improves accuracy by 12% on ImageNet. {SECTION}"


def test_chunk_text_offsets_sizes_overlap_and_headings():
    chunks = chunk_text(REF_TEXT)
    assert len(chunks) > 4
    for c in chunks:
        assert REF_TEXT[c.start : c.end] == c.text
        assert len(c.text.split()) <= CHUNK_WORDS + 40
    for a, b in zip(chunks, chunks[1:]):
        assert b.start > a.start
        if a.heading == b.heading:
            assert b.start < a.end  # consecutive chunks in a section overlap
        else:
            assert b.start > a.end  # no chunk crosses a heading
    assert chunks[0].heading == "Introduction"
    results = [c for c in chunks if "12% on ImageNet" in c.text]
    assert results and results[0].heading == "Results"
    assert "Introduction" not in chunks[0].text  # headings are metadata, not chunk text


def test_chunk_text_splits_giant_sentences():
    text = "word " * 1000
    chunks = chunk_text(text)
    assert len(chunks) >= 3 and all(len(c.text.split()) <= CHUNK_WORDS + 200 for c in chunks)


# ----------------------------------------------------------------------------- retrieval


def _bag_of_words(texts: list[str]) -> np.ndarray:
    """Deterministic fake embedding: hashed bag of words (ignores the query instruction)."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        t = t.split("Query:", 1)[-1]
        for w in re.findall(r"[a-z0-9%]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
    return out


@pytest.fixture
def fake_embed(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    async def embed(texts):
        calls.append(list(texts))
        return _bag_of_words(texts)

    monkeypatch.setattr(llm, "embed", embed)
    return calls


def _write(tmp_path, text: str) -> str:
    path = tmp_path / "ref.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_retrieval_ranks_relevant_chunk_and_includes_abstract(fake_embed, tmp_path):
    ref = Reference(key="r", raw="", fulltext_status="full", text_path=_write(tmp_path, REF_TEXT),
                    abstract="We study image classification models.")
    claim = Claim(id="s1.c1", citing_sentence_id="s1", claim_text="The model improves accuracy by 12% on ImageNet",
                  reference_keys=["r"])
    retriever = EvidenceRetriever(k=3)
    [ev] = asyncio.run(retriever.retrieve_all([claim], {"r": ref}))
    assert ev.claim_id == "s1.c1" and ev.reference_key == "r"
    abstract = [p for p in ev.passages if p.chunk_index == -1]
    chunks = [p for p in ev.passages if p.chunk_index >= 0]
    assert len(abstract) == 1 and abstract[0].heading == "Abstract"
    assert len(chunks) == 3
    assert "12% on ImageNet" in chunks[0].text and chunks[0].heading == "Results"
    assert chunks[0].score >= chunks[1].score >= chunks[2].score
    assert REF_TEXT[chunks[0].start : chunks[0].end] == chunks[0].text


def test_reference_is_embedded_once_for_many_claims(fake_embed, tmp_path):
    ref = Reference(key="r", raw="", fulltext_status="full", text_path=_write(tmp_path, REF_TEXT))
    claims = [Claim(id=f"c{i}", citing_sentence_id="s", claim_text=f"topic {i}", reference_keys=["r"]) for i in range(5)]
    asyncio.run(EvidenceRetriever().retrieve_all(claims, {"r": ref}))
    # One batch of claim queries plus one batch of chunk documents.
    assert len(fake_embed) == 2


def test_abstract_only_and_unavailable_references(fake_embed):
    abstract_only = Reference(key="a", raw="", fulltext_status="abstract_only", abstract="We find X works.")
    unavailable = Reference(key="u", raw="", fulltext_status="unavailable")
    claim = Claim(id="c", citing_sentence_id="s", claim_text="X works", reference_keys=["a", "u", "missing"])
    evs = asyncio.run(EvidenceRetriever().retrieve_all([claim], {"a": abstract_only, "u": unavailable}))
    by_key = {e.reference_key: e for e in evs}
    assert [p.text for p in by_key["a"].passages] == ["We find X works."]
    assert by_key["u"].passages == [] and by_key["missing"].passages == []


def test_query_instruction(monkeypatch):
    settings = replace(get_settings(), embed_query_instruction="Find support")
    monkeypatch.setattr(evidence_mod, "get_settings", lambda: settings)
    assert query_text("X works") == "Instruct: Find support\nQuery: X works"
    monkeypatch.setattr(evidence_mod, "get_settings", lambda: replace(settings, embed_query_instruction=""))
    assert query_text("X works") == "X works"


# ------------------------------------------------------------------ near-duplicate merge

from app.pipeline.claims import merge_near_duplicates  # noqa: E402


def _c(i, text, keys=("k",), claim_type="method", numbers=()):
    return Claim(id=f"s1.c{i}", citing_sentence_id="s1", claim_text=text, reference_keys=list(keys),
                 claim_type=claim_type, numbers=list(numbers))


@pytest.mark.parametrize(
    "claims, expected",
    [
        # Restatements merge; the more specific wording is kept.
        ([_c(1, "Adam is an optimizer"), _c(2, "Adam is used for model optimization")],
         ["Adam is used for model optimization"]),
        # A claim fully subsumed by a more specific one (words and numbers) merges into it.
        ([_c(1, "X improves accuracy", claim_type="finding"),
          _c(2, "X improves accuracy on ImageNet by 12%", claim_type="finding", numbers=["12%"])],
         ["X improves accuracy on ImageNet by 12%"]),
        # Different numbers are different statements (each can drift on its own).
        ([_c(1, "InstructGPT has a 350M model", numbers=["350M"]), _c(2, "InstructGPT has a 1.3B model", numbers=["1.3B"])],
         ["InstructGPT has a 350M model", "InstructGPT has a 1.3B model"]),
        # Different references never merge.
        ([_c(1, "X improves accuracy", keys=("a",)), _c(2, "X improves accuracy", keys=("b",))],
         ["X improves accuracy", "X improves accuracy"]),
        # Different statements about the same reference stay separate.
        ([_c(1, "GPT is a generic multi-task learner"), _c(2, "Transformer is an architecture")],
         ["GPT is a generic multi-task learner", "Transformer is an architecture"]),
        ([_c(1, "Scaling improves sample efficiency", claim_type="finding"),
          _c(2, "Scaling improves performance", claim_type="finding")],
         ["Scaling improves sample efficiency", "Scaling improves performance"]),
    ],
)
def test_merge_near_duplicates(claims, expected):
    merged = merge_near_duplicates(claims)
    assert [c.claim_text for c in merged] == expected
    assert [c.id for c in merged] == [f"s1.c{i}" for i in range(1, len(merged) + 1)]


def test_extraction_merges_restated_claims(monkeypatch):
    monkeypatch.setattr(llm, "extract", _fake_extract({"citations": [
        {"reference_ids": ["R1"], "claims": [
            {"claim_text": "X improves accuracy by 12%", "claim_type": "finding"},
            {"claim_text": "X improves the accuracy by 12%", "claim_type": "finding"}]},
        {"reference_ids": ["R2", "R3"], "claims": [
            {"claim_text": "X generalizes to clinical data", "claim_type": "finding"}]},
    ]}))
    claims = asyncio.run(extract_sentence_claims(S1, PAPER, {}))
    assert [(c.id, c.claim_text) for c in claims] == [
        ("s1.c1", "X improves the accuracy by 12%"), ("s1.c2", "X generalizes to clinical data")]


from app.models import Verdict  # noqa: E402
from app.pipeline.claims import mark_duplicate_verdicts  # noqa: E402


def _v(claim_id, key, label, attributed, confidence=0.9, escalated=False):
    return Verdict(claim_id=claim_id, reference_key=key, label=label, confidence=confidence,
                   attributed_claim=attributed, escalated=escalated)


def test_mark_duplicate_verdicts_keeps_the_most_cautious():
    claims = {
        "s4.c2": _c(2, "Adam is an optimizer", keys=("adam",)).model_copy(update={"id": "s4.c2", "citing_sentence_id": "s4"}),
        "s4.c3": _c(3, "A feedforward dimension of 4x is used", keys=("adam",)).model_copy(update={"id": "s4.c3", "citing_sentence_id": "s4"}),
        "s5.c1": _c(1, "Adam is an optimizer", keys=("adam",)).model_copy(update={"id": "s5.c1", "citing_sentence_id": "s5"}),
    }
    verdicts = [
        _v("s4.c2", "adam", "supported", "Adam is an optimizer", 0.95),
        # The judge narrowed the misattributed claim to the same statement: a duplicate.
        _v("s4.c3", "adam", "overstated", "Adam is used for model optimization", 0.8, escalated=True),
        # Same statement in a different sentence is a separate citation.
        _v("s5.c1", "adam", "supported", "Adam is an optimizer"),
    ]
    out = {v.claim_id: v for v in mark_duplicate_verdicts(verdicts, claims)}
    # The more cautious verdict (overstated) is kept; the supported restatement points at it.
    assert out["s4.c3"].duplicate_of is None
    assert out["s4.c2"].duplicate_of == "s4.c3|adam"
    assert out["s5.c1"].duplicate_of is None
    assert verdicts[0].duplicate_of is None  # inputs untouched


def test_mark_duplicate_verdicts_keeps_distinct_statements():
    claims = {cid: _c(i, t).model_copy(update={"id": cid}) for i, (cid, t) in enumerate(
        [("s1.c1", "InstructGPT has a 350M model"), ("s1.c2", "InstructGPT has a 1.3B model")], start=1)}
    verdicts = [_v("s1.c1", "k", "overstated", "InstructGPT has a 350M model"),
                _v("s1.c2", "k", "supported", "InstructGPT has a 1.3B model")]
    assert [v.duplicate_of for v in mark_duplicate_verdicts(verdicts, claims)] == [None, None]


def test_citation_clause():
    from app.models import CitingSentence
    from app.pipeline.claims import citation_clause

    text = "Intro. Errors fall with distance (Smith, 2020) as the code grows (Lee, 2021). Lee et al. [3] show that X holds [4]."
    first = text.index("Errors")
    s1 = CitingSentence(id="s1", text=text[first:text.index(" Lee et al.")], start=first, end=text.index(" Lee et al."),
                        reference_keys=["smith", "lee"],
                        marker_spans=[(text.index("(Smith"), text.index(", 2020)") + 7), (text.index("(Lee"), text.index("2021)") + 5)],
                        marker_keys=[["smith"], ["lee"]])
    assert citation_clause(s1, "smith", text) == "Errors fall with distance"
    assert citation_clause(s1, "lee", text) == "as the code grows"
    start = text.index("Lee et al.")
    s2 = CitingSentence(id="s2", text=text[start:], start=start, end=len(text), reference_keys=["lee3", "x4"],
                        marker_spans=[(start, start + len("Lee et al. [3]")), (text.index("[4]"), text.index("[4]") + 3)],
                        marker_keys=[["lee3"], ["x4"]])
    assert citation_clause(s2, "lee3", text) == "show that X holds"  # a textual citation takes what follows it
    single = s1.model_copy(update={"marker_spans": s1.marker_spans[:1], "marker_keys": [["smith"]]})
    assert citation_clause(single, "smith", text) == single.text
