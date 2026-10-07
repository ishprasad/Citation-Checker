"""Reference resolution tests with a mocked network (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import httpx
import pymupdf
import pytest

import app.http as http_mod
import app.pipeline.references as refs_mod
from app.config import get_settings
from app.models import Coverage, Reference
from app.pipeline.pdftext import PdfTextError, _clean_block, extract_pdf_text
from app.pipeline.references import (
    REASON_ABSTRACT_ONLY,
    REASON_NOT_FOUND,
    Candidate,
    best_candidate,
    candidate_from_openalex,
    crossref_retraction,
    normalize_doi,
    openalex_abstract,
    read_reference_sidecar,
    read_reference_text,
    resolve_references,
    title_similarity,
)

# ------------------------------------------------------------------------- unit helpers


def test_title_similarity_ignores_case_accents_and_punctuation():
    assert title_similarity("BERT: Pre-training of Deep Bidirectional Transformers",
                            "Bert pre training of deep bidirectional transformers") == 1.0
    assert title_similarity("Café au lait", "cafe au LAIT") == 1.0
    assert title_similarity("Attention is all you need", "Deep residual learning") < 0.5


def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1162/NECO.1997.9.8.1735") == "10.1162/neco.1997.9.8.1735"
    assert normalize_doi("doi:10.1234/ABC.") == "10.1234/abc"
    assert normalize_doi(None) is None


def test_best_candidate_threshold_and_year_check():
    ref = Reference(key="k", raw="", title="Learning to execute", year=2014)
    exact = Candidate(title="Learning to Execute", year=2014)
    wrong_year = Candidate(title="Learning to Execute", year=2021)
    other = Candidate(title="Learning to Exercise", year=2014)
    assert best_candidate(ref, [other, exact])[0] is exact
    # Same generic title, years far apart: probably a different paper.
    assert best_candidate(ref, [wrong_year])[0] is None
    assert best_candidate(ref, [])[0] is None


def test_best_candidate_accepts_shortened_title():
    ref = Reference(key="k", raw="", title="A decomposable attention model", year=2016)
    full = Candidate(title="A Decomposable Attention Model for Natural Language Inference", year=2016)
    assert best_candidate(ref, [full])[0] is full
    assert best_candidate(ref.model_copy(update={"year": 2010}), [full])[0] is None


def test_best_candidate_without_parsed_title_uses_raw_string():
    ref = Reference(key="k", raw="J. Smith. Deep networks on imagenet at scale. In CVPR, 2020.")
    assert best_candidate(ref, [Candidate(title="Deep Networks on ImageNet at Scale", year=2020)])[0] is not None
    assert best_candidate(ref, [Candidate(title="Shallow networks", year=2020)])[0] is None


def test_openalex_abstract_reconstruction():
    assert openalex_abstract({"world": [1], "Hello": [0], "again": [3], "hello": [2]}) == "Hello world hello again"
    assert openalex_abstract(None) is None


def test_candidate_from_openalex_finds_arxiv_twin_and_pdfs():
    work = {
        "id": "https://openalex.org/W2787560479",
        "doi": "https://doi.org/10.18653/v1/n18-1202",
        "title": "Deep Contextualized Word Representations",
        "publication_year": 2018,
        "is_retracted": False,
        "best_oa_location": {"landing_page_url": "https://doi.org/10.18653/v1/n18-1202",
                             "pdf_url": "https://www.aclweb.org/anthology/N18-1202.pdf"},
        "locations": [
            {"landing_page_url": "https://doi.org/10.18653/v1/n18-1202",
             "pdf_url": "https://www.aclweb.org/anthology/N18-1202.pdf"},
            {"landing_page_url": "http://arxiv.org/abs/1802.05365", "pdf_url": "https://arxiv.org/pdf/1802.05365"},
        ],
        "abstract_inverted_index": {"We": [0], "introduce": [1]},
    }
    cand = candidate_from_openalex(work)
    assert cand.arxiv_id == "1802.05365"
    assert cand.pdf_urls == ["https://www.aclweb.org/anthology/N18-1202.pdf"]  # arXiv handled separately
    assert cand.doi == "10.18653/v1/n18-1202"
    assert cand.openalex_id == "W2787560479"
    assert cand.abstract == "We introduce"
    assert cand.retracted is False


def test_crossref_retraction_field():
    # Shape recorded from api.crossref.org/works/10.1016/S0140-6736(97)11096-0 (Wakefield 1998).
    item = {
        "updated-by": [
            {"DOI": "10.1016/s0140-6736(04)15715-2", "type": "correction", "source": "retraction-watch"},
            {"DOI": "10.1016/s0140-6736(10)60175-4", "type": "retraction", "source": "retraction-watch"},
        ]
    }
    assert crossref_retraction(item) == (True, "10.1016/s0140-6736(10)60175-4")
    assert crossref_retraction({"updated-by": [{"type": "correction"}]}) == (False, None)
    assert crossref_retraction({}) == (False, None)


# ------------------------------------------------------------------------------ PDF text

BODY = (
    "Large models improve accuracy on many bench- marks. We report a 12% gain on the main task "
    "and the effect is consistent across five datasets. "
) * 6


def make_pdf(blocks: list[str]) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 50
    for block in blocks:
        rect = pymupdf.Rect(50, y, 550, y + 160)
        page.insert_textbox(rect, block, fontsize=9)
        y += 170
        if y > 650:
            page = doc.new_page()
            y = 50
    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_text_cleanup_and_references_split():
    body_with_break = BODY.replace("bench- marks", "bench-\nmarks")
    pdf = make_pdf([body_with_break, BODY, "References", "[1] A. Author. A cited paper. 2020."])
    result = extract_pdf_text(pdf)
    assert "benchmarks" in result.body  # hyphenation repaired
    assert "A cited paper" not in result.body
    assert "A cited paper" in result.references


def test_clean_block_expands_ligatures_and_joins_lines():
    assert _clean_block("the eﬀect across ﬁve data-\nsets\nshown") == "the effect across five datasets shown"


def test_pdf_text_rejects_non_pdf_and_empty():
    with pytest.raises(PdfTextError):
        extract_pdf_text(b"<html>paywall</html>")
    with pytest.raises(PdfTextError):
        extract_pdf_text(make_pdf(["tiny"]))


# ------------------------------------------------------------------ resolution, mocked

LATEX_BODY = "We show that pretraining helps in low-resource settings by 7 points. " * 60
ARXIV_TEX = (
    "\\documentclass{article}\\begin{document}\\section{Results}" + LATEX_BODY
    + "\\begin{table}\\caption{BLEU of 28.4 on WMT.}\\begin{tabular}{c}x\\end{tabular}\\end{table}"
    + "\\end{document}"
)

ATOM_ENTRY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2101.00001v2</id>
    <title>Pretraining for Low-Resource
      NLP</title>
    <summary>We study pretraining.</summary>
    <author><name>Kim Lee</name></author>
  </entry>
</feed>"""
ATOM_EMPTY = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def _tar_gz(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = text.encode()
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class Network:
    """Routes mocked requests and counts calls."""

    def __init__(self, fail_arxiv_downloads: bool = False):
        self.calls: list[str] = []
        self.fail_arxiv_downloads = fail_arxiv_downloads
        self.pdf = make_pdf([BODY, BODY])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        self.calls.append(str(url))
        host, path, params = url.host, url.path, dict(url.params)
        if host == "export.arxiv.org" and path == "/api/query":
            if "id_list" in params and "2101.00001" in params["id_list"]:
                return httpx.Response(200, text=ATOM_ENTRY)
            return httpx.Response(200, text=ATOM_EMPTY)
        if host == "arxiv.org":
            if self.fail_arxiv_downloads:
                raise httpx.ConnectError("connection reset")
            if path == "/e-print/2101.00001":
                return httpx.Response(200, content=_tar_gz("main.tex", ARXIV_TEX))
            return httpx.Response(404)
        if host == "api.openalex.org" and path == "/works":
            if params.get("filter", "").startswith("doi:"):
                dois = params["filter"][4:].split("|")
                results = []
                if "10.1234/jot.2019.42" in dois:
                    results.append({
                        "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1234/jot.2019.42",
                        "title": "Adapters for Everything", "publication_year": 2019, "is_retracted": False,
                        "best_oa_location": {"pdf_url": "https://pub.example/jones.pdf"}, "locations": [],
                        "abstract_inverted_index": {"Adapters": [0], "work.": [1]},
                    })
                return httpx.Response(200, json={"results": results})
            if "deep networks on imagenet" in params.get("search", ""):
                return httpx.Response(200, json={"results": [{
                    "id": "https://openalex.org/W2", "doi": None, "title": "Deep Networks on ImageNet",
                    "publication_year": 2020, "is_retracted": False, "best_oa_location": None, "locations": [],
                    "abstract_inverted_index": {"Deep": [0], "nets.": [1]},
                }]})
            return httpx.Response(200, json={"results": []})
        if host == "api.crossref.org":
            if path == "/works/10.1234/jot.2019.42":
                return httpx.Response(200, json={"message": {
                    "DOI": "10.1234/jot.2019.42", "title": ["Adapters for everything"],
                    "updated-by": [{"DOI": "10.1234/retraction", "type": "retraction", "source": "retraction-watch"}],
                }})
            if path == "/works":
                return httpx.Response(200, json={"message": {"items": []}})
            return httpx.Response(404)
        if host == "pub.example":
            return httpx.Response(200, content=self.pdf, headers={"content-type": "application/pdf"})
        return httpx.Response(404)


@pytest.fixture
def network(monkeypatch, tmp_path):
    def install(net: Network) -> Network:
        settings = replace(get_settings(), cache_dir=tmp_path, arxiv_min_interval=0.0, contact_email="t@example.com")
        monkeypatch.setattr(http_mod, "get_settings", lambda: settings)
        monkeypatch.setattr(refs_mod, "get_settings", lambda: settings)
        monkeypatch.setattr(http_mod, "MAX_ATTEMPTS", 1)
        http_mod.set_client(httpx.AsyncClient(transport=httpx.MockTransport(net)))
        return net

    yield install
    http_mod.set_client(None)


REFS = [
    Reference(key="lee2021", label="2", raw="Kim Lee. Pretraining for low-resource NLP. arXiv:2101.00001, 2021.",
              title="Pretraining for low-resource NLP", year=2021, arxiv_id="2101.00001"),
    Reference(key="jones2019", label="1", raw="Mary Jones. Adapters for everything. 2019.",
              title="Adapters for everything", year=2019, doi="10.1234/jot.2019.42"),
    Reference(key="smith2020", label="3", raw="John Smith. Deep networks on ImageNet. CVPR 2020.",
              title="Deep networks on ImageNet", year=2020),
    Reference(key="ghost", label="4", raw="Nobody. A paper that does not exist anywhere at all. 1999.",
              title="A paper that does not exist anywhere at all", year=1999),
]


def test_resolve_references_end_to_end(network):
    net = network(Network())
    done = []
    resolved = asyncio.run(resolve_references(REFS, on_done=lambda r: done.append(r.key)))
    by_key = {r.key: r for r in resolved}
    assert sorted(done) == sorted(r.key for r in REFS)
    assert [r.key for r in resolved] == [r.key for r in REFS]  # order preserved

    lee = by_key["lee2021"]
    assert (lee.fulltext_status, lee.fulltext_source) == ("full", "arxiv_latex")
    assert lee.title == "Pretraining for Low-Resource NLP"  # arXiv's canonical title
    assert lee.abstract == "We study pretraining."
    text = read_reference_text(lee)
    assert "by 7 points" in text and "Table: BLEU of 28.4 on WMT." in text  # captions kept
    assert "references" in read_reference_sidecar(lee)

    jones = by_key["jones2019"]
    assert (jones.fulltext_status, jones.fulltext_source) == ("full", "oa_pdf")
    assert jones.retracted is True and jones.retraction_notice == "10.1234/retraction"
    assert jones.openalex_id == "W1"
    assert "12% gain" in read_reference_text(jones)

    smith = by_key["smith2020"]
    assert smith.fulltext_status == "abstract_only"
    assert smith.status_reason == REASON_ABSTRACT_ONLY
    assert smith.abstract == "Deep nets." and smith.match_score == 1.0

    ghost = by_key["ghost"]
    assert ghost.fulltext_status == "unavailable" and ghost.status_reason == REASON_NOT_FOUND

    coverage = Coverage.of(resolved)
    assert (coverage.full, coverage.abstract_only, coverage.unavailable, coverage.retracted) == (2, 1, 1, 1)
    assert coverage.line() == "Full text found for 2 of 4 references · abstract only for 1 · unavailable for 1."

    # Second run is served entirely from the resolution cache.
    before = len(net.calls)
    again = asyncio.run(resolve_references(REFS))
    assert len(net.calls) == before
    assert [r.model_dump() for r in again] == [r.model_dump() for r in resolved]


def test_transient_arxiv_failure_degrades_and_is_not_cached(network):
    net = network(Network(fail_arxiv_downloads=True))
    [lee] = asyncio.run(resolve_references([REFS[0]]))
    # Download failed, but metadata survived: the claim can still be checked against the abstract.
    assert lee.fulltext_status == "abstract_only"
    assert lee.abstract == "We study pretraining."

    net.fail_arxiv_downloads = False
    [lee] = asyncio.run(resolve_references([REFS[0]]))
    assert lee.fulltext_status == "full"  # retried because the degraded result was not cached


def test_arxiv_twin_search_backs_off_when_rate_limited(monkeypatch):
    calls = []

    async def no_openalex(title):
        return []

    async def limited(url, namespace, **kwargs):
        calls.append(url)
        request = httpx.Request("GET", url)
        raise httpx.HTTPStatusError("429", request=request, response=httpx.Response(429, request=request))

    monkeypatch.setattr(refs_mod, "openalex_search", no_openalex)
    monkeypatch.setattr(refs_mod, "fetch_cached", limited)
    monkeypatch.setattr(refs_mod, "_arxiv_search_paused_until", 0.0)
    title = "Knowledge and attitudes towards e-cigarette use in Lebanon and their associated factors in adults"
    assert asyncio.run(refs_mod.find_arxiv_twin(title)) is None  # a refusal is "no preprint", not a failure
    assert asyncio.run(refs_mod.find_arxiv_twin(title)) is None
    assert len(calls) == 1  # paused after the first refusal
