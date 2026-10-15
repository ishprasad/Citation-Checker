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

    def __init__(self, fail_arxiv_downloads: bool = False, openalex_quota_used: bool = False):
        self.calls: list[str] = []
        self.fail_arxiv_downloads = fail_arxiv_downloads
        self.openalex_quota_used = openalex_quota_used
        self.pdf = make_pdf([BODY, BODY])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        self.calls.append(str(url))
        host, path, params = url.host, url.path, dict(url.params)
        if host == "api.openalex.org" and self.openalex_quota_used:
            return httpx.Response(429, headers={"retry-after": "23999"})
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


def test_exhausted_api_quota_fails_fast_and_falls_back(network):
    """OpenAlex refusing for hours: no waiting, other sources still used, nothing cached as final."""
    net = network(Network(openalex_quota_used=True))
    resolved = {r.key: r for r in asyncio.run(resolve_references(REFS))}
    openalex_calls = [c for c in net.calls if "api.openalex.org" in c]
    assert len(openalex_calls) == 1  # the first refusal blocks the host; later lookups skip it
    assert resolved["lee2021"].fulltext_status == "full"  # arXiv still works
    assert resolved["jones2019"].retracted is True  # Crossref still works
    assert http_mod.host_blocked("api.openalex.org") > 3600

    net.openalex_quota_used = False
    http_mod._blocked_until.clear()  # the quota is back
    again = {r.key: r for r in asyncio.run(resolve_references(REFS))}
    assert again["jones2019"].fulltext_status == "full"  # retried: the degraded results were not cached


def test_openalex_api_key_is_sent_only_to_openalex(monkeypatch, tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.host] = request.headers.get("authorization")
        return httpx.Response(200, json={"results": []})

    settings = replace(get_settings(), cache_dir=tmp_path, openalex_api_key="k123")
    monkeypatch.setattr(http_mod, "get_settings", lambda: settings)
    http_mod.set_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    try:
        asyncio.run(http_mod.fetch("https://api.openalex.org/works", {"search": "x"}))
        asyncio.run(http_mod.fetch("https://api.crossref.org/works", {"query": "x"}))
    finally:
        http_mod.set_client(None)
    assert seen == {"api.openalex.org": "Bearer k123", "api.crossref.org": None}


JATS_ARTICLE = (
    '<article><front><article-meta><abstract><p>We study e-cigarettes.</p></abstract></article-meta></front>'
    "<body><sec><title>Results</title><p>Use rose to 27.7% among students"
    '<sup><xref ref-type="bibr">3</xref>,<xref ref-type="bibr">4</xref></sup>, as earlier work found '
    '[<xref ref-type="bibr">1</xref>, <xref ref-type="bibr">2</xref>].'
    '<table-wrap><caption><p>Prevalence by year (27.7% in 2019).</p></caption><table><thead><tr><th>Year</th><th>Prevalence</th></tr></thead>'
    '<tbody><tr><td>2019</td><td>27.7%</td></tr><tr><td></td><td></td></tr></tbody></table>'
    "</table-wrap></p></sec></body><back><ref-list><ref><label>1</label><mixed-citation>Wang T. "
    "<article-title>Tobacco use among adults</article-title>. MMWR. 2018.</mixed-citation></ref></ref-list></back></article>"
)


def test_jats_to_text():
    from app.pipeline.jats import jats_to_text

    text, sidecar = jats_to_text((JATS_ARTICLE * 1).encode())
    assert text.split("\n\n") == [
        "Abstract", "We study e-cigarettes.", "Results",
        "Use rose to 27.7% among students[3,4], as earlier work found [1, 2].",
        "Table: Prevalence by year (27.7% in 2019).",
        "Year | Prevalence\n2019 | 27.7%",  # table contents: cited numbers are often only there
    ]
    assert sidecar["references"] == [{"label": "1", "title": "Tobacco use among adults",
                                      "raw": "Wang T. Tobacco use among adults. MMWR. 2018."}]


def test_europepmc_full_text_is_preferred_over_publisher_pdf(network):
    long_article = JATS_ARTICLE.replace("We study e-cigarettes.", "We study e-cigarettes. " * 120)

    class PmcNetwork(Network):
        def __call__(self, request):
            url = request.url
            if url.host == "api.openalex.org" and url.params.get("filter", "").startswith("doi:"):
                self.calls.append(str(url))
                return httpx.Response(200, json={"results": [{
                    "id": "https://openalex.org/W9", "doi": "https://doi.org/10.1371/journal.pone.1",
                    "title": "Vaping among students", "publication_year": 2021, "is_retracted": False,
                    "ids": {"pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/123456"},
                    "best_oa_location": {"pdf_url": "https://pub.example/blocked.pdf"}, "locations": [],
                }]})
            if url.host == "www.ebi.ac.uk":
                self.calls.append(str(url))
                if url.path.endswith("/search"):
                    return httpx.Response(200, json={"resultList": {"result": [
                        {"pmcid": "PMC777", "isOpenAccess": "Y"}]}})
                if url.path.endswith("/PMC123456/fullTextXML") or url.path.endswith("/PMC777/fullTextXML"):
                    return httpx.Response(200, text=long_article)
                return httpx.Response(404)
            return super().__call__(request)

    net = network(PmcNetwork())
    refs = [
        Reference(key="a", label="1", raw="Vaping among students. 2021.", title="Vaping among students",
                  year=2021, doi="10.1371/journal.pone.1"),
        Reference(key="b", label="2", raw="Another study. 2020.", title="Another study", year=2020,
                  doi="10.1186/s12889-020-1"),  # not in OpenAlex: PMC ID from a Europe PMC DOI lookup
    ]
    a, b = asyncio.run(resolve_references(refs))
    assert (a.fulltext_status, a.fulltext_source) == ("full", "europepmc")
    assert (b.fulltext_status, b.fulltext_source) == ("full", "europepmc")
    assert "Table: Prevalence by year" in read_reference_text(a)
    assert read_reference_sidecar(a)["references"][0]["label"] == "1"
    assert not any("pub.example" in c for c in net.calls)  # the publisher PDF was not needed


# ------------------------------------------------------- versions, generic titles, mangled DOIs


def test_arxiv_year_and_plausible_preprint():
    from app.pipeline.references import arxiv_year, plausible_preprint

    assert (arxiv_year("2004.09142"), arxiv_year("math/0512424"), arxiv_year("hep-th/9901001")) == (2020, 2005, 1999)
    assert plausible_preprint("1904.08128", 2021)  # preprint two years before the journal version
    assert not plausible_preprint("math/0512424", 2020)  # a 2005 survey is not a 2020 book
    assert not plausible_preprint("2004.09142", 2016)  # a 2020 paper is not a 2016 one
    assert plausible_preprint(None, 2020) and plausible_preprint("2004.09142", None)


def test_find_doi_rejoins_a_doi_broken_across_lines():
    from app.pipeline.bibliography import find_doi

    assert find_doi("PLoS Med 2016. https://doi.org/10.1371/journal.pmed. 1002151 PMID: 27755547") == "10.1371/journal.pmed.1002151"
    assert find_doi("doi: 10.1016/j.cell.2020.01.001. 2020.") == "10.1016/j.cell.2020.01.001"  # not the year
    assert find_doi("doi:10.1000/xyz. Accessed 2020") == "10.1000/xyz"


class VersionNetwork(Network):
    """OpenAlex records that link the wrong arXiv paper, or a preprint older than the cited version."""

    RECORDS = {
        "10.1007/book.1": {"title": "Orthogonal Polynomials", "publication_year": 2020,
                           "locations": [{"landing_page_url": "https://arxiv.org/abs/math/0512424"}]},
        "10.1038/nm.1": {"title": "nnU-Net: a self-configuring method", "publication_year": 2021,
                         "best_oa_location": {"pdf_url": "https://pub.example/nm.pdf"},
                         "locations": [{"landing_page_url": "https://arxiv.org/abs/2101.00001"}]},
        "10.1371/journal.pmed": {"title": "Commercial Serological Antibody Detection Tests for Tuberculosis",
                                 "publication_year": 2016, "locations": []},
    }

    def __call__(self, request):
        url = request.url
        if url.host == "api.openalex.org" and url.params.get("filter", "").startswith("doi:"):
            self.calls.append(str(url))
            dois = url.params["filter"][4:].split("|")
            return httpx.Response(200, json={"results": [
                {"id": f"https://openalex.org/W{i}", "doi": f"https://doi.org/{doi}", "is_retracted": False,
                 "best_oa_location": None, **record}
                for i, (doi, record) in enumerate(self.RECORDS.items()) if doi in dois]})
        return super().__call__(request)


def test_resolver_checks_versions_and_identifiers(network):
    net = network(VersionNetwork())
    refs = [
        Reference(key="book", raw="Foupouagnigni M, Koepf W. Orthogonal Polynomials: 2nd AIMS Workshop. 2020.",
                  title="Orthogonal Polynomials: 2nd AIMS-Volkswagen Stiftung Workshop", year=2020, doi="10.1007/book.1"),
        Reference(key="journal", raw="Isensee F. nnU-Net. Nature Methods 2021.", title="nnU-Net: a self-configuring method",
                  year=2021, doi="10.1038/nm.1"),
        Reference(key="mangled", raw="Leslie HH. Obstetric Facility Quality and Newborn Mortality in Malawi. PLoS Med 2016.",
                  title="Obstetric Facility Quality and Newborn Mortality in Malawi", year=2016, doi="10.1371/journal.pmed"),
    ]
    book, journal, mangled = asyncio.run(resolve_references(refs))
    # The 2005 arXiv survey linked to the 2020 book is not read as the book.
    assert book.arxiv_id is None and not any("0512424" in c for c in net.calls)
    # Preprint and journal version from the same year: the preprint's LaTeX source is read.
    assert journal.fulltext_source == "arxiv_latex"
    # A DOI pointing at a different paper is not trusted.
    assert mangled.doi is None and "Tuberculosis" not in (mangled.title or "")


def test_later_journal_version_is_read_before_the_preprint(network):
    net = VersionNetwork()
    net.RECORDS = {"10.1038/nm.1": {**VersionNetwork.RECORDS["10.1038/nm.1"], "publication_year": 2022}}
    network(net)
    ref = Reference(key="journal", raw="Isensee F. nnU-Net. Nature Methods 2022.", title="nnU-Net: a self-configuring method",
                    year=2022, doi="10.1038/nm.1")
    [resolved] = asyncio.run(resolve_references([ref]))
    assert (resolved.fulltext_status, resolved.fulltext_source) == ("full", "oa_pdf")
    assert not any("e-print" in c for c in net.calls)
    # When the citing paper itself gives the arXiv ID, the preprint is what it cites: read it first.
    net.RECORDS = {"10.1038/nm.2": {**VersionNetwork.RECORDS["10.1038/nm.1"], "publication_year": 2022,
                                    "title": "Pretraining for low-resource NLP"}}
    cited_preprint = Reference(key="pre", raw="Lee K. Pretraining for low-resource NLP. arXiv:2101.00001.",
                               title="Pretraining for low-resource NLP", year=2022, doi="10.1038/nm.2", arxiv_id="2101.00001")
    [preprint] = asyncio.run(resolve_references([cited_preprint]))
    assert preprint.fulltext_source == "arxiv_latex"
