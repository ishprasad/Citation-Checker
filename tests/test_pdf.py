"""PDF ingestion: layout parser, reference-entry parsing, GROBID TEI, and the arXiv fallback."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pymupdf
import pytest

import app.pipeline.ingest as ingest_mod
from app.config import get_settings
from app.pipeline import pdfparse
from app.pipeline.grobid import GrobidError, parse_tei
from app.pipeline.ingest import ArxivMeta, IngestError, ingest_arxiv, ingest_pdf
from app.pipeline.pdfparse import PdfParseError, clean, join_lines, parse_entry, parse_pdf

from .pdf_builder import PdfBuilder

INTRO = (
    "Dense widgets reduce latency by 12% on standard benchmarks (Smith et al., 2020). Lee and Park (2019a) "
    "showed that sparse widgets scale to large gadgets. Earlier work used heuristics instead (e.g., Brown, 2018; "
    "Lee and Park, 2019b). We build on these results throughout the paper and extend them further."
)


def author_year_pdf() -> bytes:
    b = PdfBuilder()
    b.title("Sparse Widgets Improve Gadget Throughput")
    b.heading("Abstract")
    b.paragraph("We study widgets. Widgets are small components that appear in many gadget designs and matter.")
    b.heading("1 Introduction")
    b.paragraph(INTRO)
    b.heading("References")
    b.reference("Tom Brown. 2018. Heuristic widget placement for gadgets. In Proceedings of the Widget Conference.")
    b.reference("Min Lee and Jae Park. 2019a. Sparse widgets at scale: an empirical study of large gadget "
                "systems. arXiv preprint arXiv:1901.00001.")
    b.reference("Min Lee and Jae Park. 2019b. Widget pruning. Journal of Gadgets, 12(3):1-10.")
    b.reference("John Smith, Ana Garcia, and Wei Chen. 2020. Dense widgets reduce latency. In NeurIPS.")
    return b.bytes()


def numeric_pdf(cite_missing: bool = False) -> bytes:
    b = PdfBuilder()
    b.title("Gadget Throughput Revisited")
    b.heading("1 Introduction")
    b.paragraph("Widgets were introduced long ago [1]. Several systems improved them [2-4], and pruning "
                "helps further [2, 4]. This paragraph has no citation at all and should be ignored by the checker.")
    if cite_missing:
        b.paragraph("Other studies disagree [7], [8], [9]. More studies disagree as well [10], [11].")
    b.heading("References")
    b.reference("[1] A. Turing. Computing machinery and intelligence. Mind, 59(236):433-460, 1950.")
    b.reference("[2] J. Smith and A. Garcia. Dense widgets reduce latency. In NeurIPS, 2020.")
    b.reference("[3] M. Lee. Sparse widgets at scale. arXiv preprint arXiv:1901.00001, 2019.")
    b.reference("[4] W. Chen, \"Widget pruning for gadgets,\" in Proc. Gadget Conf., 2021, pp. 1-10.")
    b.heading("A Additional Results")
    b.paragraph("In the appendix we repeat the experiment of Lee [3] on a larger gadget with more widgets.")
    return b.bytes()


# ---------------------------------------------------------------------------- layout parser


def test_author_year_pdf_maps_parenthetical_and_textual_citations():
    paper, sentences = parse_pdf(author_year_pdf())
    assert paper.title == "Sparse Widgets Improve Gadget Throughput"
    assert paper.parsed_from == "pdf" and paper.citation_style == "author_year"
    assert (paper.citation_keys_mapped, paper.citation_keys_total) == (4, 4) and paper.notes == []
    refs = {r.key: r for r in paper.references}
    assert refs["r2"].label == "Lee and Park, 2019a" and refs["r2"].arxiv_id == "1901.00001"
    assert refs["r4"].title == "Dense widgets reduce latency" and refs["r4"].label == "Smith et al., 2020"
    by_text = {s.text.split()[0]: s.reference_keys for s in sentences}
    assert by_text == {"Dense": ["r4"], "Lee": ["r2"], "Earlier": ["r1", "r3"]}
    # Markers cover the citation text exactly.
    s = next(s for s in sentences if s.text.startswith("Lee"))
    start, end = s.marker_spans[0]
    assert paper.text[start:end] == "Lee and Park (2019a)"
    assert [sec.title for sec in paper.sections][:2] == ["Abstract", "1 Introduction"]
    assert paper.abstract.startswith("We study widgets")


def test_numeric_pdf_ranges_lists_and_appendix_after_references():
    paper, sentences = parse_pdf(numeric_pdf())
    assert paper.citation_style == "numeric"
    assert [r.label for r in paper.references] == ["1", "2", "3", "4"]
    assert paper.references[3].title == "Widget pruning for gadgets"  # IEEE quoted title
    keys = [s.reference_keys for s in sentences]
    assert keys == [["r1"], ["r2", "r3", "r4"], ["r3"]]  # the appendix is body text again
    second = sentences[1]
    assert [paper.text[a:b] for a, b in second.marker_spans] == ["[2-4]", "[2, 4]"]
    assert "Additional Results" in [sec.title for sec in paper.sections][-1]


def test_bold_titles_in_references_do_not_end_the_list():
    # eLife sets reference titles in bold: a short, mostly bold entry looks like a heading.
    b = PdfBuilder()
    b.title("Primaquine Safety in Widget Deficiency")
    b.heading("1 Introduction")
    b.paragraph("Anaemia is common in malaria (White, 2018). Severe haemolysis is rare but documented "
                "(Yilma et al., 2023). Earlier trials used daily dosing (Alving et al., 1960).")
    b.heading("References")
    b.reference_parts([("Alving AS, Johnson CF, Tarlov AR (1960) ", False), ("Mitigation of the haemolytic effect", True)])
    b.reference_parts([("White NJ (2018) ", False), ("Anaemia and malaria in children", True)])
    b.reference_parts([("Yilma D, Groves ES, Douglas NM (2023) ", False), ("Severe hemolysis during primaquine", True)])
    b.heading("Article and author information")
    b.paragraph("Sasithon Pukrittayakamee works at the Clinical Therapeutics Unit of the Faculty of Tropical Medicine.")
    paper, sentences = parse_pdf(b.bytes())
    assert [r.label for r in paper.references] == ["Alving et al., 1960", "White, 2018", "Yilma et al., 2023"]
    assert (paper.citation_keys_mapped, paper.citation_keys_total) == (3, 3)


def test_weak_citation_matching_is_reported():
    paper, _ = parse_pdf(numeric_pdf(cite_missing=True))
    assert paper.citation_keys_total == 9 and paper.citation_keys_mapped == 4
    assert paper.notes and paper.notes[0].startswith("Only 4 of 9 in-text citations")


def test_pdf_without_references_or_text():
    b = PdfBuilder()
    b.title("No References Here")
    b.paragraph("This document mentions prior work [1] but has no reference list. " * 8)
    paper, sentences = parse_pdf(b.bytes())
    assert sentences == [] and "No reference list" in paper.notes[0]

    doc = pymupdf.open()
    doc.new_page()
    with pytest.raises(PdfParseError, match="scanned"):
        parse_pdf(doc.tobytes())
    with pytest.raises(PdfParseError):
        parse_pdf(b"%PDF-1.4 not really a pdf")


# ---------------------------------------------------------------------------- text repair


@pytest.mark.parametrize(
    "lines, expected",
    [
        (["the trans-", "duction model"], "the transduction model"),
        (["chain-of-", "thought prompting"], "chain-of-thought prompting"),
        (["state-", "of-the-art results"], "state-of-the-art results"),
        (["multi-head self-", "attention"], "multi-head self-attention"),
        (["conversation analysis-", "oriented work"], "conversation analysis-oriented work"),
        (["LARGE LAN-", "GUAGE MODELS"], "LARGE LANGUAGE MODELS"),
        (["results (Ouyang et al., 2022).The second"], "results (Ouyang et al., 2022). The second"),
        (["as in (Rebuffiet al., 2017)"], "as in (Rebuffi et al., 2017)"),
    ],
)
def test_join_lines(lines, expected):
    assert join_lines(lines) == expected


def test_clean_ligatures_and_detached_accents():
    assert clean("ﬁne-tuning by Marasovi´c and M¨uller, R´emi") == "fine-tuning by Marasović and Müller, Rémi"


# ---------------------------------------------------------------------------- reference entries


@pytest.mark.parametrize(
    "text, surnames, title, year",
    [
        # ACL / NeurIPS natbib
        ("Jason Wei, Xuezhi Wang, and Denny Zhou. 2022a. Chain of thought prompting elicits reasoning. NeurIPS.",
         ["Wei", "Wang", "Zhou"], "Chain of thought prompting elicits reasoning", "2022a"),
        ("Christopher Ré and Ana Marasović. 2018. Training classifiers with natural language explanations. ACL.",
         ["Ré", "Marasović"], "Training classifiers with natural language explanations", "2018"),
        # plainnat: year at the end, arXiv digits are not years
        ("Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. Intrinsic dimensionality explains fine-tuning. "
         "arXiv:2012.13255 [cs], December 2020.", ["Aghajanyan", "Zettlemoyer", "Gupta"],
         "Intrinsic dimensionality explains fine-tuning", "2020"),
        ("Jihun Ham and Daniel D. Lee. Grassmann discriminant analysis. In ICML, pp. 376–383, 2008.",
         ["Ham", "Lee"], "Grassmann discriminant analysis", "2008"),
        ("Tom B. Brown, Benjamin Mann, and Nick Ryder. Language models are few-shot learners. NeurIPS, 2020.",
         ["Brown", "Mann", "Ryder"], "Language models are few-shot learners", "2020"),
        # IEEE
        ('A. Vaswani, N. Shazeer, and N. Parmar, "Attention is all you need," in Proc. NeurIPS, 2017, pp. 1-11.',
         ["Vaswani", "Shazeer", "Parmar"], "Attention is all you need", "2017"),
        # APA
        ("Antaki, C., & Kent, G. (2012). Telling people what to do. Journal of Pragmatics, 44(6), 876-889.",
         ["Antaki", "Kent"], "Telling people what to do", "2012"),
        # Vancouver
        ("Wang TW, Asman K, Gentzke AS, et al. Tobacco product use among adults. MMWR. 2018;67(44):1225-32.",
         ["Wang", "Asman", "Gentzke"], "Tobacco product use among adults", "2018"),
        ("van der Tempel J, Noormohamed A. Vape, quit, tweet? Electronic cigarettes and smoking. Tob Control. 2016.",
         ["van der Tempel", "Noormohamed"], "Vape, quit, tweet? Electronic cigarettes and smoking", "2016"),
        # Nature
        ("Dufour, D. L. & Piperata, B. A. Rural-to-urban migration in Latin America. Am. J. Hum. Biol. 16, 395 (2004).",
         ["Dufour", "Piperata"], "Rural-to-urban migration in Latin America", "2004"),
        # LNCS
        ("Smith, J., Doe, A.: Widgets in the wild. In: Proc. WIDG, pp. 1–10. Springer (2019)",
         ["Smith", "Doe"], "Widgets in the wild", "2019"),
    ],
)
def test_parse_entry_styles(text, surnames, title, year):
    entry = parse_entry(text)
    assert entry.surnames == surnames
    assert entry.title == title
    assert entry.year_label == year


def test_superscript_citations_and_footnote_marks():
    sup = pdfparse.SUP_OPEN + "{}" + pdfparse.SUP_CLOSE
    paper_text = [("heading", "Introduction"),
                  ("para", "Obesity is rising" + sup.format("1,2") + ". Cities differ" + sup.format("3") + "."),
                  ("para", "Urban form matters" + sup.format("2–3") + " for health outcomes in many cities.")]
    layout = pdfparse.Layout(title="T", paragraphs=paper_text, reference_blocks=[], body_size=10)
    # Superscripts become bracketed citations only when the reference list is numbered.
    numbered = [pdfparse.RefEntry(n, f"Author {n}. Title number {n} of a paper. Journal. 2020.") for n in (1, 2, 3)]
    paper, sentences = _parse_layout(layout, numbered, True)
    assert [s.reference_keys for s in sentences] == [["r1", "r2"], ["r3"], ["r2", "r3"]]
    paper, sentences = _parse_layout(layout, [], False)
    assert "Obesity is rising." in paper.text  # footnote marks dropped


def _parse_layout(layout, entries, numbered):
    """Run parse_pdf's second half on a prepared layout."""
    orig_analyse, orig_split = pdfparse.analyse, pdfparse.split_references
    pdfparse.analyse = lambda doc: layout
    pdfparse.split_references = lambda blocks: (entries, numbered)
    try:
        return parse_pdf(PdfBuilder().bytes())
    finally:
        pdfparse.analyse, pdfparse.split_references = orig_analyse, orig_split


# ---------------------------------------------------------------------------- GROBID

TEI_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0" xmlns:xml="http://www.w3.org/XML/1998/namespace">
  <teiHeader><fileDesc><titleStmt><title level="a" type="main">Widgets Revisited</title></titleStmt>
    <sourceDesc><biblStruct><analytic><author><persName><forename>Ada</forename><surname>Byron</surname></persName></author></analytic></biblStruct></sourceDesc>
  </fileDesc>
  <profileDesc><abstract><div><p>We revisit widgets.</p></div></abstract></profileDesc></teiHeader>
  <text><body>
    <div><head n="1">Introduction</head>
      <p>Dense widgets reduce latency
         <ref type="bibr" target="#b0">(Smith et al., 2020)</ref>. Others disagree
         <ref type="bibr">(Unknown, 2001)</ref>.</p>
      <p>Pruning helps <ref type="bibr" target="#b1">(Lee, 2019)</ref> as shown in <ref type="figure">Fig. 1</ref>.</p>
    </div>
  </body>
  <back><div type="references"><listBibl>
    <biblStruct xml:id="b0"><analytic><title level="a" type="main">Dense widgets reduce latency</title>
      <author><persName><forename>John</forename><surname>Smith</surname></persName></author>
      <author><persName><forename>Ana</forename><surname>Garcia</surname></persName></author>
      <author><persName><forename>Wei</forename><surname>Chen</surname></persName></author></analytic>
      <monogr><title>NeurIPS</title><imprint><date type="published" when="2020"/></imprint></monogr>
      <idno type="arXiv">arXiv:2001.00001</idno></biblStruct>
    <biblStruct xml:id="b1"><analytic><title level="a" type="main">Widget pruning</title>
      <author><persName><forename>Min</forename><surname>Lee</surname></persName></author></analytic>
      <monogr><title>J. Gadgets</title><imprint><date when="2019-05-01"/></imprint></monogr>
      <idno type="DOI">10.1000/widgets.1</idno></biblStruct>
  </listBibl></div></back></text>
</TEI>"""


def test_parse_tei():
    paper, sentences = parse_tei(TEI_SAMPLE)
    assert paper.title == "Widgets Revisited" and paper.parsed_from == "grobid" and paper.authors == ["Ada Byron"]
    assert paper.citation_style == "author_year"
    assert (paper.citation_keys_total, paper.citation_keys_mapped) == (3, 2)
    refs = {r.key: r for r in paper.references}
    assert refs["b0"].label == "Smith et al., 2020" and refs["b0"].arxiv_id == "2001.00001"
    assert refs["b1"].doi == "10.1000/widgets.1" and refs["b1"].year == 2019
    assert [s.reference_keys for s in sentences] == [["b0"], ["b1"]]
    start, end = sentences[0].marker_spans[0]
    assert paper.text[start:end] == "(Smith et al., 2020)"
    assert "Dense widgets reduce latency (Smith" in paper.text  # line breaks inside paragraphs normalised
    with pytest.raises(GrobidError):
        parse_tei(b"<not xml")


# ---------------------------------------------------------------------------- ingest


def test_ingest_pdf_falls_back_when_grobid_fails(monkeypatch):
    monkeypatch.setattr(ingest_mod, "get_settings", lambda: replace(get_settings(), grobid_url="http://grobid:8070"))

    async def broken(data, url):
        raise GrobidError("down")

    monkeypatch.setattr(ingest_mod, "grobid_parse", broken)
    result = asyncio.run(ingest_pdf(author_year_pdf(), note="A note."))
    assert result.paper.parsed_from == "pdf" and len(result.sentences) == 3
    assert result.paper.notes == ["A note."]

    with pytest.raises(IngestError, match="scanned"):
        doc = pymupdf.open()
        doc.new_page()
        asyncio.run(ingest_pdf(doc.tobytes()))


def test_arxiv_paper_without_latex_source_is_read_from_pdf(monkeypatch):
    pdf = numeric_pdf()

    async def meta(arxiv_id):
        return ArxivMeta(id="2101.00002v1", title="Gadget Throughput Revisited", authors=["A. Author"], abstract="Abs.")

    async def fetch(url, namespace, **kwargs):
        assert url.endswith("/e-print/2101.00002v1")
        return pdf  # arXiv serves the PDF when there is no source

    monkeypatch.setattr(ingest_mod, "fetch_arxiv_metadata", meta)
    monkeypatch.setattr(ingest_mod, "fetch_cached", fetch)
    monkeypatch.setattr(ingest_mod, "get_settings", lambda: replace(get_settings(), grobid_url=""))
    result = asyncio.run(ingest_arxiv("2101.00002"))
    paper = result.paper
    assert paper.id == "arxiv:2101.00002v1" and paper.source == "arxiv" and paper.parsed_from == "pdf"
    assert paper.title == "Gadget Throughput Revisited" and paper.abstract == "Abs."
    assert paper.notes[0].startswith("arXiv has no usable LaTeX source") and len(result.sentences) == 3


def test_journal_layout_tables_sidebar_and_back_matter_are_not_body_text():
    """PLOS-style layout: 8pt references outweigh the 10pt body, a small sidebar and masthead,
    a ruled table, and back matter after the discussion."""
    b = PdfBuilder()
    b.note("JOURNAL ONE", 40, 30, size=20)  # masthead in the top margin
    b.title("Widget Use Among Students")
    b.heading("Introduction")
    b.paragraph("Widget use has grown considerably among students [1, 2]. Widgets are devices that heat a "
                "solution composed of gadgets and flavoring")
    b.note("Received: March 30, 2021", 20, 400)
    b.note("Copyright: 2021 The Authors. This is an open access article.", 20, 410)
    b.paragraph("agents for users [3]. This continues the previous paragraph after a page element.")
    b.heading("Results")
    b.table([["Characteristic", "N (%)"], ["< one year", "43 (32.8)"], ["1-3 year", "73 (55.7)"],
             ["> 3 year", "15 (11.5)"]])
    b.paragraph("Only 11.3% of students used widgets, which is lower than reported elsewhere [4].")
    b.heading("Funding")
    b.paragraph("The work was supported by the Widget Foundation (grant 12345) as described in [5].")
    b.heading("References")
    for n in range(1, 41):
        b.reference(f"{n}. Author{n} A, Writer B, Scholar C. A long study of widgets number {n} in students and "
                    f"gadget users across several countries. J Widgets. 2019;{n}:1-10.", hanging=False, size=8)
    paper, sentences = parse_pdf(b.bytes())
    text = paper.text
    assert "Received" not in text and "Copyright" not in text and "JOURNAL ONE" not in text
    assert "43 (32.8)" not in text and "Characteristic" not in text
    assert "Widget Foundation" not in text  # back matter is not checked
    assert [s.title for s in paper.sections] == ["Introduction", "Results"]
    assert "solution composed of gadgets and flavoring agents for users [3]." in text  # one paragraph again
    assert [s.reference_keys for s in sentences] == [["r1", "r2"], ["r3"], ["r4"]]
