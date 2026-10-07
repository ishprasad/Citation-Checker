"""Optional PDF ingestion through a GROBID server (GROBID_URL).

GROBID links every in-text citation to its bibliography entry, which is more reliable
than layout heuristics, so it is preferred when configured. Any failure falls back to the
built-in parser (`pdfparse`).
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

from app.models import Paper, Reference
from app.pipeline import bibliography as bib
from app.pipeline.latex import Heading, Marker
from app.pipeline.sentences import build_sections, citing_sentences

log = logging.getLogger(__name__)

TEI = "{http://www.tei-c.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
GROBID_TIMEOUT = 180.0


class GrobidError(Exception):
    pass


async def grobid_parse(data: bytes, base_url: str) -> tuple[Paper, list]:
    url = base_url.rstrip("/") + "/api/processFulltextDocument"
    try:
        async with httpx.AsyncClient(timeout=GROBID_TIMEOUT) as client:
            resp = await client.post(
                url,
                files={"input": ("paper.pdf", data, "application/pdf")},
                data={"includeRawCitations": "1", "consolidateHeader": "0", "consolidateCitations": "0"},
            )
    except httpx.HTTPError as exc:
        raise GrobidError(f"GROBID request failed: {exc}") from exc
    if resp.status_code != 200:
        raise GrobidError(f"GROBID returned HTTP {resp.status_code}")
    return parse_tei(resp.content)


def _text(el: Optional[ET.Element]) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def _person(pers: ET.Element) -> tuple[str, str]:
    forenames = " ".join(_text(f) for f in pers.findall(f"{TEI}forename"))
    surname = _text(pers.find(f"{TEI}surname"))
    return f"{forenames} {surname}".strip(), surname


class _Writer:
    """Builds the plain text, recording citation markers and headings with offsets."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.pos = 0
        self.markers: list[Marker] = []
        self.headings: list[Heading] = []
        self.citations_seen = 0
        self.citations_mapped = 0
        self.marker_texts: dict[str, list[str]] = {}

    def write(self, s: str, raw: bool = False) -> None:
        if not raw:
            s = re.sub(r"\s+", " ", s)  # TEI keeps the PDF's line breaks inside paragraphs
        self.parts.append(s)
        self.pos += len(s)

    def paragraph(self, el: ET.Element) -> None:
        if self.parts:
            self.write("\n\n", raw=True)
        self._walk(el)

    def _walk(self, el: ET.Element) -> None:
        if el.text:
            self.write(el.text)
        for child in el:
            tag = child.tag.replace(TEI, "")
            if tag == "ref" and child.get("type") == "bibr":
                label = _text(child)
                start = self.pos
                self.write(label)
                self.citations_seen += 1
                target = (child.get("target") or "").lstrip("#")
                if target:
                    self.citations_mapped += 1
                    self.markers.append(Marker(start, self.pos, [target]))
                    self.marker_texts.setdefault(target, []).append(label)
            elif tag in ("formula", "figure", "table", "note"):
                self.write(" ")
            else:
                self._walk(child)
            if child.tail:
                self.write(child.tail)

    def heading(self, title: str) -> None:
        if self.parts:
            self.write("\n\n", raw=True)
        self.headings.append(Heading(title, self.pos, 1))
        self.write(title)

    @property
    def text(self) -> str:
        return "".join(self.parts)


def parse_tei(xml: bytes) -> tuple[Paper, list]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise GrobidError(f"GROBID returned invalid XML: {exc}") from exc

    header = root.find(f".//{TEI}teiHeader")
    title = _text(header.find(f".//{TEI}titleStmt/{TEI}title")) if header is not None else ""
    authors = []
    if header is not None:
        for pers in header.findall(f".//{TEI}sourceDesc//{TEI}analytic/{TEI}author/{TEI}persName"):
            authors.append(_person(pers)[0])
    abstract = _text(root.find(f".//{TEI}profileDesc/{TEI}abstract"))

    w = _Writer()
    if abstract:
        w.heading("Abstract")
        for p in root.findall(f".//{TEI}profileDesc/{TEI}abstract//{TEI}p"):
            w.paragraph(p)
    body = root.find(f".//{TEI}text/{TEI}body")
    divs = list(body.findall(f"{TEI}div")) if body is not None else []
    back = root.find(f".//{TEI}text/{TEI}back")
    if back is not None:
        divs += [d for d in back.findall(f"{TEI}div") if d.get("type") in ("annex", "acknowledgement")]
    for div in divs:
        for child in div:
            tag = child.tag.replace(TEI, "")
            if tag == "head":
                number = child.get("n")
                w.heading(f"{number} {_text(child)}".strip() if number else _text(child))
            elif tag == "p":
                w.paragraph(child)
            elif tag == "div":  # annexes nest their sections
                for sub in child:
                    if sub.tag == f"{TEI}head":
                        w.heading(_text(sub))
                    elif sub.tag == f"{TEI}p":
                        w.paragraph(sub)

    numeric = _is_numeric(w.marker_texts)
    references = []
    for i, entry in enumerate(root.findall(f".//{TEI}listBibl/{TEI}biblStruct"), start=1):
        key = entry.get(XML_ID) or f"b{i - 1}"
        analytic = entry.find(f"{TEI}analytic")
        monogr = entry.find(f"{TEI}monogr")
        ref_title = _text(analytic.find(f"{TEI}title")) if analytic is not None else ""
        ref_title = ref_title or (_text(monogr.find(f"{TEI}title")) if monogr is not None else "")
        people = [_person(p) for p in entry.findall(f".//{TEI}author/{TEI}persName")]
        date = entry.find(f".//{TEI}imprint/{TEI}date")
        when = (date.get("when") if date is not None else None) or ""
        year = int(when[:4]) if when[:4].isdigit() else None
        raw = _text(entry.find(f"{TEI}note[@type='raw_reference']"))
        doi = _text(entry.find(f".//{TEI}idno[@type='DOI']")) or bib.find_doi(raw)
        arxiv = _text(entry.find(f".//{TEI}idno[@type='arXiv']"))
        arxiv = bib.find_arxiv_id(arxiv, raw) or (arxiv or None)
        surnames = [s for _, s in people if s]
        if numeric:
            label = _number_label(w.marker_texts.get(key, [])) or str(i)
        else:
            short = surnames[0] if len(surnames) == 1 else (
                f"{surnames[0]} and {surnames[1]}" if len(surnames) == 2 else f"{surnames[0]} et al." if surnames else "Anonymous")
            label = f"{short}, {year or 'n.d.'}"
        references.append(Reference(
            key=key, label=label, raw=raw or ref_title, title=ref_title or None, authors=[n for n, _ in people],
            year=year, doi=doi or None, arxiv_id=arxiv,
        ))

    text = w.text
    known = {r.key for r in references}
    markers = [Marker(m.start, m.end, [k for k in m.keys if k in known]) for m in w.markers]
    markers = [m for m in markers if m.keys]
    sections = build_sections(text, w.headings)
    sentences = citing_sentences(text, markers, sections, known)
    cited = {k for m in markers for k in m.keys}

    notes = []
    if w.citations_seen and w.citations_mapped < 0.5 * w.citations_seen:
        notes.append(
            f"Only {w.citations_mapped} of {w.citations_seen} in-text citations could be matched to entries in the "
            "reference list, so this report covers part of the paper's citations."
        )
    paper = Paper(
        id="", title=title, authors=authors, abstract=abstract, source="pdf", parsed_from="grobid",
        citation_style="numeric" if numeric else "author_year", text=text, sections=sections,
        references=[r for r in references if r.key in cited] or references,
        citation_keys_total=w.citations_seen, citation_keys_mapped=w.citations_mapped, notes=notes,
    )
    return paper, sentences


def _is_numeric(marker_texts: dict[str, list[str]]) -> bool:
    texts = [t for ts in marker_texts.values() for t in ts]
    if not texts:
        return True
    numeric = sum(bool(re.fullmatch(r"[\[(]?\s*\d+[\s,\-–\]);]*", t)) for t in texts)
    return numeric >= 0.5 * len(texts)


def _number_label(texts: list[str]) -> Optional[str]:
    for t in texts:
        m = re.fullmatch(r"[\[(]?\s*(\d+)[\s,\-–\]);]*", t)
        if m:
            return m.group(1)
    return None
