"""Ingest: arXiv ID -> LaTeX source -> Paper (plain text with offsets), references, citing sentences.

PDF ingestion (uploads, and arXiv papers without LaTeX source) arrives in a later milestone;
for now `NoLatexSource` is raised with the PDF bytes attached.
"""

from __future__ import annotations

import gzip
import io
import logging
import re
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

from app.http import fetch_cached
from app.models import IngestResult, Paper, Reference
from app.pipeline import bibliography as bib
from app.pipeline.latex import (
    CITE_MARKER_COMMANDS,
    LatexConverter,
    LatexProject,
    collect_macros,
    decode,
    expand_macros,
    latex_to_plain,
    split_document,
)
from app.pipeline.sentences import build_sections, citing_sentences

log = logging.getLogger(__name__)

SOURCE_EXTENSIONS = (".tex", ".bbl", ".bib", ".sty", ".ltx")
MAX_SOURCE_FILE_BYTES = 5 * 1024 * 1024
ATOM = "{http://www.w3.org/2005/Atom}"


class IngestError(Exception):
    """A problem with the input that should be shown to the user."""


class InvalidArxivId(IngestError):
    pass


class NoLatexSource(IngestError):
    def __init__(self, message: str, pdf: Optional[bytes] = None):
        super().__init__(message)
        self.pdf = pdf


# ----------------------------------------------------------------------------
# arXiv IDs
# ----------------------------------------------------------------------------

_NEW_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
_OLD_ID_RE = re.compile(r"^([a-z\-]+(?:\.[a-z]{2})?)/(\d{7})(v\d+)?$", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|html|format|e-print)/(.+?)(?:\.pdf)?/?(?:[?#].*)?$", re.IGNORECASE
)


def normalize_arxiv_id(raw: str) -> str:
    """Accept bare IDs (with/without version), old-style IDs and arxiv.org URLs.

    Returns the canonical ID, keeping the version suffix if one was given.
    """
    s = (raw or "").strip().strip("<>\"'()[]{},;. \t").strip()
    m = _ARXIV_URL_RE.search(s)
    if m:
        s = m.group(1)
    s = re.sub(r"^arxiv:\s*", "", s, flags=re.IGNORECASE).strip().rstrip(".")
    if s.lower().endswith(".pdf"):
        s = s[:-4]
    m = _NEW_ID_RE.match(s)
    if m:
        return m.group(1) + (m.group(2) or "")
    m = _OLD_ID_RE.match(s)
    if m:
        archive, _, subject = m.group(1).partition(".")
        archive = archive.lower() + (f".{subject.upper()}" if subject else "")
        return f"{archive}/{m.group(2)}{m.group(3) or ''}"
    raise InvalidArxivId(
        f"“{raw.strip()[:80]}” doesn't look like an arXiv ID. "
        "Try something like 2401.12345, hep-th/9901001, or an arxiv.org/abs/ link."
    )


def strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


# ----------------------------------------------------------------------------
# arXiv downloads
# ----------------------------------------------------------------------------


@dataclass
class ArxivMeta:
    id: str  # with version
    title: str
    authors: list[str]
    abstract: str


ARXIV_BATCH = 50


def _clean_ws(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def parse_arxiv_feed(data: bytes) -> dict[str, ArxivMeta]:
    """Atom feed -> {unversioned id: ArxivMeta}."""
    found = {}
    for entry in ET.fromstring(data).findall(f"{ATOM}entry"):
        entry_id = entry.findtext(f"{ATOM}id", "")
        if "arxiv.org/abs/" not in entry_id:
            continue  # error entries for unknown IDs
        versioned = entry_id.split("arxiv.org/abs/", 1)[1]
        found[strip_version(versioned)] = ArxivMeta(
            id=versioned,
            title=_clean_ws(entry.findtext(f"{ATOM}title")),
            authors=[_clean_ws(a.findtext(f"{ATOM}name")) for a in entry.findall(f"{ATOM}author")],
            abstract=_clean_ws(entry.findtext(f"{ATOM}summary")),
        )
    return found


async def fetch_arxiv_metadata_batch(arxiv_ids: list[str]) -> dict[str, ArxivMeta]:
    """Metadata for many papers in as few API calls as possible. Keys are unversioned IDs."""
    ids = sorted({strip_version(i) for i in arxiv_ids})
    found: dict[str, ArxivMeta] = {}
    for start in range(0, len(ids), ARXIV_BATCH):
        batch = ids[start : start + ARXIV_BATCH]
        data = await fetch_cached(
            "https://export.arxiv.org/api/query",
            "arxiv_meta",
            params={"id_list": ",".join(batch), "max_results": str(len(batch))},
            suffix=".xml",
        )
        found.update(parse_arxiv_feed(data))
    return found


async def fetch_arxiv_metadata(arxiv_id: str) -> ArxivMeta:
    data = await fetch_cached(
        "https://export.arxiv.org/api/query", "arxiv_meta", params={"id_list": arxiv_id}, suffix=".xml"
    )
    found = parse_arxiv_feed(data)
    meta = found.get(strip_version(arxiv_id))
    if meta is None:
        raise InvalidArxivId(f"arXiv has no paper with ID {arxiv_id}.")
    return meta


def unpack_source(data: bytes) -> dict[str, str]:
    """arXiv e-print bytes -> {relative path: text}. Raises NoLatexSource for PDF-only papers."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if data[:5] == b"%PDF-":
        raise NoLatexSource("arXiv has no LaTeX source for this paper (PDF only).", pdf=data)
    try:
        files = {}
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            for member in tar.getmembers():
                if not member.isfile() or member.size > MAX_SOURCE_FILE_BYTES:
                    continue
                if not member.name.lower().endswith(SOURCE_EXTENSIONS):
                    continue
                handle = tar.extractfile(member)
                if handle:
                    files[member.name] = decode(handle.read())
        return files
    except tarfile.ReadError:
        return {"main.tex": decode(data)}  # a single gzipped .tex file


# ----------------------------------------------------------------------------
# LaTeX project -> Paper
# ----------------------------------------------------------------------------

_CITE_PRESCAN_RE = re.compile(
    r"\\(" + "|".join(sorted(CITE_MARKER_COMMANDS, key=len, reverse=True)) + r")\*?"
    r"(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]*)\}"
)
_TEXTUAL_CITES = {"citet", "Citet", "textcite", "Textcite", "textcites", "citeA", "citeN", "newcite", "Newcite"}
_BARE_CITES = {"citealp", "Citealp", "citealt", "Citealt", "citenum"}
_AUTHOR_YEAR_BST = ("nat", "apa", "acl", "iclr", "icml", "named", "chicago", "harvard", "agsm", "colm", "tmlr")


@dataclass
class _Bib:
    entries: dict[str, bib.BibEntry]
    order: dict[str, int]  # key -> printed number
    author_year: bool


def _cited_keys_in_order(body: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in _CITE_PRESCAN_RE.finditer(body):
        for key in m.group(2).split(","):
            if key.strip():
                seen.setdefault(key.strip(), None)
    return list(seen)


def _load_bibliography(project: LatexProject, preamble: str, body: str, cited: list[str]) -> _Bib:
    entries: dict[str, bib.BibEntry] = {}
    natbib_entries = 0
    biblatex = False

    def add(items: list[bib.BibEntry]) -> None:
        for item in items:
            entries.setdefault(item.key, item)

    if "thebibliography" in body:
        items = bib.parse_bibitems(body)
        natbib_entries += sum(e.has_author_year_label for e in items)
        add(items)
    for name, text in project.files.items():
        if not name.endswith(".bbl"):
            continue
        if "\\entry{" in text:
            biblatex = True
            add(bib.parse_biblatex_bbl(text))
        else:
            items = bib.parse_bibitems(text)
            natbib_entries += sum(e.has_author_year_label for e in items)
            add(items)

    printed = list(entries)
    missing = [k for k in cited if k not in entries]
    if missing:
        bib_entries: dict[str, bib.BibEntry] = {}
        for name, text in project.files.items():
            if name.endswith(".bib"):
                for item in bib.parse_bibtex(text):
                    bib_entries.setdefault(item.key, item)
        for key in missing:
            if key in bib_entries:
                entries[key] = bib_entries[key]
                printed.append(key)

    # Number cited entries in printed order (bbl order, then first-citation order).
    cited_set = set(cited)
    order = {key: n for n, key in enumerate((k for k in printed if k in cited_set), start=1)}

    natbib_numbers = re.search(r"\\usepackage\s*\[[^\]]*\bnumbers\b[^\]]*\]\s*\{natbib\}", preamble)
    biblatex_style = re.search(r"\\usepackage\s*\[([^\]]*)\]\s*\{biblatex\}", preamble)
    bst = re.search(r"\\bibliographystyle\s*\{([^}]*)\}", preamble + body)
    if natbib_numbers:
        author_year = False
    elif biblatex:
        author_year = bool(biblatex_style and re.search(r"style\s*=\s*(authoryear|apa)", biblatex_style.group(1)))
    elif natbib_entries:
        author_year = natbib_entries >= 0.5 * max(1, len(printed))
    else:
        author_year = bool(bst and any(h in bst.group(1).lower() for h in _AUTHOR_YEAR_BST))
    return _Bib(entries, order, author_year)


def _make_renderers(b: _Bib):
    def ay(entry: Optional[bib.BibEntry]) -> tuple[str, str]:
        if not entry:
            return "?", "?"
        year = entry.year_label or (str(entry.year) if entry.year else "n.d.")
        return entry.short_author or "Anonymous", year

    def render(cmd: str, opts: list[str], keys: list[str]) -> str:
        prefix, postfix = ("", opts[0]) if len(opts) == 1 else (opts[0], opts[1]) if len(opts) >= 2 else ("", "")
        prefix, postfix = latex_to_plain(prefix), latex_to_plain(postfix)
        entries = [b.entries.get(k) for k in keys]
        # natbib's bare \cite is textual, but ACL-family styles (and most authors' intent)
        # make it parenthetical, so treat it as \citep.
        textual = cmd in _TEXTUAL_CITES

        if b.author_year:
            pairs = [ay(e) for e in entries]
            if textual:
                parts = [f"{a} ({y})" for a, y in pairs]
                if postfix:
                    parts[-1] = parts[-1][:-1] + f", {postfix})"
                return "; ".join(parts)
            sep = " " if cmd in ("citealt", "Citealt") else ", "
            inner = "; ".join(f"{a}{sep}{y}" for a, y in pairs)
            inner = " ".join(p for p in (prefix, inner) if p) + (f", {postfix}" if postfix else "")
            return inner if cmd in _BARE_CITES else f"({inner})"

        numbers = [str(b.order[k]) if k in b.order else "?" for k in keys]
        if textual:
            return "; ".join(
                f"{(e.short_author + ' ') if e and e.short_author else ''}[{n}]" for e, n in zip(entries, numbers)
            )
        inner = ", ".join(numbers)
        inner = " ".join(p for p in (prefix, inner) if p) + (f", {postfix}" if postfix else "")
        return f"[{inner}]"

    def cite_text(cmd: str, keys: list[str]) -> str:
        entries = [b.entries.get(k) for k in keys]
        if cmd in ("citeauthor", "Citeauthor"):
            return ", ".join(ay(e)[0] for e in entries)
        if cmd == "citeyear":
            return ", ".join(ay(e)[1] for e in entries)
        if cmd == "citeyearpar":
            return "(" + ", ".join(ay(e)[1] for e in entries) + ")"
        return ", ".join((e.title or "") for e in entries if e)

    return render, cite_text


def parse_latex_project(files: dict[str, str], keep_captions: bool = False) -> tuple[Paper, list]:
    """Pure function (no network): LaTeX sources -> (Paper without metadata, citing sentences)."""
    project = LatexProject(files)
    main = project.main_file()
    if not main:
        raise IngestError("Could not find the main .tex file in the paper's source.")

    preamble, body = split_document(project.expand(main))
    macros = collect_macros(project.macro_sources())
    body = expand_macros(body, macros)

    cited = _cited_keys_in_order(body)
    bibliography = _load_bibliography(project, preamble, body, cited)
    render, cite_text = _make_renderers(bibliography)
    result = LatexConverter(render, cite_text, keep_captions=keep_captions).convert(body)

    sections = build_sections(result.text, result.headings)
    mapped = [k for k in cited if k in bibliography.entries]
    sentences = citing_sentences(result.text, result.markers, sections, set(mapped))

    references = []
    for key in sorted(mapped, key=lambda k: bibliography.order.get(k, 10**6)):
        e = bibliography.entries[key]
        if bibliography.author_year:
            label = f"{e.short_author or 'Anonymous'}, {e.year_label or e.year or 'n.d.'}"
        else:
            label = str(bibliography.order.get(key, "?"))
        references.append(
            Reference(
                key=key,
                label=label,
                raw=e.raw,
                title=e.title,
                authors=e.authors,
                year=e.year,
                doi=e.doi,
                arxiv_id=e.arxiv_id,
                url=e.url,
            )
        )

    notes = []
    if cited and len(mapped) < 0.5 * len(cited):
        notes.append(
            f"Only {len(mapped)} of {len(cited)} cited works could be matched to bibliography entries, "
            "so this report covers part of the paper's citations."
        )
    if not cited:
        notes.append("No citations were found in this paper's source.")

    paper = Paper(
        id="",
        source="arxiv",
        parsed_from="latex",
        citation_style="author_year" if bibliography.author_year else "numeric",
        text=result.text,
        sections=sections,
        references=references,
        citation_keys_total=len(cited),
        citation_keys_mapped=len(mapped),
        notes=notes,
    )
    return paper, sentences


async def ingest_arxiv(raw_id: str) -> IngestResult:
    arxiv_id = normalize_arxiv_id(raw_id)
    meta = await fetch_arxiv_metadata(arxiv_id)
    versioned = arxiv_id if re.search(r"v\d+$", arxiv_id) else meta.id
    data = await fetch_cached(f"https://arxiv.org/e-print/{versioned}", "arxiv_src", suffix=".eprint")
    files = unpack_source(data)
    log.info("arXiv %s: %d source files", versioned, len(files))

    paper, sentences = parse_latex_project(files)
    paper.id = f"arxiv:{versioned}"
    paper.arxiv_id = versioned
    paper.title = meta.title
    paper.authors = meta.authors
    paper.abstract = meta.abstract
    return IngestResult(paper=paper, sentences=sentences)
