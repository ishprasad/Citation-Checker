"""Bibliography parsing: thebibliography/\\bibitem (natbib or numeric), biblatex .bbl, and .bib.

A compiled .bbl is preferred because its keys and order match exactly what the
paper printed. .bib files only fill in keys the .bbl does not cover.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.pipeline.latex import latex_to_plain, read_bracket, read_group

ARXIV_IN_TEXT_RE = re.compile(
    r"(?:arxiv[:\s]*|arxiv\.org/(?:abs|pdf)/|abs/|10\.48550/arxiv\.)"
    r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;{}\"<>]+)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)[a-z]?\b")
NATBIB_LABEL_RE = re.compile(r"^(.*?)\(\s*((?:19|20)\d{2}[a-z]?|n\.\s?d\.)\s*\)(.*)$", re.DOTALL)


@dataclass
class BibEntry:
    key: str
    raw: str  # plain-text rendering of the entry
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    year_label: Optional[str] = None  # e.g. "2020a" as printed
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    short_author: Optional[str] = None  # "Vaswani et al."
    has_author_year_label: bool = False


def find_arxiv_id(*texts: Optional[str]) -> Optional[str]:
    for text in texts:
        if text:
            m = ARXIV_IN_TEXT_RE.search(text)
            if m:
                return m.group(1)
    return None


def find_doi(*texts: Optional[str]) -> Optional[str]:
    for text in texts:
        if text:
            m = DOI_RE.search(text)
            if m:
                doi = m.group(1).rstrip(".")
                if doi.lower().startswith("10.48550/arxiv."):
                    continue  # an arXiv DOI; the arXiv ID is more useful
                return doi
    return None


def _year(text: str) -> Optional[int]:
    years = [int(m.group(1)) for m in YEAR_RE.finditer(text)]
    return years[-1] if years else None


def _short_from_names(families: list[str]) -> Optional[str]:
    families = [f for f in families if f]
    if not families:
        return None
    if len(families) == 1:
        return families[0]
    if len(families) == 2:
        return f"{families[0]} and {families[1]}"
    return f"{families[0]} et al."


def _family_name(full_name: str) -> str:
    name = full_name.strip()
    if "," in name:  # "Vaswani, Ashish"
        return name.split(",", 1)[0].strip()
    parts = name.split()
    return parts[-1] if parts else ""


def _split_author_list(text: str) -> list[str]:
    text = re.sub(r"\bet\.? al\.?", "", text)
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", text)
    return [p.strip(" .") for p in parts if p.strip(" .")]


# ----------------------------------------------------------------------------
# \bibitem entries
# ----------------------------------------------------------------------------

_BIBITEM_RE = re.compile(r"\\bibitem\b")
_BIB_END_RE = re.compile(r"\\end\s*\{thebibliography\}")
_URL_CMD_RE = re.compile(r"\\(?:url|href)\s*\{([^}]*)\}")


def parse_bibitems(src: str) -> list[BibEntry]:
    entries = []
    starts = [m.start() for m in _BIBITEM_RE.finditer(src)]
    for idx, start in enumerate(starts):
        i = start + len("\\bibitem")
        while i < len(src) and src[i].isspace():
            i += 1
        label = None
        if i < len(src) and src[i] == "[":
            label, i = read_bracket(src, i)
        while i < len(src) and src[i].isspace():
            i += 1
        if i >= len(src) or src[i] != "{":
            continue
        key, i = read_group(src, i)
        stop = starts[idx + 1] if idx + 1 < len(starts) else len(src)
        end = _BIB_END_RE.search(src, i, stop)
        body = src[i : end.start() if end else stop]
        entries.append(_bibitem_entry(key.strip(), label, body))
    return entries


def _bibitem_entry(key: str, label: Optional[str], body: str) -> BibEntry:
    blocks = [b for b in re.split(r"\\newblock\b", body) if b.strip()]
    plain_blocks = [latex_to_plain(b) for b in blocks]
    raw = " ".join(plain_blocks)
    url_match = _URL_CMD_RE.search(body)

    entry = BibEntry(
        key=key,
        raw=raw,
        doi=find_doi(body, raw),
        arxiv_id=find_arxiv_id(body, raw),
        url=url_match.group(1).strip() if url_match else None,
    )
    if len(plain_blocks) >= 2:
        entry.title = plain_blocks[1].strip().rstrip(".").strip() or None
        author_text = re.sub(r"\(?\b(?:19|20)\d{2}[a-z]?\)?\.?\s*$", "", plain_blocks[0]).strip()
        entry.authors = _split_author_list(author_text)

    if label:
        plain_label = latex_to_plain(label.strip().strip("{}"))
        m = NATBIB_LABEL_RE.match(plain_label)
        if m and m.group(1).strip():
            entry.short_author = m.group(1).strip().rstrip(",")
            entry.year_label = m.group(2).replace(" ", "")
            entry.has_author_year_label = True
            if entry.year_label[:4].isdigit():
                entry.year = int(entry.year_label[:4])
    if entry.year is None:
        entry.year = _year(raw)
    if not entry.short_author:
        entry.short_author = _short_from_names([_family_name(a) for a in entry.authors])
    return entry


# ----------------------------------------------------------------------------
# biblatex .bbl
# ----------------------------------------------------------------------------

_ENTRY_RE = re.compile(r"\\entry\{([^}]+)\}\{([^}]*)\}\{[^}]*\}")
_FIELD_RE = re.compile(r"\\field\{(\w+)\}(?=\{)")
_VERB_RE = re.compile(r"\\verb\{(\w+)\}\s*\\verb\s?(.*?)\s*\\endverb", re.DOTALL)
_NAME_RE = re.compile(r"\\name\{(\w+)\}\{\d+\}\{[^}]*\}(?=\{)")
_FAMILY_RE = re.compile(r"family=\{((?:[^{}]|\{[^{}]*\})*)\}")
_GIVEN_RE = re.compile(r"given=\{((?:[^{}]|\{[^{}]*\})*)\}")


def parse_biblatex_bbl(src: str) -> list[BibEntry]:
    entries = []
    for m in _ENTRY_RE.finditer(src):
        end = src.find("\\endentry", m.end())
        body = src[m.end() : end if end != -1 else len(src)]
        fields: dict[str, str] = {}
        for f in _FIELD_RE.finditer(body):
            value, _ = read_group(body, f.end())
            fields[f.group(1).lower()] = value
        for v in _VERB_RE.finditer(body):
            fields.setdefault(v.group(1).lower(), v.group(2).strip())

        families: list[str] = []
        full_names: list[str] = []
        for n in _NAME_RE.finditer(body):
            if n.group(1) != "author":
                continue
            names_block, _ = read_group(body, n.end())
            for person in re.split(r"\{\{hash=", names_block)[1:] or [names_block]:
                fam = _FAMILY_RE.search(person)
                giv = _GIVEN_RE.search(person)
                if fam:
                    family = latex_to_plain(fam.group(1))
                    families.append(family)
                    given = latex_to_plain(giv.group(1)) if giv else ""
                    full_names.append(f"{given} {family}".strip())

        title = latex_to_plain(fields.get("title", "")) or None
        venue = latex_to_plain(fields.get("journaltitle", "") or fields.get("booktitle", ""))
        year_text = fields.get("year") or fields.get("date", "")[:4]
        eprint = fields.get("eprint", "")
        arxiv_id = None
        if eprint and fields.get("eprinttype", "").lower() in ("arxiv", "") and re.match(r"^[\w.\-/]+$", eprint):
            arxiv_id = re.sub(r"v\d+$", "", eprint)
        raw = ". ".join(p for p in [", ".join(full_names), title or "", venue, year_text] if p) + "."
        entries.append(
            BibEntry(
                key=m.group(1).strip(),
                raw=raw,
                title=title,
                authors=full_names,
                year=int(year_text) if year_text[:4].isdigit() else None,
                year_label=(year_text + latex_to_plain(fields.get("extradate", "")) if year_text else None),
                doi=fields.get("doi") or find_doi(fields.get("url")),
                arxiv_id=arxiv_id or find_arxiv_id(fields.get("url"), raw),
                url=fields.get("url"),
                short_author=_short_from_names(families),
            )
        )
    return entries


# ----------------------------------------------------------------------------
# BibTeX
# ----------------------------------------------------------------------------

_BIB_START_RE = re.compile(r"@\s*(\w+)\s*\{")
_BIB_FIELD_RE = re.compile(r"\s*,?\s*([A-Za-z][\w\-:.]*)\s*=\s*")


def _bib_value(body: str, i: int) -> tuple[str, int]:
    parts = []
    while i < len(body):
        while i < len(body) and body[i].isspace():
            i += 1
        if i >= len(body):
            break
        c = body[i]
        if c == "{":
            value, i = read_group(body, i)
            parts.append(value)
        elif c == '"':
            j, depth = i + 1, 0
            while j < len(body):
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                elif body[j] == '"' and depth == 0 and body[j - 1] != "\\":
                    break
                j += 1
            parts.append(body[i + 1 : j])
            i = j + 1
        else:
            m = re.match(r"[^,#}\s]+", body[i:])
            if not m:
                break
            parts.append(m.group(0))
            i += len(m.group(0))
        while i < len(body) and body[i].isspace():
            i += 1
        if i < len(body) and body[i] == "#":
            i += 1
            continue
        break
    return "".join(parts), i


def parse_bibtex(src: str) -> list[BibEntry]:
    entries = []
    pos = 0
    while True:
        m = _BIB_START_RE.search(src, pos)
        if not m:
            break
        kind = m.group(1).lower()
        body, pos = read_group(src, m.end() - 1)
        if kind in ("comment", "preamble", "string"):
            continue
        key, _, rest = body.partition(",")
        fields: dict[str, str] = {}
        i = 0
        while i < len(rest):
            fm = _BIB_FIELD_RE.match(rest, i)
            if not fm:
                break
            value, i = _bib_value(rest, fm.end())
            fields[fm.group(1).lower()] = value
        entries.append(_bibtex_entry(key.strip(), fields))
    return entries


def _bibtex_entry(key: str, fields: dict[str, str]) -> BibEntry:
    authors = [latex_to_plain(a) for a in re.split(r"\s+and\s+", fields.get("author", "")) if a.strip()]
    authors = [a for a in authors if a.lower() != "others"]
    title = latex_to_plain(fields.get("title", "")) or None
    venue = latex_to_plain(fields.get("journal", "") or fields.get("booktitle", "") or fields.get("publisher", ""))
    year_text = latex_to_plain(fields.get("year", ""))
    eprint = fields.get("eprint", "").strip()
    arxiv_id = None
    if eprint and fields.get("archiveprefix", "arxiv").lower() == "arxiv":
        arxiv_id = re.sub(r"v\d+$", "", eprint)
    display_names = [_display_name(a) for a in authors]
    raw = ". ".join(p for p in [", ".join(display_names), title or "", venue, year_text] if p) + "."
    return BibEntry(
        key=key,
        raw=raw,
        title=title,
        authors=display_names,
        year=int(year_text[:4]) if year_text[:4].isdigit() else None,
        year_label=year_text or None,
        doi=(fields.get("doi") or "").strip() or find_doi(fields.get("url")),
        arxiv_id=arxiv_id or find_arxiv_id(fields.get("url"), fields.get("journal"), raw),
        url=(fields.get("url") or "").strip() or None,
        short_author=_short_from_names([_family_name(a) for a in authors]),
    )


def _display_name(name: str) -> str:
    if "," in name:
        family, given = name.split(",", 1)
        return f"{given.strip()} {family.strip()}".strip()
    return name
