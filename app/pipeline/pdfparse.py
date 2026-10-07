"""Citing-paper ingestion from PDF (uploads, and arXiv papers without LaTeX source).

PDF has no citation keys, so everything is reconstructed from layout:

1. PyMuPDF layout blocks -> paragraphs and headings. Running headers, page numbers,
   figure/table text, captions and footnotes are dropped by font size and shape.
2. The reference list (after the last "References" heading) is split into entries,
   using printed labels ("[12]", "12.") or the hanging indent of author-year styles.
3. Each entry is parsed into authors, title, year, DOI and arXiv ID with heuristics
   that cover the common styles (ACL/NeurIPS natbib, IEEE, APA, LNCS).
4. In-text markers are found by style: bracketed numbers, or author-year mentions
   (parenthetical "(Smith et al., 2020; Lee, 2019a)" and textual "Smith et al. (2020)"),
   and mapped to entries.

The result has the same shape as the LaTeX path (Paper + CitingSentence), so the rest of
the pipeline does not care where a paper came from. When fewer than half of the in-text
citations map to an entry, the paper gets a note that says so.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Optional

import pymupdf

from app.models import Paper, Reference
from app.pipeline import bibliography as bib
from app.pipeline.latex import Heading, Marker
from app.pipeline.pdftext import LIGATURES, MIN_TEXT_CHARS
from app.pipeline.sentences import build_sections, citing_sentences


class PdfParseError(Exception):
    """The PDF could not be turned into text (encrypted, broken, or scanned)."""


_LIGATURE_RE = re.compile("|".join(LIGATURES))
_ARXIV_STAMP_RE = re.compile(r"^arXiv:\S+\s+\[[^\]]+\]\s+\d{1,2}\s+\w{3}\s+\d{4}$")
_REFERENCES_RE = re.compile(
    r"^\s*(?:\d+\.?\s*|[A-Z]\.?\s+|[IVX]+\.\s*)?(references|bibliography|literature cited|works cited|reference list)\s*$",
    re.IGNORECASE,
)
# Back matter that never needs checking; skipped up to the next heading.
_BACK_MATTER_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(?:author contributions?|contributions|competing interests?|conflicts? of interests?|"
    r"declaration of (?:competing )?interests?|funding(?: statement)?|data availability(?: statement)?|"
    r"code availability|ethics(?: approval| statement)?|supporting information|supplementary (?:information|material)|"
    r"additional information|reporting summary|peer review information|publisher'?’?s note|open access)\b",
    re.IGNORECASE,
)
_CAPTION_RE = re.compile(
    r"^(?:(?:Figure|Fig\.?|Table|Algorithm|Listing)\s*[A-Z]?\d+[.:]"
    r"|S\d+\s+(?:Fig|Figure|Table|File|Text|Appendix|Dataset|Data)\b)",  # supporting information items
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)*\.?|[IVX]+\.)\s+\S")
_ABSTRACT_RE = re.compile(r"^\s*abstract\b", re.IGNORECASE)
_TERMINAL_RE = re.compile(r"[.!?:][\"”’)\]]*$")
_MISSING_SPACE_RE = re.compile(r"(?<=[a-z\d)\]][.!?])(?=[A-Z][a-z])")
_GLUED_ET_AL_RE = re.compile(r"(?<=[a-z]{3})et al\.")
# Line-end "self-" + "attention" is a compound, not a broken word.
_HYPHEN_PREFIXES = {
    "self", "non", "co", "cross", "multi", "semi", "well", "fine", "task", "zero", "few", "one", "two",
    "three", "high", "low", "large", "small", "long", "short", "open", "end", "state", "real", "data",
    "human", "domain", "anti", "inter", "intra", "out", "full", "half", "single", "double",
}
# Short words that end a hyphenated compound's left part ("chain-of-", "end-to-", "out-of-").
_COMPOUND_WORDS = {"of", "the", "to", "in", "on", "by", "for", "and", "at", "a", "an", "or", "per", "vs"}
_HYPHEN_SUFFIXES = {
    "based", "oriented", "specific", "level", "scale", "related", "like", "free", "aware", "driven",
    "dependent", "independent", "wise", "term", "order", "making", "child", "art", "shot", "tuning",
    "trained", "training", "world", "time", "type", "only", "sized", "friendly", "agnostic", "aligned",
}


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------


@dataclass
class Line:
    text: str
    x0: float
    size: float  # dominant size
    bold: float  # fraction of characters in bold
    max_size: float = 0.0


@dataclass
class Block:
    page: int
    x0: float
    y0: float
    x1: float
    lines: list[Line]
    page_width: float
    in_table: bool = False
    in_margin: bool = False  # page header/footer band (journal masthead, running titles)

    @property
    def text(self) -> str:
        return join_lines([ln.text for ln in self.lines])

    @property
    def chars(self) -> int:
        return sum(len(ln.text) for ln in self.lines)

    @property
    def size(self) -> float:
        counts: Counter = Counter()
        for ln in self.lines:
            counts[round(ln.size, 1)] += len(ln.text)
        return counts.most_common(1)[0][0] if counts else 0.0

    @property
    def max_size(self) -> float:
        return max((ln.max_size for ln in self.lines), default=0.0)

    @property
    def bold(self) -> float:
        total = self.chars
        return sum(ln.bold * len(ln.text) for ln in self.lines) / total if total else 0.0

    @property
    def column(self) -> int:
        return 0 if self.x0 < self.page_width / 2 - 20 else 1


# LaTeX PDFs draw accents as separate glyphs before the letter ("Marasovi´c", "R´emi"); recombine them.
_SPACING_ACCENTS = {"´": "́", "¨": "̈", "ˆ": "̂", "˜": "̃", "ˇ": "̌"}
_ACCENT_BEFORE_RE = re.compile(r"([´¨ˆ˜ˇ])\s?([A-Za-z])")


def clean(text: str) -> str:
    text = _LIGATURE_RE.sub(lambda m: LIGATURES[m.group(0)], text)
    text = _ACCENT_BEFORE_RE.sub(lambda m: unicodedata.normalize("NFC", m.group(2) + _SPACING_ACCENTS[m.group(1)]), text)
    return text.replace("­", "").replace(" ", " ")


def join_lines(lines: list[str]) -> str:
    """Join visual lines into running text, repairing hyphenation at line ends.

    "trans-" + "duction" -> "transduction", but compounds keep their hyphen:
    "chain-of-" + "thought" and "state-" + "of-the-art".
    """
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        left = out.rsplit(" ", 1)[-1]
        right = line.split(" ", 1)[0]
        caps_word = left[:-1].isalpha() and left[:-1].isupper() and right[:1].isupper()  # "LAN-" + "GUAGE"
        if out.endswith("-") and (line[:1].islower() or caps_word):
            word = re.sub(r"\W+$", "", right).lower()
            # The fragment's last piece decides: "chain-of-" + "thought" keeps it, "e-ciga-" + "rettes" does not.
            piece = re.sub(r"^\W+", "", left[:-1]).rsplit("-", 1)[-1].lower()
            if ("-" in right or len(piece) == 1 or piece in _HYPHEN_PREFIXES or piece in _COMPOUND_WORDS
                    or word in _HYPHEN_SUFFIXES):  # "e-" + "cigarettes", "t-" + "test"
                out += line
            else:
                out = out[:-1] + line
        else:
            out += " " + line
    out = _MISSING_SPACE_RE.sub(" ", out)  # tight kerning drops spaces: "(Ouyang et al., 2022).The second"
    out = _GLUED_ET_AL_RE.sub(" et al.", out)  # "Rebufﬁet al." after a ligature
    return re.sub(r"\s+", " ", out)


# Superscript numbers are citations in some styles (Nature, many medical journals) and footnote
# marks in others. They are kept as placeholders until the citation style is known.
SUP_OPEN, SUP_CLOSE = "", ""
_SUP_NUMBERS_RE = re.compile(r"\s*\d{1,3}(?:\s*[,–-]\s*\d{1,3})*\s*")
_SUP_PLACEHOLDER_RE = re.compile(f"{SUP_OPEN}([^{SUP_CLOSE}]*){SUP_CLOSE}")


_SUP_PART_RE = re.compile(r"\s*[\d,–-]+\s*")


def _span_text(span: dict, line_size: float) -> str:
    text = span["text"]
    if span["flags"] & 1 and span["size"] < line_size - 0.5 and _SUP_PART_RE.fullmatch(text):
        return f"{SUP_OPEN}{text.strip()}{SUP_CLOSE}"  # "13", "–", "15" arrive as separate spans
    return text


def _merge_superscripts(text: str) -> str:
    text = text.replace(SUP_CLOSE + SUP_OPEN, "")
    # Placeholders that are not number lists (a lone "–") go back to plain text.
    return _SUP_PLACEHOLDER_RE.sub(lambda m: m.group(0) if _SUP_NUMBERS_RE.fullmatch(m.group(1)) else m.group(1), text)


# Blocks entirely within these bands of the page are headers and footers, not body text.
MARGIN_TOP = 0.075
MARGIN_BOTTOM = 0.94


def read_blocks(doc: pymupdf.Document) -> list[Block]:
    blocks = []
    for pno, page in enumerate(doc):
        tables = _table_rects(page)
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            lines = []
            for ln in b["lines"]:
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if not spans:
                    continue
                dominant = Counter({round(s["size"], 1): len(s["text"]) for s in spans}).most_common(1)[0][0]
                text = _merge_superscripts(clean("".join(_span_text(s, dominant) for s in ln["spans"])))
                n = sum(len(s["text"]) for s in spans) or 1
                bold = sum(len(s["text"]) for s in spans if s["flags"] & 16 or "Bold" in s.get("font", "")) / n
                size = Counter({round(s["size"], 1): len(s["text"]) for s in spans}).most_common(1)[0][0]
                lines.append(Line(text=text, x0=ln["bbox"][0], size=size, bold=bold,
                                  max_size=max(s["size"] for s in spans)))
            if lines:
                x0, y0, x1, y1 = b["bbox"]
                rect = pymupdf.Rect(b["bbox"])
                in_table = any(_overlap(rect, t) >= 0.8 for t in tables)
                height = page.rect.height
                in_margin = y1 < height * MARGIN_TOP or y0 > height * MARGIN_BOTTOM
                blocks.append(Block(page=pno, x0=x0, y0=y0, x1=x1, lines=lines, page_width=page.rect.width,
                                    in_table=in_table, in_margin=in_margin))
    return blocks


def _table_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Ruled or aligned tables on the page (their cells would otherwise read as short paragraphs)."""
    try:
        return [pymupdf.Rect(t.bbox) for t in page.find_tables().tables if t.row_count >= 2 and t.col_count >= 2]
    except Exception:  # noqa: BLE001 - table detection is best effort
        return []


def _overlap(rect: pymupdf.Rect, table: pymupdf.Rect) -> float:
    """Fraction of `rect` that lies inside `table`."""
    area = rect.get_area()
    return (rect & table).get_area() / area if area else 0.0


def body_font_size(blocks: list[Block]) -> float:
    counts: Counter = Counter()
    for b in blocks:
        for ln in b.lines:
            counts[round(ln.size, 1)] += len(ln.text)
    return counts.most_common(1)[0][0] if counts else 10.0


def _repeated_texts(blocks: list[Block]) -> set[str]:
    """Running headers/footers: short texts that recur on several pages."""
    pages_by_text: dict[str, set[int]] = {}
    for b in blocks:
        t = re.sub(r"\d+", "#", b.text.strip().lower())
        if len(t) <= 120:
            pages_by_text.setdefault(t, set()).add(b.page)
    return {t for t, pages in pages_by_text.items() if len(pages) >= 3}


def is_heading(b: Block, body_size: float) -> bool:
    text = b.text.strip()
    if not text or len(b.lines) > 2 or len(text) > 120 or b.size < body_size - 0.6:
        return False
    if _REFERENCES_RE.match(text) or (_ABSTRACT_RE.match(text) and len(text) < 12):
        return True
    if _CAPTION_RE.match(text) or text.endswith((",", ";")):
        return False
    letters = [c for c in text if c.isalpha()]
    small_caps = len(letters) >= 4 and sum(c.isupper() for c in letters) >= 0.8 * len(letters)  # ICLR style
    styled = b.bold >= 0.6 or b.size >= body_size + 1 or (b.max_size >= body_size + 1.5 and small_caps)
    if not styled:
        return False
    if text.endswith(".") and not _NUMBERED_HEADING_RE.match(text):
        return False
    return text[0].isupper() or text[0].isdigit()


def _is_table_like(b: Block) -> bool:
    if len(b.lines) < 3:
        return False
    avg = b.chars / len(b.lines)
    return avg < 28


# ----------------------------------------------------------------------------
# Document structure
# ----------------------------------------------------------------------------


@dataclass
class Layout:
    title: str
    paragraphs: list[tuple[str, str]]  # (kind, text) with kind "heading" or "para"
    reference_blocks: list[Block]
    body_size: float


def analyse(doc: pymupdf.Document) -> Layout:
    blocks = read_blocks(doc)
    if sum(b.chars for b in blocks) < MIN_TEXT_CHARS:
        raise PdfParseError("This PDF has almost no extractable text. It may be a scanned document.")
    body_size = body_font_size(blocks)
    repeated = _repeated_texts(blocks)
    blocks = [
        b for b in blocks
        if not _ARXIV_STAMP_RE.match(b.text.strip())
        and not b.text.strip().isdigit()
        and re.sub(r"\d+", "#", b.text.strip().lower()) not in repeated
    ]

    blocks = _split_leading_headings(blocks, body_size)

    # References: the last matching heading (outside the first pages); they end at the next
    # heading (an appendix, which can be longer than the paper), whose content is body text again.
    ref_start = ref_end = None
    for i in range(len(blocks) - 1, len(blocks) // 10 - 1, -1):
        if _REFERENCES_RE.match(blocks[i].text.strip()):
            ref_start = i
            break
    # Body text size from the text before the references: a long reference list in a smaller
    # font (common in journals) would otherwise win, letting sidebars and table cells through.
    if ref_start:
        body_size = body_font_size([b for b in blocks[:ref_start] if not b.in_table])

    title = _title(blocks, body_size)
    headings = [is_heading(b, body_size) and not b.in_table for b in blocks]
    if ref_start is not None:
        ref_end = len(blocks)
        for j in range(ref_start + 1, len(blocks)):
            if headings[j] and not re.match(r"^\[?\d+[\].]\s", blocks[j].text):
                ref_end = j
                break
    reference_blocks = blocks[ref_start + 1 : ref_end] if ref_start is not None else []

    # Body starts at the abstract (skipping title, authors and affiliations) when there is one.
    start = 0
    for i, b in enumerate(blocks[: max(1, len(blocks) // 3)]):
        if _ABSTRACT_RE.match(b.text.strip()) and b.page <= 1:
            start = i
            break

    paragraphs: list[tuple[str, str]] = []
    skipping = False  # inside a back-matter section (funding, author contributions, ...)
    for i, b in enumerate(blocks):
        if i < start or (ref_start is not None and ref_start <= i < ref_end) or b.in_margin:
            continue
        text = b.text.strip()
        if headings[i]:
            if _CAPTION_RE.match(text) or (b.page == 0 and title and text in title):  # captions, the title
                continue
            skipping = bool(_BACK_MATTER_RE.match(text))
            if skipping:
                continue
            if _ABSTRACT_RE.match(text) and len(text) > 12:  # "Abstract. We study ..." run-in heading
                paragraphs.append(("heading", "Abstract"))
                paragraphs.append(("para", re.sub(r"^\s*abstract[\s.:—–-]*", "", text, flags=re.IGNORECASE)))
            else:
                paragraphs.append(("heading", text))
            continue
        if (
            skipping
            or _CAPTION_RE.match(text)
            or b.in_table
            or text.startswith(SUP_OPEN)  # affiliations ("¹Department of ...") and footnotes
            or b.size < body_size - 0.8
            or _is_table_like(b)
            or len(text.split()) < 4
        ):
            continue
        if _ABSTRACT_RE.match(text) and i == start:
            paragraphs.append(("heading", "Abstract"))
            text = re.sub(r"^\s*abstract[\s.:—–-]*", "", text, flags=re.IGNORECASE)
        # A paragraph broken by a column or page break continues in the next block.
        if paragraphs and paragraphs[-1][0] == "para":
            prev = paragraphs[-1][1]
            if not _TERMINAL_RE.search(prev) and (text[:1].islower() or text[:1] in "([,;"):
                paragraphs[-1] = ("para", join_lines([prev, text]))
                continue
        paragraphs.append(("para", text))
    return Layout(title=title, paragraphs=paragraphs, reference_blocks=reference_blocks, body_size=body_size)


def _split_leading_headings(blocks: list[Block], body_size: float) -> list[Block]:
    """Some layouts put a heading in the same block as what follows ("References 1. Wei, H. ...")."""
    out = []
    for b in blocks:
        first = b.lines[0]
        text = first.text.strip()
        rest_size = max((ln.size for ln in b.lines[1:]), default=0.0)
        rest_chars = sum(len(ln.text) for ln in b.lines[1:])
        if len(b.lines) > 1 and (
            _REFERENCES_RE.match(text)
            or (first.size >= body_size + 1.5 and rest_size <= first.size - 1 and len(text) < 80
                and rest_chars >= 60 and not _TERMINAL_RE.search(text))  # not "1" + "INTRODUCTION"
        ):
            out.append(replace(b, lines=b.lines[:1]))
            out.append(replace(b, lines=b.lines[1:]))
        else:
            out.append(b)
    return out


def _title(blocks: list[Block], body_size: float) -> str:
    first = [b for b in blocks if b.page == 0 and b.chars > 3 and not b.in_margin]  # not the journal masthead
    if not first:
        return ""
    biggest = max(b.size for b in first)
    if biggest < body_size + 1.5:
        return ""
    parts, started = [], False
    for b in first:
        if abs(b.size - biggest) < 0.3:
            parts.append(b.text.strip())
            started = True
        elif started:
            break
    return join_lines(parts).strip()


# ----------------------------------------------------------------------------
# Reference entries
# ----------------------------------------------------------------------------

_BRACKET_LABEL_RE = re.compile(r"^\[(\d{1,4})\]\s*")
_DOT_LABEL_RE = re.compile(r"^(\d{1,4})\.(?:\s+(?=\S)|$)")  # the label can be a line of its own


@dataclass
class RefEntry:
    number: Optional[int]
    text: str


def split_references(blocks: list[Block]) -> tuple[list[RefEntry], bool]:
    """Reference blocks -> entries, and whether they carry printed numbers."""
    lines: list[tuple[Block, Line]] = [(b, ln) for b in blocks for ln in b.lines if ln.text.strip()]
    if not lines:
        return [], False

    bracket = [i for i, (_, ln) in enumerate(lines) if _BRACKET_LABEL_RE.match(ln.text.strip())]
    dotted = [i for i, (_, ln) in enumerate(lines) if _DOT_LABEL_RE.match(ln.text.strip())]
    if len(bracket) >= 3:
        return _split_at(lines, bracket, _BRACKET_LABEL_RE), True
    if len(dotted) >= 3 and _mostly_sequential([int(_DOT_LABEL_RE.match(lines[i][1].text.strip()).group(1)) for i in dotted]):
        return _split_at(lines, dotted, _DOT_LABEL_RE), True

    # Author-year: an entry starts at a line that sits at the column's left margin when the
    # other lines of the column are indented (hanging indent).
    margins: dict[tuple[int, int], float] = {}
    for b, ln in lines:
        key = (b.page, b.column)
        margins[key] = min(margins.get(key, ln.x0), ln.x0)
    indented = sum(1 for b, ln in lines if ln.x0 > margins[(b.page, b.column)] + 4)
    starts = []
    if indented >= len(lines) * 0.2:
        starts = [i for i, (b, ln) in enumerate(lines) if ln.x0 <= margins[(b.page, b.column)] + 2]
    else:  # no hanging indent: one block per entry is the best guess
        seen = set()
        for i, (b, _) in enumerate(lines):
            if id(b) not in seen:
                seen.add(id(b))
                starts.append(i)
    return _split_at(lines, starts, None), False


def _mostly_sequential(numbers: list[int]) -> bool:
    ups = sum(1 for a, b in zip(numbers, numbers[1:]) if b == a + 1)
    return ups >= 0.7 * (len(numbers) - 1)


def _split_at(lines, starts: list[int], label_re: Optional[re.Pattern]) -> list[RefEntry]:
    entries = []
    if starts and starts[0] != 0:
        starts = [0] + starts  # text before the first label is a continuation from nowhere; keep it
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(lines)
        text = join_lines([ln.text for _, ln in lines[s:e]]).strip()
        number = None
        if label_re is not None:
            m = label_re.match(text)
            if not m:
                continue
            number = int(m.group(1))
            text = text[m.end():].strip()
        if len(text) >= 15:
            entries.append(RefEntry(number=number, text=text))
    return entries


_INITIALS_RE = re.compile(r"^(?:[A-Z]\.?|[A-Z][a-z]\.)(?:[\s-]*(?:[A-Z]\.?|[A-Z][a-z]\.)){0,3}$")  # "J.", "J.-P.", "Ch."
_YEAR_LABEL_RE = re.compile(r"\b((?:19|20)\d{2})([a-z]?)\b")
_IDENTIFIERS_RE = re.compile(r"https?://\S+|\b10\.\d{4,9}/\S+|\b(?:arXiv:|abs/)\S+|\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)


@dataclass
class ParsedEntry:
    authors: list[str] = field(default_factory=list)
    surnames: list[str] = field(default_factory=list)
    title: Optional[str] = None
    year: Optional[int] = None
    year_label: Optional[str] = None


def _author_end(text: str) -> Optional[int]:
    """Index of the period that ends the author list ("..., and Quoc V. Le. Title ...")."""
    for m in re.finditer(r"\.\s+(?=\S)", text):
        before = text[: m.start()].rsplit(" ", 1)[-1].strip(",")
        if len(before.replace(".", "")) <= 1 and before[:1].isupper():
            continue  # an initial
        if before in ("al", "et al"):
            return m.start()
        return m.start()
    return None


def _first_sentence(text: str) -> str:
    """Up to the first period that ends a sentence (not an initial or abbreviation)."""
    for m in re.finditer(r"\.(\s+|$)", text):
        before = text[: m.start()].rsplit(" ", 1)[-1]
        if len(before) <= 1 or before.lower() in ("vs", "e.g", "i.e", "et al", "al", "st"):
            continue
        return text[: m.start()]
    return text


_ABBREVIATED_WORDS = {"al", "Jr", "Sr", "eds", "Eds", "ed", "Ed"}


def _looks_like_authors(text: str) -> bool:
    """No sentence ends inside: every period follows an initial or an abbreviation."""
    tokens = text.strip().rstrip(".").split()
    return len(text) < 600 and not any(
        t.endswith(".") and len(t.strip(".,()")) >= 3 and t.strip(".,()") not in _ABBREVIATED_WORDS for t in tokens
    )


_PARTICLE = r"(?:van|von|der|den|de|del|della|da|di|dos|du|la|le|ter|ten|bin|al|el)\s"
_VANCOUVER_PERSON_RE = re.compile(rf"^(?:{_PARTICLE})*[A-Z][\w'’\-]+(?:\s(?:{_PARTICLE})*[A-Z][\w'’\-]+)*\s[A-Z]{{1,3}}$")
_SURNAME_INITIALS_RE = re.compile(
    rf"(?:(?:&|and)\s+)?((?:{_PARTICLE})*[A-Z][\w'’\-]+(?:[\s-][A-Z][\w'’\-]+){{0,2}}),\s+((?:[A-Z][a-z]?\.(?:\s?-?[A-Z]\.)*\s?){{1,4}})"
)


def _vancouver_authors(text: str) -> Optional[int]:
    """End of a Vancouver author list ("Wang TW, Asman K, et al. Title."), or None."""
    for m in re.finditer(r"\.\s+", text):
        parts = [p.strip() for p in text[: m.start()].split(",")]
        if parts and all(_VANCOUVER_PERSON_RE.match(p) or p in ("et al", "et al.") for p in parts):
            # One name with one initial is more likely "Tom B. Brown, ..." than Vancouver "Brown T."
            if len(parts) > 1 or len(parts[0].split()[-1]) > 1:
                return m.start()
        if len(parts) > 60:
            break
    return None


def _surname_initials_authors(text: str) -> Optional[int]:
    """End of a "Surname, I. I., Surname, I. & Surname, I." list (Nature, APA, LNCS), or None."""
    pos, people = 0, 0
    while True:
        m = _SURNAME_INITIALS_RE.match(text, pos)
        if not m:
            break
        people += 1
        pos = m.end()
        sep = re.match(r",\s+|\s*(?=(?:&|and)\s)|\s*et al\.,?\s*", text[pos:])
        if not sep:
            break
        pos += sep.end()
    return pos if people else None


def _split_people(author_text: str) -> list[str]:
    text = re.sub(r"\bet\.?\s*al\.?", "", author_text)
    text = text.replace("&", " and ")
    parts = [p.strip(" .,") for p in re.split(r",\s*(?:and\s+)?|\s+and\s+|;\s*", text) if p.strip(" .,")]
    people: list[str] = []
    for p in parts:
        if _INITIALS_RE.match(p) and people and " " not in people[-1].strip():
            people[-1] = f"{people[-1]}, {p}"  # "Smith, J." written as "Smith", "J."
        else:
            people.append(p)
    return [p for p in people if re.search(r"[A-Za-z]{2}", p)]


def surname(person: str) -> str:
    if "," in person:  # "Smith, J."
        return person.split(",", 1)[0].strip()
    tokens = person.split()
    if len(tokens) > 1 and re.fullmatch(r"[A-Z]{1,3}", tokens[-1]):  # Vancouver "Wang TW"
        return " ".join(tokens[:-1])
    tokens = [t for t in tokens if not _INITIALS_RE.match(t)]
    return tokens[-1] if tokens else person.strip()


def parse_entry(text: str) -> ParsedEntry:
    entry = ParsedEntry()
    # Years, but not the digits inside identifiers ("arXiv:2012.13255", URLs, DOIs).
    years = list(_YEAR_LABEL_RE.finditer(_IDENTIFIERS_RE.sub(" ", text)))
    author_text = ""
    title = None

    quoted = re.search(r",\s*[“\"]([^”\"]{8,}?)[,.]?[”\"]", text)
    apa = re.match(r"^(.*?)\s*\(((?:19|20)\d{2}[a-z]?|n\.d\.)[^)]*\)\.?\s*(.+)$", text)
    acl = re.match(r"^(.*?(?:[^\W\d_]{2}|\))\.)\s+((?:19|20)\d{2}[a-z]?)\.\s+(.+)$", text)
    vancouver = _vancouver_authors(text)
    initials = _surname_initials_authors(text)
    if quoted and _looks_like_authors(text[: quoted.start()]):  # IEEE: A. Author and B. Author, "Title," in ...
        author_text = text[: quoted.start()]
        title = quoted.group(1)
    elif apa and _looks_like_authors(apa.group(1)):  # Author, A., & Author, B. (2020). Title. Journal.
        author_text, title = apa.group(1), _first_sentence(apa.group(3))
        entry.year_label = apa.group(2) if apa.group(2)[0].isdigit() else None
    elif acl and _looks_like_authors(acl.group(1)[:-1]):  # Authors. 2020a. Title. Venue.
        author_text, title = acl.group(1), _first_sentence(acl.group(3))
        entry.year_label = acl.group(2)
    elif vancouver is not None:  # Wang TW, Asman K, et al. Title. Journal. 2018;
        author_text, title = text[:vancouver], _first_sentence(text[vancouver + 1 :].strip())
    elif initials and re.match(r"\s*[:.]?\s*\S", text[initials:]):  # Wei, H. & Li, K. Title. / Smith, J.: Title.
        author_text = text[:initials]
        title = _first_sentence(re.sub(r"^\s*[:.]?\s*", "", text[initials:]))
    else:  # Authors. Title. Venue, 2020.
        end = _author_end(text)
        if end is not None:
            author_text, title = text[:end], _first_sentence(text[end + 1 :].strip())

    if title:
        title = re.sub(r"\s+", " ", title).strip(" .,")
        title = re.sub(r",\s*(?:19|20)\d{2}[a-z]?$", "", title)
        # A title ending in "?" runs into the venue: "... in context? arXiv preprint arXiv:2204.02329"
        title = re.sub(r"(?<=[?!])\s+(?:arXiv|CoRR|In\b|Proceedings|Advances in|Journal|Transactions|[A-Z]{2,}\b).*$",
                       "", title)
        entry.title = title if len(title) >= 6 else None
    entry.authors = _split_people(author_text)[:50]
    entry.surnames = [surname(a) for a in entry.authors]
    if entry.year_label is None and years:
        m = years[-1]
        entry.year_label = m.group(1) + m.group(2)
    if entry.year_label:
        entry.year = int(entry.year_label[:4])
    return entry


# ----------------------------------------------------------------------------
# In-text citations
# ----------------------------------------------------------------------------

_NUMBER_LIST = r"\s*\d{1,4}(?:\s*[-–—]\s*\d{1,4})?(?:\s*[,;]\s*\d{1,4}(?:\s*[-–—]\s*\d{1,4})?)*\s*"
_NUMERIC_CITE_RE = re.compile(rf"\[({_NUMBER_LIST})\]")
_PAREN_NUMERIC_RE = re.compile(rf"\(({_NUMBER_LIST})\)")
_NAME = r"(?:(?:van|von|de|der|da|di|del|la|le|Le|De|Van|Von)\s+)?[A-Z][\w'’\-]+"
_NAMES = rf"{_NAME}(?:\s+{_NAME})?(?:\s+et\s+al\.?|\s+(?:and|&)\s+{_NAME}(?:\s+{_NAME})?)?"
_YEARS = r"(?:19|20)\d{2}[a-z]?(?:\s*,\s*(?:(?:19|20)\d{2}[a-z]?|[a-z]\b))*"
_TEXTUAL_RE = re.compile(rf"(?<![\w(])({_NAMES})(?:’s|'s)?\s+\(({_YEARS})\)")
_PAREN_RE = re.compile(r"\(([^()]{4,400})\)")
# Inside parentheses names are checked against the reference list, so they can be looser:
# organisations ("BIG-bench collaboration, 2021") and multi-word surnames ("Le Scao and Rush, 2021").
_PAREN_PART_RE = re.compile(rf"^(?:[a-z][\w.,]*\s+)*([A-Z][^,;()]{{0,60}}?),?\s+({_YEARS})\b")


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    return "".join(c for c in name if c.isalnum()).lower()


def _expand_years(years: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"\s*,\s*", years.strip()):
        if re.fullmatch(r"[a-z]", part) and out:
            out.append(out[-1][:4] + part)  # "2019a,b" -> 2019a, 2019b
        elif part:
            out.append(part)
    return out


class AuthorYearIndex:
    def __init__(self, refs: list[tuple[str, ParsedEntry]]):
        self.refs = refs

    def find(self, names: str, year_label: str) -> Optional[str]:
        names = re.sub(r"\bet\.?\s*al\.?", "", names).strip()
        people = [p for p in re.split(r"\s+(?:and|&)\s+", names) if p.strip()]
        if not people:
            return None
        first = _norm(people[0].split()[-1])
        second = _norm(people[1].split()[-1]) if len(people) > 1 else None

        def surname_matches(entry: ParsedEntry, idx: int, target: str) -> bool:
            if len(entry.surnames) <= idx:
                return False
            s = _norm(entry.surnames[idx])
            return s == target or s.endswith(target) or target.endswith(s) and len(s) >= 3

        candidates = [(key, e) for key, e in self.refs if surname_matches(e, 0, first)]
        if second:
            narrowed = [(k, e) for k, e in candidates if surname_matches(e, 1, second)]
            candidates = narrowed or candidates
        exact = [k for k, e in candidates if e.year_label == year_label]
        if len(exact) == 1:
            return exact[0]
        by_year = [k for k, e in candidates if e.year and str(e.year) == year_label[:4]]
        if len(by_year) == 1:
            return by_year[0]
        if len(exact) > 1:
            return exact[0]
        return None


def find_markers(text: str, numeric: bool, numbers: dict[int, str],
                 index: Optional[AuthorYearIndex], pattern: re.Pattern = None) -> tuple[list[Marker], int, int]:
    """In-text citations -> markers with reference keys, plus (citations seen, citations mapped).

    Numeric citations use `pattern` (bracketed by default; "(1, 2)" for some journals).
    """
    markers: list[Marker] = []
    seen: set[str] = set()
    mapped: set[str] = set()

    if numeric:
        highest = max(numbers, default=0)
        for m in (pattern or _NUMERIC_CITE_RE).finditer(text):
            nums: list[int] = []
            for part in re.split(r"\s*[,;]\s*", m.group(1).strip()):
                if re.search(r"[-–—]", part):
                    a, b = (int(x) for x in re.split(r"\s*[-–—]\s*", part))
                    if 0 < b - a <= 50:
                        nums.extend(range(a, b + 1))
                        continue
                    nums.extend((a, b))
                else:
                    nums.append(int(part))
            if highest and any(n > max(2 * highest, highest + 20) for n in nums):
                continue  # "[2019]", "(512)": not a citation
            keys = []
            for n in nums:
                seen.add(str(n))
                if n in numbers:
                    mapped.add(str(n))
                    if numbers[n] not in keys:
                        keys.append(numbers[n])
            if keys:
                markers.append(Marker(m.start(), m.end(), keys))
        return markers, len(seen), len(mapped)

    if index is None:
        return [], 0, 0
    taken: list[tuple[int, int]] = []
    for m in _TEXTUAL_RE.finditer(text):
        keys = []
        for year in _expand_years(m.group(2)):
            cite = f"{_norm(m.group(1))}|{year}"
            seen.add(cite)
            key = index.find(m.group(1), year)
            if key:
                mapped.add(cite)
                if key not in keys:
                    keys.append(key)
        taken.append((m.start(), m.end()))
        if keys:
            markers.append(Marker(m.start(), m.end(), keys))

    for m in _PAREN_RE.finditer(text):
        if any(a < m.end() and m.start() < b for a, b in taken):
            continue
        keys = []
        for part in re.split(r";\s*", m.group(1)):
            part = re.sub(r"^\s*(?:e\.g\.|i\.e\.|see(?: also)?|cf\.|for example|inter alia),?\s*", "", part.strip())
            pm = _PAREN_PART_RE.match(part)
            if not pm:
                continue
            for year in _expand_years(pm.group(2)):
                cite = f"{_norm(pm.group(1))}|{year}"
                seen.add(cite)
                key = index.find(pm.group(1), year)
                if key:
                    mapped.add(cite)
                    if key not in keys:
                        keys.append(key)
        if keys:
            markers.append(Marker(m.start(), m.end(), keys))
    markers.sort(key=lambda mk: mk.start)
    return markers, len(seen), len(mapped)


# ----------------------------------------------------------------------------
# Paper
# ----------------------------------------------------------------------------


def _short_author(entry: ParsedEntry) -> str:
    s = entry.surnames
    if not s:
        return "Anonymous"
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        return f"{s[0]} and {s[1]}"
    return f"{s[0]} et al."


def _usable_metadata_title(title: str) -> bool:
    """Publisher PDFs carry the real title in metadata; LaTeX ones are often empty or a file name."""
    title = (title or "").strip()
    return (
        len(title.split()) >= 3
        and not re.search(r"\.(?:dvi|tex|pdf|docx?)$|^Microsoft Word|^untitled|^Manuscript", title, re.IGNORECASE)
    )


def parse_pdf(data: bytes) -> tuple[Paper, list]:
    """Pure function (no network): PDF bytes -> (Paper without id/metadata, citing sentences)."""
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises various types
        raise PdfParseError("This file could not be opened as a PDF.") from exc
    with doc:
        if doc.needs_pass:
            raise PdfParseError("This PDF is password-protected.")
        layout = analyse(doc)
        metadata_title = (doc.metadata or {}).get("title") or ""
        metadata_author = (doc.metadata or {}).get("author") or ""

    entries, numbered = split_references(layout.reference_blocks)
    for e in entries:
        e.text = _SUP_PLACEHOLDER_RE.sub("", e.text)
    parsed = [parse_entry(e.text) for e in entries]

    # How numeric citations are printed: [12], superscript, or (12). Superscript numbers that are
    # not citations are footnote marks and are dropped.
    raw = "\n\n".join(p for _, p in layout.paragraphs)
    bare = _SUP_PLACEHOLDER_RE.sub("", raw)
    pattern = _NUMERIC_CITE_RE
    superscript = False
    if numbered and len(_NUMERIC_CITE_RE.findall(bare)) < 3:
        if len(_SUP_PLACEHOLDER_RE.findall(raw)) >= 3:
            superscript = True
        elif len(_PAREN_NUMERIC_RE.findall(bare)) >= 3:
            pattern = _PAREN_NUMERIC_RE
    numeric = numbered or (not entries and len(_NUMERIC_CITE_RE.findall(bare)) >= 5)

    # Text with headings as their own paragraphs (same shape as the LaTeX converter's output).
    text_parts: list[str] = []
    headings: list[Heading] = []
    pos = 0
    for kind, para in layout.paragraphs:
        para = _SUP_PLACEHOLDER_RE.sub(lambda m: f"[{m.group(1)}]" if superscript else "", para)
        if kind == "heading":
            headings.append(Heading(para, pos, 1))
        text_parts.append(para)
        pos += len(para) + 2
    text = "\n\n".join(text_parts)

    references: list[Reference] = []
    numbers: dict[int, str] = {}
    for i, (entry, p) in enumerate(zip(entries, parsed), start=1):
        n = entry.number if entry.number is not None else i
        key = f"r{n}" if numbered else f"r{i}"
        if numbered:
            numbers.setdefault(n, key)
        label = str(n) if numbered else f"{_short_author(p)}, {p.year_label or 'n.d.'}"
        references.append(Reference(
            key=key, label=label, raw=entry.text, title=p.title, authors=p.authors, year=p.year,
            doi=bib.find_doi(entry.text), arxiv_id=bib.find_arxiv_id(entry.text),
            url=(re.search(r"https?://\S+", entry.text) or [None])[0],
        ))
    index = None if numbered else AuthorYearIndex([(r.key, p) for r, p in zip(references, parsed)])
    markers, total, mapped = find_markers(text, numeric, numbers, index, pattern)

    sections = build_sections(text, headings)
    cited_keys = {k for m in markers for k in m.keys}
    sentences = citing_sentences(text, markers, sections, cited_keys)

    notes = []
    if not entries:
        notes.append("No reference list was found in this PDF, so its citations could not be checked.")
    elif total == 0:
        notes.append("No in-text citations could be recognised in this PDF.")
    elif mapped < 0.5 * total:
        notes.append(
            f"Only {mapped} of {total} in-text citations could be matched to entries in the reference list, "
            "so this report covers part of the paper's citations."
        )

    title = metadata_title.strip() if _usable_metadata_title(metadata_title) else layout.title
    authors = [a.strip() for a in re.split(r"\s*(?:;|,|\band\b)\s*", metadata_author) if len(a.strip()) > 2]
    paper = Paper(
        id="",
        title=title,
        authors=authors[:30],
        source="pdf",
        parsed_from="pdf",
        citation_style="numeric" if numeric else "author_year",
        text=text,
        sections=sections,
        references=[r for r in references if r.key in cited_keys] or references,
        citation_keys_total=total,
        citation_keys_mapped=mapped,
        notes=notes,
    )
    paper.abstract = next((s.text.split("\n\n", 1)[-1].strip() for s in sections if s.title.lower() == "abstract"), "")
    return paper, sentences
