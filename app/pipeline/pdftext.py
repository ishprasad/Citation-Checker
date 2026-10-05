"""PDF -> clean plain text with PyMuPDF.

Text is assembled from layout blocks (one paragraph per block), with line-end
hyphenation repaired and ligatures expanded. The trailing references section is
split off so it does not pollute retrieval, but is returned separately (useful
for naming the original work behind a secondary citation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
_LIGATURE_RE = re.compile("|".join(LIGATURES))
_HYPHEN_BREAK_RE = re.compile(r"(\w)[-­]\n(?=[a-z])")
_REFERENCES_HEADING_RE = re.compile(
    r"^\s*(?:\d+\.?\s*|[A-Z]\.?\s+)?(references|bibliography|literature cited|works cited)\s*$",
    re.IGNORECASE,
)
MIN_TEXT_CHARS = 500  # less than this means a scanned/image-only PDF


class PdfTextError(Exception):
    pass


@dataclass
class PdfText:
    body: str
    references: str
    pages: int


def _clean_block(text: str) -> str:
    text = _LIGATURE_RE.sub(lambda m: LIGATURES[m.group(0)], text)
    text = _HYPHEN_BREAK_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_text(data: bytes) -> PdfText:
    if not data.startswith(b"%PDF-"):
        raise PdfTextError("not a PDF")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises various types
        raise PdfTextError(f"could not open PDF: {exc}") from exc

    blocks: list[str] = []
    with doc:
        pages = doc.page_count
        for page in doc:
            for block in page.get_text("blocks"):
                if block[6] != 0:  # image block
                    continue
                cleaned = _clean_block(block[4])
                if cleaned and not cleaned.isdigit():  # drop bare page numbers
                    blocks.append(cleaned)

    # Split off the last "References" heading found in the second half of the document.
    cut = None
    for idx in range(len(blocks) - 1, len(blocks) // 2 - 1, -1):
        if _REFERENCES_HEADING_RE.match(blocks[idx]):
            cut = idx
            break
    body_blocks = blocks[:cut] if cut is not None else blocks
    ref_blocks = blocks[cut + 1 :] if cut is not None else []

    body = "\n\n".join(body_blocks)
    if len(body) < MIN_TEXT_CHARS:
        raise PdfTextError("PDF has almost no extractable text (scanned?)")
    return PdfText(body=body, references="\n".join(ref_blocks), pages=pages)
