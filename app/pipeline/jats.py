"""JATS XML (Europe PMC / PubMed Central full text) -> plain text with headings, and the paper's
own reference list.

Output follows the LaTeX converter's conventions so evidence chunking treats it the same way:
headings are their own paragraphs, paragraphs are separated by blank lines, and figure and table
captions are kept as "Figure: ..." / "Table: ..." (captions often hold the reported numbers),
and table contents follow their caption as "cell | cell" lines.
In-text citations stay inline; numeric ones are bracketed ("[12]") so a superscript citation
does not run into the preceding word and the secondary-citation detector can read it.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Optional


class JatsError(Exception):
    pass


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _inline(el: ET.Element) -> str:
    """Text of a paragraph-like element, with citations bracketed and floats left out."""
    out: list[str] = [el.text or ""]
    for child in el:
        tag = _local(child.tag)
        if tag == "sup" and any(_local(x.tag) == "xref" and x.get("ref-type") == "bibr" for x in child.iter()):
            out.append(f"[{_clean(''.join(child.itertext()))}]")  # superscript citations: obesity¹² -> obesity[12]
        elif tag == "xref" and child.get("ref-type") == "bibr":
            label = _clean("".join(child.itertext()))
            before = "".join(out).rstrip()
            already = before.endswith(("[", ",", "–", "-", ";"))  # inside "[1, 2]" written by the source
            out.append(f"[{label}]" if re.fullmatch(r"[\d,\s–-]+", label) and not already else label)
        elif tag in ("fig", "table-wrap", "disp-formula", "supplementary-material", "fn", "table"):
            pass  # floats are rendered separately (captions) or skipped
        else:
            out.append(_inline(child))
        out.append(child.tail or "")
    return "".join(out)


def _caption(el: ET.Element, kind: str) -> Optional[str]:
    caption = next((c for c in el.iter() if _local(c.tag) == "caption"), None)
    if caption is None:
        return None
    text = _clean(" ".join(_inline(p) for p in caption if _local(p.tag) in ("title", "p")) or "".join(caption.itertext()))
    return f"{kind}: {text}" if text else None


MAX_TABLE_ROWS = 80
MAX_TABLE_CHARS = 6000


def _table_rows(el: ET.Element) -> Optional[str]:
    """Table contents as "cell | cell" lines: cited numbers are often only in a table."""
    lines = []
    for tr in el.iter():
        if _local(tr.tag) != "tr":
            continue
        cells = [_clean("".join(c.itertext())) for c in tr if _local(c.tag) in ("td", "th")]
        if any(cells):
            lines.append(" | ".join(cells).strip(" |"))
    return "\n".join(lines[:MAX_TABLE_ROWS])[:MAX_TABLE_CHARS] or None


def _float(el: ET.Element, tag: str, parts: list[str]) -> None:
    cap = _caption(el, "Figure" if tag == "fig" else "Table")
    if cap:
        parts.append(cap)
    rows = _table_rows(el) if tag == "table-wrap" else None
    if rows:
        parts.append(rows)


def _walk(el: ET.Element, parts: list[str]) -> None:
    for child in el:
        tag = _local(child.tag)
        if tag == "sec":
            if child.get("sec-type", "") in ("supplementary-material", "COI-statement"):
                continue
            title = next((c for c in child if _local(c.tag) == "title"), None)
            if title is not None:
                label = next((c for c in child if _local(c.tag) == "label"), None)
                heading = _clean(f"{''.join(label.itertext()) if label is not None else ''} {''.join(title.itertext())}")
                if heading:
                    parts.append(heading.rstrip(".:"))
            _walk(child, parts)
        elif tag == "p":
            text = _clean(_inline(child))
            if text:
                parts.append(text)
            for float_el in child.iter():  # figures and tables are often nested inside paragraphs
                ftag = _local(float_el.tag)
                if ftag in ("fig", "table-wrap"):
                    _float(float_el, ftag, parts)
        elif tag in ("fig", "table-wrap"):
            _float(child, tag, parts)
        elif tag in ("list", "list-item", "boxed-text", "disp-quote"):
            _walk(child, parts)


def _reference(ref: ET.Element) -> dict:
    label_el = next((c for c in ref if _local(c.tag) == "label"), None)
    cit = next((c for c in ref.iter() if _local(c.tag) in ("mixed-citation", "element-citation", "citation")), None)
    raw = _clean("".join(cit.itertext())) if cit is not None else _clean("".join(ref.itertext()))
    title_el = next((c for c in ref.iter() if _local(c.tag) in ("article-title", "chapter-title")), None)
    title = _clean("".join(title_el.itertext())) if title_el is not None else None
    label = _clean("".join(label_el.itertext())).strip(".[]") if label_el is not None else ""
    if not label:  # author-year journals: "Smith et al., 2020"
        surnames = [_clean("".join(s.itertext())) for s in ref.iter() if _local(s.tag) == "surname"]
        year = next((_clean("".join(y.itertext())) for y in ref.iter() if _local(y.tag) == "year"), "")
        if surnames:
            who = surnames[0] if len(surnames) == 1 else (
                f"{surnames[0]} and {surnames[1]}" if len(surnames) == 2 else f"{surnames[0]} et al.")
            label = f"{who}, {year}".strip(", ")
    return {"label": label, "title": title, "raw": raw}


def jats_to_text(xml: bytes) -> tuple[str, dict]:
    """Returns (plain text, sidecar with the paper's own references)."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise JatsError(f"invalid JATS XML: {exc}") from exc
    parts: list[str] = []
    abstract = next((e for e in root.iter() if _local(e.tag) == "abstract" and not e.get("abstract-type")), None)
    if abstract is not None:
        paragraphs = [_clean(_inline(p)) for p in abstract.iter() if _local(p.tag) == "p"]
        if any(paragraphs):
            parts.append("Abstract")
            parts += [p for p in paragraphs if p]
    body = next((e for e in root.iter() if _local(e.tag) == "body"), None)
    if body is not None:
        _walk(body, parts)
    refs = [_reference(r) for r in root.iter() if _local(r.tag) == "ref"]
    return "\n\n".join(parts), {"references": refs}
