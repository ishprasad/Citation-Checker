"""Sentence splitting that keeps citation markers attached to their sentence."""

from __future__ import annotations

import re
from bisect import bisect_right

from app.models import CitingSentence, Section
from app.pipeline.latex import Marker

ABBREVIATIONS = {
    "e.g", "i.e", "et al", "al", "fig", "figs", "eq", "eqs", "sec", "secs", "tab", "ref", "refs",
    "cf", "vs", "approx", "resp", "no", "nos", "vol", "pp", "p", "dr", "mr", "ms", "prof", "st",
    "jr", "inc", "ltd", "ca", "c.f", "w.r.t", "a.k.a", "viz", "ch", "chap", "app", "appx", "def",
    "thm", "lem", "prop", "eqn", "alg", "obs", "est", "incl", "max", "min", "avg", "std", "u.s",
}

# A candidate boundary: terminal punctuation (plus closing quotes/brackets), whitespace,
# then something that can start a sentence.
_BOUNDARY_RE = re.compile(r"[.!?][\"”’)\]]*(\s+)(?=[A-Z0-9“\"(\[•§]|$)")
_WORD_BEFORE_RE = re.compile(r"([A-Za-z.]+)\.?[\"”’)\]]*$")


def _is_abbreviation(text: str, dot_index: int) -> bool:
    if text[dot_index] != ".":
        return False
    m = _WORD_BEFORE_RE.search(text, 0, dot_index + 1)
    if not m:
        return False
    word = m.group(1).rstrip(".").lower()
    if word in ABBREVIATIONS:
        return True
    # Initials such as "J. Smith" or "A.B."
    return len(word.replace(".", "")) == 1 and word.isalpha()


def split_sentences(text: str, markers: list[Marker]) -> list[tuple[int, int]]:
    """Return (start, end) spans of sentences in `text`."""
    marker_starts = {m.start: m for m in markers}
    spans = []
    para_start = 0
    for para in re.split(r"(\n\s*\n)", text):
        if not para.strip() or para.startswith("\n"):
            para_start += len(para)
            continue
        start = para_start + (len(para) - len(para.lstrip()))
        para_end = para_start + len(para.rstrip())
        for m in _BOUNDARY_RE.finditer(text, start, para_end):
            punct = m.start()
            if _is_abbreviation(text, punct) or _inside_marker(punct, markers):
                continue
            cut = m.start(1)  # end of current sentence (after punctuation)
            nxt = m.end(1)
            # A citation placed after the period ("... results. [3] Next sentence") belongs
            # to the preceding sentence, unless it is the subject of the next one
            # ("[3] proposed ...").
            marker = marker_starts.get(nxt)
            if marker is not None:
                after = text[marker.end : para_end]
                stripped = after.lstrip()
                if not stripped or stripped[0].isupper() or stripped[0] in "“\"(":
                    cut = marker.end
                    nxt = marker.end + len(after) - len(stripped)
            if cut <= start:
                continue
            spans.append((start, cut))
            start = nxt
        if start < para_end:
            spans.append((start, para_end))
        para_start += len(para)
    return [s for s in spans if text[s[0] : s[1]].strip()]


def _inside_marker(pos: int, markers: list[Marker]) -> bool:
    return any(m.start <= pos < m.end for m in markers)


def build_sections(text: str, headings: list) -> list[Section]:
    sections = []
    starts = [(h.start, h.title) for h in headings]
    if not starts or starts[0][0] > 0:
        starts.insert(0, (0, ""))
    for idx, (start, title) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(text)
        if end > start and text[start:end].strip():
            sections.append(Section(title=title, text=text[start:end], start=start, end=end))
    return sections


def citing_sentences(
    text: str,
    markers: list[Marker],
    sections: list[Section],
    known_keys: set[str],
) -> list[CitingSentence]:
    spans = split_sentences(text, markers)
    section_starts = [s.start for s in sections]
    ordered = sorted(markers, key=lambda m: m.start)
    result = []
    for start, end in spans:
        inside = [m for m in ordered if start <= m.start < end]
        keys: list[str] = []
        for m in inside:
            for k in m.keys:
                if k in known_keys and k not in keys:
                    keys.append(k)
        if not keys:
            continue
        idx = bisect_right(section_starts, start) - 1
        result.append(
            CitingSentence(
                id=f"s{len(result) + 1}",
                text=text[start:end],
                section=sections[idx].title if idx >= 0 else "",
                start=start,
                end=end,
                reference_keys=keys,
                marker_spans=[(m.start, m.end) for m in inside],
            )
        )
    return result
