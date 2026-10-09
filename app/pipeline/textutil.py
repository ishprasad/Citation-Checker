"""Small text helpers shared by several stages: number extraction and quote normalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

# A number with optional thousands separators, decimals, exponent and unit.
# The look-behind skips digits inside identifiers ("GPT-3", "β1", "v2").
NUMBER_RE = re.compile(
    r"(?<![\w.\-−])"
    r"(?P<sign>[-−+])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
    r"(?:\s?(?P<op>\^|×\s?10\^)(?P<exp>[-−]?\d+))?"
    r"(?P<unit>\s(?:billion|million|thousand)\b|\s?%|\s?percent\b|\s?pp\b|\s?(?:x|×)(?![\w])|-?fold\b|[kKMB](?![\w]))?"
    r"(?![\d.]\d)"
)


@dataclass(frozen=True)
class Number:
    raw: str
    value: float
    unit: str  # "", "%", "x", "k", "M", "B", "pp"

    def same_quantity(self, other: "Number", rel_tol: float = 1e-6) -> bool:
        if self.unit != other.unit:
            return False
        return abs(self.value - other.value) <= rel_tol * max(abs(self.value), abs(other.value), 1.0)


_UNIT_MAP = {"%": "%", "percent": "%", "pp": "pp", "x": "x", "×": "x", "fold": "x", "-fold": "x",
             "k": "k", "K": "k", "M": "M", "B": "B", "thousand": "k", "million": "M", "billion": "B"}
SCALE = {"k": 1e3, "M": 1e6, "B": 1e9}


def extract_numbers(text: str) -> list[Number]:
    found = []
    for m in NUMBER_RE.finditer(text):
        digits = m.group("num").replace(",", "") + (m.group("dec") or "")
        value = float(digits)
        if m.group("sign") in ("-", "−"):
            value = -value
        if m.group("exp"):
            exp = int(m.group("exp").replace("−", "-"))
            # "10^-9" is a power; "2×10^-9" is scientific notation.
            value = value**exp if m.group("op") == "^" else value * 10.0**exp
        unit_raw = (m.group("unit") or "").strip()
        found.append(Number(raw=m.group(0).strip(), value=value, unit=_UNIT_MAP.get(unit_raw, unit_raw)))
    return found


def number_strings(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for n in extract_numbers(text):
        seen.setdefault(n.raw, None)
    return list(seen)


_QUOTES = {"“": '"', "”": '"', "„": '"', "‟": '"', "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'"}
_DASHES = {"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-", "­": ""}


def normalize_for_match(text: str) -> str:
    """Normalization used to check that a quote is verbatim: quotes, dashes, ligatures, whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(_QUOTES.get(c, _DASHES.get(c, c)) for c in text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # hyphenation left over from PDFs
    return re.sub(r"\s+", " ", text).strip().lower()


def find_normalized(needle: str, haystack: str) -> Optional[int]:
    """Index of `needle` in `haystack` after normalization (index into the normalized text)."""
    n, h = normalize_for_match(needle), normalize_for_match(haystack)
    if not n:
        return None
    idx = h.find(n)
    return idx if idx >= 0 else None


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Character-level normalization that remembers where each output character came from.

    Folds quotes, dashes, ligatures and accents, lowercases, collapses whitespace and
    removes PDF line-break hyphenation ("bench- marks" -> "benchmarks"). Returns the
    normalized text and, for each of its characters, the index in `text`.
    """
    chars: list[str] = []
    origin: list[int] = []
    for i, c in enumerate(text or ""):
        if c in _QUOTES:
            piece = _QUOTES[c]
        elif c in _DASHES:
            piece = _DASHES[c]
        else:
            piece = "".join(ch for ch in unicodedata.normalize("NFKD", c) if not unicodedata.combining(ch))
        for ch in piece.lower():
            if ch.isspace():
                if not chars or chars[-1] == " ":
                    continue
                ch = " "
            chars.append(ch)
            origin.append(i)
    while chars and chars[-1] == " ":
        chars.pop()
        origin.pop()
    # Drop "x- y" hyphenation joins.
    out, out_origin = [], []
    k = 0
    while k < len(chars):
        if (
            chars[k] == "-" and 0 < k and k + 2 < len(chars) and chars[k + 1] == " "
            and chars[k - 1].isalpha() and chars[k + 2].isalpha()
        ):
            k += 2
            continue
        out.append(chars[k])
        out_origin.append(origin[k])
        k += 1
    # Drop inline citation markers ("[12]", "[3, 5-7]", "(smith et al., 2019)"): models
    # routinely leave them out when quoting, and they carry nothing to verify.
    text = "".join(out)
    keep = [True] * len(text)
    for m in _CITATION_MARKER_RE.finditer(text):
        for j in range(m.start(), m.end()):
            keep[j] = False
    return "".join(c for c, k in zip(text, keep) if k), [o for o, k in zip(out_origin, keep) if k]


_CITATION_MARKER_RE = re.compile(
    r" ?\[\d+(?:\s?[,–-]\s?\d+)*\]"
    r"| ?\((?:see |e\.g\., )?[a-z][^()]{0,80}?,? (?:19|20)\d{2}[a-z]?(?:[;,] [^()]{0,80}?(?:19|20)\d{2}[a-z]?)*\)"
)


_ELLIPSIS_RE = re.compile(r"\s*(?:\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…)\s*")
MAX_FRAGMENT_GAP = 2000  # normalized chars allowed between pieces of an elided quote


def locate_quote(quote: Optional[str], text: str) -> Optional[tuple[int, int]]:
    """(start, end) of `quote` in `text` under normalized matching, or None if it is not verbatim.

    A quote elided with "..." matches if every piece is found, in order, close together.
    """
    if not quote or not text:
        return None
    cleaned = quote.strip().strip("\"'“”‘’").strip()
    fragments = [f for f in _ELLIPSIS_RE.split(cleaned) if f.strip()]
    norm_text, origin = normalize_with_map(text)
    spans: list[tuple[int, int]] = []
    pos = 0
    for frag in fragments:
        norm_frag = normalize_with_map(frag)[0].strip(" .,;:")
        min_len = 6 if len(fragments) == 1 else 12
        if len(norm_frag) < min_len:
            if len(fragments) == 1:
                return None
            continue
        idx = norm_text.find(norm_frag, pos)
        if idx < 0 or (spans and idx - spans[-1][1] > MAX_FRAGMENT_GAP):
            return None
        spans.append((idx, idx + len(norm_frag)))
        pos = idx + len(norm_frag)
    if not spans:
        return None
    return origin[spans[0][0]], origin[spans[-1][1] - 1] + 1


def _decimals(raw: str) -> int:
    m = re.search(r"\.(\d+)", raw)
    return len(m.group(1)) if m else 0


def numbers_match(claim: Number, source: Number) -> Optional[bool]:
    """Do two numbers state the same value, allowing for the claim's rounding?

    Returns None when they are not comparable (different kinds of unit), so callers can
    avoid flagging what they cannot judge.
    """
    cu, su = claim.unit, source.unit
    cv = claim.value * SCALE.get(cu, 1.0)
    sv = source.value * SCALE.get(su, 1.0)
    # A percentage stated as a fraction ("12%" vs "0.12").
    if cu == "%" and su == "" and abs(source.value) <= 1:
        sv *= 100
    elif su == "%" and cu == "" and abs(claim.value) <= 1:
        cv *= 100
    elif (cu in SCALE or su in SCALE) and (cu in SCALE or cu == "") and (su in SCALE or su == ""):
        pass  # "175B" vs "175 billion" vs "175000000000"
    elif cu != su:
        return None
    # The claim may round the source ("31%" for "30.9%", "2x" for "2.1x") or truncate it
    # ("18 kpc" for "18.7 kpc").
    step = 10 ** (-_decimals(claim.raw)) * SCALE.get(cu, 1.0)
    eps = 1e-9 * max(abs(cv), abs(sv), 1.0)
    if abs(cv - sv) <= 0.5 * step + eps:
        return True
    return abs(cv) < abs(sv) and abs(cv - sv) < step - eps and (cv >= 0) == (sv >= 0)
