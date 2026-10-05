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
    r"(?P<unit>\s?%|\s?percent\b|\s?pp\b|\s?(?:x|×)(?![\w])|-?fold\b|[kKMB](?![\w]))?"
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
             "k": "k", "K": "k", "M": "M", "B": "B"}


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
