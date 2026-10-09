"""Build small synthetic PDFs for parser tests (no binary fixtures in the repo)."""

from __future__ import annotations

import pymupdf

BODY = 10
LEFT = 72
WIDTH = 468


class PdfBuilder:
    def __init__(self) -> None:
        self.doc = pymupdf.open()
        self.page = None
        self.y = 0.0
        self.new_page()

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=612, height=792)
        self.y = 72.0

    def _room(self, height: float) -> None:
        if self.y + height > 740:
            self.new_page()

    def title(self, text: str) -> None:
        self.page.insert_text((LEFT, self.y + 18), text, fontsize=18, fontname="hebo")
        self.y += 40

    def heading(self, text: str) -> None:
        self._room(30)
        self.page.insert_text((LEFT, self.y + 12), text, fontsize=12, fontname="hebo")
        self.y += 26

    def paragraph(self, text: str, size: float = BODY, left: float = LEFT, width: float = WIDTH) -> None:
        lines = _wrap(text, size, width)
        height = len(lines) * size * 1.25 + 8
        self._room(height)
        for i, line in enumerate(lines):
            self.page.insert_text((left, self.y + size + i * size * 1.25), line, fontsize=size, fontname="helv")
        self.y += height + 4

    def reference(self, text: str, hanging: bool = True, size: float = BODY) -> None:
        """One bibliography entry; continuation lines are indented when `hanging`."""
        lines = _wrap(text, size, WIDTH - 10)
        height = len(lines) * size * 1.25 + 6
        self._room(height)
        for i, line in enumerate(lines):
            x = LEFT + (10 if hanging and i else 0)
            self.page.insert_text((x, self.y + size + i * size * 1.25), line, fontsize=size, fontname="helv")
        self.y += height

    def reference_parts(self, parts: list[tuple[str, bool]], size: float = BODY) -> None:
        """A one-line entry in mixed fonts, e.g. [("White NJ (2018) ", False), ("Title", True)]."""
        self._room(size * 1.25 + 6)
        x = LEFT
        for text, bold in parts:
            font = "hebo" if bold else "helv"
            self.page.insert_text((x, self.y + size), text, fontsize=size, fontname=font)
            x += pymupdf.get_text_length(text, fontname=font, fontsize=size)
        self.y += size * 1.25 + 6

    def note(self, text: str, x: float, y: float, size: float = 7) -> None:
        """Free-standing small text, e.g. a journal sidebar or masthead."""
        self.page.insert_text((x, y), text, fontsize=size, fontname="helv")

    def table(self, rows: list[list[str]], size: float = BODY) -> None:
        """A ruled table (grid lines, so PDF table detection sees it)."""
        cols = len(rows[0])
        row_h, col_w = size * 1.8, WIDTH / cols
        height = row_h * len(rows)
        self._room(height + 10)
        top = self.y
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                self.page.insert_text((LEFT + c * col_w + 4, top + r * row_h + size + 2), cell, fontsize=size,
                                      fontname="helv")
        for r in range(len(rows) + 1):
            self.page.draw_line((LEFT, top + r * row_h), (LEFT + WIDTH, top + r * row_h))
        for c in range(cols + 1):
            self.page.draw_line((LEFT + c * col_w, top), (LEFT + c * col_w, top + height))
        self.y += height + 14

    def bytes(self, title: str = "") -> bytes:
        if title:
            self.doc.set_metadata({"title": title})
        return self.doc.tobytes()


def _wrap(text: str, size: float, width: float) -> list[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        candidate = f"{line} {w}".strip()
        if pymupdf.get_text_length(candidate, fontname="helv", fontsize=size) > width and line:
            lines.append(line)
            line = w
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines
