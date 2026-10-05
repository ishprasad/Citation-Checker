"""LaTeX project handling and LaTeX -> plain text conversion with character offsets.

The converter is a small hand-written scanner rather than a chain of regex
substitutions, so it knows exactly where each emitted character lands in the
output. That lets us record precise spans for section headings and citation
markers, which the UI uses for highlighting.
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

# ----------------------------------------------------------------------------
# Project loading: pick the main file, inline \input/\include, strip comments,
# collect simple macros.
# ----------------------------------------------------------------------------

_COMMENT_LINE_RE = re.compile(r"^[ \t]*(?<!\\)%[^\n]*\n", re.MULTILINE)
_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")
_IFFALSE_RE = re.compile(r"\\iffalse\b.*?\\fi\b", re.DOTALL)
_INPUT_RE = re.compile(r"\\(input|include|subfile)(?![A-Za-z@])\s*(?:\{([^}]+)\}|([^\s{}\\]+))")
_IMPORT_RE = re.compile(r"\\(?:sub)?import\*?\s*\{([^}]*)\}\s*\{([^}]+)\}")
_DOCCLASS_RE = re.compile(r"\\documentclass\b")
_BEGIN_DOC_RE = re.compile(r"\\begin\s*\{document\}")
_END_DOC_RE = re.compile(r"\\end\s*\{document\}")

MAX_INPUT_DEPTH = 12


def strip_comments(src: str) -> str:
    src = _COMMENT_LINE_RE.sub("", src)
    src = _COMMENT_RE.sub("", src)
    return _IFFALSE_RE.sub("", src)


def decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


class LatexProject:
    """A set of source files (paths relative to the project root)."""

    def __init__(self, files: dict[str, str]):
        self.files = {posixpath.normpath(k.replace("\\", "/")).lstrip("./"): v for k, v in files.items()}

    def find(self, name: str, base_dir: str = "") -> Optional[str]:
        name = name.strip().strip('"')
        candidates = []
        for directory in (base_dir, ""):
            path = posixpath.normpath(posixpath.join(directory, name))
            candidates += [path, path + ".tex"]
        for c in candidates:
            if c in self.files:
                return c
        return None

    def main_file(self) -> Optional[str]:
        mains = [
            name
            for name, text in self.files.items()
            if name.endswith(".tex")
            and _DOCCLASS_RE.search(strip_comments(text))
            and _BEGIN_DOC_RE.search(strip_comments(text))
        ]
        if not mains:
            return None
        # Several candidates (e.g. a supplement): prefer the largest expanded body.
        return max(mains, key=lambda m: len(self.expand(m)))

    def expand(self, name: str, depth: int = 0, base_dir: Optional[str] = None) -> str:
        """Return the file's text with comments stripped and inputs inlined."""
        text = strip_comments(self.files.get(name, ""))
        if depth >= MAX_INPUT_DEPTH:
            return text
        base_dir = posixpath.dirname(name) if base_dir is None else base_dir

        def sub_import(m: re.Match) -> str:
            directory = posixpath.join(base_dir, m.group(1))
            found = self.find(m.group(2), directory)
            return self.expand(found, depth + 1, directory) if found else ""

        def sub_input(m: re.Match) -> str:
            target = m.group(2) or m.group(3)
            found = self.find(target, base_dir)
            if not found:
                return ""
            body = self.expand(found, depth + 1, base_dir)
            return f"\\clearpage {body} \\clearpage" if m.group(1) == "include" else body

        text = _IMPORT_RE.sub(sub_import, text)
        return _INPUT_RE.sub(sub_input, text)

    def macro_sources(self) -> list[str]:
        return [strip_comments(t) for n, t in self.files.items() if n.endswith((".tex", ".sty"))]


def split_document(expanded: str) -> tuple[str, str]:
    """Split expanded main file into (preamble, body)."""
    begin = _BEGIN_DOC_RE.search(expanded)
    if not begin:
        return "", expanded
    end = _END_DOC_RE.search(expanded, begin.end())
    return expanded[: begin.start()], expanded[begin.end() : end.start() if end else len(expanded)]


# ----------------------------------------------------------------------------
# Macros
# ----------------------------------------------------------------------------

_NEWCOMMAND_RE = re.compile(
    r"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\*?\s*"
    r"(?:\{\s*\\([A-Za-z@]+)\s*\}|\\([A-Za-z@]+))\s*(\[\d\])?\s*(?=\{)"
)
_DEF_RE = re.compile(r"\\(?:def|gdef|edef)\s*\\([A-Za-z@]+)\s*(#\d)?\s*(?=\{)")

# Common author shortcuts, used only when the paper does not define them itself.
DEFAULT_MACROS = {
    "eg": "e.g.",
    "ie": "i.e.",
    "etal": "et al.",
    "etc": "etc.",
    "wrt": "w.r.t.",
    "vs": "vs.",
    "cf": "cf.",
    "aka": "a.k.a.",
}


def read_group(src: str, i: int) -> tuple[str, int]:
    """src[i] == '{'. Return (content, index after the closing brace)."""
    depth = 0
    j = i
    while j < len(src):
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1 : j], j + 1
        j += 1
    return src[i + 1 :], len(src)


def read_bracket(src: str, i: int) -> tuple[str, int]:
    """src[i] == '['. Return (content, index after ']'), respecting braces."""
    depth = 0
    j = i + 1
    while j < len(src):
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "]" and depth <= 0:
            return src[i + 1 : j], j + 1
        j += 1
    return src[i + 1 :], len(src)


def collect_macros(sources: list[str]) -> dict[str, str]:
    """Zero-argument macro definitions: name -> replacement LaTeX."""
    macros: dict[str, str] = {}
    for src in sources:
        for m in _NEWCOMMAND_RE.finditer(src):
            if m.group(3):  # takes arguments
                continue
            body, _ = read_group(src, m.end())
            macros[m.group(1) or m.group(2)] = body
        for m in _DEF_RE.finditer(src):
            if m.group(2):
                continue
            body, _ = read_group(src, m.end())
            macros[m.group(1)] = body
    # Keep only simple text-like macros (e.g. \def\method{FooNet\xspace}). Style files
    # redefine \maketitle, \large etc. with internals that must not be expanded.
    simple = {}
    for name, body in macros.items():
        body = body.replace("\\xspace", "")
        if body.endswith("\\@"):  # sentence-spacing hint, e.g. \def\etal{et al.\@}
            body = body[:-2]
        if len(body) > 200 or "@" in name or _COMPLEX_MACRO_RE.search(body):
            continue
        simple[name] = body
    return simple


_COMPLEX_MACRO_RE = re.compile(
    r"@|#|\\(?:par|begin|end|[vh]box|def|let|if\w*|fi|else|setlength|global|renewcommand|newcommand|"
    r"fontsize|selectfont|vskip|hskip|vspace|hspace|relax|expandafter|csname|makeatletter|"
    r"usepackage|RequirePackage|color|newpage|clearpage)(?![A-Za-z])"
)


def expand_macros(body: str, macros: dict[str, str], passes: int = 3) -> str:
    table = {**DEFAULT_MACROS, **macros}
    # Never expand names the converter handles itself.
    table = {k: v for k, v in table.items() if k not in RESERVED_COMMANDS and "#" not in v}
    if not table:
        return body
    pattern = re.compile(r"\\(" + "|".join(sorted(map(re.escape, table), key=len, reverse=True)) + r")(?![A-Za-z@])")
    for _ in range(passes):
        new = pattern.sub(lambda m: table[m.group(1)], body)
        if new == body:
            break
        body = new
    return body


# ----------------------------------------------------------------------------
# Converter
# ----------------------------------------------------------------------------

SECTION_COMMANDS = {"chapter": 0, "section": 1, "subsection": 2, "subsubsection": 3}
RUNIN_HEADINGS = {"paragraph", "subparagraph"}

CITE_MARKER_COMMANDS = {
    "cite", "citep", "citet", "citealp", "citealt", "citenum", "Cite", "Citep", "Citet",
    "Citealp", "Citealt", "parencite", "Parencite", "textcite", "Textcite", "autocite",
    "Autocite", "footcite", "smartcite", "Smartcite", "supercite", "cites", "parencites",
    "textcites", "autocites", "citeA", "citeN", "shortcite", "newcite", "Newcite",
}
CITE_TEXT_COMMANDS = {"citeauthor", "Citeauthor", "citeyear", "citeyearpar", "citetitle"}

SKIP_ENVS = {
    "figure", "figure*", "table", "table*", "wrapfigure", "wraptable", "tabular", "tabular*",
    "tabularx", "longtable", "algorithm", "algorithm*", "algorithmic", "algorithm2e",
    "tikzpicture", "lstlisting", "verbatim", "verbatim*", "minted", "comment",
    "thebibliography", "filecontents", "filecontents*", "subfigure", "sidewaystable",
    "sidewaysfigure", "pgfpicture", "picture", "titlepage", "acks",
}
MATH_ENVS = {
    "equation", "equation*", "align", "align*", "gather", "gather*", "multline", "multline*",
    "eqnarray", "eqnarray*", "displaymath", "math", "flalign", "flalign*", "alignat", "alignat*",
    "split",
}
LIST_ENVS = {"itemize", "enumerate", "description", "list", "compactitem", "compactenum"}
ENV_ARGS = {"minipage": 1, "multicols": 1, "list": 2}

# command -> number of brace arguments to discard
DROP_ARGS = {
    "label": 1, "vspace": 1, "hspace": 1, "includegraphics": 1, "bibliographystyle": 1,
    "bibliography": 1, "addbibresource": 1, "nocite": 1, "setlength": 2, "addtolength": 2,
    "setcounter": 2, "addtocounter": 2, "newcommand": 2, "renewcommand": 2,
    "providecommand": 2, "newenvironment": 3, "renewenvironment": 3, "definecolor": 3,
    "color": 1, "pagestyle": 1, "thispagestyle": 1, "fontsize": 2, "usepackage": 1,
    "hypersetup": 1, "title": 1, "thanks": 1, "date": 1, "graphicspath": 1, "input": 1,
    "include": 1, "pagenumbering": 1, "captionsetup": 1, "newtheorem": 2, "crefname": 3,
    "Crefname": 3, "resizebox": 2, "scalebox": 1, "raisebox": 1, "rule": 2, "vskip": 0,
    "caption": 1, "bibitem": 1, "linenumbers": 0, "theoremstyle": 1, "doi": 1, "orcid": 1,
    "markboth": 2, "markright": 1, "runningtitle": 1, "runningauthor": 1, "icmltitle": 1,
    "icmltitlerunning": 1, "icmlsetsymbol": 2, "icmlcorrespondingauthor": 2, "icmlkeywords": 1,
    "keywords": 1, "DeclareMathOperator": 2, "newif": 0, "setcitestyle": 1, "tnotetext": 1,
    "fntext": 1, "ead": 1, "cortext": 1, "address": 1, "email": 1, "institute": 1, "affil": 1,
    "affiliation": 1, "author": 1, "authorrunning": 1, "titlerunning": 1, "acmConference": 1,
}
# command -> number of leading brace arguments to discard before keeping the next one
SKIP_THEN_KEEP = {"textcolor": 1, "colorbox": 1, "href": 1, "fcolorbox": 2, "foreignlanguage": 1}
TYPED_REF_COMMANDS = {"autoref", "Autoref", "cref", "Cref", "vref"}
REF_KINDS = {
    "fig": "Figure", "figure": "Figure", "tab": "Table", "table": "Table", "sec": "Section",
    "section": "Section", "subsec": "Section", "eq": "Equation", "eqn": "Equation",
    "app": "Appendix", "appendix": "Appendix", "alg": "Algorithm", "thm": "Theorem",
    "lem": "Lemma", "def": "Definition", "prop": "Proposition", "cor": "Corollary",
}
FRONT_MATTER_HINTS = ("author", "affil", "address", "email", "institut", "keyword", "orcid")
REF_COMMANDS = {"ref", "eqref", "autoref", "cref", "Cref", "pageref", "nameref", "Autoref", "vref"}

SYMBOLS = {
    "ldots": "…", "dots": "…", "textellipsis": "…", "textendash": "–", "textemdash": "—",
    "textasciitilde": "~", "textasciigrave": "`", "textbackslash": "\\", "textbar": "|",
    "textless": "<", "textgreater": ">", "textquoteleft": "‘", "textquoteright": "’",
    "textquotedblleft": "“", "textquotedblright": "”", "textdegree": "°", "S": "§", "P": "¶",
    "copyright": "©", "textregistered": "®", "texttrademark": "™", "dag": "†", "ddag": "‡",
    "LaTeX": "LaTeX", "TeX": "TeX", "ss": "ß", "ae": "æ", "AE": "Æ", "oe": "œ", "OE": "Œ",
    "o": "ø", "O": "Ø", "l": "ł", "L": "Ł", "aa": "å", "AA": "Å", "i": "ı", "j": "ȷ",
    "textpercent": "%", "texttimes": "×", "textpm": "±", "textmu": "µ", "euro": "€",
    "pounds": "£", "quad": " ", "qquad": " ", "enspace": " ", "space": " ", "newline": " ",
    "linebreak": " ", "par": "\n\n", "noindent": "", "centering": "", "item": "",
}

ACCENTS = {
    "'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303", "=": "\u0304",
    ".": "\u0307", "u": "\u0306", "v": "\u030c", "H": "\u030b", "c": "\u0327", "k": "\u0328",
    "r": "\u030a", "d": "\u0323", "b": "\u0331",
}

MATH_SYMBOLS = {
    "times": "×", "pm": "±", "mp": "∓", "sim": "~", "approx": "≈", "leq": "≤", "le": "≤",
    "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠", "cdot": "·", "cdots": "⋯", "ldots": "…",
    "dots": "…", "infty": "∞", "rightarrow": "→", "to": "→", "leftarrow": "←", "Rightarrow": "⇒",
    "in": "∈", "subset": "⊂", "subseteq": "⊆", "cup": "∪", "cap": "∩", "sum": "∑", "prod": "∏",
    "int": "∫", "partial": "∂", "nabla": "∇", "ll": "≪", "gg": "≫", "propto": "∝", "%": "%",
    "langle": "⟨", "rangle": "⟩", "circ": "∘", "degree": "°", "prime": "′", "times\\!": "×",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}

RESERVED_COMMANDS = (
    set(SECTION_COMMANDS) | RUNIN_HEADINGS | CITE_MARKER_COMMANDS | CITE_TEXT_COMMANDS
    | set(DROP_ARGS) | REF_COMMANDS | {"begin", "end", "footnote", "item", "url", "emph"}
)

_MATH_TEXT_CMD_RE = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|mathcal|mathbb|text|textrm|textbf|textit|operatorname|boldsymbol|bm)\s*\{([^{}]*)\}"
)
_MATH_CMD_RE = re.compile(r"\\([A-Za-z]+|.)")


def simplify_math(raw: str) -> str:
    s = raw
    for _ in range(3):
        s = _MATH_TEXT_CMD_RE.sub(r"\1", s)
    s = _MATH_CMD_RE.sub(lambda m: MATH_SYMBOLS.get(m.group(1), m.group(1) if not m.group(1).isalpha() else ""), s)
    s = re.sub(r"\^\s*\{([^{}]*)\}", r"^\1", s)  # keep exponents readable: 10^{-9} -> 10^-9
    s = re.sub(r"[{}_$]", "", s)
    s = s.replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Marker:
    start: int
    end: int
    keys: list[str]


@dataclass
class Heading:
    title: str
    start: int
    level: int


@dataclass
class ConvertResult:
    text: str
    headings: list[Heading] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)


# (command, optional args, keys) -> display text, or None to drop
CiteRenderer = Callable[[str, list[str], list[str]], str]


class LatexConverter:
    def __init__(self, cite_renderer: Optional[CiteRenderer] = None, cite_text: Optional[Callable[[str, list[str]], str]] = None):
        self.cite_renderer = cite_renderer
        self.cite_text = cite_text
        self.out: list[str] = []
        self.pos = 0
        self.pending = 0  # 0 none, 1 space, 2 paragraph break
        self.last = "\n"
        self.headings: list[Heading] = []
        self.markers: list[Marker] = []

    # -- emission --------------------------------------------------------------

    def _write(self, s: str) -> None:
        self.out.append(s)
        self.pos += len(s)
        self.last = s[-1]

    def _flush(self) -> None:
        if self.pending == 2 and self.pos > 0:
            self._write("\n\n")
        elif self.pending == 1 and self.pos > 0 and not self.last.isspace():
            self._write(" ")
        self.pending = 0

    def space(self) -> None:
        self.pending = max(self.pending, 1)

    def para(self) -> None:
        self.pending = 2

    def emit(self, s: str) -> None:
        for chunk in re.split(r"(\s+)", s):
            if not chunk:
                continue
            if chunk.isspace():
                if chunk.count("\n") >= 2:
                    self.para()
                else:
                    self.space()
            else:
                self._flush()
                self._write(chunk)

    # -- scanning --------------------------------------------------------------

    def convert(self, src: str) -> ConvertResult:
        self._run(src)
        return ConvertResult("".join(self.out), self.headings, self.markers)

    def _run(self, src: str) -> None:
        i, n = 0, len(src)
        while i < n:
            c = src[i]
            if c == "\\":
                i = self._command(src, i)
            elif c == "{" or c == "}":
                i += 1
            elif c == "$":
                if src.startswith("$$", i):
                    end = src.find("$$", i + 2)
                    end = n if end == -1 else end + 2
                    self.space()
                    i = end
                else:
                    j = i + 1
                    while j < n and not (src[j] == "$" and src[j - 1] != "\\"):
                        j += 1
                    self.emit(simplify_math(src[i + 1 : j]))
                    i = j + 1
            elif c == "~":
                self.space()
                i += 1
            elif c == "-" and src.startswith("---", i):
                self.emit("—")
                i += 3
            elif c == "-" and src.startswith("--", i):
                self.emit("–")
                i += 2
            elif c == "`" and src.startswith("``", i):
                self.emit("“")
                i += 2
            elif c == "'" and src.startswith("''", i):
                self.emit("”")
                i += 2
            elif c == "`":
                self.emit("‘")
                i += 1
            elif c in "&#^_":
                if c == "&":
                    self.space()
                i += 1
            else:
                j = i
                while j < n and src[j] not in "\\{}$~-`'&#^_":
                    j += 1
                if j == i:  # lone '-' or "'"
                    j = i + 1
                self.emit(src[i:j])
                i = j

    def _skip_ws(self, src: str, i: int) -> int:
        while i < len(src) and src[i] in " \t\n":
            i += 1
        return i

    def _opt_args(self, src: str, i: int, max_n: int = 3) -> tuple[list[str], int]:
        opts = []
        while len(opts) < max_n:
            j = self._skip_ws(src, i)
            if j < len(src) and src[j] == "[":
                content, i = read_bracket(src, j)
                opts.append(content)
            else:
                break
        return opts, i

    def _arg(self, src: str, i: int) -> tuple[Optional[str], int]:
        j = self._skip_ws(src, i)
        if j < len(src) and src[j] == "{":
            return read_group(src, j)
        if j < len(src) and src[j] == "\\":  # \cmd as a single-token argument
            m = re.match(r"\\[A-Za-z]+|\\.", src[j:])
            return (m.group(0), j + len(m.group(0))) if m else (None, i)
        if j < len(src) and src[j] not in "}[":
            return src[j], j + 1
        return None, i

    def _sub(self, latex: str) -> None:
        """Convert a nested fragment into the current output."""
        self._run(latex)

    def _command(self, src: str, i: int) -> int:
        n = len(src)
        if i + 1 >= n:
            return n
        nxt = src[i + 1]
        if not nxt.isalpha() and nxt != "@":
            return self._symbol_command(src, i, nxt)

        m = re.match(r"[A-Za-z@]+", src[i + 1 :])
        name = m.group(0)
        i = i + 1 + len(name)
        star = i < n and src[i] == "*"
        if star:
            i += 1

        if name == "begin":
            return self._begin(src, i)
        if name == "end":
            env, i = self._arg(src, i)
            if env in LIST_ENVS or env in {"abstract", "quote", "quotation", "center"}:
                self.para()
            return i
        if name in SECTION_COMMANDS:
            _, i = self._opt_args(src, i, 1)
            title, i = self._arg(src, i)
            self._heading(title or "", SECTION_COMMANDS[name])
            return i
        if name in RUNIN_HEADINGS:
            _, i = self._opt_args(src, i, 1)
            title, i = self._arg(src, i)
            self.para()
            self._sub(title or "")
            self.emit(". " if title and not title.rstrip().endswith((".", "?", "!", ":")) else " ")
            return i
        if name in CITE_MARKER_COMMANDS:
            return self._cite(src, i, name)
        if name in CITE_TEXT_COMMANDS:
            _, i = self._opt_args(src, i, 2)
            keys, i = self._arg(src, i)
            if self.cite_text and keys:
                self.emit(self.cite_text(name, _split_keys(keys)))
            return i
        if name in REF_COMMANDS:
            label, i = self._arg(src, i)
            kind = ""
            if name in TYPED_REF_COMMANDS and label:
                kind = REF_KINDS.get(label.split(":", 1)[0].strip().lower(), "")
            self.emit(f"{kind} §" if kind else "§")
            return i
        if name == "item":
            opts, i = self._opt_args(src, i, 1)
            self.para()
            self.emit("• " + (opts[0] + " " if opts else ""))
            return i
        if name == "footnote":
            _, i = self._opt_args(src, i, 1)
            content, i = self._arg(src, i)
            if content:
                self.emit(" (")
                self._sub(content)
                self.emit(")")
            return i
        if name in ("url", "nolinkurl"):
            content, i = self._arg(src, i)
            self.emit(content or "")
            return i
        if name in DROP_ARGS:
            _, i = self._opt_args(src, i, 2)
            for _ in range(DROP_ARGS[name]):
                _, i = self._arg(src, i)
            return i
        if name in SKIP_THEN_KEEP:
            _, i = self._opt_args(src, i, 1)
            for _ in range(SKIP_THEN_KEEP[name]):
                _, i = self._arg(src, i)
            return i  # the kept argument is scanned as an ordinary group
        if name in SYMBOLS:
            value = SYMBOLS[name]
            if value == "\n\n":
                self.para()
            elif value.isspace():
                self.space()
            else:
                self.emit(value)
            return i
        if name in ACCENTS and len(name) == 1:
            return self._accent(src, i, name)
        if name in ("appendix",):
            return i
        if any(s in name.lower() for s in FRONT_MATTER_HINTS):
            # Author blocks, affiliations etc. (often conference-specific macros).
            _, i = self._opt_args(src, i, 1)
            while True:
                arg, j = self._arg(src, i)
                if arg is None or not src[self._skip_ws(src, i)] == "{":
                    return i
                i = j
        # Unknown command: drop the command, keep its arguments' content.
        _, i = self._opt_args(src, i, 1)
        return i

    def _symbol_command(self, src: str, i: int, c: str) -> int:
        if c in ACCENTS:
            return self._accent(src, i + 2, c)
        if c == "\\":
            _, j = self._opt_args(src, i + 2, 1)
            self.space()
            return j
        if c in "%&_#${}":
            self.emit(c)
            return i + 2
        if c == "[":  # display math \[ ... \]
            end = src.find("\\]", i + 2)
            self.space()
            return len(src) if end == -1 else end + 2
        if c == "(":  # inline math \( ... \)
            end = src.find("\\)", i + 2)
            end = len(src) if end == -1 else end
            self.emit(simplify_math(src[i + 2 : end]))
            return end + 2
        if c in " \n\t":
            self.space()
            return i + 2
        return i + 2  # \, \; \! \@ \/ etc.

    def _accent(self, src: str, i: int, accent: str) -> int:
        arg, j = self._arg(src, i)
        if arg is None:
            return i
        base = arg.strip("{}").strip()
        if base in ("\\i", "\\j"):  # dotless i/j under an accent: \'{\i} -> í
            base = base[1]
        elif base.startswith("\\"):
            base = SYMBOLS.get(base[1:], base[1:])
        if not base:
            return j
        self.emit(unicodedata.normalize("NFC", base[0] + ACCENTS[accent]) + base[1:])
        return j

    def _heading(self, title_latex: str, level: int) -> None:
        self.para()
        sub = LatexConverter()
        title = sub.convert(title_latex).text.strip()
        self._flush()
        self.headings.append(Heading(title, self.pos, level))
        self.emit(title)
        self.para()

    def _begin(self, src: str, i: int) -> int:
        env, i = self._arg(src, i)
        env = (env or "").strip()
        if env in SKIP_ENVS or env in MATH_ENVS:
            end = _find_env_end(src, i, env)
            if env in MATH_ENVS:
                self.space()
            else:
                self.para()
            return end
        if env == "abstract":
            self._heading("Abstract", 1)
            return i
        if env in LIST_ENVS or env in {"quote", "quotation", "center"}:
            self.para()
        if env in ENV_ARGS:
            _, i = self._opt_args(src, i, 1)
            for _ in range(ENV_ARGS[env]):
                _, i = self._arg(src, i)
        return i

    def _cite(self, src: str, i: int, name: str) -> int:
        opts, i = self._opt_args(src, i, 2)
        groups = []
        multi = name.endswith("cites")
        while True:
            arg, j = self._arg(src, i)
            if arg is None or (groups and not multi):
                break
            groups.append(arg)
            i = j
            if not multi:
                break
            _, i = self._opt_args(src, i, 2)
        keys = [k for g in groups for k in _split_keys(g)]
        if not keys or not self.cite_renderer:
            return i
        display = self.cite_renderer(name, opts, keys)
        if not display:
            return i
        if self.pending == 0 and self.pos > 0 and (self.last.isalnum() or self.last in ")."):
            self.pending = 1
        self._flush()
        start = self.pos
        self._write(display)
        self.markers.append(Marker(start, self.pos, keys))
        return i


def _split_keys(group: str) -> list[str]:
    return [k.strip() for k in group.split(",") if k.strip()]


def _find_env_end(src: str, i: int, env: str) -> int:
    begin_pat = re.compile(r"\\begin\s*\{" + re.escape(env) + r"\}")
    end_pat = re.compile(r"\\end\s*\{" + re.escape(env) + r"\}")
    depth = 1
    pos = i
    while depth:
        e = end_pat.search(src, pos)
        if not e:
            return len(src)
        b = begin_pat.search(src, pos, e.start())
        if b:
            depth += 1
            pos = b.end()
        else:
            depth -= 1
            pos = e.end()
    return pos


def latex_to_plain(fragment: str) -> str:
    """Convert a short fragment (e.g. a bibliography field) to plain text."""
    text = LatexConverter().convert(fragment.replace("\\newblock", " ")).text
    return re.sub(r"\s+", " ", text).strip()
