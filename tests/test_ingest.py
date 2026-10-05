"""Ingest tests: arXiv IDs, LaTeX conversion, bibliography parsing, citing sentences (no network)."""

from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import pytest

from app.pipeline.ingest import (
    InvalidArxivId,
    NoLatexSource,
    normalize_arxiv_id,
    parse_latex_project,
    unpack_source,
)
from app.pipeline.latex import LatexConverter, simplify_math

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, str]:
    root = FIXTURES / name
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in root.rglob("*") if p.is_file()}


# ----------------------------------------------------------------------------- arXiv IDs


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2401.12345", "2401.12345"),
        ("2401.12345v2", "2401.12345v2"),
        ("  arXiv:2401.12345 ", "2401.12345"),
        ("https://arxiv.org/abs/2401.12345v3", "2401.12345v3"),
        ("https://arxiv.org/pdf/2401.12345", "2401.12345"),
        ("https://arxiv.org/pdf/2401.12345v1.pdf", "2401.12345v1"),
        ("arxiv.org/abs/1706.03762?context=cs", "1706.03762"),
        ("<https://arxiv.org/abs/2401.12345>.", "2401.12345"),
        ("hep-th/9901001", "hep-th/9901001"),
        ("https://arxiv.org/abs/math.GT/0309136v2", "math.GT/0309136v2"),
        ("0704.0001", "0704.0001"),
    ],
)
def test_normalize_arxiv_id(raw, expected):
    assert normalize_arxiv_id(raw) == expected


@pytest.mark.parametrize("raw", ["", "hello", "12345", "https://example.com/abs/2401.12345", "2401.123"])
def test_normalize_arxiv_id_rejects(raw):
    with pytest.raises(InvalidArxivId):
        normalize_arxiv_id(raw)


# ------------------------------------------------------------------------- source files


def _tar_gz(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_unpack_tarball_keeps_only_source_files():
    files = unpack_source(_tar_gz({"main.tex": "x", "refs.bib": "y", "fig.png": "z", "./sec/a.tex": "w"}))
    assert set(files) == {"main.tex", "refs.bib", "./sec/a.tex"}


def test_unpack_single_gzipped_tex():
    assert unpack_source(gzip.compress(b"\\documentclass{article}")) == {"main.tex": "\\documentclass{article}"}


def test_unpack_pdf_only():
    with pytest.raises(NoLatexSource) as exc:
        unpack_source(gzip.compress(b"%PDF-1.5 ..."))
    assert exc.value.pdf.startswith(b"%PDF-")


# ---------------------------------------------------------------------------- converter


def test_converter_basics():
    text = LatexConverter().convert(
        r"Caf\'e na\"ive Garc{\'\i}a -- 10--20 ``quoted'' \emph{emph} \textbf{bold} 50\% \& more\\ next"
    ).text
    assert text == "Café naïve García – 10–20 “quoted” emph bold 50% & more next"


def test_converter_skips_math_environments_and_keeps_inline_math():
    text = LatexConverter().convert(
        "Loss is low \\begin{equation} L = \\sum x \\end{equation} and $p < 0.05$ with $\\alpha=0.1$."
    ).text
    assert text == "Loss is low and p < 0.05 with α=0.1."


@pytest.mark.parametrize(
    "math, expected",
    [(r"10^{-9}", "10^-9"), (r"12\%", "12%"), (r"\mathrm{BLEU}_{4} \approx 28.4", "BLEU4 ≈ 28.4"), (r"3\times", "3×")],
)
def test_simplify_math(math, expected):
    assert simplify_math(math) == expected


# ------------------------------------------------------------------- fixture paper (numeric)


@pytest.fixture(scope="module")
def numeric_paper():
    return parse_latex_project(load_fixture("latex_numeric"))


def test_fixture_references(numeric_paper):
    paper, _ = numeric_paper
    refs = {r.key: r for r in paper.references}
    assert paper.citation_style == "numeric"
    assert [r.label for r in paper.references] == ["1", "2", "3"]  # .bbl order
    assert refs["jones2019"].title == "Adapters for everything"
    assert refs["jones2019"].doi == "10.1234/jot.2019.42"
    assert refs["jones2019"].year == 2019
    assert refs["lee2021"].arxiv_id == "2101.00001"
    assert refs["lee2021"].title == "Pretraining for low-resource NLP"
    assert "García" in refs["lee2021"].raw
    assert refs["smith2020"].title == "Deep networks on ImageNet"
    # \cite{missingkey} is counted but cannot be mapped.
    assert (paper.citation_keys_total, paper.citation_keys_mapped) == (4, 3)


def test_fixture_text(numeric_paper):
    paper, _ = numeric_paper
    text = paper.text
    assert "FooNet" in text and "BarSet" in text  # macros expanded
    assert "should not appear" not in text  # commented-out definition
    assert "caption" not in text  # figure environment skipped
    assert "disabled" not in text  # \iffalse block skipped
    assert "Ada Lovelace" not in text and "A Fixture Paper" not in text  # front matter dropped
    assert "\\" not in text and "{" not in text
    assert [s.title for s in paper.sections] == ["Abstract", "Introduction", "Method", "Related Work"]
    for section in paper.sections:
        assert text[section.start : section.end] == section.text


def test_fixture_citing_sentences(numeric_paper):
    paper, sentences = numeric_paper
    by_text = {s.text: s for s in sentences}

    s = by_text["Deep networks dominate vision benchmarks [3, 1]."]
    assert s.section == "Introduction"
    assert s.reference_keys == ["smith2020", "jones2019"]
    # The comment line inside the intro paragraph did not split it.
    assert "However, they are data hungry [2]." in by_text

    s = by_text["The model reaches 12% higher accuracy than the baseline [3]."]
    assert s.reference_keys == ["smith2020"]

    s = by_text["Lee et al. [2] report that pretraining helps in low-resource settings."]
    assert [paper.text[a:b] for a, b in s.marker_spans] == ["Lee et al. [2]"]

    assert "Training is stable for learning rates up to 10^-3 (see [3, Sec. 3])." in by_text
    # "e.g." and "et al." do not end sentences.
    assert by_text["Several methods exist, e.g. adapters and prompts [1, 2]."].reference_keys == [
        "jones2019",
        "lee2021",
    ]
    # Unmapped citation does not produce a citing sentence.
    assert not any("Early work" in t for t in by_text)

    for sent in sentences:
        assert paper.text[sent.start : sent.end] == sent.text
        for a, b in sent.marker_spans:
            assert sent.start <= a < b <= sent.end


# --------------------------------------------------------------- author-year (natbib .bbl)

AUTHOR_YEAR_MAIN = r"""\documentclass{article}
\usepackage{natbib}
\begin{document}
\section{Intro}
Language models scale well \citep{kaplan2020,brown2020}. \citet{brown2020} introduced few-shot prompting.
Chain of thought helps \citep[see][p.~4]{wei2022a}, as noted by \citeauthor{kaplan2020}.
\end{document}
"""

AUTHOR_YEAR_BBL = r"""\begin{thebibliography}{3}
\expandafter\ifx\csname natexlab\endcsname\relax\def\natexlab#1{#1}\fi

\bibitem[{Brown et~al.(2020)Brown, Mann, Ryder, and Subbiah}]{brown2020}
Tom Brown, Benjamin Mann, Nick Ryder, and Melanie Subbiah. 2020.
\newblock \href {https://arxiv.org/abs/2005.14165} {Language models are few-shot learners}.
\newblock \emph{NeurIPS}.

\bibitem[{Kaplan et~al.(2020)}]{kaplan2020}
Jared Kaplan, Sam McCandlish, and Tom Henighan. 2020.
\newblock Scaling laws for neural language models.
\newblock \emph{arXiv preprint arXiv:2001.08361}.

\bibitem[{Wei et~al.(2022{\natexlab{a}})}]{wei2022a}
Jason Wei and Xuezhi Wang. 2022{\natexlab{a}}.
\newblock Chain of thought prompting.
\newblock \emph{NeurIPS}. \doi{10.5555/abc.123}.
\end{thebibliography}
"""


def test_author_year_rendering():
    paper, sentences = parse_latex_project({"main.tex": AUTHOR_YEAR_MAIN, "main.bbl": AUTHOR_YEAR_BBL})
    assert paper.citation_style == "author_year"
    texts = [s.text for s in sentences]
    assert texts == [
        "Language models scale well (Kaplan et al., 2020; Brown et al., 2020).",
        "Brown et al. (2020) introduced few-shot prompting.",
        "Chain of thought helps (see Wei et al., 2022a, p. 4), as noted by Kaplan et al..",
    ]
    refs = {r.key: r for r in paper.references}
    assert refs["brown2020"].label == "Brown et al., 2020"
    assert refs["brown2020"].title == "Language models are few-shot learners"
    assert refs["brown2020"].arxiv_id == "2005.14165"
    assert refs["wei2022a"].label == "Wei et al., 2022a"
    assert refs["kaplan2020"].arxiv_id == "2001.08361"


def test_natbib_numbers_option_forces_numeric():
    main = AUTHOR_YEAR_MAIN.replace(r"\usepackage{natbib}", r"\usepackage[numbers]{natbib}")
    paper, sentences = parse_latex_project({"main.tex": main, "main.bbl": AUTHOR_YEAR_BBL})
    assert paper.citation_style == "numeric"
    assert sentences[0].text == "Language models scale well [2, 1]."


# ------------------------------------------------------------------------- biblatex .bbl

BIBLATEX_BBL = r"""
\refsection{0}
  \datalist[entry]{nty/global//global/global}
    \entry{vaswani2017}{inproceedings}{}
      \name{author}{2}{}{%
        {{hash=a1}{%
           family={Vaswani},
           familyi={V\bibinitperiod},
           given={Ashish},
           giveni={A\bibinitperiod},
        }}%
        {{hash=a2}{%
           family={Shazeer},
           familyi={S\bibinitperiod},
           given={Noam},
           giveni={N\bibinitperiod},
        }}%
      }
      \field{booktitle}{Advances in Neural Information Processing Systems}
      \field{title}{Attention is All you Need}
      \field{year}{2017}
      \field{eprinttype}{arXiv}
      \field{eprint}{1706.03762}
      \verb{doi}
      \verb 10.5555/3295222.3295349
      \endverb
    \endentry
  \enddatalist
\endrefsection
"""


def test_biblatex_bbl():
    main = r"""\documentclass{article}
\usepackage[style=authoryear]{biblatex}
\begin{document}
Transformers use attention \parencite{vaswani2017}. \textcite{vaswani2017} proposed it.
\end{document}"""
    paper, sentences = parse_latex_project({"main.tex": main, "main.bbl": BIBLATEX_BBL})
    [ref] = paper.references
    assert ref.title == "Attention is All you Need"
    assert ref.arxiv_id == "1706.03762"
    assert ref.doi == "10.5555/3295222.3295349"
    assert ref.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert paper.citation_style == "author_year"
    assert [s.text for s in sentences] == [
        "Transformers use attention (Vaswani and Shazeer, 2017).",
        "Vaswani and Shazeer (2017) proposed it.",
    ]


# --------------------------------------------------------------------------- .bib only

BIBTEX = r"""
@string{neurips = "NeurIPS"}
@article{he2016,
  title = {Deep Residual Learning for Image Recognition},
  author = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal = neurips,
  year = 2016,
  doi = {10.1109/CVPR.2016.90},
}
@misc{radford2019,
  title = "Language Models are {Unsupervised} Multitask Learners",
  author = "Radford, Alec and others",
  year = {2019},
  eprint = {1234.56789},
  archivePrefix = {arXiv},
}
"""


def test_bibtex_only():
    main = r"""\documentclass{article}
\begin{document}
Residual connections ease optimization \cite{he2016}. GPT-2 is a language model \cite{radford2019}.
\bibliographystyle{plain}
\bibliography{refs}
\end{document}"""
    paper, sentences = parse_latex_project({"main.tex": main, "refs.bib": BIBTEX})
    refs = {r.key: r for r in paper.references}
    assert refs["he2016"].title == "Deep Residual Learning for Image Recognition"
    assert refs["he2016"].doi == "10.1109/CVPR.2016.90"
    assert refs["he2016"].authors[0] == "Kaiming He"
    assert refs["he2016"].year == 2016
    assert refs["radford2019"].title == "Language Models are Unsupervised Multitask Learners"
    assert refs["radford2019"].arxiv_id == "1234.56789"
    # Numbered in first-citation order when there is no .bbl.
    assert [s.text for s in sentences] == [
        "Residual connections ease optimization [1].",
        "GPT-2 is a language model [2].",
    ]


# ------------------------------------------------------------------------ sentence edges


def _sentences(body: str) -> list[str]:
    main = "\\documentclass{article}\\begin{document}\n" + body + "\n\\end{document}"
    bbl = "\\begin{thebibliography}{2}\\bibitem{a} A. Title A. 2020.\\bibitem{b} B. Title B. 2021.\\end{thebibliography}"
    _, sentences = parse_latex_project({"main.tex": main, "main.bbl": bbl})
    return [s.text for s in sentences]


def test_marker_after_period_attaches_to_previous_sentence():
    assert _sentences(r"Prior work exists. \cite{a} Next we go on.") == ["Prior work exists. [1]"]


def test_marker_starting_sentence_stays_with_it():
    assert _sentences(r"Prior work exists. \cite{a} proposed a method.") == ["[1] proposed a method."]


def test_abbreviations_and_initials_do_not_split():
    assert _sentences(r"As in Fig. 3 and Sec. 2, J. Smith showed this \cite{a}. Another one \cite{b}.") == [
        "As in Fig. 3 and Sec. 2, J. Smith showed this [1].",
        "Another one [2].",
    ]
