# Citation Checker

For every citation in a research paper, Citation Checker asks one question: does the cited source actually say what the citing author claims it says? It runs on NVIDIA Nemotron models served through Nebius Token Factory.

> Work in progress. See `CITATION_CHECKER_BRIEF.md` for the full plan. Done so far: milestone 1 (scaffold, config, LLM wrapper) and milestone 2 (arXiv LaTeX ingest).

## Setup

```bash
pip install -r requirements-dev.txt
cp .env.example .env          # then add NEBIUS_API_KEY
python scripts/list_models.py # copy the exact Nemotron / embedding model IDs into .env
python scripts/smoke_test.py  # one call to each configured model, with token usage
python scripts/ingest_paper.py 1706.03762  # parse a paper: references and citing sentences
uvicorn app.main:app --reload
```

Run the tests with `python -m pytest`. The tests use a fake client and do not call the API.

## Model roles

| Role | Env var | Model | Reasoning |
|------|---------|-------|-----------|
| Claim extraction (high volume) | `MODEL_EXTRACT` | Nemotron 3 Nano | off |
| Verdicts | `MODEL_JUDGE` | Nemotron 3 Super | on |
| Escalation (full paper, long context) | `MODEL_ESCALATE` | Nemotron 3 Ultra | on |
| Scanned pages (optional) | `MODEL_VISION` | Nemotron 3 Nano Omni | off |
| Retrieval | `MODEL_EMBED` | Token Factory embedding model | n/a |

`app/llm.py` handles everything about model access:

- **Reasoning toggle.** Sent as `chat_template_kwargs.enable_thinking` by default. Set `REASONING_CONTROL=system_prompt` for models that use `/think` and `/no_think` instead.
- **Structured output.** Tries `json_schema` first, then `json_object`, then a prompt-only instruction, falling back automatically per model. Replies are validated with pydantic, and an invalid reply gets one corrective retry.
- **Retries.** Exponential backoff on 429 and 5xx errors.
- **Concurrency.** A global semaphore set by `MAX_CONCURRENCY`.
- **Cache.** A SQLite disk cache in `.cache/llm_cache.sqlite`, keyed by a hash of (model, messages, params). Embeddings are cached per text.
- **Usage tracking.** Tokens and estimated cost are recorded per call, per model and per job, which feeds the UI cost panel. Prices are set per role with `PRICE_*` variables.
- **Token budget.** An optional per-job limit set by `MAX_TOKENS_PER_JOB`.

## Ingest (arXiv LaTeX source)

For an arXiv ID, the app downloads the paper's LaTeX source rather than the PDF. Citation commands and the compiled bibliography map every citation to its reference exactly, which PDF parsing cannot do.

- **Source handling.** `\input`, `\include` and `\import` are inlined. Comments and `\iffalse` blocks are stripped. Simple user macros (such as `\def\method{FooNet\xspace}`) are expanded.
- **Text conversion.** `app/pipeline/latex.py` converts LaTeX to plain text with exact character offsets for section headings and citation markers. Figures, tables, display math and front matter are skipped. Inline math is simplified (for example, `10^{-9}` becomes `10^-9`).
- **Bibliography.** `app/pipeline/bibliography.py` reads natbib or numeric `\bibitem` entries (in a `.bbl` file or inline), biblatex `.bbl` files and `.bib` files, in that order of preference. It extracts title, authors, year, DOI and arXiv ID.
- **Citation markers.** Markers are rendered the way the paper shows them: numeric (`[3, 5]`, `Lee et al. [2]`) or author-year (`(Kaplan et al., 2020)`, `Brown et al. (2020)`).
- **Sentences.** `app/pipeline/sentences.py` splits sentences without breaking on abbreviations such as `et al.`, `e.g.` and `Fig.`, and keeps each citation marker attached to its sentence.
