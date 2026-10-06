# Citation Checker

For every citation in a research paper, Citation Checker asks one question: does the cited source actually say what the citing author claims it says? It runs on NVIDIA Nemotron models served through Nebius Token Factory.

> Work in progress. See `CITATION_CHECKER_BRIEF.md` for the full plan. Done so far: milestones 1–5 (scaffold and LLM wrapper, arXiv LaTeX ingest, reference resolution and full-text fetching, claim extraction and evidence retrieval, judging with quote verification and escalation).

## Setup

```bash
pip install -r requirements-dev.txt
cp .env.example .env          # then add NEBIUS_API_KEY
python scripts/list_models.py # copy the exact Nemotron / embedding model IDs into .env
python scripts/smoke_test.py  # one call to each configured model, with token usage
python scripts/ingest_paper.py 1706.03762  # parse a paper: references and citing sentences
python scripts/resolve_refs.py 1706.03762  # find every cited paper, fetch its text, print coverage
python scripts/sample_claims.py 2310.06825 # extracted claims and top evidence passages for a few citations
python scripts/check_paper.py 2310.06825   # whole pipeline: verdicts, quotes, escalations, token usage
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

## Reference resolution (`app/pipeline/references.py`)

Each reference is matched to a real paper, and its text is fetched using free APIs only.

- **Identify.** The app first uses the arXiv ID or DOI from the bibliography entry. arXiv metadata is looked up in batches, as are OpenAlex records by DOI. Otherwise it runs an OpenAlex title search, then a Crossref bibliographic search. A match must reach a title-similarity threshold of 0.88 and pass a year sanity check. A shortened cited title is accepted if it is a prefix of the real title.
- **Fetch text.** The app tries these sources in order:
  1. arXiv LaTeX source, converted with figure and table captions kept, since captions often hold the cited numbers.
  2. The arXiv PDF.
  3. An open-access PDF listed by OpenAlex.
  4. An arXiv preprint "twin" of a published paper.
  5. The abstract alone, marked `abstract_only`.

  PDF text comes from PyMuPDF, with hyphenation and ligatures repaired. The cited paper's own reference list is stored next to its text, for the secondary-citation detector.
- **Retractions.** The app uses OpenAlex `is_retracted` and Crossref `updated-by` entries with `type: "retraction"`. Crossref includes Retraction Watch data. If neither source has a record, `retracted` stays `null`.
- **Caching and politeness.** Every HTTP response and every resolved reference is cached under `.cache/`. Requests carry a `mailto` User-Agent, and arXiv requests are spaced 3 seconds apart. A transient download failure degrades to the abstract and is not cached, so the next run retries.

Coverage on real papers:

| Paper | Full text | Abstract only | Unavailable |
|---|---|---|---|
| Attention Is All You Need | 32 of 40 | 8 | 0 |
| Chain-of-Thought Prompting | 82 of 86 | 2 | 2 |

## Claim extraction (`app/pipeline/claims.py`, Nemotron Nano)

Each citing sentence goes to the extraction model with its citation markers replaced by short ids (`[R1, R2]`), along with the neighbouring sentences and the citing paper's title. For each marker, the model writes what the sentence attributes to the cited work: the statements that work itself must contain for the citation to be accurate. Framing it this way stops a small model from attributing the citing authors' own results to the works they cite ("our changes to FlashAttention [R1] give a 2x speedup" yields only "FlashAttention is an implementation of attention").

Code checks guard the output:
- Unknown reference ids are dropped.
- Claims whose content words mostly don't appear in the citing sentence are dropped.
- Numbers are re-extracted from the claim text.
- A reference the model skipped gets a lenient `background` claim built from the phrase its marker is attached to, never the whole sentence.

Reference titles are deliberately not shown to the model, because it tended to copy them into claims.

## Evidence retrieval (`app/pipeline/evidence.py`, embeddings)

Each cited paper is split into chunks of about 400 tokens with about 80 tokens of overlap, on sentence boundaries. Chunks never cross a section heading, and each keeps its heading and character offsets. For PDF text, only numbered headings or standard section names count as headings. Chunks are embedded once per reference (cached on disk). Claims are embedded with a retrieval instruction, which Qwen3-Embedding expects; set it with `EMBED_QUERY_INSTRUCTION`. Each (claim, reference) pair gets the top 5 chunks by cosine similarity, plus the abstract.

## Judging (`app/pipeline/judge.py`, Nemotron Super) and escalation (Nemotron Ultra)

For each (claim, reference) pair, the judge gets the citing sentence and its neighbours, the extracted claim (marked as a hint), and the retrieved passages. It returns a label (`supported`, `overstated`, `not_found`, `contradicted` or `unverifiable`), a confidence, a quote, a neutral explanation, and structured signals for the failure-mode detectors.

The judge first works out what the citation marker is actually attached to. This catches extraction mistakes, such as a marker that only names the GSM8K benchmark being read as support for the citing paper's own result.

Hard rules are enforced in code:
- A `supported`, `overstated` or `contradicted` verdict needs a quote found verbatim in the cited text. Matching ignores differences in quotes, dashes, ligatures, accents, PDF hyphenation and inline citation markers, and allows "..." elisions. An unverified quote halves the confidence and adds the flag `quote_unverified`.
- Background claims can only be `supported` or `unverifiable`.
- A `not_found` judged against an abstract alone becomes `unverifiable`.
- References with no available text get `unverifiable` with the reason, without a model call.
- Explanations are passed through a tone filter, so no "error", "wrong" or "misconduct".

A verdict is escalated to Ultra, with the full text of the cited paper (up to `ESCALATE_MAX_CHARS`) and the first verdict as context, when any of these hold:
- confidence is below 0.6;
- the label is `not_found` or `contradicted`;
- the quote failed verification;
- the label is `unverifiable` although the full text is available, which usually means retrieval missed the relevant part.

Accusatory labels on abstract-only references are also escalated. Ultra's verdict replaces the judge's, and the original label is kept. In a 20-sentence run on the Chain-of-Thought paper, 8 of 46 verdicts were escalated and Ultra changed the label in 7. Most of those were `not_found` becoming `supported` or `overstated`, so escalation is mainly what removes false alarms.
