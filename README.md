# Citation Checker

For every citation in a research paper, Citation Checker asks one question: does the cited source actually say what the citing author claims it says? It runs on NVIDIA Nemotron models served through Nebius Token Factory.

> Work in progress. See `CITATION_CHECKER_BRIEF.md` for the full plan. Done so far: milestones 1–9 (through the UI, demo mode and evaluation), plus four extra failure-mode detectors (scaffold and LLM wrapper, arXiv LaTeX ingest, reference resolution and full-text fetching, claim extraction and evidence retrieval, judging with quote verification and escalation, failure-mode detectors).

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
python scripts/plant_errors.py             # build the demo paper (planted errors) and its ground truth
python scripts/evaluate.py --snapshot      # score a check of the demo paper; bundle it for offline demos
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

## Failure-mode detectors (`app/pipeline/detectors.py`)

After judging, each verdict can carry one or more failure-mode tags, each with a short note for the UI. The judge returns structured signals for this, at no extra model cost. Every tag also needs rule-based evidence, so one mistaken signal can't put a tag on a citation.

| Tag | Fires when | Example note from a real run |
|---|---|---|
| `number_drift` | The judge pairs a claim number with the source's value for the same quantity. Both numbers must actually appear in the claim and in the verified quote, and they must differ beyond the claim's rounding (31% for 30.9% is fine; fractions and percentages, and 175B and "175 billion", are compared correctly). A `supported` verdict with drift becomes `overstated`. | "6.7B in the claim; 6B in the cited work." (Chain-of-Thought citing InstructGPT's model sizes) |
| `scope_inflation` | An `overstated` or `contradicted` verdict where the judge found the claim broader than who or what was studied. Both scopes must be stated and must differ, and a scope that is only a list of numbers doesn't count. | "Claim scope: tasks that require reasoning abilities. Source scope: specific reasoning tasks…" |
| `hypothesis_as_finding` | An `overstated` finding where the source hedges (the judge's signal, or hedge words such as "may" or "suggests" in the quote) and the claim doesn't. | 'The cited work hedges this point ("may"); the claim states it as fact.' |
| `secondary_citation` | The judge says the supporting passage credits the point to another work, and a citation marker is visible in the source text. The original work is named from the cited paper's own bibliography. | 'The supporting passage credits Brown et al. (2020): "Language models are few-shot learners".' |
| `retracted_source` | The reference is marked as retracted (Crossref / Retraction Watch, OpenAlex). This applies to every claim citing it. | "The cited work has been retracted (notice: doi:…)." |
| `off_topic_citation` | The work cited may be the wrong one. Fires on a `not_found` or `unverifiable` verdict when the judge says the cited work isn't about the claim's topic, *and* the embeddings agree. Either another reference in the same bibliography fits the claim's abstract-level similarity at least 0.15 better (that reference is then named), or the best passage is an outlier among this paper's citations. | "Possibly the wrong reference. The claim matches Cobbe et al., 2021 in this paper's bibliography more closely." (a swapped GSM8K citation, with the original recovered) |
| `correlation_as_causation` | The judge's signals say the source is associational and the claim is causal, *and* the claim uses causal wording ("reduces", "leads to") while the verified source text uses associational wording ("associated with", "correlated"). A `supported` verdict becomes `overstated`. | 'The cited work reports an association ("associated"); the claim states a causal effect ("reduces").' |
| `significance_inflation` | The claim says "significantly", "proven" or similar, the judge says the source reports the result as not significant, *and* the source text says so ("not significant", "p = 0.12", "n.s."). A `supported` verdict becomes `overstated`. It also fires when a claim says "statistically significant" about a result the source never tested. "Significantly" alone is common ML wording and isn't flagged. | 'The cited work reports this as not statistically significant ("p = 0.12")…' |
| `corrected_source`, `expression_of_concern` | Crossref `updated-by` lists a correction, erratum or corrigendum, or an expression of concern, for the cited work. This applies to every claim citing it. | "A correction was published for the cited work (doi:…, 2004-03-06)." |

## API (`app/main.py`) and orchestration (`app/pipeline/run.py`)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | Start a check. Send `{"arxiv_id": "...", "mode": "quick" \| "full"}`; `arxiv_id` can be a bare ID, a versioned ID, an old-style ID or any arxiv.org URL. A multipart PDF upload (`file`, `mode`) is validated (content type, `%PDF-` magic bytes, `MAX_PDF_MB`); PDF parsing arrives in milestone 11. |
| `POST` | `/api/jobs/{id}/expand` | Upgrade a finished quick check to a full check, reusing everything already computed. |
| `GET` | `/api/jobs/{id}` | Status, per-stage progress and live token usage. |
| `GET` | `/api/jobs/{id}/events` | Server-sent events. First a `snapshot` of everything so far, then `progress`, `paper`, `reference`, `coverage`, `claims`, `escalating`, `verdict`, `status` and finally `done`. |
| `GET` | `/api/jobs/{id}/report` | Full report JSON: paper text with offsets, references, sentences, claims, verdicts, and a summary (label counts, failure modes, escalation stats, coverage line). |
| `GET` | `/api/jobs/{id}/export?format=md\|json` | Download the report. |

Stages overlap so results stream early:
- Claim extraction starts while references are still downloading.
- Each (claim, reference) pair goes through retrieval, judging and escalation as soon as its reference is ready.
- A final pass re-runs the detectors that need paper-level context.

Every intermediate result is stored in SQLite. When the server restarts it resumes unfinished jobs, skipping finished work; this was tested by killing the server mid-job. Expanding a quick check only processes the new sentences. A failure in one citation becomes an `unverifiable` verdict and never fails the job. Hitting the token budget stops the job and marks it `partial`, keeping its results.

## UI (`static/`)

The UI is plain HTML, CSS and JavaScript, with no build step and no CDN. Run `uvicorn app.main:app` and open http://localhost:8000.

- **Start.** Enter an arXiv ID or any arxiv.org link, or drop a PDF. Choose a quick check (the first 20 citing sentences) or a full check, or try the demo paper.
- **Progress.** The report opens as soon as the paper is parsed. A six-stage tracker fills in live from the event stream (for example "Judging claims: 37 / 84"). Citation markers change from "being checked" to their verdict as results arrive, and the coverage line appears once reference resolution finishes.
- **Report.** The paper text is on the left. Each citation is coloured by verdict *and* has its own underline style and glyph (✓ supported, ! overstated, ✕ contradicted, ? not found, – could not be verified), so colour is never the only cue. The detail panel on the right shows the citing sentence and, for each cited work:
  - the claim checked;
  - a link to the cited paper and whether its full text was available;
  - the verdict and confidence;
  - failure-mode chips with notes;
  - the cited paper's text with the supporting quote highlighted;
  - an "Escalated to Nemotron Ultra" badge and a "Needs human check" banner where relevant.

  With nothing selected, the panel shows a legend and the list of citations that need a human check.
- **Top bar.** Verdict counts double as filters. There's also a search box (over sentences, claims, reference titles and explanations), a "Show only issues" toggle, and "Next issue" (or press `j`). Export gives Markdown or JSON. A cost panel shows tokens and estimated cost per model, plus how many verdicts needed the large model.
- **Layout.** Light and dark themes follow `prefers-color-scheme`. The layout is two-column at laptop widths and single-column below 900px. The URL (`#/job/<id>`) reopens a report after a reload.

## Demo paper and evaluation (`app/demo.py`, `scripts/plant_errors.py`, `scripts/evaluate.py`)

The demo is the Chain-of-Thought Prompting paper (arXiv:2201.11903). Its references are mostly open (full text for 78 of 82), and 10 of its citations have been deliberately altered.

**How errors are planted.** `plant_errors.py` first checks the unmodified paper. Errors are planted only on citations the checker judged supported with high confidence and no tags, so a caught error can't be a pre-existing problem. The four mutation types:

- **Number change.** A number the cited work actually reports, inside the clause the citation is attached to, is changed. Years and list enumerators are excluded.
- **Scope inflation.** A sweeping phrase is added before a parenthetical citation that closes its clause, supports a finding, and cites at most two works.
- **Hedge removal.** "may" or "suggests" becomes "will" or "shows", or "likely" is deleted.
- **Citation swap.** The citation is replaced by the bibliography entry least related to the claim. It must be at least 0.15 less similar than the original, and each replacement is used only once.

Edits are applied from the end of the text backwards, so every offset stays exact; this is covered by tests. The output is `demo/paper.json` and `demo/ground_truth.json`, which also records the citations flagged in the unmodified paper.

**Demo button (`POST /api/demo`).** It returns the latest finished check of the demo paper instantly. If there is none, it runs one live; with warm caches that takes about 30 seconds. With no API key configured, it shows the bundled snapshot of a recorded check (`demo/snapshot.json`). In the UI, the banner shows the score, each planted citation shows what was changed and its original sentence, and the issue list marks planted entries.

**Calibration.** Flags on untouched citations were reviewed by hand (`demo/annotations.json`: genuine, false alarm or debatable, with notes; checked against the cited texts where it mattered). Most false alarms fell into two groups:

- **Group citations read literally.** "Improving factuality is an open direction (A; B; C)" was flagged because A "does not mention chain-of-thought".
- **Identification citations judged on the citing authors' own description.** BIG-bench was flagged for "not describing" the Date Understanding task.

The fixes:

- A marker citing three or more works for a statement without numbers is judged like a background citation: supported or "could not be verified".
- An identification claim without numbers can no longer be "not found" or "overstated". It still escalates, and explicit contradictions stay strict.
- The judge prompt now treats the citing authors' own details about a resource (subset, licence, URL, a task description) as theirs, accepts standard terms and acronyms as paraphrases, and treats sweeping qualifiers ("universally", "in all cases") as part of the claim.

**Results** (`python scripts/evaluate.py`, full check of all 70 citing sentences, before and after calibration):

| | Before | After |
|---|---|---|
| Planted errors caught (recall) | 7 of 10 | **9 of 10** |
| Label accuracy on planted errors | 50% | 70% |
| Expected failure-mode tag present | 50% | 80% |
| Flags on untouched citations | 33 | 13 |
| ... of which false alarms (human review) | 21 | 7 |
| ... genuine citation problems found | 3 | 3 |
| Precision with human review (debatable left out) | 32% | **63%** |
| Estimated cost of a full check | $1.44 | $1.16 |

The planted errors caught after calibration break down as: all 4 citation swaps (each tagged "possibly the wrong reference"), both number changes (both tagged "number drift", e.g. "86.9% in the claim; 54.3% in the cited work"), 2 of 3 scope inflations (tagged "scope inflation"), and the hedge removal. The one miss is a scope inflation on a sentence the extractor classified as a background citation, which is judged leniently by design.

The genuine problems found in the unmodified paper:

- InstructGPT and LaMDA model sizes that the cited papers do not report (350M and 6.7B; 422M and 68B).
- "Cobbe et al. extend Ling et al. by creating a *larger* dataset": GSM8K has 8.5K problems and AQuA has 100K.

Calibration was done on this paper, so these are development-set numbers; a second paper with planted errors would give a held-out estimate.

- **Escalation:** 30 of 147 verdicts went to Nemotron Ultra, which changed the label in 13.
- **Cost:** a full check costs about $1.16 (1.65M tokens) the first time. Repeat runs are served from the cache in about 30 seconds.
