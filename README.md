# Citation Checker

For every citation in a research paper, Citation Checker asks one question: does the cited source actually say what the citing author claims it says? It runs on NVIDIA Nemotron models served through Nebius Token Factory.

> Work in progress. See `CITATION_CHECKER_BRIEF.md` for the full plan. Milestone 1 (scaffold, config, LLM wrapper) is done.

## Setup

```bash
pip install -r requirements-dev.txt
cp .env.example .env          # then add NEBIUS_API_KEY
python scripts/list_models.py # copy the exact Nemotron / embedding model IDs into .env
python scripts/smoke_test.py  # one call to each configured model, with token usage
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
