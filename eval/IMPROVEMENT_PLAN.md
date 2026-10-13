# Improving recall and precision: diagnosis and strategies

Branch `PerfImpr`. Goal: raise recall (planted errors caught) and precision (correct flags among all flags) of the whole pipeline.

## Evaluation set

The 18 papers already screened and planted: round 1 (`eval/heldout/`, 6 papers, 44 planted errors) and round 2 (`eval/heldout-2/`, 12 papers, 103 planted errors). Every flag on an untouched citation from the last runs has been reviewed by hand.

These papers are now **development data**. Improvements are tuned on them, so their numbers will be optimistic; the final measurement needs a fresh sample (round 3) with the paired old-pipeline run.

| Starting point | Recall | Reviewed precision | Genuine / false alarms / debatable |
|---|---|---|---|
| Round 2, current pipeline (`PIPELINE_VERSION` 2) | 89/103 = 86% | 60% | 17 / 71 / 49 |
| Round 1, current pipeline (development baseline) | 38/44 = 86% | 62% | 6 / 27 / 23 |
| Round 1, round-1 pipeline (as evaluated) | 35/44 = 80% | 54% | 5 / 34 / 24 |

Round 1 was evaluated with round 1's pipeline, and its papers were the development data for the fixes in the current one, so its current-pipeline gain (4 planted errors gained, 1 lost; false alarms 34 → 27) is not evidence that the fixes generalise. Round 2's paired run is that evidence, and it shows no gain.

## Working with the harness (E0.1–E0.4, done)

All commands are in `scripts/heldout.py` and run on both rounds unless `--rounds` says otherwise.

- **Frozen inputs (`freeze`).** Each paper's references and cited texts are copied from its development baseline into `.cache/<round>/frozen/` (not committed). Later resolver or parser changes cannot alter them. Each text is checked against the passages the baseline retrieved from it.
  - Round 2: 455 of 457 texts restored byte-identical (3,823 retrieved passages all found). Europe PMC currently returns server errors for the other 2 (`biology-elife.74777` r25 and r109, PMC4214605 and PMC5286201), which are kept as references whose text is missing.
  - Round 1: refrozen from its new baseline, all 1,391 passages found.
- **Development baselines.** Round 2's baseline is its evaluation check. Round 1's is the run `current` (the current pipeline from reference resolution, $3.36), set with `set-baseline`.
- **Reproduction check.** The run `repro` re-ran round 2 from claim extraction on the frozen inputs. 1,114 of 1,116 verdicts were identical, all from the model cache ($0.01). The 2 that differ, and one extra unreviewed flag, involve r25, whose text is missing.
- **Experiments.** `experiment NAME --from STAGE --note "..."` re-runs from `resolve`, `extract`, `retrieve`, `judge` (judging and escalation) or `detect` on the baseline's stored results. Unchanged model calls come from the cache, so an experiment costs only the calls its change alters. It is resumable, and records the commit, base jobs and cost paid in `eval/<round>/experiments/NAME.json`.
- **Comparison.** `compare NAME [--against RUN]` reports, per round and in total:
  - planted errors gained and lost, with an exact McNemar test;
  - flags added and removed, by review verdict;
  - precision over reviewed flags, with unreviewed flags counted separately;
  - how many verdicts changed label or tags;
  - model cost paid.
- **Incremental review (`review-new NAME`).** Writes `review-NAME.md` sheets with only the flags no one has reviewed yet, and adds blank entries to `annotations.json` without touching earlier ones. All flags of both baselines are now reviewed.
- **Not yet done: E0.5 (better planting).** It changes the evaluation set itself, so it belongs with round 3.

## Diagnosis (round 2, current pipeline)

### Misses (14 of 103)

| Cause | Misses |
|---|---|
| Sweeping qualifier accepted ("universally, in all cases", "without exception"): judged *supported*. In some cases the judge's attributed claim keeps the qualifier and still says supported; in others the qualifier is dropped before judging. | 6 of 7 missed scope inflations |
| Hedge removal accepted ("could" → "will", "likely" deleted): judged *supported* | 4 of 4 (one claim was typed *background*, so it could not be flagged) |
| Number change read as a different quantity (30 ± 5% for 30 ± 3%) | 1 |
| Reference available only as an abstract | 2 |
| Identification leniency on a swapped citation | 1 |

### False alarms on untouched citations (71), by root cause

| Root cause | Count | Stage |
|---|---|---|
| Co-cited works each judged on the whole sentence | 12 | extraction / judging |
| Values the citing authors derived from cited data or a cited formula | 9 | extraction / judging |
| Citing authors' own comparisons ("consistent with", "in agreement with") | 5 | extraction / judging |
| Citing authors' own details or hypotheses in the cited sentence | 6 | extraction / judging |
| Marker attachment (a clause before or after the marker) | 4 | ingest / extraction |
| Over-literal wording (terms, "first", causal pedantry) | 12 | judging |
| Reasoning slips (arithmetic, unit conversion, inference) | 11 | judging / retrieval |
| Number parsing or pairing | 5 | ingest / detectors |
| Reference available only as an abstract | 6 | resolution / judging |
| Wrong cited text (a book matched to its review) | 1 | resolution |

**36 of 71 (51%) are attribution-scoping errors:** the cited work is held responsible for more of the sentence than it is cited for.

### Per-stage diagnostics

| Stage | Finding |
|---|---|
| Reference resolution | Of 674 cited references, 440 have full text (65%), 109 an abstract only, 125 nothing. Errors in a third of citations cannot be found at all; planted recall does not show this, because errors were planted only on readable citations. Papers with title-less bibliographies (AAS, REVTeX) cannot be resolved and were excluded at screening. |
| Claim extraction | 92 claims typed *background* and 188 *method*, both judged leniently. Flags on method claims: 1 genuine, 14 false alarms. One planted hedge removal was missed because its claim was typed *background*. Extraction occasionally returns truncated JSON and falls back to background claims. |
| Evidence retrieval | When Ultra re-read a full paper and found a verified supporting quote, that quote was **not among the retrieved passages in 18 of 63 cases (29%)**. |
| Judging | Labels: *overstated* flags on untouched citations are 4 genuine / 39 false / 31 debatable; *not found* 9 / 30 / 17; *contradicted* 3 / 0 / 0. |
| Escalation | 221 of 1,116 verdicts escalated, 49 labels changed (17 unverifiable → not found, 12 not found → supported). Flags that were **not** escalated: 3 genuine / 37 false alarms (7% precision); escalated flags 14 / 34. Unescalated flags still carry 21 caught scope inflations, so they cannot simply be hidden. |
| Detectors | Tag precision (planted + genuine vs false alarms): off-topic 96%, scope inflation 72%, hypothesis-as-finding 69%, number drift 62%, secondary citation 40%, correlation-as-causation 0% (4 flags). |
| Co-cited markers | Flags on markers citing 2+ works: 4 genuine / 40 false (9%); single-work markers 13 / 31 (30%). Hiding co-cited flags would raise precision to 75% (round 2) and 70% (round 1), but lose 11 planted scope inflations in round 2, so co-cited works need joint judging, not suppression. |

## Strategies by pipeline stage

Each strategy lists what it targets, the evidence, and how to test it. "Addresses" counts are from the diagnosis above, not predictions.

### 0. Evaluation harness (prerequisite)

- **E0.1 Rebuild the cited texts** for the 18 papers (they were deleted in the rewind) and freeze each paper's resolved references, so runs differ only by the pipeline change. No model cost.
- **E0.2 Stage-level re-run harness.** Re-run from any stage (extraction, retrieval, judging, escalation, detectors) on stored inputs, so a judge change costs only judge calls. Report paired results against the baseline: planted errors gained and lost (McNemar), flags added and removed, and reviewed precision with unreviewed flags counted separately.
- **E0.3 Incremental review.** Annotations are keyed by (sentence, reference), so only flags new to a run need review; report them as unreviewed until then.
- **E0.4 Overfitting guard.** Develop on round 2, check every change on round 1 (they disagreed on some rules in the analysis), and measure the final pipeline on a fresh round 3.
- **E0.5 Better planting.**
  - Make the planted distortions more realistic. All scope inflations use 3 fixed phrases, and one planted hedge removal ("could potentially" → "will potentially") is still hedged. A rule keyed on those phrases would inflate recall without real value.
  - Add new types: population and species generalisation, correlation to causation, reversed direction, subtle number changes, and swaps to topically close papers.
  - Plant on co-cited markers too.

### 1. Ingest (LaTeX and PDF)

- **I1 Marker-to-clause attachment.** Record which clause each marker closes, so the text after a marker and the citing authors' own continuation are not attributed to it. Addresses 4 marker-attachment false alarms and supports A1–A4.
- **I2 Number fidelity in PDF text.** Restore superscripts ("10−17" is 10^-17), keep ranges and "±" as single values, and normalise Unicode minus signs. Addresses number-parsing false alarms (the citing paper's own text is parsed too).
- **I3 Title-less bibliographies.** Parse AAS and REVTeX entries ("Author 2007, ApJ, 474, L21", bare arXiv IDs) into journal, volume and page fields for resolution (R3). Removes a screening exclusion and raises coverage in physics and astronomy.

### 2. Reference resolution and coverage

- **R1 More open full text.** Add Unpaywall, Semantic Scholar open-access PDFs, PubMed Central author manuscripts and publisher HTML full text. A third of cited references are unreadable today; this raises the recall ceiling and removes abstract-only false alarms (6) and misses (2).
- **R2 Type-aware matching.** A reference with a publisher or ISBN is a book: reject journal-article candidates (book reviews share the title). Check the first author and year for short or generic titles.
- **R3 Bibliographic lookup without titles.** Use Crossref and ADS-style journal + volume + first-page queries, and treat bare arXiv IDs in REVTeX entries as IDs.
- **R4 Report coverage as a metric** in every evaluation (full / abstract / none), alongside recall and precision.

### 3. Claim extraction and attribution

- **A1 Citation-function classification.** For each marker, classify its role: source of data, method, formula or definition; prior finding being reported; comparison or agreement; example in a list; credit for a term. Judge each role by its own rubric: for a data or method source, "does the cited work provide this data or method?"; for a comparison, "does it report the compared value?". Addresses the derived-value (9), comparison (5) and own-detail (6) false alarms.
- **A2 Sub-claim decomposition.** Split the sentence into clauses and assign each to the markers that support it, using I1. The citing authors' own speculation or results get no marker. Addresses own-detail (6) and marker-attachment (4).
- **A3 Joint judging of co-cited works.** Judge each marker group in one call, with every work's evidence. Ask which work supports which part; flag a work only if a part assigned to it is unsupported, and flag the group if a part is supported by none. Addresses the co-cited false alarms (12, and most of the 40 co-cited flags) while keeping planted sweeping qualifiers, which no work supports.
- **A4 Derived-value cues.** Phrases such as "we get / obtain / fit / derive", "the relation we get for the X sample" or "linear regression on the data from X" mark the value as the citing authors' own; check only that the cited work provides the data.
- **A5 Keep qualifiers and modality verbatim.** In code, check that the claim and the judge's attributed claim keep the sentence's quantifiers ("all", "always", "universally") and modal verbs ("may", "will"); re-ask when they are dropped. Addresses the misses where the qualifier or hedge vanished before judging.
- **A6 Tighter background typing.** Reserve *background* for pointers and lists of examples; a sentence that states a finding is a finding. Addresses the hedge removal missed as background. Retry truncated JSON once with a larger token limit before falling back to background claims.

### 4. Evidence retrieval

- **V1 Full-text judging when it fits.** If the cited paper fits the judge's context, judge against the whole text instead of five passages. Addresses the 29% retrieval misses and reasoning slips caused by missing context (tables, definitions elsewhere in the paper).
- **V2 Hybrid retrieval.** Combine embeddings with lexical matching (BM25) on the claim's numbers, named entities and rare terms.
- **V3 Number-anchored passages.** Always include passages containing the claim's numbers or named resources, and table rows that match them.
- **V4 Adaptive k.** Retrieve more passages when the top scores are close or low, fewer when one passage clearly matches.
- **V5 Retrieval metric.** Track "supporting quote found by Ultra but not retrieved" (18 of 63 today) as a retrieval-recall measure for tuning chunk size, overlap and query phrasing.

### 5. Judging

- **J1 Materiality rubric.** Before flagging, the judge must say whether the difference would matter to a domain expert (substance) or is wording (terms, "first", plain-language "significant"). Wording-only differences are not flagged. Addresses over-literal wording (12).
- **J2 Explicit qualifier and hedge checks.** Have the judge compare the claim's quantifiers and modality with the source's in dedicated signal fields, and require a statement of whether the source supports the claim's universality and certainty. Addresses the 10 qualifier and hedge misses.
- **J3 Show the work for numbers.** For numeric claims, require the judge to show the arithmetic or unit conversion connecting the claim to the source (10 lost per 1 netted × 10; St^-0.5 from the tabulated exponent). Addresses reasoning slips (11).
- **J4 Few-shot calibration.** Add short examples of false alarms and genuine problems from the reviewed dev data, rotated so that no evaluation paper's own examples are in its prompt.
- **J5 Model and reasoning choice.** Compare Super with Ultra as the first-pass judge, and reasoning-effort settings, on the dev set: accuracy against cost.

### 6. Escalation and verification

- **X1 Verify every flag.** Escalate *overstated* flags too (today only "not found", "contradicted", low confidence and unverified quotes escalate). Unescalated flags are 7% precise on untouched citations but carry 21 caught scope inflations, so a second look by Ultra with the full paper should remove false alarms while keeping real ones. Expected cost: about 90 more Ultra calls across round 2's 12 papers (68 unescalated flags on untouched citations, 23 on planted errors).
- **X2 Support search before flagging.** For a flag, first ask for the strongest supporting passage in the full text; keep the flag only if none is found, or the remaining gap is material (J1).
- **X3 Self-consistency.** Judge accusatory labels three times and flag on two-of-three agreement; disagreement escalates.
- **X4 Two-sided escalation.** Ultra also re-checks high-confidence *supported* verdicts on numeric and qualified claims, where the misses are; it already turns 17 unverifiable verdicts into "not found".

### 7. Failure-mode detectors and code rules

- **D1 Number drift.** Pair numbers only with matching units and quantity type; treat ranges and "±" as intervals; drop drift when the claim value lies within the source's range; never pair the citing authors' own values (A4) with the source's. 5 false alarms; tag precision 62%.
- **D2 Hedge-to-certainty rule.** Extend *hypothesis as finding* to *supported* verdicts when the verified quote is hedged and the claim uses a certainty verb ("will", "shows", "proves"). Use a general hedge lexicon, never the planting phrases, and check new flags on untouched citations.
- **D3 Sweeping-qualifier rule.** When the claim contains a universal quantifier absent from the verified quote's context and the source's scope is narrower, raise to *overstated*. Same safeguards as D2.
- **D4 Abstract-only rule.** Do not flag *overstated* or *not found* against an abstract unless it contradicts the claim or the work is off-topic. Round 2: removes 5 false alarms and 10 debatable flags, and loses 3 planted errors (fewer once R1 raises coverage).
- **D5 Retire or fix weak tags.** Correlation-as-causation (0 of 4) and secondary citation (40%): require stronger evidence or show them as notes, not flags.
- **D6 Keep the off-topic tag**: 96% precise.

### 8. Aggregation and display

- **P1 Tiers.** Likely problems first (contradicted; off-topic; not found after Ultra), then "worth a second look" (overstated).
- **P2 Confidence threshold.** Hide flags below 0.90 by default behind a toggle. Earlier analysis: +7–9 points precision for −3–5 points recall, consistent across both rounds.
- **P3 Report coverage per paper**, so readers know how much could not be checked.

### 9. Cost

- **C1 Spend Ultra where the data says it pays** (X1, X4), and stop escalating verdicts the rules make lenient anyway.
- **C2 Cache by stage**, so experiments re-run only the stages they change (E0.2).

## Suggested order

1. **Harness:** E0.1–E0.4 and the baseline re-runs on rounds 1 and 2 (done; see above).
2. **Cheap, high-yield, mostly code (done; see "Step 2 results" below):**
   - D1 and I2 (number fidelity);
   - D4 (abstract-only);
   - A5 and A6 (qualifiers and background typing);
   - D2 and D3 (hedge and qualifier rules, with safeguards);
   - D5 (weak tags).
3. **Attribution and verification:**
   - A3 (joint co-citation judging);
   - A1, A2 and A4 (citation function and decomposition);
   - X1 and X2 (verify every flag).
   These target over half the false alarms; re-review new flags after each.
4. **Retrieval and coverage:** V1–V3, R1–R3, I3.
5. **Judge calibration:** J1–J5 and X3, compared on cost.
6. **Measure on a fresh round 3**, paired with the current pipeline, with better planting (E0.5).

## Step 2 results: code rules (experiment `cheap-fixes`)

The rules are all in the detectors (`app/pipeline/detectors.py`), so the experiment re-ran only that stage, with no model calls. Every new flag has been reviewed.

| | Baseline | `cheap-fixes` |
|---|---|---|
| Planted errors caught, round 1 | 38/44 = 86% | 40/44 = 91% |
| Planted errors caught, round 2 | 89/103 = 86% | 96/103 = 93% |
| Both rounds: gained / lost | | 9 / 0 (exact McNemar p = 0.004) |
| False alarms (rounds 1 + 2) | 27 + 71 | 26 + 65 |
| Debatable flags (rounds 1 + 2) | 23 + 49 | 23 + 44 |
| Reviewed precision, round 1 / round 2 | 62% / 60% | 64% / 63% |

What changed:

- **D1 and I2, number drift.** A claim number that differs from the source's is not drift when it doesn't conflict:
  - the source value lies within a range the claim states (works co-cited for "between 3 and 6.6");
  - the claim value lies within the source's range or ± interval;
  - the claim states a bound the source satisfies ("at least 100").

  "10−17" in PDF text now reads as 10^-17. A unitless value from the judge ("5") now matches "5%" in the claim, which caught the planted "30±5%". All 3 flags the drift rule itself created were false alarms, and are gone. No planted number change lost its drift tag: 8 of the 12 now carry it, up from 7.
- **D4, abstract-only references.** "Overstated" against an abstract alone now needs a concrete mismatch tag (scope, number, hedge, causation or significance). Untagged ones were "the abstract doesn't mention it": 5 false alarms and 5 debatable flags, no planted error.
- **D3, sweeping qualifiers.** A finding whose citing clause says "without exception", "universally", "always", "in all cases" and the like, against a supporting passage that makes no such general statement, becomes overstated (scope inflation). It reads the citing sentence's own clause (`citation_clause`), so a qualifier dropped by extraction still counts. 6 planted scope inflations gained.
- **D2, hedge removal.** A "supported" finding becomes overstated (hypothesis as finding) when the verified quote hedges with a modal ("may", "could", "likely", "hypothesize") and the citing clause asserts certainty ("will", "is known to", "demonstrates") without a hedge of its own. 2 planted hedge removals gained, and 1 new flag on an untouched citation: a false alarm (Najita & Kenyon, 2014, whose conclusions are firmer than its abstract).
  - A first version used any hedge word, including "suggests" and "appears to". It added 7 flags, which were 6 false alarms and 1 debatable: papers routinely report their own results with "this study suggests". That version was discarded.

Not done, and why:

- **A5 (qualifiers kept by extraction).** No longer needed for detection: D2 and D3 read the citing clause directly. It may still help the judge, but it means re-running extraction and everything after it.
- **A6.**
  - Fallback "background" claims (23 verdicts) carry no planted error or flag. The truncated extractions behind some of them are repetition loops, not outputs that ran out of room.
  - Two planted errors were missed because extraction typed the claim as background. Applying D2 and D3 to background claims would catch neither (only one already-planted sentence matches), so background typing needs a prompt change: a step-3 item.
- **D5 (weak tags).** Correlation-as-causation and secondary-citation tags never create a flag (the judge had already said "overstated"), so they do not affect recall or precision. They belong with the display work (P1).

**Caveat: the gains depend on how errors were planted.** All 6 sweeping-qualifier gains are the planting script's fixed phrases, and both hedge gains are its "may/could → will". On untouched citations D3 fired on none and D2 on one (a false alarm), so they cost almost no precision here. Whether they catch real overstatements can only be measured on round 3, with different planted wordings (E0.5).

## Caveats

- One reviewer (Claude) judged every flag; 49 debatable flags are left out of precision.
- The planted set is dominated by citation swaps (which the checker already catches: 53 of 54) and fixed-phrase scope inflations; recall on real, subtler errors is likely lower.
- Rules tuned on 18 papers can overfit; every change is checked on both rounds before round 3.
