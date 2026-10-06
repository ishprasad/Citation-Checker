"""Report assembly (for the UI and the export endpoint) from a job's stored results."""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from app import demo, llm
from app.models import Claim, Coverage, IngestResult, Job, Reference, Verdict
from app.pipeline.detectors import failure_mode_counts
from app.pipeline.judge import escalation_stats
from app.pipeline.run import JobStore

LABEL_ORDER = ("contradicted", "not_found", "overstated", "unverifiable", "supported")
LABEL_TEXT = {
    "supported": "Supported",
    "overstated": "Possibly overstated",
    "not_found": "Not found in cited text",
    "contradicted": "Contradicted by cited text",
    "unverifiable": "Could not be verified",
}
FAILURE_TEXT = {
    "scope_inflation": "Scope inflation",
    "number_drift": "Number drift",
    "hypothesis_as_finding": "Hypothesis stated as finding",
    "secondary_citation": "Secondary citation",
    "retracted_source": "Retracted source",
    "off_topic_citation": "Possibly the wrong reference",
    "correlation_as_causation": "Correlation stated as causation",
    "significance_inflation": "Significance overstated",
    "corrected_source": "Source has a correction",
    "expression_of_concern": "Expression of concern on source",
}


def _load(store: JobStore, job: Job) -> dict[str, Any]:
    ingest = store.get(job.id, "ingest", "paper", IngestResult)
    resolved = store.all(job.id, "reference", Reference)
    claims_by_sentence = store.all_lists(job.id, "claims", Claim)
    verdicts = list(store.all(job.id, "verdict", Verdict).values())
    return {"ingest": ingest, "resolved": resolved, "claims_by_sentence": claims_by_sentence, "verdicts": verdicts}


def summarize(job: Job, references: list[Reference], verdicts: list[Verdict]) -> dict[str, Any]:
    coverage = Coverage.of(references)
    labels = Counter(v.label for v in verdicts)
    return {
        "labels": {label: labels.get(label, 0) for label in LABEL_ORDER},
        "failure_modes": failure_mode_counts(verdicts),
        "escalation": escalation_stats(verdicts).model_dump(),
        "coverage": {**coverage.model_dump(), "line": coverage.line()},
        "sentences_total": job.sentences_total,
        "sentences_checked": job.sentences_checked,
        "quick": job.mode == "quick" and job.sentences_checked < job.sentences_total,
    }


def _sort_key(v: Verdict) -> tuple:
    sentence, _, claim = v.claim_id.partition(".c")
    return (int(sentence.lstrip("s") or 0), int(claim or 0), v.reference_key)


def build_report(store: JobStore, job: Job, include_text: bool = True) -> dict[str, Any]:
    data = _load(store, job)
    ingest: Optional[IngestResult] = data["ingest"]
    job = job.model_copy(update={"usage": llm.get_llm().usage_summary(job.id)})
    if ingest is None:
        return {"job": job.model_dump(mode="json"), "paper": None, "references": [], "sentences": [],
                "claims": [], "verdicts": [], "summary": summarize(job, [], [])}

    checked_ids = {s.id for s in ingest.sentences[: job.sentences_checked]} if job.sentences_checked else set()
    needed = {k for s in ingest.sentences if s.id in checked_ids for k in s.reference_keys}
    resolved = data["resolved"]
    references = [resolved.get(r.key, r) for r in ingest.paper.references]
    checked_refs = [r for r in references if r.key in needed and r.key in resolved]
    # Verdicts that restate another verdict for the same sentence and reference are left out
    # (listed under "duplicates" so clients know those pairs are merged, not pending).
    verdicts = sorted((v for v in data["verdicts"] if not v.duplicate_of), key=_sort_key)
    duplicates = [{"pair_id": f"{v.claim_id}|{v.reference_key}", "duplicate_of": v.duplicate_of}
                  for v in data["verdicts"] if v.duplicate_of]
    claims = [c for s in ingest.sentences for c in data["claims_by_sentence"].get(s.id, [])]

    paper = ingest.paper.model_dump(mode="json", exclude={"references"})
    demo_info = None
    if job.source == "demo" and (truth := demo.load_ground_truth()) is not None:
        claims_by_id = {c.id: c for c in claims}
        demo_info = {
            "planted": len(truth.planted),
            "planted_by_sentence": demo.planted_by_sentence(truth),
            "eval": demo.score(truth, data["verdicts"], claims_by_id) if job.status in ("done", "partial") else None,
        }
    if not include_text:
        paper.pop("text", None)
        paper.pop("sections", None)
    return {
        "job": job.model_dump(mode="json"),
        "paper": paper,
        "references": [r.model_dump(mode="json") for r in references],
        "sentences": [{**s.model_dump(mode="json"), "checked": s.id in checked_ids} for s in ingest.sentences],
        "claims": [c.model_dump(mode="json") for c in claims],
        "verdicts": [v.model_dump(mode="json") for v in verdicts],
        "duplicates": duplicates,
        "demo": demo_info,
        "summary": summarize(job, checked_refs, verdicts),
    }


def snapshot(store: JobStore, job: Job) -> dict[str, Any]:
    """Everything so far except the paper text, for a client (re)connecting to the event stream."""
    return build_report(store, job, include_text=False)


# ----------------------------------------------------------------------------
# Markdown export
# ----------------------------------------------------------------------------


def _md_escape(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def export_markdown(report: dict[str, Any]) -> str:
    job, paper, summary = report["job"], report["paper"] or {}, report["summary"]
    refs = {r["key"]: r for r in report["references"]}
    claims = {c["id"]: c for c in report["claims"]}
    sentences = {s["id"]: s for s in report["sentences"]}
    lines = [
        f"# Citation check: {paper.get('title') or job.get('arxiv_id') or job['id']}",
        "",
        f"- Source: {paper.get('arxiv_id') and 'arXiv:' + paper['arxiv_id'] or paper.get('source', '')}",
        f"- Citing sentences checked: {summary['sentences_checked']} of {summary['sentences_total']}"
        + (" (quick check)" if summary["quick"] else ""),
        f"- Coverage: {summary['coverage']['line']}",
        f"- Status: {job['status']}" + (f" ({job['message']})" if job.get("message") else ""),
    ]
    for note in paper.get("notes") or []:
        lines.append(f"- Note: {note}")
    lines += ["", "## Summary", "", "| Verdict | Citations |", "|---|---|"]
    for label, count in summary["labels"].items():
        lines.append(f"| {LABEL_TEXT[label]} | {count} |")
    if summary["failure_modes"]:
        lines += ["", "| Failure mode | Citations |", "|---|---|"]
        lines += [f"| {FAILURE_TEXT.get(m, m)} | {n} |" for m, n in summary["failure_modes"].items()]
    esc = summary["escalation"]
    lines += ["", f"Escalated to the long-context model: {esc['escalated']} of {esc['verdicts']} "
                  f"(label changed in {esc['label_changed']})."]
    usage = job.get("usage") or {}
    if usage.get("per_model"):
        lines += ["", "| Model | Role | Input tokens | Output tokens | Est. cost (USD) |", "|---|---|---|---|---|"]
        for m in usage["per_model"]:
            lines.append(f"| {m['model']} | {m['role']} | {m['prompt_tokens']:,} | {m['completion_tokens']:,} | "
                         f"{m['estimated_cost_usd']:.4f} |")

    def section(title: str, items: list[dict]) -> None:
        if not items:
            return
        lines.extend(["", f"## {title}", ""])
        for v in items:
            claim = claims.get(v["claim_id"], {})
            sentence = sentences.get(claim.get("citing_sentence_id", ""), {})
            ref = refs.get(v["reference_key"], {})
            lines.append(f"### {LABEL_TEXT[v['label']]}: [{ref.get('label', v['reference_key'])}] "
                         f"{_md_escape(ref.get('title') or '')}")
            lines.append("")
            if sentence:
                lines.append(f"> {_md_escape(sentence.get('text', ''))}")
                lines.append("")
            lines.append(f"- **Claim checked:** {_md_escape(v.get('attributed_claim') or claim.get('claim_text', ''))}")
            lines.append(f"- **Verdict:** {LABEL_TEXT[v['label']]} (confidence {v['confidence']:.2f})"
                         + (" · escalated to the long-context model" if v.get("escalated") else ""))
            if v.get("quote"):
                status = "" if v.get("quote_verified") else " (not found verbatim in the cited text)"
                lines.append(f"- **Cited text{status}:** \"{_md_escape(v['quote'])}\"")
            lines.append(f"- **Explanation:** {_md_escape(v.get('explanation', ''))}")
            for mode in v.get("failure_modes") or []:
                lines.append(f"- **{FAILURE_TEXT.get(mode, mode)}:** {_md_escape(v['failure_notes'].get(mode, ''))}")
            lines.append("")

    verdicts = report["verdicts"]
    section("Needs human check", [v for v in verdicts if v["label"] in ("contradicted", "not_found", "overstated")])
    section("Supported, with notes", [v for v in verdicts if v["label"] == "supported" and v.get("failure_modes")])
    section("Could not be verified", [v for v in verdicts if v["label"] == "unverifiable"])
    lines += ["", "_Verdicts are automated and need human review. For references marked abstract only or "
                  "unavailable, results are limited by open access, not by the checker._", ""]
    return "\n".join(lines)
