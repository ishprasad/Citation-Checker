"""Demo paper with planted citation errors, and scoring against its ground truth.

`scripts/plant_errors.py` writes:
    demo/paper.json         the mutated paper (an IngestResult: text with offsets, references, sentences)
    demo/ground_truth.json  which citations were altered, how, and what a correct check should report

The same scoring is used by the report (shown in the UI) and by `scripts/evaluate.py`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.config import ROOT_DIR
from app.models import Claim, IngestResult, Verdict

DEMO_DIR = ROOT_DIR / "demo"
PAPER_PATH = DEMO_DIR / "paper.json"
GROUND_TRUTH_PATH = DEMO_DIR / "ground_truth.json"
SNAPSHOT_PATH = DEMO_DIR / "snapshot.json"
ISSUE_LABELS = {"overstated", "contradicted", "not_found"}


class PlantedError(BaseModel):
    id: str
    sentence_id: str
    reference_keys: list[str]  # the citation(s) the error is attached to
    mutation: str  # number_change | scope_inflation | hedge_removal | citation_swap
    description: str  # human readable, e.g. 'Changed "8.5K" to "13.6K"'
    original: str  # sentence before the change
    mutated: str  # sentence after the change
    expected_labels: list[str]
    expected_failure_modes: list[str] = Field(default_factory=list)


class GroundTruth(BaseModel):
    arxiv_id: str
    title: str
    planted: list[PlantedError]
    # Citations flagged when checking the unmodified paper: (sentence_id, reference_key, label).
    baseline_flags: list[tuple[str, str, str]] = Field(default_factory=list)


@lru_cache
def _load_ground_truth(mtime: float) -> Optional[GroundTruth]:
    return GroundTruth.model_validate_json(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


def load_ground_truth() -> Optional[GroundTruth]:
    if not GROUND_TRUTH_PATH.exists():
        return None
    return _load_ground_truth(GROUND_TRUTH_PATH.stat().st_mtime)


def load_demo_paper() -> Optional[IngestResult]:
    if not PAPER_PATH.exists():
        return None
    return IngestResult.model_validate_json(PAPER_PATH.read_text(encoding="utf-8"))


def demo_available() -> bool:
    return PAPER_PATH.exists() and GROUND_TRUTH_PATH.exists()


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


def is_flag(v: Verdict) -> bool:
    return v.label in ISSUE_LABELS or "off_topic_citation" in v.failure_modes


def score(ground_truth: GroundTruth, verdicts: list[Verdict], claims: dict[str, Claim]) -> dict[str, Any]:
    """Compare a check of the demo paper against the planted errors.

    caught:          a planted citation got an issue label (or was tagged as the wrong reference)
    label correct:   ...and the label is one a correct check should give for that mutation
    other flags:     issues on citations that were not altered; `preexisting` ones were also
                     flagged in the unmodified paper (often genuine citation problems)
    """
    live = [v for v in verdicts if not v.duplicate_of and v.claim_id in claims]
    by_sentence: dict[str, list[Verdict]] = {}
    for v in live:
        by_sentence.setdefault(claims[v.claim_id].citing_sentence_id, []).append(v)

    planted_sentences = {p.sentence_id for p in ground_truth.planted}
    details = []
    for p in ground_truth.planted:
        relevant = [v for v in by_sentence.get(p.sentence_id, []) if v.reference_key in p.reference_keys]
        flagged = [v for v in relevant if is_flag(v)]
        modes = sorted({m for v in relevant for m in v.failure_modes})
        details.append({
            "id": p.id,
            "sentence_id": p.sentence_id,
            "mutation": p.mutation,
            "description": p.description,
            "checked": bool(relevant),
            "caught": bool(flagged),
            "label_correct": any(v.label in p.expected_labels for v in relevant),
            "labels": sorted({v.label for v in relevant}),
            "failure_modes": modes,
            "expected_mode_found": bool(set(p.expected_failure_modes) & set(modes)),
            "escalated": any(v.escalated for v in relevant),
        })

    baseline = {(s, k) for s, k, _ in ground_truth.baseline_flags}
    untouched = [v for v in live if claims[v.claim_id].citing_sentence_id not in planted_sentences]
    other_flags = [v for v in untouched if is_flag(v)]
    preexisting = [v for v in other_flags if (claims[v.claim_id].citing_sentence_id, v.reference_key) in baseline]

    caught = sum(d["caught"] for d in details)
    checked = sum(d["checked"] for d in details)
    total = len(details)
    precision = caught / (caught + len(other_flags)) if caught + len(other_flags) else 0.0
    new_flags = len(other_flags) - len(preexisting)
    precision_new = caught / (caught + new_flags) if caught + new_flags else 0.0
    by_mutation: dict[str, dict[str, int]] = {}
    for d in details:
        row = by_mutation.setdefault(d["mutation"], {"planted": 0, "caught": 0})
        row["planted"] += 1
        row["caught"] += d["caught"]
    return {
        "planted": total,
        "checked": checked,
        "caught": caught,
        "recall": caught / total if total else 0.0,
        "label_accuracy": sum(d["label_correct"] for d in details) / total if total else 0.0,
        "expected_mode_rate": sum(d["expected_mode_found"] for d in details) / total if total else 0.0,
        "untouched_verdicts": len(untouched),
        "other_flags": len(other_flags),
        "other_flags_preexisting": len(preexisting),
        "precision": precision,
        "precision_excluding_preexisting": precision_new,
        "by_mutation": by_mutation,
        "details": details,
        "summary_line": f"Caught {caught} of {total} planted errors · {len(other_flags)} other citation"
                        f"{'' if len(other_flags) == 1 else 's'} flagged for review"
                        + (f" ({len(preexisting)} also flagged in the original paper)" if preexisting else ""),
    }


def planted_by_sentence(ground_truth: GroundTruth) -> dict[str, dict[str, Any]]:
    """What the UI shows on a planted citation: what was changed and the original text."""
    return {
        p.sentence_id: {"mutation": p.mutation, "description": p.description, "original": p.original,
                        "reference_keys": p.reference_keys}
        for p in ground_truth.planted
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
