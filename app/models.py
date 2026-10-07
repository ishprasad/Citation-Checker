"""Core pydantic schemas shared by the pipeline, API and UI (brief section 4.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

FulltextStatus = Literal["full", "abstract_only", "unavailable"]
FulltextSource = Literal["arxiv_latex", "arxiv_pdf", "oa_pdf"]
ClaimType = Literal["finding", "method", "definition", "background"]
VerdictLabel = Literal["supported", "overstated", "not_found", "contradicted", "unverifiable"]
FailureMode = Literal[
    "scope_inflation",
    "number_drift",
    "hypothesis_as_finding",
    "secondary_citation",
    "retracted_source",
    "off_topic_citation",
    "correlation_as_causation",
    "significance_inflation",
    "corrected_source",
    "expression_of_concern",
]
Relation = Literal["causal", "associational", "none"]
SourceStatistics = Literal["significant", "not_significant", "not_tested", "unclear"]
JobStatus = Literal["queued", "running", "done", "partial", "failed"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Section(BaseModel):
    title: str
    text: str
    start: int  # character offset of `text` in the paper's plain text
    end: int


class EditorialUpdate(BaseModel):
    type: str  # Crossref update type: retraction, correction, erratum, expression_of_concern, ...
    doi: Optional[str] = None  # DOI of the notice
    source: Optional[str] = None  # "publisher" or "retraction-watch"
    date: Optional[str] = None  # ISO date


class Reference(BaseModel):
    key: str  # citation key as used in the source, e.g. "smith2020" (LaTeX) or "12" (PDF)
    label: str = ""  # as displayed in the paper text, e.g. "12" or "Smith et al., 2020"
    raw: str
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    openalex_id: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    fulltext_status: FulltextStatus = "unavailable"
    fulltext_source: Optional[FulltextSource] = None
    status_reason: Optional[str] = None  # shown in the UI when text is limited
    match_score: Optional[float] = None  # title similarity of the resolved record
    retracted: Optional[bool] = None  # None = could not be checked
    retraction_notice: Optional[str] = None  # DOI of the retraction notice, if known
    # Corrections, errata, expressions of concern... (Crossref `updated-by`, incl. Retraction Watch).
    updates: list[EditorialUpdate] = Field(default_factory=list)
    text_path: Optional[str] = None


class Coverage(BaseModel):
    total: int = 0
    full: int = 0
    abstract_only: int = 0
    unavailable: int = 0
    retracted: int = 0

    @classmethod
    def of(cls, references: list[Reference]) -> "Coverage":
        return cls(
            total=len(references),
            full=sum(r.fulltext_status == "full" for r in references),
            abstract_only=sum(r.fulltext_status == "abstract_only" for r in references),
            unavailable=sum(r.fulltext_status == "unavailable" for r in references),
            retracted=sum(bool(r.retracted) for r in references),
        )

    def line(self) -> str:
        return (
            f"Full text found for {self.full} of {self.total} references · abstract only for "
            f"{self.abstract_only} · unavailable for {self.unavailable}."
        )


class Paper(BaseModel):
    id: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    source: Literal["arxiv", "pdf"]
    arxiv_id: Optional[str] = None  # including version, e.g. "1706.03762v7"
    parsed_from: Literal["latex", "pdf", "grobid"] = "latex"
    citation_style: Literal["numeric", "author_year"] = "numeric"
    text: str = ""  # full plain text; section/sentence offsets index into this
    sections: list[Section] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    # In-text citation keys seen vs. those mapped to a bibliography entry.
    citation_keys_total: int = 0
    citation_keys_mapped: int = 0
    notes: list[str] = Field(default_factory=list)  # shown in the report header



class CitingSentence(BaseModel):
    id: str
    text: str
    section: str = ""
    start: int
    end: int
    reference_keys: list[str]
    # Offsets of each citation marker within the paper text, for highlighting.
    marker_spans: list[tuple[int, int]] = Field(default_factory=list)
    # Keys cited by each marker, aligned with marker_spans.
    marker_keys: list[list[str]] = Field(default_factory=list)
    context_before: str = ""  # previous sentence in the same paragraph
    context_after: str = ""  # next sentence in the same paragraph


class IngestResult(BaseModel):
    paper: Paper
    sentences: list[CitingSentence]

class Claim(BaseModel):
    id: str
    citing_sentence_id: str
    claim_text: str
    reference_keys: list[str]
    claim_type: ClaimType = "finding"
    numbers: list[str] = Field(default_factory=list)
    scope: Optional[str] = None
    # True when extraction failed and the whole sentence is used as the claim.
    fallback: bool = False


class Passage(BaseModel):
    text: str
    chunk_index: int  # -1 for the abstract
    score: float
    heading: str = ""  # section heading the chunk falls under
    start: int = -1  # character offsets in the cited paper's text (-1 for the abstract)
    end: int = -1


class Evidence(BaseModel):
    claim_id: str
    reference_key: str
    passages: list[Passage] = Field(default_factory=list)


class NumberComparison(BaseModel):
    claim_value: str
    source_value: str
    same_quantity: bool  # the two numbers refer to the same measured quantity


class JudgeSignals(BaseModel):
    """Structured observations from the judge, used by the failure-mode detectors."""

    claim_scope: Optional[str] = None
    source_scope: Optional[str] = None
    scope_broader_than_source: bool = False
    source_hedged: bool = False
    claim_hedged: bool = False
    numbers: list[NumberComparison] = Field(default_factory=list)
    support_attributed_to_other_work: bool = False
    other_work: Optional[str] = None
    claim_relation: Relation = "none"  # how the claim links its variables
    source_relation: Relation = "none"  # how the cited passage links them
    source_statistics: SourceStatistics = "unclear"  # statistical support for the point in the source
    cited_work_on_topic: bool = True  # False if the cited work is not about the claim's topic at all


class Verdict(BaseModel):
    claim_id: str
    reference_key: str
    label: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    # What the citing sentence attributes to this reference, as the judge read it
    # (may correct an imperfect extracted claim).
    attributed_claim: str = ""
    quote: Optional[str] = None
    quote_verified: bool = False
    quote_start: Optional[int] = None  # offsets of the quote in quote_context
    quote_end: Optional[int] = None
    quote_context: Optional[str] = None  # surrounding text from the cited paper, for display
    explanation: str = ""
    signals: JudgeSignals = Field(default_factory=JudgeSignals)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    # One short human-readable note per failure mode, e.g. {"number_drift": "34B in the claim, 33B in the source"}.
    failure_notes: dict[str, str] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)  # e.g. "quote_unverified"
    escalated: bool = False
    original_label: Optional[VerdictLabel] = None  # judge label before escalation
    original_confidence: Optional[float] = None
    model_used: str = ""
    # Set when this verdict restates another verdict for the same sentence and reference
    # (pair id "claim_id|reference_key" of the verdict that is kept); hidden from reports.
    duplicate_of: Optional[str] = None


class ModelUsage(BaseModel):
    model: str
    role: str
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


class UsageSummary(BaseModel):
    per_model: list[ModelUsage] = Field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_estimated_cost_usd: float = 0.0


STAGES = (
    ("parse", "Parse"),
    ("resolve", "Resolve references"),
    ("extract", "Extract claims"),
    ("retrieve", "Retrieve evidence"),
    ("judge", "Judge"),
    ("escalate", "Escalate"),
)


class StageProgress(BaseModel):
    stage: str
    label: str = ""
    done: int = 0
    total: int = 0
    state: Literal["pending", "running", "done"] = "pending"


def default_progress() -> list[StageProgress]:
    return [StageProgress(stage=key, label=label) for key, label in STAGES]


class Job(BaseModel):
    id: str
    status: JobStatus = "queued"
    mode: Literal["quick", "full"] = "quick"
    source: Literal["arxiv", "pdf", "demo"] = "arxiv"
    arxiv_id: Optional[str] = None  # as requested (normalized); versioned id is on the paper
    title: str = ""
    progress: list[StageProgress] = Field(default_factory=default_progress)
    created_at: datetime = Field(default_factory=_now)
    paper_id: Optional[str] = None
    sentences_total: int = 0  # citing sentences in the paper
    sentences_checked: int = 0  # citing sentences included in this check
    references_total: int = 0  # entries in the paper's bibliography
    pipeline_version: int = 0  # results from an older pipeline are not reused for new requests
    estimate_seconds: Optional[int] = None  # expected duration of this check when nothing is cached
    rest_estimate_seconds: Optional[int] = None  # expected duration of "Check the rest" for a quick check
    started_at: Optional[datetime] = None  # start of the latest run (a check can run again when expanded)
    finished_at: Optional[datetime] = None
    usage: UsageSummary = Field(default_factory=UsageSummary)
    message: Optional[str] = None
