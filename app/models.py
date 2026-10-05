"""Core pydantic schemas shared by the pipeline, API and UI (brief section 4.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

FulltextStatus = Literal["full", "abstract_only", "unavailable"]
ClaimType = Literal["finding", "method", "definition", "background"]
VerdictLabel = Literal["supported", "overstated", "not_found", "contradicted", "unverifiable"]
FailureMode = Literal[
    "scope_inflation",
    "number_drift",
    "hypothesis_as_finding",
    "secondary_citation",
    "retracted_source",
]
JobStatus = Literal["queued", "running", "done", "partial", "failed"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Section(BaseModel):
    title: str
    text: str
    start: int  # character offset of `text` in the paper's plain text
    end: int


class Reference(BaseModel):
    key: str  # as cited in text, e.g. "12" or "Smith2020"
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
    retracted: Optional[bool] = None
    text_path: Optional[str] = None


class Paper(BaseModel):
    id: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    source: Literal["arxiv", "pdf"]
    parsed_from: Literal["latex", "pdf", "grobid"] = "latex"
    text: str = ""  # full plain text; section/sentence offsets index into this
    sections: list[Section] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)


class CitingSentence(BaseModel):
    id: str
    text: str
    section: str = ""
    start: int
    end: int
    reference_keys: list[str]
    # Offsets of each citation marker within the paper text, for highlighting.
    marker_spans: list[tuple[int, int]] = Field(default_factory=list)


class Claim(BaseModel):
    id: str
    citing_sentence_id: str
    claim_text: str
    reference_keys: list[str]
    claim_type: ClaimType = "finding"
    numbers: list[str] = Field(default_factory=list)
    scope: Optional[str] = None


class Passage(BaseModel):
    text: str
    chunk_index: int  # -1 for the abstract
    score: float


class Evidence(BaseModel):
    claim_id: str
    reference_key: str
    passages: list[Passage] = Field(default_factory=list)


class Verdict(BaseModel):
    claim_id: str
    reference_key: str
    label: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    quote: Optional[str] = None
    explanation: str = ""
    failure_modes: list[FailureMode] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)  # e.g. "quote_unverified"
    escalated: bool = False
    original_label: Optional[VerdictLabel] = None  # judge label before escalation
    model_used: str = ""


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


class StageProgress(BaseModel):
    stage: str
    done: int = 0
    total: int = 0


class Job(BaseModel):
    id: str
    status: JobStatus = "queued"
    mode: Literal["quick", "full"] = "quick"
    progress: list[StageProgress] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    paper_id: Optional[str] = None
    usage: UsageSummary = Field(default_factory=UsageSummary)
    message: Optional[str] = None
