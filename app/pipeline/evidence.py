"""Evidence retrieval (MODEL_EMBED).

Each cited paper's text is split into ~400-token chunks (approximated as ~300 words)
with ~80 tokens (~60 words) of overlap, on sentence boundaries. Each chunk keeps the
section heading it falls under and its character offsets. Chunks are embedded once per
reference (the LLM wrapper caches every embedding on disk). For each (claim, reference)
pair the top-k chunks by cosine similarity are returned, plus the abstract, since many
claims summarize abstracts.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Union

import numpy as np

from app import llm
from app.config import get_settings
from app.models import Claim, Evidence, Passage, Reference
from app.pipeline.references import read_reference_text
from app.pipeline.sentences import split_sentences

log = logging.getLogger(__name__)

CHUNK_WORDS = 300  # ~400 tokens
OVERLAP_WORDS = 60  # ~80 tokens
MAX_UNIT_WORDS = 200  # longer "sentences" (flattened tables, run-ons) are split by words
TOP_K = 5

OnDone = Callable[[Evidence], Union[None, Awaitable[None]]]


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int
    heading: str
    text: str


_WORD_RE = re.compile(r"\S+")


_SECTION_NAMES = (
    "abstract|introduction|background|related work|related works|preliminaries|method|methods|methodology|"
    "approach|model|experiments|experimental setup|experimental results|evaluation|results|analysis|"
    "discussion|limitations|conclusion|conclusions|future work|acknowledgments|acknowledgements|appendix"
)
# "3", "3.1", "A.2", "IV." followed by a capitalized title, or a standard section name.
_STRICT_HEADING_RE = re.compile(
    rf"^(?:(?:\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)+\.?|[IVX]+\.)\s+[A-Z][A-Za-z]{{2,}}[^.]*"
    rf"|(?:\d+\.?\s+)?(?i:{_SECTION_NAMES}))$"
)


def _is_heading(para: str, strict: bool = False) -> bool:
    """LaTeX-derived text has headings as their own short paragraphs. PDF text is full of
    short layout blocks (table cells, figure labels), so there only numbered or standard
    section names count."""
    p = re.sub(r"\s+", " ", para.strip())
    if not p or len(p) > 120 or p.endswith((".", ":", ";", ",", "?", "!")):
        return False
    if p.startswith(("Table:", "Figure:", "•")):
        return False
    if strict:
        return bool(_STRICT_HEADING_RE.match(p)) and len(p.split()) <= 12 and p[-1].isalnum()
    words = p.split()
    return len(words) <= 12 and (p[0].isupper() or p[0].isdigit())


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for part in re.split(r"(\n\s*\n)", text):
        if part.strip() and not part.startswith("\n"):
            lead = len(part) - len(part.lstrip())
            spans.append((pos + lead, pos + len(part.rstrip())))
        pos += len(part)
    return spans


def _units(text: str, strict_headings: bool) -> list[tuple[int, int, str, int]]:
    """(start, end, heading, word count) for each sentence-sized unit."""
    units = []
    heading = ""
    for ps, pe in _paragraph_spans(text):
        para = text[ps:pe]
        if _is_heading(para, strict_headings):
            heading = re.sub(r"\s+", " ", para.strip())
            continue
        for s, e in split_sentences(para, []):
            words = list(_WORD_RE.finditer(para, s, e))
            if len(words) <= MAX_UNIT_WORDS:
                units.append((ps + s, ps + e, heading, len(words)))
                continue
            for i in range(0, len(words), MAX_UNIT_WORDS):
                piece = words[i : i + MAX_UNIT_WORDS]
                units.append((ps + piece[0].start(), ps + piece[-1].end(), heading, len(piece)))
    return units


def chunk_text(text: str, strict_headings: bool = False) -> list[Chunk]:
    """Use strict_headings=True for text extracted from PDFs."""
    units = _units(text, strict_headings)
    chunks: list[Chunk] = []
    i = 0
    while i < len(units):
        heading = units[i][2]
        j, words = i, 0
        # Chunks never cross a section heading, so each chunk's heading is accurate.
        while j < len(units) and (words < CHUNK_WORDS or j == i) and units[j][2] == heading:
            words += units[j][3]
            j += 1
        start, end = units[i][0], units[j - 1][1]
        chunks.append(Chunk(len(chunks), start, end, heading, text[start:end]))
        if j >= len(units):
            break
        if units[j][2] != heading:
            i = j  # new section: start fresh, no overlap
            continue
        # Step back so the next chunk overlaps this one by ~OVERLAP_WORDS.
        k, back = j, 0
        while k - 1 > i and back + units[k - 1][3] <= OVERLAP_WORDS:
            k -= 1
            back += units[k][3]
        i = max(k, i + 1)
    return chunks


def query_text(claim_text: str) -> str:
    instruction = get_settings().embed_query_instruction
    return f"Instruct: {instruction}\nQuery: {claim_text}" if instruction else claim_text


def _chunk_document(chunk: Chunk) -> str:
    return f"{chunk.heading}\n{chunk.text}" if chunk.heading else chunk.text


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


@dataclass
class RefIndex:
    chunks: list[Chunk]
    matrix: np.ndarray  # (n_chunks, dim), unit rows
    abstract: Optional[str]
    abstract_vec: Optional[np.ndarray]


class EvidenceRetriever:
    def __init__(self, k: int = TOP_K):
        self.k = k
        self._indexes: dict[str, asyncio.Future] = {}

    async def index(self, ref: Reference) -> Optional[RefIndex]:
        """Chunk and embed a reference once, even when many claims ask concurrently."""
        key = f"{ref.text_path}|{ref.fulltext_status}|{ref.key}"
        if key not in self._indexes:
            self._indexes[key] = asyncio.ensure_future(self._build(ref))
        return await self._indexes[key]

    async def _build(self, ref: Reference) -> Optional[RefIndex]:
        text = read_reference_text(ref) if ref.fulltext_status == "full" else None
        strict = ref.fulltext_source in ("arxiv_pdf", "oa_pdf")
        chunks = await asyncio.to_thread(chunk_text, text, strict) if text else []
        abstract = (ref.abstract or "").strip() or None
        docs = [_chunk_document(c) for c in chunks] + ([abstract] if abstract else [])
        if not docs:
            return None
        vectors = _normalize(await llm.embed(docs))
        return RefIndex(
            chunks=chunks,
            matrix=vectors[: len(chunks)],
            abstract=abstract,
            abstract_vec=vectors[len(chunks)] if abstract else None,
        )

    async def query_vectors(self, claims: list[Claim]) -> dict[str, np.ndarray]:
        if not claims:
            return {}
        vectors = _normalize(await llm.embed([query_text(c.claim_text) for c in claims]))
        return {c.id: v for c, v in zip(claims, vectors)}

    async def abstract_scores(
        self, claims: list[Claim], refs_by_key: dict[str, Reference]
    ) -> dict[str, dict[str, float]]:
        """For each claim, similarity to the abstract of every reference that has one.

        Abstract-level similarity is comparable across references (unlike best-chunk scores,
        which grow with paper length), so it can tell when a claim fits a different
        reference in the same bibliography much better than the one it cites.
        """
        query_vecs = await self.query_vectors(claims)
        keys = list(refs_by_key)
        indexes = await asyncio.gather(*(self.index(refs_by_key[k]) for k in keys))
        with_abstract = [(k, ix.abstract_vec) for k, ix in zip(keys, indexes) if ix is not None and ix.abstract_vec is not None]
        if not with_abstract:
            return {c.id: {} for c in claims}
        matrix = np.stack([vec for _, vec in with_abstract])
        return {
            c.id: {k: float(s) for (k, _), s in zip(with_abstract, matrix @ query_vecs[c.id])}
            for c in claims
        }

    async def retrieve(self, claim: Claim, ref: Reference, query_vec: Optional[np.ndarray] = None) -> Evidence:
        evidence = Evidence(claim_id=claim.id, reference_key=ref.key)
        index = await self.index(ref)
        if index is None:
            return evidence
        if query_vec is None:
            query_vec = (await self.query_vectors([claim]))[claim.id]

        passages: list[Passage] = []
        if index.abstract is not None:
            passages.append(
                Passage(text=index.abstract, chunk_index=-1, score=float(index.abstract_vec @ query_vec), heading="Abstract")
            )
        if len(index.chunks):
            scores = index.matrix @ query_vec
            for i in np.argsort(-scores)[: self.k]:
                c = index.chunks[int(i)]
                passages.append(
                    Passage(text=c.text, chunk_index=c.index, score=float(scores[i]), heading=c.heading, start=c.start, end=c.end)
                )
        evidence.passages = passages
        return evidence

    async def retrieve_all(
        self,
        claims: list[Claim],
        refs_by_key: dict[str, Reference],
        on_done: Optional[OnDone] = None,
    ) -> list[Evidence]:
        """Evidence for every (claim, reference) pair. Failures yield empty evidence, not errors."""
        query_vecs = await self.query_vectors(claims)

        async def run(claim: Claim, key: str) -> Evidence:
            ref = refs_by_key.get(key)
            try:
                evidence = (
                    await self.retrieve(claim, ref, query_vecs[claim.id])
                    if ref is not None
                    else Evidence(claim_id=claim.id, reference_key=key)
                )
            except llm.TokenBudgetExceeded:
                raise
            except Exception:  # noqa: BLE001 - one bad reference must not fail the job
                log.exception("evidence for %s / %s failed", claim.id, key)
                evidence = Evidence(claim_id=claim.id, reference_key=key)
            if on_done is not None:
                result = on_done(evidence)
                if inspect.isawaitable(result):
                    await result
            return evidence

        return list(await asyncio.gather(*(run(c, k) for c in claims for k in c.reference_keys)))
