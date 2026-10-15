"""Resolve references to real papers and fetch their text.

Identification order: arXiv ID or DOI from the bibliography entry; then an OpenAlex
title search; then a Crossref bibliographic search. A match must clear a title
similarity threshold (with a year sanity check) to avoid checking claims against
the wrong paper.

Text order: arXiv LaTeX source (cleanest), arXiv PDF, an open-access PDF listed by
OpenAlex, and finally the abstract (`abstract_only`).

Retractions: OpenAlex `is_retracted`, and Crossref `updated-by` entries with
type "retraction" (Crossref includes Retraction Watch data, source "retraction-watch").
If neither source has a record, `retracted` stays None rather than guessing.

Every HTTP response is cached on disk, and each fully resolved reference is cached
too, so re-running a paper (or another paper citing the same work) is instant.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union

from app.config import get_settings
import httpx

from app.http import DownloadTooLarge, cache_path, fetch_cached
from app.models import EditorialUpdate, Reference
from app.pipeline.ingest import (
    ArxivMeta,
    NoLatexSource,
    fetch_arxiv_metadata_batch,
    parse_arxiv_feed,
    parse_latex_project,
    strip_version,
    unpack_source,
)
from app.pipeline.jats import JatsError, jats_to_text
from app.pipeline.pdftext import PdfTextError, extract_pdf_text

log = logging.getLogger(__name__)

RESOLVER_VERSION = 9  # bump to invalidate cached resolutions after logic changes
TITLE_MATCH_THRESHOLD = 0.88
MIN_FULLTEXT_CHARS = 3000
MAX_PDF_BYTES = 40 * 1024 * 1024
MAX_TEXT_CHARS = 800_000
MAX_PDF_ATTEMPTS = 3
CONCURRENCY = 8

OPENALEX_WORKS = "https://api.openalex.org/works"
OPENALEX_SELECT = "id,doi,title,publication_year,locations,best_oa_location,abstract_inverted_index,is_retracted,ids"
OPENALEX_DOI_BATCH = 50
CROSSREF_WORKS = "https://api.crossref.org/works"
CROSSREF_SELECT = "DOI,title,issued,abstract,updated-by,type"

REASON_ABSTRACT_ONLY = (
    "Only the abstract of this paper is openly available, so claims about details may not be verifiable."
)
REASON_NOT_OPEN = "The full text of this paper is not openly available."
REASON_NOT_FOUND = "This reference could not be matched to a known paper."
REASON_LOOKUP_FAILED = "Looking up this reference failed."

OnDone = Callable[[Reference], Union[None, Awaitable[None]]]


# ----------------------------------------------------------------------------
# Matching helpers
# ----------------------------------------------------------------------------


def normalize_title(s: Optional[str]) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def title_similarity(a: Optional[str], b: Optional[str]) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi.strip(), flags=re.IGNORECASE)
    return doi.lower().rstrip(".") or None


def arxiv_doi(arxiv_id: str) -> str:
    return f"10.48550/arxiv.{strip_version(arxiv_id).lower()}"


_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([a-z\-]+(?:\.[a-z]{2})?/\d{7}|\d{4}\.\d{4,5})", re.I)
_ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", re.I)


def _arxiv_from(*urls: Optional[str]) -> Optional[str]:
    for url in urls:
        if not url:
            continue
        m = _ARXIV_URL_RE.search(url) or _ARXIV_DOI_RE.search(url)
        if m:
            return m.group(1)
    return None


def arxiv_year(arxiv_id: Optional[str]) -> Optional[int]:
    """Submission year encoded in an arXiv ID: "2004.09142" -> 2020, "math/0512424" -> 2005."""
    m = re.match(r"(?:[a-z\-]+(?:\.[a-z]{2})?/)?(\d{2})\d{2}[.\d]", arxiv_id or "", re.I)
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 91 else 2000 + yy


# A preprint is posted up to a few years before the published version, or shortly after it.
PREPRINT_YEARS_BEFORE = 6
PREPRINT_YEARS_AFTER = 1


def plausible_preprint(arxiv_id: Optional[str], year: Optional[int]) -> bool:
    """Can this arXiv paper be a version of a work published in `year`? Catches records that
    link a different paper with the same generic title (a 2005 survey for a 2020 book)."""
    ay = arxiv_year(arxiv_id)
    return ay is None or year is None or year - PREPRINT_YEARS_BEFORE <= ay <= year + PREPRINT_YEARS_AFTER


DOI_TITLE_MIN_SIMILARITY = 0.5


def doi_title_mismatch(ref: Reference, cand: "Candidate") -> bool:
    """The record found for the entry's DOI has a clearly different title: the DOI was probably
    mangled (e.g. broken across lines in a PDF) and points at another paper."""
    if not ref.title or not cand.title or len(normalize_title(ref.title)) < 15:
        return False
    if normalize_title(cand.title) in normalize_title(ref.raw):
        return False
    return title_similarity(ref.title, cand.title) < DOI_TITLE_MIN_SIMILARITY


@dataclass
class Candidate:
    title: str
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    openalex_id: Optional[str] = None
    abstract: Optional[str] = None
    pdf_urls: list[str] = field(default_factory=list)
    retracted: Optional[bool] = None
    source: str = "openalex"
    pmcid: Optional[str] = None  # PubMed Central: full text as JATS XML through Europe PMC


def match_score(ref: Reference, cand: Candidate) -> float:
    if ref.title:
        score = title_similarity(ref.title, cand.title)
        # Older bibliographies sometimes cite a shortened title
        # ("A decomposable attention model" for "... for Natural Language Inference").
        nr, nc = normalize_title(ref.title), normalize_title(cand.title)
        if len(nr) >= 25 and nc.startswith(nr + " ") and (not ref.year or not cand.year or abs(ref.year - cand.year) <= 1):
            score = max(score, 0.9)
    else:
        # No parsed title (e.g. PDF references): accept if the candidate's title
        # appears verbatim inside the raw reference string.
        nt, nr = normalize_title(cand.title), normalize_title(ref.raw)
        score = 1.0 if len(nt) >= 15 and nt in nr else 0.0
    if ref.year and cand.year:
        # Preprint vs. publication years often differ by one; a bigger gap suggests a
        # different paper with the same (generic) title.
        gap = abs(ref.year - cand.year)
        score -= 0.0 if gap <= 1 else 0.05 if gap == 2 else 0.2
    return score


def best_candidate(ref: Reference, candidates: list[Candidate]) -> tuple[Optional[Candidate], float]:
    scored = sorted(((match_score(ref, c), c) for c in candidates), key=lambda x: -x[0])
    if scored and scored[0][0] >= TITLE_MATCH_THRESHOLD:
        return scored[0][1], scored[0][0]
    return None, scored[0][0] if scored else 0.0


def openalex_abstract(inverted: Optional[dict[str, list[int]]]) -> Optional[str]:
    if not inverted:
        return None
    positions = [(pos, word) for word, poss in inverted.items() for pos in poss]
    return " ".join(word for _, word in sorted(positions)) or None


def candidate_from_openalex(work: dict[str, Any]) -> Candidate:
    pdf_urls: list[str] = []
    arxiv_id = None
    best = work.get("best_oa_location") or {}
    for loc in [best, *(work.get("locations") or [])]:
        arxiv_id = arxiv_id or _arxiv_from(loc.get("landing_page_url"), loc.get("pdf_url"))
        pdf = loc.get("pdf_url")
        if pdf and pdf not in pdf_urls and not _arxiv_from(pdf):
            pdf_urls.append(pdf)
    arxiv_id = arxiv_id or _arxiv_from(work.get("doi"))
    return Candidate(
        title=work.get("title") or "",
        year=work.get("publication_year"),
        doi=normalize_doi(work.get("doi")),
        arxiv_id=arxiv_id,
        openalex_id=(work.get("id") or "").rsplit("/", 1)[-1] or None,
        abstract=openalex_abstract(work.get("abstract_inverted_index")),
        pdf_urls=pdf_urls,
        retracted=work.get("is_retracted"),
        source="openalex",
        pmcid=_pmcid((work.get("ids") or {}).get("pmcid")),
    )


def _pmcid(value: Optional[str]) -> Optional[str]:
    m = re.search(r"(?:PMC)?(\d{4,9})/?$", value or "")
    return f"PMC{m.group(1)}" if m else None


def _strip_jats(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^abstract\s*", "", text, flags=re.IGNORECASE) or None


def candidate_from_crossref(item: dict[str, Any]) -> Candidate:
    parts = (item.get("issued") or {}).get("date-parts") or [[None]]
    year = parts[0][0] if parts and parts[0] else None
    retracted, _ = crossref_retraction(item)
    return Candidate(
        title=((item.get("title") or [""])[0]) or "",
        year=year,
        doi=normalize_doi(item.get("DOI")),
        abstract=_strip_jats(item.get("abstract")),
        retracted=retracted,
        source="crossref",
    )


RETRACTION_TYPES = {"retraction", "withdrawal", "removal"}


def crossref_updates(item: dict[str, Any]) -> list[EditorialUpdate]:
    """Editorial notices attached to a work (`updated-by`): retractions, corrections, errata,
    expressions of concern... Crossref merges in Retraction Watch data (source "retraction-watch")."""
    updates = []
    for update in item.get("updated-by") or []:
        kind = (update.get("type") or "").lower().replace("-", "_").replace(" ", "_")
        if not kind:
            continue
        date = ((update.get("updated") or {}).get("date-time") or "")[:10] or None
        updates.append(EditorialUpdate(type=kind, doi=normalize_doi(update.get("DOI")),
                                       source=update.get("source"), date=date))
    return updates


def crossref_retraction(item: dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Crossref marks retracted works with an `updated-by` entry of type "retraction"
    (also "withdrawal" / "removal")."""
    for update in crossref_updates(item):
        if update.type in RETRACTION_TYPES:
            return True, update.doi
    return False, None


# ----------------------------------------------------------------------------
# API calls (all cached on disk)
# ----------------------------------------------------------------------------


def _polite(params: dict[str, str]) -> dict[str, str]:
    email = get_settings().contact_email
    return {**params, "mailto": email} if email else params


# Failures of metadata APIs (rate limits, outages) for the reference being resolved: the
# resolver continues with other sources, but does not cache a result built without them.
_lookup_failures: contextvars.ContextVar[Optional["Downloads"]] = contextvars.ContextVar("lookup_failures", default=None)


async def _get_json(url: str, namespace: str, params: Optional[dict] = None) -> Optional[dict]:
    try:
        data = await fetch_cached(url, namespace, params=params, suffix=".json", allow_missing=True)
    except httpx.HTTPError as exc:
        log.warning("%s lookup failed (%s); continuing with other sources", namespace, exc)
        tracker = _lookup_failures.get()
        if tracker is not None:
            tracker.transient_failure = True
        return None
    return json.loads(data) if data else None


async def openalex_by_dois(dois: list[str]) -> dict[str, dict]:
    """{normalized doi: work} for all DOIs OpenAlex knows, in batches."""
    unique = sorted({d for d in (normalize_doi(x) for x in dois) if d and "|" not in d and "," not in d})
    found: dict[str, dict] = {}
    for start in range(0, len(unique), OPENALEX_DOI_BATCH):
        batch = unique[start : start + OPENALEX_DOI_BATCH]
        payload = await _get_json(
            OPENALEX_WORKS,
            "openalex",
            _polite({"filter": "doi:" + "|".join(batch), "per-page": str(len(batch)), "select": OPENALEX_SELECT}),
        )
        for work in (payload or {}).get("results", []):
            doi = normalize_doi(work.get("doi"))
            if doi:
                found[doi] = work
    return found


async def openalex_search(query: str) -> list[dict]:
    q = normalize_title(query)[:300]
    if len(q) < 8:
        return []
    payload = await _get_json(
        OPENALEX_WORKS, "openalex", _polite({"search": q, "per-page": "5", "select": OPENALEX_SELECT})
    )
    return (payload or {}).get("results", [])


async def crossref_search(query: str) -> list[dict]:
    q = re.sub(r"\s+", " ", query).strip()[:400]
    if len(q) < 8:
        return []
    payload = await _get_json(
        CROSSREF_WORKS, "crossref", _polite({"query.bibliographic": q, "rows": "5", "select": CROSSREF_SELECT})
    )
    return ((payload or {}).get("message") or {}).get("items", [])


ARXIV_TWIN_THRESHOLD = 0.93
# The arXiv search API rate-limits title queries hard. After a refusal, skip twin searches for a
# while: most papers without open text (e.g. in medicine) have no arXiv preprint anyway.
ARXIV_SEARCH_COOLDOWN = 600.0
_arxiv_search_paused_until = 0.0


async def find_arxiv_twin(title: str, year: Optional[int] = None) -> Optional[str]:
    """arXiv ID of a preprint with (nearly) the same title and a compatible year: OpenAlex
    first, then the arXiv API."""
    global _arxiv_search_paused_until
    for work in await openalex_search(title):
        cand = candidate_from_openalex(work)
        if (cand.arxiv_id and title_similarity(title, cand.title) >= ARXIV_TWIN_THRESHOLD
                and plausible_preprint(cand.arxiv_id, year)):
            return cand.arxiv_id
    words = normalize_title(title)
    if len(words) < 12 or time.monotonic() < _arxiv_search_paused_until:
        return None
    try:
        data = await fetch_cached(
            "https://export.arxiv.org/api/query",
            "arxiv_meta",
            params={"search_query": f'ti:"{words}"', "max_results": "3"},
            suffix=".xml",
            allow_missing=True,
        )
    except (httpx.HTTPError, DownloadTooLarge) as exc:
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            _arxiv_search_paused_until = time.monotonic() + ARXIV_SEARCH_COOLDOWN
            log.warning("arXiv search is rate-limiting; skipping preprint look-ups for %d s", ARXIV_SEARCH_COOLDOWN)
        else:
            log.info("arXiv preprint search failed for %r: %s", title[:60], exc)
        return None
    for arxiv_id, meta in (parse_arxiv_feed(data) if data else {}).items():
        if title_similarity(title, meta.title) >= ARXIV_TWIN_THRESHOLD and plausible_preprint(arxiv_id, year):
            return arxiv_id
    return None


async def crossref_work(doi: str) -> Optional[dict]:
    payload = await _get_json(f"{CROSSREF_WORKS}/{doi}", "crossref", _polite({}))
    return (payload or {}).get("message")


# ----------------------------------------------------------------------------
# Text fetching
# ----------------------------------------------------------------------------


@dataclass
class FetchedText:
    text: str
    source: str  # FulltextSource
    sidecar: dict  # the cited paper's own references, for secondary-citation detection


def _latex_text(data: bytes) -> FetchedText:
    files = unpack_source(data)  # may raise NoLatexSource (carrying the PDF)
    paper, _ = parse_latex_project(files, keep_captions=True)
    sidecar = {"references": [{"label": r.label, "title": r.title, "raw": r.raw} for r in paper.references]}
    return FetchedText(paper.text, "arxiv_latex", sidecar)


def _pdf_text(data: bytes, source: str) -> FetchedText:
    pdf = extract_pdf_text(data)
    return FetchedText(pdf.body, source, {"references_text": pdf.references})


class Downloads:
    """Downloads for one reference. Records transient failures (network errors on arXiv) so
    that a degraded result is not cached as if it were final."""

    def __init__(self) -> None:
        self.transient_failure = False

    async def get(self, url: str, namespace: str, suffix: str, transient_ok: bool = True) -> Optional[bytes]:
        try:
            return await fetch_cached(url, namespace, suffix=suffix, allow_missing=True, max_bytes=MAX_PDF_BYTES)
        except DownloadTooLarge as exc:
            log.info("skipping %s", exc)
        except httpx.HTTPError as exc:
            log.warning("download %s failed: %s", url, exc)
            self.transient_failure = self.transient_failure or transient_ok
        return None


async def fetch_arxiv_text(arxiv_id: str, downloads: Optional[Downloads] = None) -> Optional[FetchedText]:
    downloads = downloads or Downloads()
    arxiv_id = strip_version(arxiv_id)
    source = await downloads.get(f"https://arxiv.org/e-print/{arxiv_id}", "arxiv_src", ".eprint")
    if source:
        try:
            fetched = await asyncio.to_thread(_latex_text, source)
            if len(fetched.text) >= MIN_FULLTEXT_CHARS:
                return fetched
            log.info("arXiv %s: LaTeX text too short (%d chars); trying PDF", arxiv_id, len(fetched.text))
        except NoLatexSource as exc:
            if exc.pdf:
                try:
                    return await asyncio.to_thread(_pdf_text, exc.pdf, "arxiv_pdf")
                except PdfTextError as pdf_exc:
                    log.info("arXiv %s: %s", arxiv_id, pdf_exc)
                    return None
        except Exception as exc:  # noqa: BLE001 - unusual LaTeX should not fail the job
            log.warning("arXiv %s: LaTeX conversion failed (%s); trying PDF", arxiv_id, exc)

    pdf = await downloads.get(f"https://arxiv.org/pdf/{arxiv_id}", "arxiv_pdf", ".pdf")
    if pdf:
        try:
            return await asyncio.to_thread(_pdf_text, pdf, "arxiv_pdf")
        except PdfTextError as exc:
            log.info("arXiv %s PDF: %s", arxiv_id, exc)
    return None


EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
MIN_FULLTEXT_CHARS = 2000


async def europepmc_pmcid(doi: str) -> Optional[str]:
    """PMC ID of an open-access article in Europe PMC, by DOI (free API, no key)."""
    payload = await _get_json(f"{EUROPEPMC}/search", "europepmc",
                              {"query": f'DOI:"{doi}"', "format": "json", "resultType": "lite"})
    for hit in ((payload or {}).get("resultList") or {}).get("result", []):
        if hit.get("pmcid") and hit.get("isOpenAccess") == "Y":
            return hit["pmcid"]
    return None


async def fetch_europepmc_text(pmcid: str, downloads: Optional[Downloads] = None) -> Optional[FetchedText]:
    """Full text from Europe PMC's open-access subset (structured XML: no PDF layout problems)."""
    downloads = downloads or Downloads()
    data = await downloads.get(f"{EUROPEPMC}/{pmcid}/fullTextXML", "europepmc_xml", ".xml")
    if not data:
        return None
    try:
        text, sidecar = await asyncio.to_thread(jats_to_text, data)
    except JatsError as exc:
        log.info("Europe PMC %s: %s", pmcid, exc)
        return None
    return FetchedText(text, "europepmc", sidecar) if len(text) >= MIN_FULLTEXT_CHARS else None


async def fetch_oa_pdf_text(urls: list[str], downloads: Optional[Downloads] = None) -> Optional[FetchedText]:
    downloads = downloads or Downloads()
    for url in urls[:MAX_PDF_ATTEMPTS]:
        # Publisher sites often block bots outright; treat their errors as final.
        data = await downloads.get(url, "oa_pdf", ".pdf", transient_ok=False)
        if not data or not data.startswith(b"%PDF-"):
            continue  # landing page or paywall HTML
        try:
            return await asyncio.to_thread(_pdf_text, data, "oa_pdf")
        except PdfTextError as exc:
            log.info("OA PDF %s: %s", url, exc)
    return None


def _store_text(identity: str, fetched: FetchedText) -> str:
    path = cache_path("fulltext", identity, ".txt")
    path.write_text(fetched.text[:MAX_TEXT_CHARS], encoding="utf-8")
    path.with_suffix(".refs.json").write_text(json.dumps(fetched.sidecar), encoding="utf-8")
    return str(path)


def read_reference_text(ref: Reference) -> Optional[str]:
    if ref.text_path and Path(ref.text_path).exists():
        return Path(ref.text_path).read_text(encoding="utf-8")
    return None


def read_reference_sidecar(ref: Reference) -> dict:
    if ref.text_path:
        path = Path(ref.text_path).with_suffix(".refs.json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


# ----------------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------------


def _resolution_cache_key(ref: Reference) -> str:
    return json.dumps(
        [RESOLVER_VERSION, ref.raw, ref.title, normalize_doi(ref.doi), ref.arxiv_id, ref.year], ensure_ascii=False
    )


_RESOLVED_FIELDS = (
    "title", "authors", "year", "doi", "arxiv_id", "openalex_id", "url", "abstract", "fulltext_status",
    "fulltext_source", "status_reason", "match_score", "retracted", "retraction_notice", "text_path", "updates",
)


def _load_cached_resolution(ref: Reference) -> Optional[Reference]:
    path = cache_path("resolved", _resolution_cache_key(ref), ".json")
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("text_path") and not Path(data["text_path"]).exists():
        return None
    return Reference.model_validate({**ref.model_dump(), **{k: data[k] for k in _RESOLVED_FIELDS if k in data}})


def _save_resolution(original: Reference, resolved: Reference) -> None:
    path = cache_path("resolved", _resolution_cache_key(original), ".json")
    path.write_text(json.dumps(resolved.model_dump(include=set(_RESOLVED_FIELDS))), encoding="utf-8")


class ReferenceResolver:
    def __init__(self, concurrency: int = CONCURRENCY):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.arxiv_meta: dict[str, ArxivMeta] = {}
        self.openalex: dict[str, dict] = {}
        self.prefetch_degraded = False  # a batch lookup failed: results are not cached as final
        self.incomplete = 0  # references resolved with a failed lookup or download (not final)

    async def resolve_all(self, refs: list[Reference], on_done: Optional[OnDone] = None) -> list[Reference]:
        """Resolve and fetch text for each reference. Never raises for a single reference."""
        cached: dict[int, Reference] = {}
        todo: list[tuple[int, Reference]] = []
        for idx, ref in enumerate(refs):
            hit = _load_cached_resolution(ref)
            if hit is not None:
                cached[idx] = hit
            else:
                todo.append((idx, ref))

        results: dict[int, Reference] = dict(cached)
        for idx in sorted(cached):
            await _call(on_done, cached[idx])

        if todo:
            await self._prefetch([r for _, r in todo])

            async def run(idx: int, ref: Reference) -> None:
                async with self.semaphore:
                    try:
                        resolved, final = await self.resolve_one(ref)
                        self.incomplete += not final
                        if final and not self.prefetch_degraded:
                            _save_resolution(ref, resolved)
                    except Exception:  # noqa: BLE001 - isolate failures per reference
                        log.exception("reference %s: resolution failed", ref.key)
                        self.incomplete += 1
                        resolved = ref.model_copy(
                            update={"fulltext_status": "unavailable", "status_reason": REASON_LOOKUP_FAILED}
                        )
                results[idx] = resolved
                await _call(on_done, resolved)

            await asyncio.gather(*(run(i, r) for i, r in todo))
        return [results[i] for i in range(len(refs))]

    async def _prefetch(self, refs: list[Reference]) -> None:
        """Batch lookups: arXiv metadata and OpenAlex records for known DOIs / arXiv IDs."""
        arxiv_ids = [r.arxiv_id for r in refs if r.arxiv_id]
        dois = [r.doi for r in refs if r.doi] + [arxiv_doi(a) for a in arxiv_ids]
        meta_task = fetch_arxiv_metadata_batch(arxiv_ids) if arxiv_ids else _empty()
        tracker = Downloads()
        token = _lookup_failures.set(tracker)
        try:
            results = await asyncio.gather(meta_task, openalex_by_dois(dois), return_exceptions=True)
        finally:
            _lookup_failures.reset(token)
        for result, name in zip(results, ("arXiv metadata", "OpenAlex DOI batch")):
            if isinstance(result, Exception):
                log.warning("%s lookup failed: %s", name, result)
        self.prefetch_degraded = tracker.transient_failure or any(isinstance(r, Exception) for r in results)
        self.arxiv_meta = results[0] if isinstance(results[0], dict) else {}
        self.openalex = results[1] if isinstance(results[1], dict) else {}

    async def resolve_one(self, ref: Reference) -> tuple[Reference, bool]:
        """Returns (resolved reference, final). Not final when a download failed transiently,
        so the result is not cached and the next run tries again."""
        ref = ref.model_copy(deep=True)
        downloads = Downloads()
        _lookup_failures.set(downloads)  # this task's own context: metadata API failures count as transient
        ref.doi = normalize_doi(ref.doi)
        cand: Optional[Candidate] = None
        score: Optional[float] = None

        # 1. Identifiers from the bibliography entry.
        meta = self.arxiv_meta.get(strip_version(ref.arxiv_id)) if ref.arxiv_id else None
        if ref.arxiv_id and meta and ref.title and title_similarity(ref.title, meta.title) < 0.6:
            log.info("reference %s: arXiv %s title mismatch; searching instead", ref.key, ref.arxiv_id)
            ref.arxiv_id, meta = None, None
        if ref.arxiv_id and not meta:
            log.info("reference %s: arXiv %s not found", ref.key, ref.arxiv_id)
            ref.arxiv_id = None
        if ref.doi and ref.doi in self.openalex:
            cand = candidate_from_openalex(self.openalex[ref.doi])
            if doi_title_mismatch(ref, cand):
                log.info("reference %s: DOI %s is a different paper (%r); searching instead", ref.key, ref.doi, cand.title)
                ref.doi, cand = None, None
        elif ref.arxiv_id and arxiv_doi(ref.arxiv_id) in self.openalex:
            cand = candidate_from_openalex(self.openalex[arxiv_doi(ref.arxiv_id)])
        entry_arxiv = ref.arxiv_id  # the citing paper itself cites the arXiv version

        # 2. Search when the entry has no usable identifier.
        if cand is None and not ref.arxiv_id and not ref.doi:
            cand, score = best_candidate(ref, [candidate_from_openalex(w) for w in await openalex_search(ref.title or ref.raw)])
            if cand is None:
                cr, cr_score = best_candidate(ref, [candidate_from_crossref(i) for i in await crossref_search(ref.raw)])
                score = max(score or 0.0, cr_score)
                if cr is not None:
                    works = await openalex_by_dois([cr.doi]) if cr.doi else {}
                    cand = candidate_from_openalex(works[cr.doi]) if cr.doi in works else cr
                    score = cr_score

        # 3. Merge what we learned.
        crossref_item = await crossref_work(ref.doi or cand.doi) if (ref.doi or (cand and cand.doi)) else None
        if cand is None and crossref_item is not None:
            cand = candidate_from_crossref(crossref_item)
        if cand is not None:
            ref.title = cand.title or ref.title
            ref.doi = ref.doi or cand.doi
            if cand.arxiv_id and not plausible_preprint(cand.arxiv_id, cand.year or ref.year):
                log.info("reference %s: ignoring arXiv %s linked to a %s record", ref.key, cand.arxiv_id, cand.year)
                cand.arxiv_id = None
            ref.arxiv_id = ref.arxiv_id or cand.arxiv_id
            ref.openalex_id = cand.openalex_id
            ref.year = ref.year or cand.year
            ref.abstract = cand.abstract
            ref.retracted = cand.retracted
        if meta is not None:
            ref.title = meta.title or ref.title
            ref.abstract = meta.abstract or ref.abstract
        if crossref_item is not None and not ref.doi.startswith("10.48550/"):
            retracted, notice = crossref_retraction(crossref_item)
            ref.retracted = bool(ref.retracted) or retracted
            ref.retraction_notice = notice
            ref.updates = crossref_updates(crossref_item)
            ref.abstract = ref.abstract or _strip_jats(crossref_item.get("abstract"))
        if score is not None:
            ref.match_score = round(score, 3)
        elif cand is not None or meta is not None:
            ref.match_score = 1.0  # matched by identifier

        identified = cand is not None or meta is not None

        # 4. Text. The arXiv version first (LaTeX source reads best), unless the citing paper cites
        # a journal version published in a later year than the preprint: its numbers may have
        # changed, so its own open text is tried first.
        fetched: Optional[FetchedText] = None
        identity = None
        published_first = bool(
            ref.arxiv_id and not entry_arxiv and ref.doi and not ref.doi.startswith("10.48550/")
            and cand is not None and cand.year and (arxiv_year(ref.arxiv_id) or cand.year) < cand.year
        )

        async def arxiv() -> None:
            nonlocal fetched, identity
            if fetched is None and ref.arxiv_id:
                fetched = await fetch_arxiv_text(ref.arxiv_id, downloads)
                identity = f"arxiv:{strip_version(ref.arxiv_id)}"

        async def published() -> None:
            nonlocal fetched, identity
            if fetched is None and (ref.doi or (cand is not None and cand.pmcid)):
                # PubMed Central's open-access subset: clean XML, and reachable where publisher sites
                # refuse automated downloads (most biomedical open-access journals).
                pmcid = (cand.pmcid if cand is not None else None) or await europepmc_pmcid(ref.doi)
                if pmcid:
                    fetched = await fetch_europepmc_text(pmcid, downloads)
                    identity = f"pmc:{pmcid}"
            if fetched is None and cand is not None and cand.pdf_urls:
                fetched = await fetch_oa_pdf_text(cand.pdf_urls, downloads)
                identity = f"pdf:{ref.doi or cand.pdf_urls[0]}"

        for attempt in ((published, arxiv) if published_first else (arxiv, published)):
            await attempt()
        if fetched is None and not ref.arxiv_id and ref.title:
            # Published versions often have an arXiv preprint that OpenAlex lists as a separate work.
            twin = await find_arxiv_twin(ref.title, ref.year)
            if twin:
                log.info("reference %s: found arXiv preprint %s", ref.key, twin)
                ref.arxiv_id = twin
                fetched = await fetch_arxiv_text(twin, downloads)
                identity = f"arxiv:{twin}"
            identified = identified or twin is not None

        if fetched is not None:
            ref.text_path = await asyncio.to_thread(_store_text, identity, fetched)
            ref.fulltext_status = "full"
            ref.fulltext_source = fetched.source
            ref.status_reason = None
        elif ref.abstract:
            ref.fulltext_status = "abstract_only"
            ref.status_reason = REASON_ABSTRACT_ONLY
        else:
            ref.fulltext_status = "unavailable"
            ref.status_reason = REASON_NOT_OPEN if identified else REASON_NOT_FOUND
        return ref, not downloads.transient_failure


async def _empty() -> dict:
    return {}


async def _call(callback: Optional[OnDone], ref: Reference) -> None:
    if callback is None:
        return
    result = callback(ref)
    if inspect.isawaitable(result):
        await result


async def resolve_references(refs: list[Reference], on_done: Optional[OnDone] = None) -> list[Reference]:
    return await ReferenceResolver().resolve_all(refs, on_done)

