"""Settings loaded from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

ROLES = ("extract", "judge", "escalate", "vision", "embed")

DEFAULT_EMBED_QUERY_INSTRUCTION = (
    "Given a claim that a paper attributes to a cited work, retrieve passages from the cited work "
    "that support or contradict the claim"
)


def _env(name: str, default: str = "") -> str:
    value = (os.getenv(name, default) or default).strip()
    # python-dotenv keeps "# comment" as the value of an empty `KEY=   # comment` line.
    if value.startswith("#"):
        return default
    if " #" in value:
        value = value.split(" #", 1)[0].strip()
    return value


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class ModelPrice:
    """Estimated USD per 1M tokens."""

    input: float = 0.0
    output: float = 0.0


@dataclass(frozen=True)
class Settings:
    nebius_api_key: str
    nebius_base_url: str
    models: dict[str, str]
    reasoning_control: str
    max_concurrency: int
    cache_dir: Path
    db_path: Path
    contact_email: str
    prices: dict[str, ModelPrice]
    arxiv_min_interval: float = 3.0  # seconds between requests to arxiv.org (arXiv's policy)
    escalate_max_chars: int = 300_000  # cited-paper text sent to the long-context model (~75k tokens)
    # Instruction prefix for retrieval queries (instruction-aware embedders such as Qwen3-Embedding).
    # Empty string = embed the raw claim.
    embed_query_instruction: str = ""

    # Per-role generation parameters.
    max_tokens: dict[str, int] = field(default_factory=dict)
    temperature: dict[str, float] = field(default_factory=dict)
    reasoning: dict[str, bool] = field(default_factory=dict)

    # Limits (section 10).
    quick_check_max_citations: int = 20
    max_references: int = 150
    max_pdf_mb: int = 25
    max_jobs_per_ip_per_hour: int = 5
    max_previews_per_ip_per_hour: int = 30  # paper look-ups that download from arXiv (cache hits are free)
    max_concurrent_jobs: int = 3
    max_tokens_per_job: int = 0
    judge_mode_password: str = ""
    grobid_url: str = ""
    trust_proxy_headers: bool = False  # use X-Forwarded-For for rate limiting (only behind a trusted proxy)

    def model_for(self, role: str) -> str:
        model = self.models.get(role, "")
        if not model:
            raise RuntimeError(
                f"No model configured for role '{role}'. Set MODEL_{role.upper()} in .env "
                "(run scripts/list_models.py to see available IDs)."
            )
        return model


@lru_cache
def get_settings() -> Settings:
    load_dotenv(ROOT_DIR / ".env")
    cache_dir = Path(_env("CACHE_DIR", ".cache"))
    if not cache_dir.is_absolute():
        cache_dir = ROOT_DIR / cache_dir
    db_path = Path(_env("DB_PATH", str(cache_dir / "app.sqlite")))
    if not db_path.is_absolute():
        db_path = ROOT_DIR / db_path

    return Settings(
        nebius_api_key=_env("NEBIUS_API_KEY"),
        nebius_base_url=_env("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
        models={role: _env(f"MODEL_{role.upper()}") for role in ROLES},
        reasoning_control=_env("REASONING_CONTROL", "chat_template_kwargs"),
        max_concurrency=_env_int("MAX_CONCURRENCY", 8),
        cache_dir=cache_dir,
        db_path=db_path,
        contact_email=_env("CONTACT_EMAIL"),
        arxiv_min_interval=_env_float("ARXIV_MIN_INTERVAL", 3.0),
        escalate_max_chars=_env_int("ESCALATE_MAX_CHARS", 300_000),
        embed_query_instruction=(
            "" if _env("EMBED_QUERY_INSTRUCTION").lower() == "none"
            else _env("EMBED_QUERY_INSTRUCTION", DEFAULT_EMBED_QUERY_INSTRUCTION)
        ),
        prices={
            role: ModelPrice(
                input=_env_float(f"PRICE_{role.upper()}_IN", 0.0),
                output=_env_float(f"PRICE_{role.upper()}_OUT", 0.0),
            )
            for role in ROLES
        },
        max_tokens={
            "extract": _env_int("MAX_TOKENS_EXTRACT", 2048),
            "judge": _env_int("MAX_TOKENS_JUDGE", 8192),
            "escalate": _env_int("MAX_TOKENS_ESCALATE", 16384),
            "vision": _env_int("MAX_TOKENS_VISION", 4096),
        },
        temperature={
            "extract": _env_float("TEMPERATURE_EXTRACT", 0.0),
            "judge": _env_float("TEMPERATURE_JUDGE", 0.2),
            "escalate": _env_float("TEMPERATURE_ESCALATE", 0.2),
            "vision": _env_float("TEMPERATURE_VISION", 0.0),
        },
        # Reasoning on for judging and escalation, off for high-volume extraction.
        reasoning={
            "extract": _env("REASONING_EXTRACT", "false").lower() in ("1", "true", "yes", "on"),
            "judge": True,
            "escalate": True,
            "vision": False,
        },
        quick_check_max_citations=_env_int("QUICK_CHECK_MAX_CITATIONS", 20),
        max_references=_env_int("MAX_REFERENCES", 150),
        max_pdf_mb=_env_int("MAX_PDF_MB", 25),
        max_jobs_per_ip_per_hour=_env_int("MAX_JOBS_PER_IP_PER_HOUR", 5),
        max_previews_per_ip_per_hour=_env_int("MAX_PREVIEWS_PER_IP_PER_HOUR", 30),
        max_concurrent_jobs=_env_int("MAX_CONCURRENT_JOBS", 3),
        max_tokens_per_job=_env_int("MAX_TOKENS_PER_JOB", 0),
        judge_mode_password=_env("JUDGE_MODE_PASSWORD"),
        grobid_url=_env("GROBID_URL"),
        trust_proxy_headers=_env("TRUST_PROXY_HEADERS", "false").lower() in ("1", "true", "yes", "on"),
    )
