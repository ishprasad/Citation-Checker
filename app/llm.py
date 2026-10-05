"""Token Factory client wrapper.

All model access goes through here so that caching, retries, concurrency limits,
reasoning toggles, structured output and usage tracking live in one place.

Role functions:
    extract()   Nemotron Nano, reasoning off  -> pydantic model
    judge()     Nemotron Super, reasoning on  -> pydantic model
    escalate()  Nemotron Ultra, reasoning on  -> pydantic model
    embed()     embedding model               -> np.ndarray (n, dim)

Structured-output failures raise `StructuredOutputError` after one corrective retry;
pipeline stages catch it and mark the item as an error instead of failing the job.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import random
import re
from contextvars import ContextVar
from typing import Any, Iterator, Optional, TypeVar

import numpy as np
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings
from app.db import Database, LLMCache
from app.models import ModelUsage, UsageSummary

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# The job a call is billed to. Set once per job; asyncio tasks inherit it.
current_job_id: ContextVar[Optional[str]] = ContextVar("current_job_id", default=None)

MAX_ATTEMPTS = 6
EMBED_BATCH = 64
# Structured-output strategies, strongest first. We fall back per model if the
# endpoint rejects one.
FORMAT_MODES = ("json_schema", "json_object", "prompt")


class LLMError(Exception):
    pass


class StructuredOutputError(LLMError):
    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


class TokenBudgetExceeded(LLMError):
    pass


@contextlib.contextmanager
def job_context(job_id: Optional[str]) -> Iterator[None]:
    token = current_job_id.set(job_id)
    try:
        yield
    finally:
        current_job_id.reset(token)


def cache_key(model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "params": params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def extract_json_text(content: str) -> str:
    """Strip reasoning traces and markdown fences, returning the JSON payload."""
    text = _THINK_RE.sub("", content or "")
    if "</think>" in text:  # opening tag was in the prompt template, not the output
        text = text.split("</think>", 1)[1]
    text = text.strip()
    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    if text and text[0] not in "{[":
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


def _schema_instruction(schema: type[BaseModel]) -> str:
    return (
        "Respond with a single JSON object and nothing else (no prose, no markdown). "
        "It must validate against this JSON schema:\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )


def _retry_delay(attempt: int, exc: Exception) -> float:
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(60.0, float(retry_after))
            except ValueError:
                pass
    return min(60.0, 2.0**attempt) + random.uniform(0, 1)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        client: Any = None,
        db: Optional[Database] = None,
        cache: Optional[LLMCache] = None,
    ):
        self.settings = settings
        self.client = client or AsyncOpenAI(
            base_url=settings.nebius_base_url,
            api_key=settings.nebius_api_key or "missing",
            max_retries=0,  # we do our own backoff
            timeout=600,
        )
        self.db = db or Database(settings.db_path)
        self.cache = cache or LLMCache(settings.cache_dir / "llm_cache.sqlite")
        self.semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._format_mode: dict[str, str] = {}

    # ------------------------------------------------------------------ helpers

    async def _with_retries(self, fn, what: str):
        for attempt in range(MAX_ATTEMPTS):
            try:
                async with self.semaphore:
                    return await fn()
            except Exception as exc:  # noqa: BLE001 - classified below
                if not _is_retryable(exc) or attempt == MAX_ATTEMPTS - 1:
                    raise
                delay = _retry_delay(attempt, exc)
                log.warning("%s failed (%s); retry %d in %.1fs", what, type(exc).__name__, attempt + 1, delay)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    def _check_budget(self, job_id: Optional[str]) -> None:
        budget = self.settings.max_tokens_per_job
        if budget and job_id and self.db.job_token_total(job_id) >= budget:
            raise TokenBudgetExceeded(f"Job {job_id} exceeded its token budget of {budget}")

    def _record(self, role: str, model: str, usage: dict[str, int], cached: bool) -> None:
        self.db.record_usage(
            current_job_id.get(),
            role,
            model,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            cached,
        )

    def _reasoning_messages_and_extra(
        self, role: str, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply the configured reasoning on/off mechanism for this role."""
        enabled = self.settings.reasoning.get(role, False)
        mode = self.settings.reasoning_control
        if mode == "chat_template_kwargs":
            return messages, {"chat_template_kwargs": {"enable_thinking": enabled}}
        if mode == "system_prompt":
            flag = "/think" if enabled else "/no_think"
            if messages and messages[0]["role"] == "system":
                first = {**messages[0], "content": f"{flag}\n{messages[0]['content']}"}
                return [first, *messages[1:]], {}
            return [{"role": "system", "content": flag}, *messages], {}
        return messages, {}

    # --------------------------------------------------------------- chat calls

    async def _chat(
        self,
        role: str,
        messages: list[dict[str, Any]],
        response_format: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """One chat completion, cached. Returns {content, reasoning, finish_reason, usage}."""
        model = self.settings.model_for(role)
        messages, extra_body = self._reasoning_messages_and_extra(role, messages)
        params: dict[str, Any] = {
            "temperature": self.settings.temperature.get(role, 0.0),
            "max_tokens": self.settings.max_tokens.get(role, 4096),
        }
        if response_format:
            params["response_format"] = response_format
        if extra_body:
            params["extra_body"] = extra_body

        key = cache_key(model, messages, params)
        cached = self.cache.get_chat(key)
        if cached is not None:
            self._record(role, model, cached["usage"], cached=True)
            return cached

        self._check_budget(current_job_id.get())

        async def call():
            return await self.client.chat.completions.create(model=model, messages=messages, **params)

        resp = await self._with_retries(call, f"{role} chat ({model})")
        choice = resp.choices[0]
        extra = getattr(choice.message, "model_extra", None) or {}
        result = {
            "content": choice.message.content or "",
            "reasoning": extra.get("reasoning_content") or extra.get("reasoning"),
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
            },
        }
        self.cache.put_chat(key, model, result)
        self._record(role, model, result["usage"], cached=False)
        return result

    async def complete_text(self, role: str, messages: list[dict[str, Any]]) -> str:
        """Plain-text completion (no schema). Reasoning traces are stripped."""
        result = await self._chat(role, messages)
        text = _THINK_RE.sub("", result["content"])
        return text.split("</think>", 1)[1].strip() if "</think>" in text else text.strip()

    async def chat_json(self, role: str, messages: list[dict[str, Any]], schema: type[T]) -> T:
        model = self.settings.model_for(role)
        instruction = _schema_instruction(schema)
        if messages and messages[0]["role"] == "system":
            messages = [
                {**messages[0], "content": f"{messages[0]['content']}\n\n{instruction}"},
                *messages[1:],
            ]
        else:
            messages = [{"role": "system", "content": instruction}, *messages]

        result = await self._chat_with_format_fallback(role, model, messages, schema)
        try:
            return self._parse(result, schema)
        except StructuredOutputError as first_error:
            log.info("%s: invalid structured output, retrying once: %s", role, first_error)
            retry_messages = [
                *messages,
                {"role": "assistant", "content": extract_json_text(result["content"]) or result["content"]},
                {
                    "role": "user",
                    "content": (
                        f"That reply was not valid: {first_error}\n"
                        "Reply again with only the corrected JSON object."
                    ),
                },
            ]
            result = await self._chat_with_format_fallback(role, model, retry_messages, schema)
            return self._parse(result, schema)

    async def _chat_with_format_fallback(
        self, role: str, model: str, messages: list[dict[str, Any]], schema: type[BaseModel]
    ) -> dict[str, Any]:
        start = FORMAT_MODES.index(self._format_mode.get(model, FORMAT_MODES[0]))
        for mode in FORMAT_MODES[start:]:
            if mode == "json_schema":
                response_format = {
                    "type": "json_schema",
                    "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
                }
            elif mode == "json_object":
                response_format = {"type": "json_object"}
            else:
                response_format = None
            try:
                result = await self._chat(role, messages, response_format)
                self._format_mode[model] = mode
                return result
            except openai.BadRequestError as exc:
                if mode == FORMAT_MODES[-1]:
                    raise
                log.warning("%s rejected response_format=%s (%s); falling back", model, mode, exc)
                self._format_mode[model] = FORMAT_MODES[FORMAT_MODES.index(mode) + 1]
        raise AssertionError("unreachable")

    @staticmethod
    def _parse(result: dict[str, Any], schema: type[T]) -> T:
        raw = result["content"]
        text = extract_json_text(raw)
        if not text:
            hint = " (output was truncated: max_tokens reached)" if result.get("finish_reason") == "length" else ""
            raise StructuredOutputError(f"empty response{hint}", raw)
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()[:5]
            )
            raise StructuredOutputError(errors, raw) from exc

    # ---------------------------------------------------------------- embeddings

    async def embed(self, texts: list[str]) -> np.ndarray:
        model = self.settings.model_for("embed")
        clean = [t if t.strip() else " " for t in texts]
        keys = [hashlib.sha256(f"{model}\0{t}".encode("utf-8")).hexdigest() for t in clean]
        found = self.cache.get_embeddings(list(set(keys)))
        missing = {k: t for k, t in zip(keys, clean) if k not in found}
        missing_items = list(missing.items())
        batches = [missing_items[i : i + EMBED_BATCH] for i in range(0, len(missing_items), EMBED_BATCH)]

        async def run_batch(batch: list[tuple[str, str]]) -> None:
            self._check_budget(current_job_id.get())

            async def call():
                return await self.client.embeddings.create(model=model, input=[t for _, t in batch])

            resp = await self._with_retries(call, f"embed ({model})")
            vectors = {
                k: np.asarray(item.embedding, dtype=np.float32)
                for (k, _), item in zip(batch, sorted(resp.data, key=lambda d: d.index))
            }
            self.cache.put_embeddings(model, vectors)
            found.update(vectors)
            tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
            self._record("embed", model, {"prompt_tokens": tokens}, cached=False)

        await asyncio.gather(*(run_batch(b) for b in batches))
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([found[k] for k in keys])

    # -------------------------------------------------------------------- usage

    def usage_summary(self, job_id: str) -> UsageSummary:
        per: dict[tuple[str, str], ModelUsage] = {}
        for row in self.db.usage_rows(job_id):
            item = per.setdefault(
                (row["role"], row["model"]), ModelUsage(model=row["model"], role=row["role"])
            )
            item.calls += row["calls"]
            if row["cached"]:
                item.cached_calls += row["calls"]
            item.prompt_tokens += row["prompt_tokens"] or 0
            item.completion_tokens += row["completion_tokens"] or 0

        summary = UsageSummary()
        for item in per.values():
            price = self.settings.prices.get(item.role)
            if price:
                item.estimated_cost_usd = round(
                    (item.prompt_tokens * price.input + item.completion_tokens * price.output) / 1e6, 6
                )
            summary.per_model.append(item)
            summary.total_prompt_tokens += item.prompt_tokens
            summary.total_completion_tokens += item.completion_tokens
            summary.total_estimated_cost_usd += item.estimated_cost_usd
        summary.total_estimated_cost_usd = round(summary.total_estimated_cost_usd, 6)
        return summary


_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(get_settings())
    return _client


def set_llm(client: Optional[LLMClient]) -> None:
    """Override the global client (tests)."""
    global _client
    _client = client


async def extract(messages: list[dict[str, Any]], schema: type[T]) -> T:
    return await get_llm().chat_json("extract", messages, schema)


async def judge(messages: list[dict[str, Any]], schema: type[T]) -> T:
    return await get_llm().chat_json("judge", messages, schema)


async def escalate(messages: list[dict[str, Any]], schema: type[T]) -> T:
    return await get_llm().chat_json("escalate", messages, schema)


async def embed(texts: list[str]) -> np.ndarray:
    return await get_llm().embed(texts)
