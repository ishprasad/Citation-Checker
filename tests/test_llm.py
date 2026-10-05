"""LLM wrapper tests against a fake OpenAI client (no network)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest
from pydantic import BaseModel

from app import llm
from app.config import ModelPrice, Settings
from app.db import Database, LLMCache


class Answer(BaseModel):
    label: str
    confidence: float


def _status_error(cls, status: int):
    response = httpx.Response(status, request=httpx.Request("POST", "http://test/v1/chat/completions"))
    return cls(f"HTTP {status}", response=response, body=None)


def _completion(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    message = SimpleNamespace(content=content, model_extra={})
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeClient:
    def __init__(self, replies=(), dim: int = 4):
        self.replies = list(replies)
        self.chat_calls: list[dict] = []
        self.embed_calls: list[list[str]] = []
        self.dim = dim
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.embeddings = SimpleNamespace(create=self._embed)

    async def _create(self, **kwargs):
        self.chat_calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _completion(reply)

    async def _embed(self, model, input):
        self.embed_calls.append(list(input))
        data = [
            SimpleNamespace(index=i, embedding=[float(len(t)), float(i), 1.0, 0.5][: self.dim])
            for i, t in enumerate(input)
        ]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=3 * len(input)))


def make_client(tmp_path: Path, replies=(), **overrides) -> tuple[llm.LLMClient, FakeClient]:
    settings = Settings(
        nebius_api_key="test",
        nebius_base_url="http://test/v1/",
        models={"extract": "nano", "judge": "super", "escalate": "ultra", "vision": "", "embed": "emb"},
        reasoning_control=overrides.pop("reasoning_control", "chat_template_kwargs"),
        max_concurrency=4,
        cache_dir=tmp_path,
        db_path=tmp_path / "app.sqlite",
        contact_email="",
        prices={"judge": ModelPrice(input=1.0, output=2.0)},
        max_tokens={"extract": 100, "judge": 200, "escalate": 300},
        temperature={"extract": 0.0, "judge": 0.2, "escalate": 0.2},
        reasoning={"extract": False, "judge": True, "escalate": True},
        **overrides,
    )
    fake = FakeClient(replies)
    client = llm.LLMClient(
        settings, client=fake, db=Database(settings.db_path), cache=LLMCache(tmp_path / "cache.sqlite")
    )
    return client, fake


MESSAGES = [{"role": "user", "content": "Is the claim supported?"}]


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(llm, "_retry_delay", lambda attempt, exc: 0)


def test_structured_output_and_cache(tmp_path):
    client, fake = make_client(tmp_path, ['{"label": "supported", "confidence": 0.9}'])

    async def run():
        with llm.job_context("job1"):
            first = await client.chat_json("judge", MESSAGES, Answer)
            second = await client.chat_json("judge", MESSAGES, Answer)
        return first, second

    first, second = asyncio.run(run())
    assert first == second == Answer(label="supported", confidence=0.9)
    assert len(fake.chat_calls) == 1  # second call served from cache

    call = fake.chat_calls[0]
    assert call["model"] == "super"
    assert call["response_format"]["type"] == "json_schema"
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert "JSON schema" in call["messages"][0]["content"]


def test_usage_tracking_and_cost(tmp_path):
    client, _ = make_client(tmp_path, ['{"label": "supported", "confidence": 0.9}'])

    async def run():
        with llm.job_context("job1"):
            await client.chat_json("judge", MESSAGES, Answer)
            await client.chat_json("judge", MESSAGES, Answer)

    asyncio.run(run())
    summary = client.usage_summary("job1")
    [item] = summary.per_model
    assert (item.calls, item.cached_calls) == (2, 1)
    assert (item.prompt_tokens, item.completion_tokens) == (20, 10)
    assert item.estimated_cost_usd == pytest.approx((20 * 1.0 + 10 * 2.0) / 1e6)
    assert client.usage_summary("other-job").per_model == []


def test_reasoning_off_for_extract(tmp_path):
    client, fake = make_client(tmp_path, ['{"label": "x", "confidence": 0.1}'])
    asyncio.run(client.chat_json("extract", MESSAGES, Answer))
    assert fake.chat_calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_reasoning_system_prompt_mode(tmp_path):
    client, fake = make_client(
        tmp_path, ['{"label": "x", "confidence": 0.1}'], reasoning_control="system_prompt"
    )
    asyncio.run(client.chat_json("extract", MESSAGES, Answer))
    call = fake.chat_calls[0]
    assert "extra_body" not in call
    assert call["messages"][0]["content"].startswith("/no_think\n")


def test_think_tags_and_fences_are_stripped(tmp_path):
    reply = '<think>The passage says 31%...</think>\n```json\n{"label": "overstated", "confidence": 0.7}\n```'
    client, _ = make_client(tmp_path, [reply])
    result = asyncio.run(client.chat_json("judge", MESSAGES, Answer))
    assert result.label == "overstated"


def test_validation_retry_includes_error(tmp_path):
    client, fake = make_client(
        tmp_path, ['{"label": "supported"}', '{"label": "supported", "confidence": 0.8}']
    )
    result = asyncio.run(client.chat_json("judge", MESSAGES, Answer))
    assert result.confidence == 0.8
    assert len(fake.chat_calls) == 2
    correction = fake.chat_calls[1]["messages"][-1]["content"]
    assert "confidence" in correction and "Field required" in correction


def test_validation_fails_after_one_retry(tmp_path):
    client, fake = make_client(tmp_path, ["not json", "still not json"])
    with pytest.raises(llm.StructuredOutputError):
        asyncio.run(client.chat_json("judge", MESSAGES, Answer))
    assert len(fake.chat_calls) == 2


def test_backoff_on_429_and_5xx(tmp_path):
    client, fake = make_client(
        tmp_path,
        [
            _status_error(openai.RateLimitError, 429),
            _status_error(openai.InternalServerError, 503),
            '{"label": "supported", "confidence": 0.9}',
        ],
    )
    result = asyncio.run(client.chat_json("judge", MESSAGES, Answer))
    assert result.label == "supported"
    assert len(fake.chat_calls) == 3


def test_no_retry_on_auth_error(tmp_path):
    client, fake = make_client(tmp_path, [_status_error(openai.AuthenticationError, 401)])
    with pytest.raises(openai.AuthenticationError):
        asyncio.run(client.chat_json("judge", MESSAGES, Answer))
    assert len(fake.chat_calls) == 1


def test_response_format_fallback(tmp_path):
    client, fake = make_client(
        tmp_path,
        [
            _status_error(openai.BadRequestError, 400),
            '{"label": "supported", "confidence": 0.9}',
            '{"label": "supported", "confidence": 0.5}',
        ],
    )

    async def run():
        await client.chat_json("judge", MESSAGES, Answer)
        # The fallback is remembered per model.
        await client.chat_json("judge", [{"role": "user", "content": "another"}], Answer)

    asyncio.run(run())
    assert fake.chat_calls[0]["response_format"]["type"] == "json_schema"
    assert fake.chat_calls[1]["response_format"] == {"type": "json_object"}
    assert fake.chat_calls[2]["response_format"] == {"type": "json_object"}


def test_token_budget(tmp_path):
    client, fake = make_client(
        tmp_path,
        ['{"label": "a", "confidence": 0.1}', '{"label": "b", "confidence": 0.1}'],
        max_tokens_per_job=15,
    )

    async def run():
        with llm.job_context("job1"):
            await client.chat_json("judge", MESSAGES, Answer)  # spends 15 tokens
            await client.chat_json("judge", [{"role": "user", "content": "more"}], Answer)

    with pytest.raises(llm.TokenBudgetExceeded):
        asyncio.run(run())
    assert len(fake.chat_calls) == 1


def test_embed_caches_per_text(tmp_path):
    client, fake = make_client(tmp_path)

    async def run():
        with llm.job_context("job1"):
            a = await client.embed(["alpha", "beta"])
            b = await client.embed(["beta", "gamma", "alpha"])
        return a, b

    a, b = asyncio.run(run())
    assert a.shape == (2, 4) and b.shape == (3, 4)
    assert np.array_equal(a[1], b[0]) and np.array_equal(a[0], b[2])
    assert fake.embed_calls == [["alpha", "beta"], ["gamma"]]
    [item] = client.usage_summary("job1").per_model
    assert item.role == "embed" and item.prompt_tokens == 9


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"a": 1}', '{"a": 1}'),
        ('<think>hmm</think>{"a": 1}', '{"a": 1}'),
        ('reasoning without opening tag</think>\n{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('Here you go: {"a": {"b": 2}} done', '{"a": {"b": 2}}'),
    ],
)
def test_extract_json_text(raw, expected):
    assert llm.extract_json_text(raw) == expected
