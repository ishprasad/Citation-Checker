"""Make one call to each configured model through app.llm and print token usage.

    python scripts/smoke_test.py

Calls are cached, so rerunning is free. Delete .cache/llm_cache.sqlite to force
fresh responses.
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app import llm  # noqa: E402
from app.config import get_settings  # noqa: E402


class Probe(BaseModel):
    answer: str
    confidence: float


PROBE_MESSAGES = [
    {"role": "system", "content": "You are a careful scientific assistant."},
    {
        "role": "user",
        "content": (
            'Source passage: "In a pilot study of 24 mice, the compound reduced tumor volume by 31%." '
            'Claim: "The compound reduces tumor volume in humans." '
            "Is the claim supported by the passage? Answer in a few words."
        ),
    },
]


async def probe_chat(role: str) -> None:
    fn = {"extract": llm.extract, "judge": llm.judge, "escalate": llm.escalate}[role]
    t0 = time.perf_counter()
    result = await fn(PROBE_MESSAGES, Probe)
    print(f"  [ok] {role:<9} {time.perf_counter() - t0:5.1f}s  {result.model_dump()}")


async def probe_vision() -> None:
    t0 = time.perf_counter()
    text = await llm.get_llm().complete_text(
        "vision", [{"role": "user", "content": "Reply with the single word: ready"}]
    )
    print(f"  [ok] vision    {time.perf_counter() - t0:5.1f}s  {text[:80]!r}")


async def probe_embed() -> None:
    t0 = time.perf_counter()
    vecs = await llm.embed(["tumor volume reduction in mice", "the compound shrank tumors in a mouse study"])
    a, b = vecs
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"  [ok] embed     {time.perf_counter() - t0:5.1f}s  dim={vecs.shape[1]} cosine={cos:.3f}")


async def main() -> int:
    settings = get_settings()
    if not settings.nebius_api_key:
        print("NEBIUS_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    job_id = f"smoke-{uuid.uuid4().hex[:8]}"
    probes = {
        "extract": lambda: probe_chat("extract"),
        "judge": lambda: probe_chat("judge"),
        "escalate": lambda: probe_chat("escalate"),
        "vision": probe_vision,
        "embed": probe_embed,
    }
    failures = 0
    print(f"Reasoning control: {settings.reasoning_control}\n")
    with llm.job_context(job_id):
        for role, run in probes.items():
            if not settings.models.get(role):
                print(f"  [--] {role:<9} not configured (MODEL_{role.upper()})")
                if role != "vision":
                    failures += 1
                continue
            try:
                await run()
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures += 1
                print(f"  [!!] {role:<9} {type(exc).__name__}: {exc}")

    summary = llm.get_llm().usage_summary(job_id)
    print("\nUsage for this run:")
    for item in summary.per_model:
        print(
            f"  {item.role:<9} {item.model:<50} calls={item.calls} (cached {item.cached_calls})"
            f" in={item.prompt_tokens} out={item.completion_tokens} ${item.estimated_cost_usd:.5f}"
        )
    print(f"  total estimated cost: ${summary.total_estimated_cost_usd:.5f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
