"""List every model available on Nebius Token Factory.

Highlights Nemotron models and likely embedding models so you can fill in .env.

    python scripts/list_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from app.config import get_settings  # noqa: E402

EMBED_HINTS = ("embed", "bge", "e5-", "gte", "nv-embed", "retriever")


def main() -> int:
    settings = get_settings()
    if not settings.nebius_api_key:
        print("NEBIUS_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    client = OpenAI(base_url=settings.nebius_base_url, api_key=settings.nebius_api_key)
    ids = sorted(m.id for m in client.models.list())

    nemotron = [i for i in ids if "nemotron" in i.lower()]
    embedding = [i for i in ids if any(h in i.lower() for h in EMBED_HINTS)]

    print(f"{len(ids)} models available:\n")
    for model_id in ids:
        tags = []
        if model_id in nemotron:
            tags.append("NEMOTRON")
        if model_id in embedding:
            tags.append("EMBEDDING")
        marker = f"  <-- {', '.join(tags)}" if tags else ""
        print(f"  {model_id}{marker}")

    print("\nNemotron models:")
    for model_id in nemotron or ["(none found)"]:
        print(f"  {model_id}")
    print("\nLikely embedding models:")
    for model_id in embedding or ["(none found)"]:
        print(f"  {model_id}")

    print("\nCurrently configured:")
    for role, model_id in settings.models.items():
        status = "ok" if model_id in ids else ("NOT SET" if not model_id else "NOT IN LISTING")
        print(f"  MODEL_{role.upper():<9} = {model_id or '-':<55} [{status}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
