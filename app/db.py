"""SQLite storage: per-call usage records and the LLM response cache.

Two separate files are used on purpose: the jobs/usage database is per-deployment
state, while the LLM cache is a reusable artifact (e.g. shipped with the demo paper
so the demo runs offline).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


class _SQLite:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row

    def execute(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            self._conn.commit()
            return rows

    def executemany(self, sql: str, seq: list[tuple]) -> None:
        with self._lock:
            self._conn.executemany(sql, seq)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class Database(_SQLite):
    def __init__(self, path: Path):
        super().__init__(path)
        self.execute(
            """CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                role TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                cached INTEGER NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        self.execute("CREATE INDEX IF NOT EXISTS usage_job ON usage(job_id)")

    def record_usage(
        self,
        job_id: Optional[str],
        role: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached: bool,
    ) -> None:
        self.execute(
            "INSERT INTO usage (job_id, role, model, prompt_tokens, completion_tokens, cached, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, role, model, prompt_tokens, completion_tokens, int(cached), time.time()),
        )

    def usage_rows(self, job_id: str) -> list[sqlite3.Row]:
        return self.execute(
            "SELECT role, model, cached, COUNT(*) AS calls,"
            " SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens"
            " FROM usage WHERE job_id = ? GROUP BY role, model, cached",
            (job_id,),
        )

    def job_token_total(self, job_id: str) -> int:
        """Tokens actually spent (cache hits excluded) — used for the per-job budget."""
        rows = self.execute(
            "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS t"
            " FROM usage WHERE job_id = ? AND cached = 0",
            (job_id,),
        )
        return int(rows[0]["t"])


class LLMCache(_SQLite):
    """Content-addressed cache for chat completions and embeddings."""

    def __init__(self, path: Path):
        super().__init__(path)
        self.execute(
            """CREATE TABLE IF NOT EXISTS chat (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        self.execute(
            """CREATE TABLE IF NOT EXISTS embedding (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL
            )"""
        )

    def get_chat(self, key: str) -> Optional[dict[str, Any]]:
        rows = self.execute("SELECT response FROM chat WHERE key = ?", (key,))
        return json.loads(rows[0]["response"]) if rows else None

    def put_chat(self, key: str, model: str, response: dict[str, Any]) -> None:
        self.execute(
            "INSERT OR REPLACE INTO chat (key, model, response, created_at) VALUES (?, ?, ?, ?)",
            (key, model, json.dumps(response), time.time()),
        )

    def get_embeddings(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        # SQLite caps bound parameters; query in slices.
        for i in range(0, len(keys), 500):
            batch = keys[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            for row in self.execute(
                f"SELECT key, vector FROM embedding WHERE key IN ({placeholders})", tuple(batch)
            ):
                found[row["key"]] = np.frombuffer(row["vector"], dtype=np.float32)
        return found

    def put_embeddings(self, model: str, items: dict[str, np.ndarray]) -> None:
        self.executemany(
            "INSERT OR REPLACE INTO embedding (key, model, dim, vector) VALUES (?, ?, ?, ?)",
            [
                (k, model, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes())
                for k, v in items.items()
            ],
        )
