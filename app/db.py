"""SQLite storage: jobs and their intermediate results, per-call usage records, and the LLM response cache.

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
        self.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        # Intermediate results per job: ingest, reference, claims, evidence, judged, verdict.
        self.execute(
            """CREATE TABLE IF NOT EXISTS artifacts (
                job_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                item_id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY (job_id, kind, item_id)
            )"""
        )

    # -- jobs -------------------------------------------------------------------

    def save_job(self, job_id: str, data: str, status: str, created_at: float) -> None:
        self.execute(
            "INSERT INTO jobs (id, data, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET data = excluded.data, status = excluded.status,"
            " updated_at = excluded.updated_at",
            (job_id, data, status, created_at, time.time()),
        )

    def load_job(self, job_id: str) -> Optional[str]:
        rows = self.execute("SELECT data FROM jobs WHERE id = ?", (job_id,))
        return rows[0]["data"] if rows else None

    def jobs_with_status(self, statuses: tuple[str, ...]) -> list[str]:
        placeholders = ",".join("?" * len(statuses))
        return [r["data"] for r in self.execute(f"SELECT data FROM jobs WHERE status IN ({placeholders})", statuses)]

    # -- artifacts ----------------------------------------------------------------

    def put_artifact(self, job_id: str, kind: str, item_id: str, data: str) -> None:
        self.execute(
            "INSERT OR REPLACE INTO artifacts (job_id, kind, item_id, data) VALUES (?, ?, ?, ?)",
            (job_id, kind, item_id, data),
        )

    def get_artifact(self, job_id: str, kind: str, item_id: str) -> Optional[str]:
        rows = self.execute(
            "SELECT data FROM artifacts WHERE job_id = ? AND kind = ? AND item_id = ?", (job_id, kind, item_id)
        )
        return rows[0]["data"] if rows else None

    def artifacts(self, job_id: str, kind: str) -> dict[str, str]:
        rows = self.execute("SELECT item_id, data FROM artifacts WHERE job_id = ? AND kind = ?", (job_id, kind))
        return {r["item_id"]: r["data"] for r in rows}

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


_database: Optional[Database] = None


def get_database() -> Database:
    """The app-wide jobs/usage database."""
    global _database
    if _database is None:
        from app.config import get_settings

        _database = Database(get_settings().db_path)
    return _database


def set_database(db: Optional[Database]) -> None:
    """Override the app-wide database (tests)."""
    global _database
    _database = db
