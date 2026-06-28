"""SQLite-backed query and eval logging."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import LOGS_DIR, ensure_data_dirs
from app.models import QueryResult


class QueryLogger:
    def __init__(self, db_path: Path | None = None) -> None:
        ensure_data_dirs()
        self.db_path = db_path or (LOGS_DIR / "queries.db")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT,
                    refused INTEGER,
                    confidence REAL,
                    faithfulness REAL,
                    chunk_recall REAL,
                    response_relevance REAL,
                    hallucination INTEGER,
                    retrieval_ms REAL,
                    total_ms REAL,
                    cost_usd REAL,
                    alerts TEXT,
                    payload TEXT
                );
                CREATE TABLE IF NOT EXISTS eval_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    summary TEXT NOT NULL
                );
            """)

    def log_query(self, result: QueryResult) -> None:
        scores = result.scores
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO query_logs
                   (timestamp, query, answer, refused, confidence,
                    faithfulness, chunk_recall, response_relevance,
                    hallucination, retrieval_ms, total_ms, cost_usd, alerts, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.timestamp.isoformat(),
                    result.query,
                    result.answer,
                    int(result.refused),
                    result.confidence,
                    scores.faithfulness if scores else None,
                    scores.chunk_recall if scores else None,
                    scores.response_relevance if scores else None,
                    int(scores.hallucination_detected) if scores else 0,
                    result.latency.retrieval_ms + result.latency.rerank_ms,
                    result.latency.total_ms,
                    result.cost.total_cost_usd,
                    json.dumps(scores.alerts if scores else []),
                    json.dumps(result.to_dict()),
                ),
            )

    def log_eval_summary(self, summary: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO eval_summaries (timestamp, summary) VALUES (?, ?)",
                (datetime.now(timezone.utc).isoformat(), json.dumps(summary)),
            )

    def recent_queries(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM query_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_eval_summary(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM eval_summaries ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return json.loads(row["summary"]) if row else None