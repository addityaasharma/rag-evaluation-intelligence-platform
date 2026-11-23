import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ragkit.config import Settings
from ragkit.evaluation.runner import EvalReport

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    label TEXT NOT NULL,
    config TEXT NOT NULL,
    summary TEXT NOT NULL,
    report TEXT NOT NULL
)
"""


class RunRecord(BaseModel):
    id: int
    created_at: str
    label: str
    config: dict[str, Any]
    summary: dict[str, float]


def config_snapshot(settings: Settings, dataset: str) -> dict[str, Any]:
    """The pipeline settings that determine evaluation results."""
    return {
        "dataset": dataset,
        "llm_model": settings.llm_model,
        "judge_model": settings.judge_model,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "retrieval_mode": settings.retrieval_mode,
    }


class HistoryStore:
    """SQLite log of evaluation runs, so pipeline versions can be compared over time."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(_SCHEMA)

    def record(self, label: str, config: dict[str, Any], report: EvalReport) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO runs (created_at, label, config, summary, report) VALUES (?,?,?,?,?)",
                (
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    label,
                    json.dumps(config),
                    json.dumps(report.summary),
                    report.model_dump_json(),
                ),
            )
            return int(cursor.lastrowid or 0)

    def runs(self, limit: int | None = None) -> list[RunRecord]:
        """Most recent first."""
        query = "SELECT id, created_at, label, config, summary FROM runs ORDER BY id DESC"
        params: tuple[int, ...] = ()
        if limit is not None:
            query, params = f"{query} LIMIT ?", (limit,)
        with self._connect() as db:
            rows = db.execute(query, params).fetchall()
        return [
            RunRecord(
                id=id_,
                created_at=created,
                label=label,
                config=json.loads(config),
                summary=json.loads(summary),
            )
            for id_, created, label, config, summary in rows
        ]

    def report(self, run_id: int) -> EvalReport:
        with self._connect() as db:
            row = db.execute("SELECT report FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return EvalReport.model_validate_json(row[0])

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)
