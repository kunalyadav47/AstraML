from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from backend.app.config import logger, settings

CREATE_EXPERIMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    dataset_filename TEXT NOT NULL,
    target_column TEXT NOT NULL,
    problem_type TEXT NOT NULL,
    best_algorithm TEXT NOT NULL,
    best_metric REAL NOT NULL,
    metric_name TEXT NOT NULL,
    session_blob TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments (created_at DESC);
"""

CREATE_CHAT_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    messages_blob TEXT NOT NULL,
    context_blob TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_experiment ON chat_sessions (experiment_id);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    target_path = db_path or settings.sqlite_db_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(target_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Performance and concurrency pragmas
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def init_db(db_path: Path | None = None) -> None:
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executescript(CREATE_EXPERIMENTS_TABLE)
            conn.executescript(CREATE_CHAT_SESSIONS_TABLE)
        logger.info("Initialized AstraML database tables successfully.")
    finally:
        conn.close()


@contextmanager
def get_db(db_path: Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
