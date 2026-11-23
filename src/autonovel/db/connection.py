"""SQLite WAL connection factory and context managers that always close handles on Windows."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def open_sqlite_connection(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection configured with WAL mode and Row factory."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


@contextmanager
def read_connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection, commit any pending changes, and close on exit."""
    conn = open_sqlite_connection(db_path)
    try:
        yield conn
        if conn.in_transaction:
            conn.commit()
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def atomic_transaction(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Execute a block of SQL operations inside a single atomic transaction and close connection."""
    conn = open_sqlite_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
