"""Hot snapshot backup manager using SQLite Online Backup API and rolling retention."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autonovel.db.connection import open_sqlite_connection, read_connection
from autonovel.db.project_repo import ProjectRepository


class SnapshotManager:
    """Manages periodic (rolling 5), milestone (permanent), and manual snapshots."""

    def __init__(
        self,
        project_db_path: Path,
        snapshot_dir: Path,
        max_periodic_snapshots: int = 5,
    ) -> None:
        """Initialize snapshot manager with target database and backup directory."""
        self.project_db_path = project_db_path
        self.snapshot_dir = snapshot_dir
        self.max_periodic_snapshots = max_periodic_snapshots
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def create_snapshot(
        self, kind: str = "periodic", chapter_num: int = 0, label: str = ""
    ) -> dict[str, Any]:
        """Create a consistent online backup of novel_project.db and enforce retention."""
        if kind not in ("periodic", "milestone", "manual"):
            raise ValueError(f"Unsupported snapshot kind: {kind}")
        self._seq += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safe_label = "".join(c for c in label if c.isalnum() or c in ("_", "-"))[:24]
        suffix = f"_{safe_label}" if safe_label else ""
        filename = f"snap_{kind}_ch{chapter_num:04d}_{ts}_{self._seq:03d}{suffix}.db"
        dest_path = self.snapshot_dir / filename
        self._perform_online_backup(self.project_db_path, dest_path)
        info = self._inspect_snapshot_file(dest_path)
        if kind == "periodic":
            self._prune_periodic_snapshots()
        return info

    @staticmethod
    def _perform_online_backup(src_path: Path, dest_path: Path) -> None:
        """Copy a SQLite database atomically through a temp file.

        Using the Online Backup API straight onto ``dest_path`` risks a half-written
        database if the process dies, and for restores it leaves pre-restore ``-wal``/
        ``-shm`` files behind that SQLite would replay into the restored file.
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_name(dest_path.name + ".tmp")
        for suffix in ("", "-wal", "-shm"):
            tmp_path.with_name(tmp_path.name + suffix).unlink(missing_ok=True)
        src_conn = open_sqlite_connection(src_path)
        dest_conn = sqlite3.connect(str(tmp_path))
        try:
            src_conn.backup(dest_conn)
            dest_conn.execute("PRAGMA journal_mode=DELETE;")
            dest_conn.commit()
        finally:
            dest_conn.close()
            src_conn.close()
        # Drop stale WAL/SHM sidecars before replacing, otherwise SQLite would
        # replay the old transaction log on top of the restored database.
        for suffix in ("-wal", "-shm"):
            dest_path.with_name(dest_path.name + suffix).unlink(missing_ok=True)
        os.replace(tmp_path, dest_path)

    def _snapshot_sort_key(self, path: Path) -> tuple[float, str]:
        """Sort by real write time, falling back to the name for identical mtimes."""
        try:
            return (path.stat().st_mtime, path.name)
        except OSError:
            return (0.0, path.name)

    def _prune_periodic_snapshots(self) -> None:
        """Retain only the most recent N periodic snapshots on disk."""
        periodic_files = sorted(
            self.snapshot_dir.glob("snap_periodic_*.db"),
            key=self._snapshot_sort_key,
            reverse=True,
        )
        for old_file in periodic_files[self.max_periodic_snapshots :]:
            for suffix in ("", "-wal", "-shm"):
                candidate = old_file.with_name(old_file.name + suffix)
                candidate.unlink(missing_ok=True)

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List all available snapshots ordered from newest to oldest."""
        files = sorted(
            self.snapshot_dir.glob("snap_*.db"),
            key=self._snapshot_sort_key,
            reverse=True,
        )
        return [self._inspect_snapshot_file(f) for f in files]

    @staticmethod
    def _inspect_snapshot_file(path: Path) -> dict[str, Any]:
        """Extract metadata (kind, completed chapters, total words) from a snapshot file."""
        parts = path.stem.split("_")
        kind = parts[1] if len(parts) > 1 else "manual"
        completed_chapters = 0
        total_words = 0
        try:
            with read_connection(path) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS c, COALESCE(SUM(word_count), 0) AS w FROM chapter_metrics;"
                ).fetchone()
                if row:
                    completed_chapters = int(row["c"])
                    total_words = int(row["w"])
        except sqlite3.Error:
            completed_chapters = 0
        stat = path.stat()
        return {
            "snapshot_id": path.name,
            "kind": kind,
            "completed_chapters": completed_chapters,
            "total_words": total_words,
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }

    def restore_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Restore a snapshot over novel_project.db, keeping a pre-restore safety copy."""
        src_path = (self.snapshot_dir / snapshot_id).resolve()
        if src_path.parent != self.snapshot_dir.resolve() or not src_path.exists():
            raise FileNotFoundError(f"Snapshot '{snapshot_id}' not found.")
        # Keep the current state so picking the wrong snapshot is reversible.
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safety_name = f"snap_manual_prerestore_{ts}.db"
        self._perform_online_backup(
            self.project_db_path, self.snapshot_dir / safety_name
        )
        self._perform_online_backup(src_path, self.project_db_path)
        repo = ProjectRepository(self.project_db_path)
        rolled_back = repo.rollback_staging_chapters()
        repo.update_project_config(pipeline_state="PAUSED")
        return {
            "restored_snapshot": snapshot_id,
            "safety_snapshot": safety_name,
            "rolled_back_staging_chapters": rolled_back,
        }
