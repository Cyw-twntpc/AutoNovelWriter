"""Periodic deep maintenance executed sequentially every 10 chapters."""

from __future__ import annotations

import json
from typing import Any

from autonovel.db.connection import atomic_transaction
from autonovel.db.project_repo import ProjectRepository


class DeepOrganizer:
    """Performs alias deduplication, inactive minor NPC archiving, and timeline sync."""

    def __init__(
        self, repo: ProjectRepository, inactive_chapter_threshold: int = 30
    ) -> None:
        """Initialize organizer with project repository and archive threshold."""
        self.repo = repo
        self.inactive_threshold = inactive_chapter_threshold

    def run_maintenance(self, current_chapter_num: int) -> dict[str, Any]:
        """Execute all 3 maintenance tasks inside an atomic transaction."""
        with atomic_transaction(self.repo.db_path) as conn:
            merged_aliases = self._deduplicate_character_aliases(conn)
            archived_npcs = self._archive_inactive_minor_characters(
                conn, current_chapter_num
            )
            synced_tracks = self._align_timeline_sync_points(conn, current_chapter_num)
        return {
            "chapter_num": current_chapter_num,
            "merged_alias_records": merged_aliases,
            "archived_characters": archived_npcs,
            "synced_tracks": synced_tracks,
        }

    @staticmethod
    def _deduplicate_character_aliases(conn: Any) -> int:
        """Clean and deduplicate JSON alias arrays across all characters."""
        rows = conn.execute("SELECT id, name, aliases FROM characters;").fetchall()
        updated = 0
        for row in rows:
            raw_list = json.loads(row["aliases"] or "[]")
            cleaned = sorted(
                {str(a).strip() for a in raw_list if str(a).strip() and str(a).strip() != row["name"]}
            )
            if cleaned != raw_list:
                conn.execute(
                    "UPDATE characters SET aliases = ? WHERE id = ?;",
                    (json.dumps(cleaned, ensure_ascii=False), int(row["id"])),
                )
                updated += 1
        return updated

    def _archive_inactive_minor_characters(
        self, conn: Any, current_chapter_num: int
    ) -> list[str]:
        """Archive ALIVE supporting characters unseen for >=30 chapters without pending foreshadowing."""
        pending_rows = conn.execute(
            "SELECT related_character_ids FROM foreshadowing WHERE status = 'PENDING';"
        ).fetchall()
        protected_names: set[str] = set()
        for pr in pending_rows:
            for item in json.loads(pr["related_character_ids"] or "[]"):
                protected_names.add(str(item).strip())
        candidates = conn.execute(
            """
            SELECT id, name, role_type, last_appeared_chapter
            FROM characters
            WHERE status = 'ALIVE' AND role_type NOT IN ('PROTAGONIST', 'MAJOR', 'ANTAGONIST');
            """
        ).fetchall()
        archived: list[str] = []
        for ch in candidates:
            name = str(ch["name"])
            cid_str = str(ch["id"])
            last_seen = int(ch["last_appeared_chapter"] or 0)
            if (
                (current_chapter_num - last_seen) >= self.inactive_threshold
                and name not in protected_names
                and cid_str not in protected_names
            ):
                conn.execute(
                    "UPDATE characters SET status = 'ARCHIVED' WHERE id = ?;",
                    (int(ch["id"]),),
                )
                archived.append(name)
        return archived

    @staticmethod
    def _align_timeline_sync_points(conn: Any, current_chapter_num: int) -> int:
        """Align elapsed_days across parallel tracks at periodic synchronization checkpoints."""
        rows = conn.execute(
            "SELECT track_name, MAX(elapsed_days) AS max_d FROM timeline GROUP BY track_name;"
        ).fetchall()
        if len(rows) <= 1:
            return 0
        global_max = max(float(r["max_d"] or 0.0) for r in rows)
        sync_label = f"SYNC_CH_{current_chapter_num}"
        aligned = 0
        for r in rows:
            track = str(r["track_name"])
            track_max = float(r["max_d"] or 0.0)
            if track_max < global_max:
                conn.execute(
                    """
                    INSERT INTO timeline (
                        chapter_num, track_name, elapsed_days, chapter_duration_days,
                        event_summary, sync_point_id
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        current_chapter_num,
                        track,
                        global_max,
                        round(global_max - track_max, 2),
                        f"多線時間軸於第 {current_chapter_num} 章同步對齊",
                        sync_label,
                    ),
                )
                aligned += 1
        return aligned
