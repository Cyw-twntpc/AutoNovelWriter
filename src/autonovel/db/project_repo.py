"""Repository methods for novel_project.db including two-phase commit and hard-fact validation."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from autonovel.db.connection import atomic_transaction, read_connection
from autonovel.db.project_schema import init_project_database
from autonovel.schemas.domain import ChapterStatus, HierarchyChangeType
from autonovel.schemas.llm_contracts import (
    CharacterDeltaContract,
    ChapterExtractionContract,
)

PERMITTED_NON_MONOTONIC_TYPES = {
    HierarchyChangeType.INJURY_REGRESSION.value,
    HierarchyChangeType.SEALED.value,
    HierarchyChangeType.RE_CULTIVATION.value,
    HierarchyChangeType.ARTIFACT_DAMAGED.value,
    HierarchyChangeType.FORTUITOUS_LEAP.value,
}

logger = logging.getLogger(__name__)


class ProjectRepository:
    """Data access and transactional integrity manager for novel_project.db."""

    # Re-running the full DDL with BEGIN IMMEDIATE on every route-level repo
    # construction serialises requests for no benefit; remember which files are ready.
    _schema_ready: set[str] = set()

    def __init__(self, db_path: Path) -> None:
        """Initialize repository and ensure schema exists."""
        self.db_path = db_path
        key = str(db_path.resolve())
        if (
            key in self._schema_ready
            and db_path.exists()
            and db_path.stat().st_size > 0
        ):
            return
        init_project_database(self.db_path)
        self._schema_ready.add(key)

    def get_project_config(self) -> dict[str, Any]:
        """Return the singleton project_config record as a dictionary."""
        with read_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM project_config WHERE id = 1;").fetchone()
            return dict(row) if row else {}

    def update_project_config(self, **fields: Any) -> None:
        """Update specified columns on the singleton project_config row."""
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values())
        sql = f"UPDATE project_config SET {assignments}, updated_at = datetime('now') WHERE id = 1;"
        with atomic_transaction(self.db_path) as conn:
            conn.execute(sql, values)

    def rollback_staging_chapters(self) -> list[int]:
        """Roll back interrupted STAGING, WRITING, or REVISING chapters on startup."""
        with atomic_transaction(self.db_path) as conn:
            rows = conn.execute(
                "SELECT chapter_num, final_content FROM chapters WHERE status IN ('STAGING', 'WRITING', 'REVISING');"
            ).fetchall()
            rolled_back: list[int] = []
            for row in rows:
                ch_num = int(row["chapter_num"])
                target_status = (
                    ChapterStatus.COMPLETED.value
                    if row["final_content"]
                    else ChapterStatus.UNWRITTEN.value
                )
                conn.execute(
                    "UPDATE chapters SET staging_content = '', status = ?, updated_at = datetime('now') WHERE chapter_num = ?;",
                    (target_status, ch_num),
                )
                rolled_back.append(ch_num)
            return rolled_back

    def save_staging_chapter(
        self,
        chapter_num: int,
        title: str,
        staging_content: str,
        status: str = "STAGING",
    ) -> None:
        """Save draft content in STAGING/WRITING/REVISING status before atomic commit."""
        with atomic_transaction(self.db_path) as conn:
            conn.execute(
                """
                UPDATE chapters
                SET title = ?, staging_content = ?, status = ?, updated_at = datetime('now')
                WHERE chapter_num = ?;
                """,
                (title, staging_content, status, chapter_num),
            )

    def set_volume_status(self, volume_num: int, status: str) -> None:
        """Update the lifecycle status of a volume inside an atomic transaction."""
        with atomic_transaction(self.db_path) as conn:
            conn.execute(
                "UPDATE volumes SET status = ? WHERE volume_num = ?;",
                (status, volume_num),
            )

    @staticmethod
    def validate_hierarchy_transition(
        old_level: int,
        new_level: int,
        max_allowed_level: int,
        change_type: str,
        reason: str,
    ) -> tuple[bool, str]:
        """Validate controlled rank transition rules against regression or illegal leaps."""
        if new_level > max_allowed_level:
            return (
                False,
                f"Rank {new_level} exceeds volume stage gate ceiling ({max_allowed_level}).",
            )
        delta = new_level - old_level
        if delta == 0:
            return True, ""
        if delta == 1 and change_type == HierarchyChangeType.NORMAL_BREAKTHROUGH.value:
            return True, ""
        if change_type in PERMITTED_NON_MONOTONIC_TYPES and reason.strip():
            return True, ""
        return (
            False,
            f"Illegal rank transition {old_level}->{new_level} (type={change_type}, reason='{reason}').",
        )

    def validate_extraction_hard_facts(
        self, volume_num: int, extraction: ChapterExtractionContract
    ) -> list[str]:
        """Check extracted deltas against SQLite hard facts (dead characters, rank ceiling, timeline)."""
        errors: list[str] = []
        with read_connection(self.db_path) as conn:
            vol_row = conn.execute(
                "SELECT max_allowed_rank_level FROM volumes WHERE volume_num = ?;",
                (volume_num,),
            ).fetchone()
            max_rank = int(vol_row["max_allowed_rank_level"]) if vol_row else 99
            for delta in extraction.character_deltas:
                char_row = self._find_character_by_name_or_alias(conn, delta.character_name)
                if not char_row:
                    continue
                if char_row["status"] == "DEAD" and delta.new_status == "ALIVE":
                    errors.append(
                        f"Hard conflict: Dead character '{char_row['name']}' reappeared as ALIVE."
                    )
                if delta.new_rank_level is not None:
                    ok, msg = self.validate_hierarchy_transition(
                        int(char_row["rank_level"]),
                        delta.new_rank_level,
                        max_rank,
                        delta.change_type.value,
                        delta.change_reason,
                    )
                    if not ok:
                        errors.append(f"Hard conflict for '{char_row['name']}': {msg}")
        return errors

    @staticmethod
    def _find_character_by_name_or_alias(
        conn: sqlite3.Connection, name: str
    ) -> sqlite3.Row | None:
        """Resolve a character record by canonical name or JSON alias list."""
        row = conn.execute(
            "SELECT * FROM characters WHERE name = ?;", (name.strip(),)
        ).fetchone()
        if row:
            return row
        # Deterministic ORDER BY makes alias resolution stable when two characters
        # share an alias; stop at the first hit instead of scanning every row.
        all_rows = conn.execute(
            "SELECT * FROM characters ORDER BY id ASC;"
        ).fetchall()
        for candidate in all_rows:
            try:
                aliases = json.loads(candidate["aliases"] or "[]")
            except json.JSONDecodeError:
                continue
            if name.strip() in aliases:
                return candidate
        return None

    def commit_chapter_atomically(
        self,
        chapter_num: int,
        final_content: str,
        extraction: ChapterExtractionContract,
        literary_score: float,
        rubric_payload: dict[str, Any],
        best_paragraph_text: str,
        best_paragraph_score: float,
        revision_count: int,
        micro_rewrite_count: int,
        contradiction_retry_count: int,
        duration_sec: float,
    ) -> None:
        """Commit chapter text, metrics, dynamic anchor, and entity deltas in one transaction."""
        status = (
            ChapterStatus.REVISED.value
            if revision_count > 0
            else ChapterStatus.COMPLETED.value
        )
        with atomic_transaction(self.db_path) as conn:
            conn.execute(
                """
                UPDATE chapters
                SET title = ?, final_content = ?, staging_content = '',
                    best_paragraph_text = ?, best_paragraph_score = ?,
                    status = ?, updated_at = datetime('now')
                WHERE chapter_num = ?;
                """,
                (
                    extraction.chapter_title,
                    final_content,
                    best_paragraph_text,
                    best_paragraph_score,
                    status,
                    chapter_num,
                ),
            )
            if best_paragraph_score >= 85.0 and best_paragraph_text.strip():
                conn.execute(
                    """
                    UPDATE project_config
                    SET dynamic_anchor_text = ?, dynamic_anchor_score = ?, updated_at = datetime('now')
                    WHERE id = 1;
                    """,
                    (best_paragraph_text.strip(), best_paragraph_score),
                )
            self._clear_chapter_derived_state(conn, chapter_num)
            self._insert_chapter_metrics(
                conn,
                chapter_num,
                extraction,
                len(final_content),
                literary_score,
                rubric_payload,
                revision_count,
                micro_rewrite_count,
                contradiction_retry_count,
                duration_sec,
            )
            self._apply_entity_deltas(conn, chapter_num, extraction)

    @staticmethod
    def _clear_chapter_derived_state(conn: sqlite3.Connection, chapter_num: int) -> None:
        """Drop state produced by a previous commit of the same chapter.

        Re-writing a chapter must not append a second timeline row, duplicate
        character history, or plant the same foreshadowing twice.
        """
        conn.execute("DELETE FROM timeline WHERE chapter_num = ?;", (chapter_num,))
        conn.execute("DELETE FROM character_history WHERE chapter_num = ?;", (chapter_num,))
        conn.execute(
            "DELETE FROM foreshadowing WHERE planted_chapter = ? AND status = 'PENDING';",
            (chapter_num,),
        )
        conn.execute(
            "UPDATE foreshadowing SET status = 'PENDING', resolved_chapter = NULL, "
            "resolution_notes = '' WHERE resolved_chapter = ?;",
            (chapter_num,),
        )

    @staticmethod
    def _insert_chapter_metrics(
        conn: sqlite3.Connection,
        chapter_num: int,
        extraction: ChapterExtractionContract,
        word_count: int,
        literary_score: float,
        rubric_payload: dict[str, Any],
        revision_count: int,
        micro_rewrite_count: int,
        contradiction_retry_count: int,
        duration_sec: float,
    ) -> None:
        """Insert or replace chapter metrics record inside active transaction."""
        conn.execute(
            """
            INSERT OR REPLACE INTO chapter_metrics (
                chapter_num, title, word_count, tension_intensity, goal_progress,
                emotional_payoff, literary_score, revision_count, micro_rewrite_count,
                contradiction_retry_count, generation_duration_sec, literary_rubric_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                chapter_num,
                extraction.chapter_title,
                word_count,
                extraction.tension_intensity,
                extraction.goal_progress,
                extraction.emotional_payoff,
                literary_score,
                revision_count,
                micro_rewrite_count,
                contradiction_retry_count,
                duration_sec,
                json.dumps(rubric_payload, ensure_ascii=False),
            ),
        )

    def _apply_entity_deltas(
        self,
        conn: sqlite3.Connection,
        chapter_num: int,
        extraction: ChapterExtractionContract,
    ) -> None:
        """Apply character, foreshadowing, relation, and timeline updates in active transaction."""
        for char_delta in extraction.character_deltas:
            self._upsert_character_delta(conn, chapter_num, char_delta)
        for f_delta in extraction.foreshadow_deltas:
            self._apply_foreshadow_delta(conn, chapter_num, f_delta)
        for rel in extraction.relation_deltas:
            self._apply_relation_delta(conn, chapter_num, rel)
        for tl in extraction.timeline_deltas:
            self._apply_timeline_delta(conn, chapter_num, tl)

    def _upsert_character_delta(
        self,
        conn: sqlite3.Connection,
        chapter_num: int,
        delta: CharacterDeltaContract,
    ) -> None:
        """Insert new character or update existing character state and history log."""
        loc_id = self._ensure_named_entity(conn, "locations", delta.location_name)
        fac_id = self._ensure_named_entity(conn, "factions", delta.faction_name)
        existing = self._find_character_by_name_or_alias(conn, delta.character_name)
        if not existing:
            conn.execute(
                """
                INSERT INTO characters (
                    name, aliases, status, rank_level, faction_id, current_location_id,
                    description, last_appeared_chapter, last_updated_chapter
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    delta.character_name.strip(),
                    json.dumps(delta.aliases, ensure_ascii=False),
                    delta.new_status,
                    delta.new_rank_level or 1,
                    fac_id,
                    loc_id,
                    delta.description_update,
                    chapter_num,
                    chapter_num,
                ),
            )
            return
        self._update_existing_character(conn, existing, chapter_num, delta, fac_id, loc_id)

    @staticmethod
    def _update_existing_character(
        conn: sqlite3.Connection,
        existing: sqlite3.Row,
        chapter_num: int,
        delta: CharacterDeltaContract,
        fac_id: int | None,
        loc_id: int | None,
    ) -> None:
        """Record history changes and update existing character row."""
        char_id = int(existing["id"])
        old_rank = int(existing["rank_level"])
        new_rank = delta.new_rank_level if delta.new_rank_level is not None else old_rank
        if new_rank != old_rank:
            conn.execute(
                """
                INSERT INTO character_history (
                    character_id, chapter_num, attribute_name, old_value, new_value, change_type, reason
                ) VALUES (?, ?, 'rank_level', ?, ?, ?, ?);
                """,
                (
                    char_id,
                    chapter_num,
                    str(old_rank),
                    str(new_rank),
                    delta.change_type.value,
                    delta.change_reason or "Plot progression",
                ),
            )
        if delta.new_status != existing["status"]:
            # Keep the lifecycle enum in characters.status but preserve the model's
            # free-form 2-6 char term in the history trail for auditability.
            status_value = (
                f"{delta.new_status}（{delta.status_note}）"
                if delta.status_note.strip()
                else delta.new_status
            )
            conn.execute(
                """
                INSERT INTO character_history (
                    character_id, chapter_num, attribute_name, old_value, new_value, change_type, reason
                ) VALUES (?, ?, 'status', ?, ?, ?, ?);
                """,
                (
                    char_id,
                    chapter_num,
                    existing["status"],
                    status_value,
                    delta.change_type.value,
                    delta.change_reason or "Status change",
                ),
            )
        merged_aliases = sorted(
            set(json.loads(existing["aliases"] or "[]")) | set(delta.aliases)
        )
        desc = delta.description_update or existing["description"]
        conn.execute(
            """
            UPDATE characters
            SET aliases = ?, status = ?, rank_level = ?,
                faction_id = COALESCE(?, faction_id),
                current_location_id = COALESCE(?, current_location_id),
                description = ?, last_appeared_chapter = ?, last_updated_chapter = ?
            WHERE id = ?;
            """,
            (
                json.dumps(merged_aliases, ensure_ascii=False),
                delta.new_status,
                new_rank,
                fac_id,
                loc_id,
                desc,
                chapter_num,
                chapter_num,
                char_id,
            ),
        )

    @staticmethod
    def _ensure_named_entity(
        conn: sqlite3.Connection, table: str, name: str
    ) -> int | None:
        """Ensure a location or faction exists by name and return its primary key."""
        if table not in ("locations", "factions"):
            raise ValueError(f"Unsupported entity table: {table}")
        cleaned = name.strip()
        if not cleaned:
            return None
        conn.execute(
            f"INSERT OR IGNORE INTO {table} (name) VALUES (?);", (cleaned,)
        )
        row = conn.execute(
            f"SELECT id FROM {table} WHERE name = ?;", (cleaned,)
        ).fetchone()
        return int(row["id"]) if row else None

    @staticmethod
    def _find_pending_foreshadow_by_text(
        conn: sqlite3.Connection, description: str, chapter_num: int
    ) -> int | None:
        """Match a RESOLVE description to a pending foreshadowing row.

        The model cannot know database ids, so fall back to exact text, containment,
        then character-bigram overlap. Returns None when nothing matches convincingly.
        """
        target = description.strip()
        if not target:
            return None
        rows = conn.execute(
            "SELECT id, description FROM foreshadowing "
            "WHERE status = 'PENDING' AND planted_chapter < ? ORDER BY id ASC;",
            (chapter_num,),
        ).fetchall()
        if not rows:
            return None
        for row in rows:
            if str(row["description"]).strip() == target:
                return int(row["id"])
        for row in rows:
            stored = str(row["description"]).strip()
            if stored and (stored in target or target in stored):
                return int(row["id"])
        target_grams = ProjectRepository._char_bigrams(target)
        best_id: int | None = None
        best_score = 0.0
        for row in rows:
            stored = str(row["description"]).strip()
            score = len(target_grams & ProjectRepository._char_bigrams(stored)) / max(
                1, len(target_grams)
            )
            if score > best_score:
                best_id, best_score = int(row["id"]), score
        return best_id if best_score >= 0.5 else None

    @staticmethod
    def _char_bigrams(text: str) -> set[str]:
        """Return adjacent character pairs, whitespace removed (language-agnostic similarity)."""
        compact = "".join(ch for ch in text if not ch.isspace())
        return {compact[i : i + 2] for i in range(len(compact) - 1)}

    @staticmethod
    def _apply_foreshadow_delta(
        conn: sqlite3.Connection, chapter_num: int, f_delta: Any
    ) -> None:
        """Plant new foreshadowing or mark an existing pending item as resolved."""
        vol_row = conn.execute(
            """
            SELECT v.target_chapters FROM volumes v
            JOIN chapters c ON c.volume_num = v.volume_num
            WHERE c.chapter_num = ?;
            """,
            (chapter_num,),
        ).fetchone()
        vol_span = int(vol_row["target_chapters"]) if vol_row else 50
        span_map = {"SHORT": 15, "VOLUME": vol_span, "MAIN": 500}
        if f_delta.action == "PLANT":
            conn.execute(
                """
                INSERT INTO foreshadowing (
                    description, scope, planted_chapter, expected_span, status, related_character_ids
                ) VALUES (?, ?, ?, ?, 'PENDING', ?);
                """,
                (
                    f_delta.description,
                    f_delta.scope.value,
                    chapter_num,
                    span_map.get(f_delta.scope.value, 15),
                    json.dumps(f_delta.related_characters, ensure_ascii=False),
                ),
            )
        elif f_delta.action == "RESOLVE":
            target_id = ProjectRepository._find_pending_foreshadow_by_text(
                conn, f_delta.description, chapter_num
            )
            if target_id is None:
                logger.warning(
                    "Ignoring foreshadow RESOLVE without a matching pending item: %s",
                    f_delta.description[:60],
                )
                return
            conn.execute(
                """
                UPDATE foreshadowing
                SET status = 'RESOLVED', resolved_chapter = ?, resolution_notes = ?
                WHERE id = ? AND status = 'PENDING';
                """,
                (chapter_num, f_delta.resolution_notes, target_id),
            )

    def _apply_relation_delta(
        self, conn: sqlite3.Connection, chapter_num: int, rel: Any
    ) -> None:
        """Insert or update an interpersonal relationship edge without creating duplicates."""
        src = self._find_character_by_name_or_alias(conn, rel.source_name)
        tgt = self._find_character_by_name_or_alias(conn, rel.target_name)
        if not src or not tgt:
            return
        src_id = int(src["id"])
        tgt_id = int(tgt["id"])
        existing = conn.execute(
            """
            SELECT id FROM entity_relations
            WHERE source_id = ? AND source_type = 'CHARACTER'
              AND target_id = ? AND target_type = 'CHARACTER'
            LIMIT 1;
            """,
            (src_id, tgt_id),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE entity_relations
                SET relation_type = ?, status = ?, established_chapter = ?
                WHERE id = ?;
                """,
                (rel.relation_type, rel.status, chapter_num, int(existing["id"])),
            )
        else:
            conn.execute(
                """
                INSERT INTO entity_relations (
                    source_id, source_type, relation_type, target_id, target_type, status, established_chapter
                ) VALUES (?, 'CHARACTER', ?, ?, 'CHARACTER', ?, ?);
                """,
                (src_id, rel.relation_type, tgt_id, rel.status, chapter_num),
            )

    def _apply_timeline_delta(
        self, conn: sqlite3.Connection, chapter_num: int, tl: Any
    ) -> None:
        """Append monotonically non-decreasing timeline event for a narrative track."""
        loc_id = self._ensure_named_entity(conn, "locations", tl.location_name)
        # Base elapsed time on prior chapters only, keeping re-commits idempotent.
        last_row = conn.execute(
            "SELECT MAX(elapsed_days) AS max_days FROM timeline "
            "WHERE track_name = ? AND chapter_num < ?;",
            (tl.track_name, chapter_num),
        ).fetchone()
        prev_days = float(last_row["max_days"]) if last_row and last_row["max_days"] is not None else 0.0
        duration = max(0.0, float(tl.duration_days))
        elapsed = prev_days + duration
        sync_id = tl.sync_point_label if tl.is_sync_point and tl.sync_point_label else None
        conn.execute(
            """
            INSERT INTO timeline (
                chapter_num, track_name, elapsed_days, chapter_duration_days,
                in_universe_calendar, location_id, event_summary, sync_point_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                chapter_num,
                tl.track_name,
                elapsed,
                duration,
                tl.in_universe_calendar,
                loc_id,
                tl.event_summary,
                sync_id,
            ),
        )
        # Rewriting an older chapter changes its duration, so later rows of the same
        # track must be shifted; otherwise the monotonicity audit fails spuriously.
        ProjectRepository._resync_track_timeline(conn, tl.track_name)

    @staticmethod
    def _resync_track_timeline(conn: sqlite3.Connection, track_name: str) -> None:
        """Recompute cumulative elapsed_days for one track from per-chapter durations."""
        rows = conn.execute(
            "SELECT id, chapter_duration_days FROM timeline "
            "WHERE track_name = ? ORDER BY chapter_num ASC, id ASC;",
            (track_name,),
        ).fetchall()
        total = 0.0
        for row in rows:
            total += max(0.0, float(row["chapter_duration_days"] or 0.0))
            conn.execute(
                "UPDATE timeline SET elapsed_days = ? WHERE id = ?;",
                (total, int(row["id"])),
            )

    def log_anomaly(
        self,
        chapter_num: int,
        anomaly_type: str,
        severity: str,
        resolution_method: str,
        affected_chapters: list[int],
        detail_diff: dict[str, Any] | None = None,
    ) -> None:
        """Persist an anomaly detection and resolution event to anomaly_logs."""
        with atomic_transaction(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO anomaly_logs (
                    chapter_num, anomaly_type, severity, resolution_method,
                    affected_chapters, detail_diff_json
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    chapter_num,
                    anomaly_type,
                    severity,
                    resolution_method,
                    json.dumps(affected_chapters),
                    json.dumps(detail_diff or {}, ensure_ascii=False),
                ),
            )

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a read-only SQL query and return rows as dictionaries."""
        with read_connection(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
