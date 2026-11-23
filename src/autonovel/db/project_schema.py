"""DDL schema definitions and initialization for novel_project.db."""

from __future__ import annotations

from pathlib import Path

from autonovel.db.connection import open_sqlite_connection

PROJECT_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS project_config (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    title TEXT NOT NULL DEFAULT '未命名長篇小說',
    genre_primary TEXT NOT NULL DEFAULT '奇幻',
    genre_sub TEXT NOT NULL DEFAULT '修仙',
    min_word_count INTEGER NOT NULL DEFAULT 5000000,
    qa_history_json TEXT NOT NULL DEFAULT '[]',
    global_outline_json TEXT NOT NULL DEFAULT '{}',
    style_contract_text TEXT NOT NULL DEFAULT '',
    static_anchor_text TEXT NOT NULL DEFAULT '',
    dynamic_anchor_text TEXT NOT NULL DEFAULT '',
    dynamic_anchor_score REAL NOT NULL DEFAULT 0.0,
    blacklist_json TEXT NOT NULL DEFAULT '{}',
    forward_retcon_note TEXT NOT NULL DEFAULT '',
    pipeline_state TEXT NOT NULL DEFAULT 'WIZARD',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS volumes (
    volume_num INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    outline_text TEXT NOT NULL DEFAULT '',
    target_chapters INTEGER NOT NULL DEFAULT 50,
    target_word_quota INTEGER NOT NULL DEFAULT 150000,
    max_allowed_rank_level INTEGER NOT NULL DEFAULT 3,
    status TEXT NOT NULL DEFAULT 'PLANNED',
    review_report_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chapters (
    chapter_num INTEGER PRIMARY KEY,
    volume_num INTEGER NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    beats_json TEXT NOT NULL DEFAULT '[]',
    staging_content TEXT NOT NULL DEFAULT '',
    final_content TEXT NOT NULL DEFAULT '',
    best_paragraph_text TEXT NOT NULL DEFAULT '',
    best_paragraph_score REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'UNWRITTEN',
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (volume_num) REFERENCES volumes(volume_num)
);

CREATE TABLE IF NOT EXISTS factions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    aliases TEXT NOT NULL DEFAULT '[]',
    parent_id INTEGER,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ACTIVE'
);

CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    aliases TEXT NOT NULL DEFAULT '[]',
    parent_id INTEGER,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ACTIVE'
);

CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    aliases TEXT NOT NULL DEFAULT '[]',
    role_type TEXT NOT NULL DEFAULT 'SUPPORTING',
    status TEXT NOT NULL DEFAULT 'ALIVE',
    rank_level INTEGER NOT NULL DEFAULT 1,
    faction_id INTEGER,
    current_location_id INTEGER,
    description TEXT NOT NULL DEFAULT '',
    last_appeared_chapter INTEGER NOT NULL DEFAULT 0,
    last_updated_chapter INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (faction_id) REFERENCES factions(id),
    FOREIGN KEY (current_location_id) REFERENCES locations(id)
);

CREATE TABLE IF NOT EXISTS character_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_id INTEGER NOT NULL,
    chapter_num INTEGER NOT NULL,
    attribute_name TEXT NOT NULL,
    old_value TEXT NOT NULL,
    new_value TEXT NOT NULL,
    change_type TEXT NOT NULL DEFAULT 'NORMAL_BREAKTHROUGH',
    reason TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (character_id) REFERENCES characters(id)
);

CREATE TABLE IF NOT EXISTS foreshadowing (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'SHORT',
    planted_chapter INTEGER NOT NULL,
    expected_span INTEGER NOT NULL DEFAULT 15,
    status TEXT NOT NULL DEFAULT 'PENDING',
    resolved_chapter INTEGER,
    resolution_notes TEXT NOT NULL DEFAULT '',
    related_character_ids TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS hierarchy_levels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL DEFAULT 'CULTIVATION',
    level_num INTEGER NOT NULL,
    level_name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    UNIQUE(category, level_num)
);

CREATE TABLE IF NOT EXISTS world_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_text TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL DEFAULT 'CORE'
);

CREATE TABLE IF NOT EXISTS entity_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'CHARACTER',
    relation_type TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    target_type TEXT NOT NULL DEFAULT 'CHARACTER',
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    established_chapter INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS timeline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_num INTEGER NOT NULL,
    track_name TEXT NOT NULL DEFAULT '主角線',
    elapsed_days REAL NOT NULL,
    chapter_duration_days REAL NOT NULL DEFAULT 1.0,
    in_universe_calendar TEXT NOT NULL DEFAULT '',
    location_id INTEGER,
    event_summary TEXT NOT NULL,
    sync_point_id TEXT,
    FOREIGN KEY (location_id) REFERENCES locations(id)
);

CREATE TABLE IF NOT EXISTS chapter_metrics (
    chapter_num INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    tension_intensity INTEGER NOT NULL CHECK (tension_intensity BETWEEN 1 AND 5),
    goal_progress INTEGER NOT NULL CHECK (goal_progress BETWEEN 1 AND 5),
    emotional_payoff INTEGER NOT NULL CHECK (emotional_payoff BETWEEN 1 AND 5),
    literary_score REAL NOT NULL,
    revision_count INTEGER NOT NULL DEFAULT 0,
    micro_rewrite_count INTEGER NOT NULL DEFAULT 0,
    contradiction_retry_count INTEGER NOT NULL DEFAULT 0,
    generation_duration_sec REAL NOT NULL DEFAULT 0.0,
    literary_rubric_json TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS anomaly_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    chapter_num INTEGER NOT NULL,
    anomaly_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'WARNING',
    resolution_method TEXT NOT NULL,
    affected_chapters TEXT NOT NULL DEFAULT '[]',
    detail_diff_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chapters_volume_status ON chapters(volume_num, status);
CREATE INDEX IF NOT EXISTS idx_characters_faction ON characters(faction_id);
CREATE INDEX IF NOT EXISTS idx_characters_location ON characters(current_location_id);
CREATE INDEX IF NOT EXISTS idx_characters_status_rank ON characters(status, rank_level);
CREATE INDEX IF NOT EXISTS idx_character_history_char_num ON character_history(character_id, chapter_num);
CREATE INDEX IF NOT EXISTS idx_foreshadowing_status_planted ON foreshadowing(status, planted_chapter);
CREATE INDEX IF NOT EXISTS idx_timeline_track_elapsed ON timeline(track_name, elapsed_days);
CREATE INDEX IF NOT EXISTS idx_entity_relations_source_target ON entity_relations(source_id, target_id, status);
CREATE INDEX IF NOT EXISTS idx_anomaly_logs_chapter ON anomaly_logs(chapter_num);
"""


def init_project_database(db_path: Path) -> None:
    """Create all project database tables and ensure singleton project_config row."""
    # executescript() implicitly COMMITs any open transaction, so running it inside
    # atomic_transaction left the init non-atomic. Drive an explicit transaction here;
    # a failure rolls the whole schema back instead of leaving half of it applied.
    script = (
        "BEGIN IMMEDIATE;\n"
        + PROJECT_SCHEMA_DDL
        + "\nINSERT OR IGNORE INTO project_config (id, title)"
        " VALUES (1, '未命名長篇小說');\nCOMMIT;"
    )
    conn = open_sqlite_connection(db_path)
    try:
        conn.executescript(script)
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()
