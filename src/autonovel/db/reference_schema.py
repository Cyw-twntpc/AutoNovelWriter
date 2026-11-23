"""DDL schema definitions and initialization for reference_library.db."""

from __future__ import annotations

from pathlib import Path

from autonovel.db.connection import atomic_transaction

REFERENCE_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS ref_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_sha256 TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    imported_at TEXT NOT NULL DEFAULT (datetime('now')),
    chunk_count INTEGER NOT NULL DEFAULT 0,
    entry_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ref_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    description TEXT NOT NULL,
    desc_embedding BLOB NOT NULL,
    snippet TEXT NOT NULL,
    subject_role TEXT NOT NULL DEFAULT '',
    object_role TEXT NOT NULL DEFAULT '',
    interaction_dynamic TEXT NOT NULL DEFAULT '',
    relational_dynamic TEXT NOT NULL DEFAULT '',
    chapter_index INTEGER NOT NULL DEFAULT 0,
    exemplar_score REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (source_id) REFERENCES ref_sources(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ref_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id INTEGER NOT NULL,
    axis_type TEXT NOT NULL DEFAULT 'RHETORIC',
    tag_name TEXT NOT NULL,
    tag_embedding BLOB NOT NULL,
    FOREIGN KEY (entry_id) REFERENCES ref_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ref_entries_source ON ref_entries(source_id);
CREATE INDEX IF NOT EXISTS idx_ref_tags_entry ON ref_tags(entry_id);
CREATE INDEX IF NOT EXISTS idx_ref_tags_name ON ref_tags(tag_name);
CREATE INDEX IF NOT EXISTS idx_ref_tags_axis ON ref_tags(axis_type);

CREATE TABLE IF NOT EXISTS ref_book_blueprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL UNIQUE,
    novel_title TEXT NOT NULL,
    genre_primary TEXT NOT NULL,
    genre_sub TEXT NOT NULL,
    macro_outline_json TEXT NOT NULL DEFAULT '{}',
    character_network_json TEXT NOT NULL DEFAULT '{}',
    faction_geopolitics_json TEXT NOT NULL DEFAULT '{}',
    hierarchy_rules_json TEXT NOT NULL DEFAULT '{}',
    world_laws_json TEXT NOT NULL DEFAULT '[]',
    factual_pacing_curve TEXT NOT NULL DEFAULT '[]',
    total_chapters_analyzed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (source_id) REFERENCES ref_sources(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ref_blueprints_source ON ref_book_blueprints(source_id);

CREATE TABLE IF NOT EXISTS ref_pacing_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER,
    name TEXT NOT NULL UNIQUE,
    genre_primary TEXT NOT NULL,
    genre_sub TEXT NOT NULL,
    archetype TEXT NOT NULL,
    pacing_tags TEXT NOT NULL DEFAULT '[]',
    description TEXT NOT NULL DEFAULT '',
    total_stages INTEGER NOT NULL DEFAULT 4,
    total_chapters INTEGER NOT NULL DEFAULT 0,
    chapter_curve_data TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (source_id) REFERENCES ref_sources(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS ref_pacing_stages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    template_id INTEGER NOT NULL,
    stage_index INTEGER NOT NULL,
    offset_percent REAL NOT NULL,
    pacing_phase TEXT NOT NULL,
    target_tension INTEGER NOT NULL,
    target_progress INTEGER NOT NULL,
    target_density INTEGER NOT NULL,
    dramatic_purpose TEXT NOT NULL,
    FOREIGN KEY (template_id) REFERENCES ref_pacing_templates(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ref_pacing_stages_template ON ref_pacing_stages(template_id);
"""


def init_reference_database(db_path: Path) -> None:
    """Create all reference library tables and indexes according to standard schema."""
    with atomic_transaction(db_path) as conn:
        conn.executescript(REFERENCE_SCHEMA_DDL)
