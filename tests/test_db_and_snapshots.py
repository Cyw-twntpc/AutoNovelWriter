"""Tests for dual SQLite schemas, controlled hierarchy transitions, atomic commit, and snapshots."""

from __future__ import annotations

from pathlib import Path

from autonovel.db.connection import read_connection
from autonovel.db.project_repo import ProjectRepository
from autonovel.db.reference_repo import ReferenceRepository
from autonovel.db.snapshot_manager import SnapshotManager
from autonovel.schemas.domain import HierarchyChangeType
from autonovel.schemas.llm_contracts import (
    CharacterDeltaContract,
    ChapterExtractionContract,
    TimelineDeltaContract,
)


def test_dual_sqlite_schema_and_controlled_hierarchy(tmp_path: Path) -> None:
    """Verify 14 project tables, 3 reference tables, and controlled rank transition rules."""
    proj_db = tmp_path / "novel_project.db"
    ref_db = tmp_path / "reference_library.db"
    proj_repo = ProjectRepository(proj_db)
    ref_repo = ReferenceRepository(ref_db)

    with read_connection(proj_db) as conn:
        tables = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
    expected_project_tables = {
        "project_config",
        "volumes",
        "chapters",
        "factions",
        "locations",
        "characters",
        "character_history",
        "foreshadowing",
        "hierarchy_levels",
        "world_rules",
        "entity_relations",
        "timeline",
        "chapter_metrics",
        "anomaly_logs",
    }
    assert expected_project_tables.issubset(tables)
    assert ref_repo.get_library_stats() == {"sources": 0, "entries": 0, "tags": 0}

    # Normal breakthrough +1 within ceiling -> True
    ok, _ = proj_repo.validate_hierarchy_transition(
        1, 2, 4, HierarchyChangeType.NORMAL_BREAKTHROUGH.value, "築基成功"
    )
    assert ok is True

    # Unexplained regression 3 -> 1 with NORMAL_BREAKTHROUGH -> False
    ok_reg_bad, _ = proj_repo.validate_hierarchy_transition(
        3, 1, 4, HierarchyChangeType.NORMAL_BREAKTHROUGH.value, ""
    )
    assert ok_reg_bad is False

    # Explained regression 3 -> 1 with INJURY_REGRESSION and reason -> True
    ok_reg_good, _ = proj_repo.validate_hierarchy_transition(
        3, 1, 4, HierarchyChangeType.INJURY_REGRESSION.value, "遭玄陰老祖重創跌落境界"
    )
    assert ok_reg_good is True

    # Exceeding stage gate ceiling -> False
    ok_ceil, _ = proj_repo.validate_hierarchy_transition(
        4, 5, 4, HierarchyChangeType.NORMAL_BREAKTHROUGH.value, "強行突破"
    )
    assert ok_ceil is False


def test_two_phase_atomic_commit_anchor_fallback_and_snapshots(tmp_path: Path) -> None:
    """Verify STAGING rollback, dynamic anchor >=85 update/fallback, and rolling 5 snapshots."""
    proj_db = tmp_path / "novel_project.db"
    snap_dir = tmp_path / "snapshots"
    repo = ProjectRepository(proj_db)
    snap_mgr = SnapshotManager(proj_db, snap_dir, max_periodic_snapshots=5)

    with read_connection(proj_db) as conn:
        conn.execute(
            "INSERT INTO volumes (volume_num, title, target_chapters, max_allowed_rank_level) VALUES (1, '第一卷', 10, 3);"
        )
        for idx in range(1, 8):
            conn.execute(
                "INSERT INTO chapters (chapter_num, volume_num, title, status) VALUES (?, 1, ?, 'UNWRITTEN');",
                (idx, f"第{idx}章"),
            )
        conn.commit()

    # Save chapter 1 as STAGING and test startup crash rollback
    repo.save_staging_chapter(1, "第1章 暫存", "未完成草稿")
    rolled = repo.rollback_staging_chapters()
    assert rolled == [1]
    ch1 = repo.fetch_all("SELECT status, staging_content FROM chapters WHERE chapter_num = 1;")[0]
    assert ch1["status"] == "UNWRITTEN"
    assert ch1["staging_content"] == ""

    # Commit chapter 1 with best_paragraph_score = 90.0 (>= 85 updates dynamic anchor)
    ext1 = ChapterExtractionContract(
        chapter_title="第1章 寒淵驚變",
        tension_intensity=3,
        goal_progress=3,
        emotional_payoff=4,
        character_deltas=[CharacterDeltaContract(character_name="林墨", new_rank_level=1)],
        timeline_deltas=[TimelineDeltaContract(track_name="主角線", duration_days=1.5, event_summary="初入寒淵")],
    )
    repo.commit_chapter_atomically(
        chapter_num=1,
        final_content="正式正文第一章內容",
        extraction=ext1,
        literary_score=88.0,
        rubric_payload={"composite_score": 88.0},
        best_paragraph_text="第一高分段落錨點文字",
        best_paragraph_score=90.0,
        revision_count=0,
        micro_rewrite_count=1,
        contradiction_retry_count=0,
        duration_sec=12.5,
    )
    assert repo.get_project_config()["dynamic_anchor_text"] == "第一高分段落錨點文字"

    # Commit chapter 2 with best_paragraph_score = 81.0 (< 85 retains previous dynamic anchor)
    repo.commit_chapter_atomically(
        chapter_num=2,
        final_content="正式正文第二章內容",
        extraction=ext1,
        literary_score=82.0,
        rubric_payload={"composite_score": 82.0},
        best_paragraph_text="未達85分之段落",
        best_paragraph_score=81.0,
        revision_count=1,
        micro_rewrite_count=0,
        contradiction_retry_count=0,
        duration_sec=14.0,
    )
    assert repo.get_project_config()["dynamic_anchor_text"] == "第一高分段落錨點文字"

    # Create 7 periodic snapshots and 1 milestone snapshot -> only 5 periodic + 1 milestone remain
    for i in range(1, 8):
        snap_mgr.create_snapshot("periodic", chapter_num=i)
    ms_snap = snap_mgr.create_snapshot("milestone", chapter_num=2, label="vol1")
    all_snaps = snap_mgr.list_snapshots()
    periodic_snaps = [s for s in all_snaps if s["kind"] == "periodic"]
    milestone_snaps = [s for s in all_snaps if s["kind"] == "milestone"]
    assert len(periodic_snaps) == 5
    assert len(milestone_snaps) == 1

    # Restore milestone snapshot and verify state transitions to PAUSED
    res = snap_mgr.restore_snapshot(ms_snap["snapshot_id"])
    assert res["restored_snapshot"] == ms_snap["snapshot_id"]
    assert repo.get_project_config()["pipeline_state"] == "PAUSED"
