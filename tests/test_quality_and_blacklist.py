"""Tests for Gate A/B quality gates, non-destructive BlacklistScanner, and 3D pacing telemetry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autonovel.db.connection import atomic_transaction
from autonovel.db.project_repo import ProjectRepository
from autonovel.engine.blacklist_scanner import BlacklistScanner
from autonovel.engine.entity_resolver import EntityResolver
from autonovel.engine.pacing_monitor import PacingMonitor
from autonovel.engine.quality_gate import QualityGateEngine, compute_weighted_rubric_score
from autonovel.schemas.domain import BlacklistConfig, HierarchyChangeType
from autonovel.schemas.llm_contracts import (
    ChapterExtractionContract,
    CharacterDeltaContract,
    RubricItemResult,
)


@pytest.mark.asyncio
async def test_gate_a_b_and_entity_resolution(
    tmp_path: Path, mock_llm_client: Any
) -> None:
    """Verify BlacklistScanner does not alter prose, EntityResolver matches exact aliases, and Gate A catches dead characters."""
    db_path = tmp_path / "novel_project.db"
    repo = ProjectRepository(db_path)
    bl = BlacklistConfig()
    repo.update_project_config(blacklist_json=bl.model_dump_json())

    with atomic_transaction(db_path) as conn:
        conn.execute(
            "INSERT INTO volumes (volume_num, title, max_allowed_rank_level) VALUES (1, '卷一', 3);"
        )
        conn.execute(
            "INSERT INTO characters (name, aliases, status, rank_level) VALUES ('沈蒼', '[\"沈長老\"]', 'DEAD', 3);"
        )
        conn.execute(
            "INSERT INTO factions (name, aliases) VALUES ('玄陰閣', '[\"玄陰魔宗\"]');"
        )
        conn.execute(
            "INSERT INTO locations (name, aliases) VALUES ('北境寒淵', '[\"極北寒淵\"]');"
        )

    scanner = BlacklistScanner(repo, llm_client=mock_llm_client)
    resolver = EntityResolver(repo)
    gate = QualityGateEngine(repo, scanner, resolver, llm_client=mock_llm_client)

    raw_draft = (
        "林墨立於斷崖古松之下，凝視著眼前的古老陣紋，神色微斂，在心中推演著破陣的因果氣機與退路。\n\n"
        * 25
    )
    bad_extraction = ChapterExtractionContract(
        chapter_title="第1章 測試",
        tension_intensity=3,
        goal_progress=3,
        emotional_payoff=3,
        character_deltas=[
            CharacterDeltaContract(
                character_name="沈長老",
                new_status="ALIVE",
                new_rank_level=5,
                change_type=HierarchyChangeType.NORMAL_BREAKTHROUGH,
                faction_name="玄陰魔宗",
                location_name="極北寒淵",
            )
        ],
    )
    gate_a_res = await gate.run_gate_a(1, raw_draft, bad_extraction)
    assert gate_a_res["passed"] is False
    assert gate_a_res["micro_rewrite_count"] == 0
    assert gate_a_res["sanitized_text"].strip() == raw_draft.strip()
    assert scanner.scan_violations(gate_a_res["sanitized_text"]) == []
    assert any("Dead character '沈蒼'" in err for err in gate_a_res["hard_errors"])
    assert gate_a_res["extraction"].character_deltas[0].faction_name == "玄陰閣"
    assert gate_a_res["extraction"].character_deltas[0].location_name == "北境寒淵"

    gate_b_res = await gate.run_gate_b(gate_a_res["sanitized_text"])
    assert gate_b_res["composite_score"] >= 80.0
    assert gate_b_res["passed"] is True

    # Verify exact 40% RHETORIC / 35% DIALOGUE / 25% SCENE weighting on AI verdicts
    weighted = compute_weighted_rubric_score(
        [
            RubricItemResult(item_id="R1", dimension="RHETORIC", verdict="PASS"),
            RubricItemResult(item_id="D1", dimension="DIALOGUE", verdict="FAIL"),
            RubricItemResult(item_id="S1", dimension="SCENE", verdict="PASS"),
        ],
    )
    assert weighted["composite_score"] == 65.0


def test_3d_pacing_sliding_window_and_foreshadow_expiration(tmp_path: Path) -> None:
    """Verify quantitative stagnation telemetry, climax fatigue, payoff drought alerts, and non-destructive foreshadow reminder."""
    db_path = tmp_path / "novel_project.db"
    repo = ProjectRepository(db_path)
    monitor = PacingMonitor(repo)

    with atomic_transaction(db_path) as conn:
        for ch in range(1, 6):
            conn.execute(
                """
                INSERT INTO chapter_metrics (
                    chapter_num, title, word_count, tension_intensity, goal_progress,
                    emotional_payoff, literary_score
                ) VALUES (?, ?, 2500, 1, 1, 2, 86.0);
                """,
                (ch, f"第{ch}章"),
            )
        conn.execute(
            """
            INSERT INTO foreshadowing (description, scope, planted_chapter, expected_span, status)
            VALUES ('玄鐵古令秘密', 'SHORT', 1, 10, 'PENDING');
            """
        )

    eval_stagnant = monitor.evaluate_sliding_window(6)
    codes = {a["code"] for a in eval_stagnant["alerts"]}
    assert "STAGNATION_ALERT" in codes

    exp_res = monitor.check_and_expire_foreshadowing(22)
    assert len(exp_res["expired_ids"]) == 1
    fs_row = repo.fetch_all("SELECT status FROM foreshadowing WHERE id = ?;", (exp_res["expired_ids"][0],))[0]
    assert fs_row["status"] == "PENDING"
