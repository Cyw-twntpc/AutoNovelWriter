"""Unit and integration tests for Genesis Q&A dual-role separation and turn-by-turn loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from autonovel.config import RuntimeConfig
from autonovel.web.app import create_app
from autonovel.wizard.qa_engine import QUESTION_PROGRESSION_ASPECTS, WizardQAEngine


def _attach_mock_llm(app: Any, mock_client: Any) -> None:
    daemon = app.state.daemon
    daemon.llm_client = mock_client
    daemon.qa_engine.llm_client = mock_client
    daemon.blueprint_builder.llm_client = mock_client


def test_pick_balanced_aspect_distribution() -> None:
    """Verify that _pick_balanced_aspect_static evenly distributes across 7 dimensions."""
    history: list[dict[str, Any]] = []
    seen_aspects: list[str] = []

    # First pass: 7 questions should cover all 7 unique aspects
    for i in range(7):
        asp, desc = WizardQAEngine._pick_balanced_aspect_static(history)
        seen_aspects.append(asp)
        history.append({
            "step_index": i + 1,
            "question": f"問題 {i + 1}",
            "aspect": asp,
            "answer": f"回答 {i + 1}",
        })

    unique_first_pass = set(seen_aspects)
    assert len(unique_first_pass) == 7
    for asp, _ in QUESTION_PROGRESSION_ASPECTS:
        assert asp in unique_first_pass

    # Second pass: next 7 questions should deepen round 2
    second_pass_aspects: list[str] = []
    for i in range(7, 14):
        asp, desc = WizardQAEngine._pick_balanced_aspect_static(history)
        second_pass_aspects.append(asp)
        assert "深化" in desc
        history.append({
            "step_index": i + 1,
            "question": f"問題 {i + 1}",
            "aspect": asp,
            "answer": f"回答 {i + 1}",
        })

    assert len(set(second_pass_aspects)) == 7


@pytest.mark.asyncio
async def test_auto_answer_and_step_auto_turn(tmp_path: Path, mock_llm_client: Any) -> None:
    """Verify Role B answering and turn-by-turn execution updates DB history."""
    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        llama_model_path="models/nonexistent_offline.gguf",
    )
    app = create_app(cfg)
    _attach_mock_llm(app, mock_llm_client)

    daemon = app.state.daemon
    engine: WizardQAEngine = daemon.qa_engine

    # Step turn 1
    res = await engine.step_auto_turn(
        question_title="請決定「敘事文風與語言節奏」",
        question_aspect="敘事文風與推進節奏",
        options=["古典內斂", "熱血快節奏", "冷酷壓抑"],
    )

    assert res["accepted"] is True
    assert res["step_index"] == 1
    assert res["total_answered"] == 1
    assert len(res["answered_selection"]) > 0
    assert "方針" in res["history"][0]["answer"] or len(res["history"][0]["answer"]) > 0
    assert res["next_question"]["question_title"] != ""


@pytest.mark.asyncio
async def test_autocomplete_step_api_endpoint(tmp_path: Path, mock_llm_client: Any) -> None:
    """Verify HTTP endpoint /api/wizard/autocomplete_step returns compliant structure."""
    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        llama_model_path="models/nonexistent_offline.gguf",
    )
    app = create_app(cfg)
    _attach_mock_llm(app, mock_llm_client)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Setup project
        r_setup = await client.post(
            "/api/wizard/setup",
            json={
                "title": "《太玄問仙錄》",
                "genre_primary": "仙俠",
                "genre_sub": "古典修仙",
                "min_word_count": 500000,
            },
        )
        assert r_setup.status_code == 200
        setup_data = r_setup.json()
        first_q = setup_data["next_question"]

        # 2. Call autocomplete_step
        r_step = await client.post(
            "/api/wizard/autocomplete_step",
            json={
                "question_title": first_q["question_title"],
                "question_aspect": first_q.get("question_aspect", "力量／資源階梯體系"),
                "options": first_q["options"],
            },
        )
        assert r_step.status_code == 200
        step_data = r_step.json()
        assert step_data["accepted"] is True
        assert step_data["total_answered"] == 1
        assert step_data["qa_threshold"] == 7
        assert step_data["can_build_blueprint"] is False
        assert "next_question" in step_data
        assert step_data["next_question"]["question_title"] != ""


def test_get_blueprint_unlock_threshold_tiers() -> None:
    """Verify threshold adapts dynamically to word count tiers."""
    from autonovel.wizard.qa_engine import get_blueprint_unlock_threshold

    # < 1,000,000 words -> 7 questions
    assert get_blueprint_unlock_threshold(100_000) == 7
    assert get_blueprint_unlock_threshold(500_000) == 7
    assert get_blueprint_unlock_threshold(999_999) == 7

    # 1,000,000 to < 3,000,000 words -> 10 questions
    assert get_blueprint_unlock_threshold(1_000_000) == 10
    assert get_blueprint_unlock_threshold(2_000_000) == 10
    assert get_blueprint_unlock_threshold(2_999_999) == 10

    # >= 3,000,000 words -> 14 questions
    assert get_blueprint_unlock_threshold(3_000_000) == 14
    assert get_blueprint_unlock_threshold(5_000_000) == 14
    assert get_blueprint_unlock_threshold(10_000_000) == 14
