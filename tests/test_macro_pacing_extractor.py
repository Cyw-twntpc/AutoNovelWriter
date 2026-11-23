"""Unit tests for MacroPacingExtractor and pacing template repository operations."""

from pathlib import Path
from typing import Any
import pytest

from autonovel.db.reference_repo import ReferenceRepository
from autonovel.db.reference_schema import init_reference_database
from autonovel.reference.macro_extractor import (
    MacroPacingExtractor,
    apply_pacing_jitter,
    build_factual_pacing_curve,
    extract_table_of_contents,
    select_high_value_chapters_zscore,
)
from autonovel.schemas.llm_contracts import (
    ChapterFactExtractContract,
    PacingStageTemplateContract,
)


@pytest.fixture
def temp_ref_repo(tmp_path: Path) -> ReferenceRepository:
    """Provide a fresh ReferenceRepository with initialized schema."""
    db_file = tmp_path / "reference_library.db"
    init_reference_database(db_file)
    return ReferenceRepository(db_file)


def test_toc_extraction():
    sample_text = (
        "楔子 前世之誓\n天地玄黃，宇宙洪荒。\n"
        "第一章 少年出山\n風雲激盪，劍指蒼穹。\n"
        "第二章 靈器風波\n「這把劍，我要了！」少年冷聲道。\n"
        "第3章 宗門大比\n擂台之上，萬眾矚目。\n"
        "尾聲 餘韻悠長\n夕陽西下，劍客遠行。\n"
    )
    toc = extract_table_of_contents(sample_text)
    assert len(toc) == 5
    assert "楔子" in toc[0]["title"]
    assert "第一章" in toc[1]["title"]
    assert "第二章" in toc[2]["title"]
    assert "第3章" in toc[3]["title"]
    assert "尾聲" in toc[4]["title"]


def test_pacing_jitter():
    stages = [
        PacingStageTemplateContract(
            stage_index=1,
            offset_percent=0.0,
            pacing_phase="BUILDUP",
            target_tension=2,
            target_progress=2,
            target_density=2,
            dramatic_purpose="蓄勢",
        ),
        PacingStageTemplateContract(
            stage_index=2,
            offset_percent=30.0,
            pacing_phase="COMPLICATION",
            target_tension=3,
            target_progress=3,
            target_density=3,
            dramatic_purpose="拉扯",
        ),
        PacingStageTemplateContract(
            stage_index=3,
            offset_percent=75.0,
            pacing_phase="CLIMAX",
            target_tension=5,
            target_progress=5,
            target_density=5,
            dramatic_purpose="決戰",
        ),
        PacingStageTemplateContract(
            stage_index=4,
            offset_percent=100.0,
            pacing_phase="RESOLUTION",
            target_tension=2,
            target_progress=2,
            target_density=2,
            dramatic_purpose="收束",
        ),
    ]
    jittered = apply_pacing_jitter(stages, jitter_ratio=0.15, seed=42)
    assert len(jittered) == 4
    assert jittered[0].offset_percent == 0.0
    assert jittered[3].offset_percent == 100.0
    assert jittered[1].offset_percent != 30.0  # Jitter applied
    assert 0.0 < jittered[1].offset_percent < jittered[2].offset_percent < 100.0


@pytest.mark.asyncio
async def test_repo_pacing_template_crud_and_matching(
    temp_ref_repo: ReferenceRepository, mock_llm_client: Any
):
    extractor = MacroPacingExtractor(temp_ref_repo, llm_client=mock_llm_client)
    sample_text = (
        "第一章 煉氣開端\n山門沉寂。\n"
        "第二章 危機漸近\n殺機四伏。\n"
        "第三章 生死博弈\n命懸一線。\n"
        "第四章 斬道破局\n神劍出鞘。\n"
        "第五章 踏入化神\n天地開闢。\n"
    )
    tpl1 = await extractor.extract_from_novel_text(
        text=sample_text,
        novel_title="凡人修仙",
        genre_primary="仙俠",
        genre_sub="凡人修真",
        jitter_ratio=0.0,
    )
    assert tpl1.template_name == "凡人修仙·標準宏觀節奏波浪"
    assert len(tpl1.stages) == 5
    assert len(tpl1.chapter_curve_data) == 5
    assert tpl1.chapter_curve_data[0]["pct"] == 20.0
    assert tpl1.chapter_curve_data[-1]["pct"] == 100.0

    # Retrieve
    all_tpls = temp_ref_repo.list_pacing_templates()
    assert len(all_tpls) == 1
    t_id = all_tpls[0]["id"]
    fetched = temp_ref_repo.get_pacing_template(t_id)
    assert fetched is not None
    assert fetched["name"] == "凡人修仙·標準宏觀節奏波浪"
    assert len(fetched["stages"]) == 5
    assert len(fetched["chapter_curve_data"]) == 5
    assert fetched["chapter_curve_data"][0]["pct"] == 20.0
    assert fetched["chapter_curve_data"][-1]["pct"] == 100.0

    # Add second template
    await extractor.extract_from_novel_text(
        text=sample_text,
        novel_title="詭秘之塔",
        genre_primary="懸疑",
        genre_sub="克蘇魯",
        jitter_ratio=0.0,
    )
    assert len(temp_ref_repo.list_pacing_templates()) == 2

    # Three-tier matching
    matched_xianxia = temp_ref_repo.match_pacing_template(
        genre_primary="仙俠",
        genre_sub="凡人修真",
        keywords=["#起承轉合", "#蓄勢爆發"],
    )
    assert matched_xianxia is not None
    assert "凡人修仙" in matched_xianxia["name"]

    matched_mystery = temp_ref_repo.match_pacing_template(
        genre_primary="懸疑",
        genre_sub="克蘇魯",
    )
    assert matched_mystery is not None
    assert "詭秘之塔" in matched_mystery["name"]

    # Blend templates
    t2_id = [t["id"] for t in temp_ref_repo.list_pacing_templates() if "詭秘" in t["name"]][0]
    hybrid_id = temp_ref_repo.blend_pacing_templates(t_id, t2_id, weight_a=0.6)
    hybrid_tpl = temp_ref_repo.get_pacing_template(hybrid_id)
    assert hybrid_tpl is not None
    assert "混血" in hybrid_tpl["name"]
    assert len(hybrid_tpl["stages"]) == 5

    # Stage override
    updated_stages = [
        {"stage_index": 1, "offset_percent": 0.0, "pacing_phase": "BUILDUP", "target_tension": 1, "target_progress": 1, "target_density": 1, "dramatic_purpose": "壓低"},
        {"stage_index": 2, "offset_percent": 50.0, "pacing_phase": "COMPLICATION", "target_tension": 3, "target_progress": 3, "target_density": 3, "dramatic_purpose": "拉扯"},
        {"stage_index": 3, "offset_percent": 85.0, "pacing_phase": "CLIMAX", "target_tension": 5, "target_progress": 5, "target_density": 5, "dramatic_purpose": "反常規大爆發"},
        {"stage_index": 4, "offset_percent": 100.0, "pacing_phase": "RESOLUTION", "target_tension": 1, "target_progress": 1, "target_density": 1, "dramatic_purpose": "餘波"},
    ]
    ok = temp_ref_repo.override_pacing_stages(hybrid_id, updated_stages)
    assert ok is True
    hybrid_updated = temp_ref_repo.get_pacing_template(hybrid_id)
    assert hybrid_updated["total_stages"] == 4
    assert hybrid_updated["stages"][2]["dramatic_purpose"] == "反常規大爆發"

    # Delete
    deleted = temp_ref_repo.delete_pacing_template(hybrid_id)
    assert deleted is True
    assert temp_ref_repo.get_pacing_template(hybrid_id) is None


def test_build_factual_pacing_curve_uses_ai_pacing_phase():
    """Verify build_factual_pacing_curve preserves AI-quantified pacing_phase and full core_event."""
    facts = [
        ChapterFactExtractContract(
            chapter_index=1,
            chapter_title="第一章 開端",
            core_event="主角初入宗門了解規則",
            pacing_phase="BUILDUP",
            actual_tension=2,
            actual_progress=2,
        ),
        ChapterFactExtractContract(
            chapter_index=2,
            chapter_title="第二章 暗流",
            core_event="外門試煉暗藏殺機",
            pacing_phase="COMPLICATION",
            actual_tension=3,
            actual_progress=3,
        ),
        ChapterFactExtractContract(
            chapter_index=3,
            chapter_title="第三章 爆發",
            core_event="主角在絕境中反殺強敵",
            pacing_phase="CLIMAX",
            actual_tension=5,
            actual_progress=5,
        ),
        ChapterFactExtractContract(
            chapter_index=4,
            chapter_title="第四章 收束",
            core_event="戰後清點收穫並晉升內門",
            pacing_phase="RESOLUTION",
            actual_tension=2,
            actual_progress=3,
        ),
    ]
    curve_data, stages = build_factual_pacing_curve(facts, 4)
    assert len(curve_data) == 4
    assert len(stages) == 4
    assert stages[0].pacing_phase == "BUILDUP"
    assert stages[1].pacing_phase == "COMPLICATION"
    assert stages[2].pacing_phase == "CLIMAX"
    assert stages[3].pacing_phase == "RESOLUTION"


def test_select_high_value_chapters_zscore():
    """Verify Z-score adaptive selection on AI-quantified chapter scores."""
    facts = [
        ChapterFactExtractContract(
            chapter_index=i,
            chapter_title=f"第{i}章",
            core_event="事件",
            actual_tension=5 if i in (3, 8) else 2,
            actual_progress=5 if i in (3, 8) else 2,
            score_causal=4.8 if i in (3, 8) else 2.5,
            score_craft=4.7 if i in (3, 8) else 2.5,
            score_depth=4.6 if i in (3, 8) else 2.5,
        )
        for i in range(1, 11)
    ]
    selected, scores_map, stats = select_high_value_chapters_zscore(
        facts, min_pct=0.20, max_pct=0.30
    )
    assert 3 in selected
    assert 8 in selected
    assert stats["selected_count"] == 2
    assert stats["global_mean_score"] > 0.0


def test_continuous_500_chapters_normalization_and_factual_curve():
    """Verify 500 chapters direct percentage normalization (each 0.2%) and factual curve milestones."""
    facts = []
    for i in range(1, 501):
        tension = 5 if (440 <= i <= 455 or i in (30, 95, 200, 310)) else (1 if i in (10, 45, 120, 240, 360, 480) else 3)
        progress = 5 if tension == 5 else 3
        phase = "CLIMAX" if tension == 5 else ("RESOLUTION" if tension == 1 else "COMPLICATION")
        facts.append(
            ChapterFactExtractContract(
                chapter_index=i,
                chapter_title=f"第{i}章 冒險",
                core_event="事件進展",
                pacing_phase=phase,
                actual_tension=tension,
                actual_progress=progress,
            )
        )

    curve_data, stages = build_factual_pacing_curve(facts, 500)
    assert len(curve_data) == 500
    assert curve_data[0]["pct"] == 0.2
    assert curve_data[1]["pct"] == 0.4
    assert curve_data[249]["pct"] == 50.0
    assert curve_data[499]["pct"] == 100.0

    tensions = [p["t"] for p in curve_data]
    assert min(tensions) == 1.0
    assert max(tensions) == 5.0

    assert len(stages) >= 4
    assert stages[0].offset_percent == 0.0
    assert stages[-1].offset_percent == 100.0


@pytest.mark.asyncio
async def test_chapter_fact_extractor_retry_and_error():
    """Verify that extract_chapter_facts retries and raises RuntimeError if all attempts fail."""
    from unittest.mock import AsyncMock, MagicMock
    from autonovel.reference.macro_extractor import ChapterFactExtractor

    mock_llm = MagicMock()
    mock_llm.server_manager.is_healthy.return_value = True
    mock_llm.generate_structured = AsyncMock(side_effect=RuntimeError("JSON EOF truncation"))

    extractor = ChapterFactExtractor(mock_llm)
    with pytest.raises(RuntimeError, match="Failed to extract chapter facts"):
        await extractor.extract_chapter_facts(
            chapter_index=85,
            chapter_title="第八十五章 風雲突變",
            chapter_text="文本內容" * 100,
        )
    assert mock_llm.generate_structured.call_count == 3
