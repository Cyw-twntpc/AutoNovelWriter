"""Tests for Chapter 3D Value Pre-Scoring and Track A Targeted Z-Score Sampling."""

from __future__ import annotations

import random
from pathlib import Path

from autonovel.schemas.llm_contracts import (
    ChapterFactExtractContract,
    calculate_chapter_exemplar_score,
)
from autonovel.reference.chunk_extractor import split_targeted_chapters_into_chunks
from autonovel.db.reference_schema import init_reference_database
from autonovel.db.reference_repo import ReferenceRepository
from scripts.run_auto_extraction import select_high_value_chapters_zscore


def test_calculate_chapter_exemplar_score_formula():
    """Verify peak-led formula S = S_max * 1.4 + S_mid * 0.4 + S_min * 0.2."""
    # Peak single-dimension (5.0, 1.0, 1.0) -> 5*1.4 + 1*0.4 + 1*0.2 = 7.6
    score_peak = calculate_chapter_exemplar_score(5.0, 1.0, 1.0)
    assert score_peak == 7.6

    # Flat mediocrity (3.0, 3.0, 3.0) -> 3*1.4 + 3*0.4 + 3*0.2 = 6.0
    score_flat = calculate_chapter_exemplar_score(3.0, 3.0, 3.0)
    assert score_flat == 6.0

    # Single-dimension peak must outrank flat mediocrity
    assert score_peak > score_flat

    # Balanced excellence (4.5, 3.5, 2.0) -> 6.3 + 1.4 + 0.4 = 8.1
    score_balanced = calculate_chapter_exemplar_score(4.5, 3.5, 2.0)
    assert score_balanced == 8.1

    # Absolute min (1.0, 1.0, 1.0) -> 2.0
    assert calculate_chapter_exemplar_score(1.0, 1.0, 1.0) == 2.0

    # Absolute max (5.0, 5.0, 5.0) -> 10.0
    assert calculate_chapter_exemplar_score(5.0, 5.0, 5.0) == 10.0


def test_select_high_value_chapters_zscore_distribution():
    """Verify dual-tier Z-score proportion sampling across 100 chapters."""
    random.seed(42)
    chapter_facts: list[ChapterFactExtractContract] = []
    
    # Generate 100 chapters with baseline scores around 2.5 ~ 3.5
    for i in range(1, 101):
        # Insert intentional peaks at specific chapters
        if i in (15, 35, 55, 75, 95):
            # Climax chapters: high score
            causal, craft, depth = 4.8, 4.5, 4.2
        elif i in (10, 30, 50, 70, 90):
            # Moderate peaks
            causal, craft, depth = 3.8, 3.5, 3.0
        else:
            # Baseline normal distribution
            causal = round(random.uniform(2.0, 3.2), 1)
            craft = round(random.uniform(2.0, 3.2), 1)
            depth = round(random.uniform(2.0, 3.0), 1)

        cf = ChapterFactExtractContract(
            chapter_index=i,
            chapter_title=f"第{i}章 測試情節",
            core_event=f"第{i}章的核心事件說明",
            character_dynamics=[f"主角在第{i}章的狀態位移"],
            faction_shifts=["無"],
            actual_tension=3,
            actual_progress=3,
            score_causal=causal,
            score_craft=craft,
            score_depth=depth,
        )
        chapter_facts.append(cf)

    selected, scores_map, stats = select_high_value_chapters_zscore(
        chapter_facts=chapter_facts,
        min_pct=0.08,
        max_pct=0.12,
        vol_batch_size=20,
    )

    # 1. Bounded proportion window check: 100 * 0.08 = 8, 100 * 0.12 = 12
    assert 8 <= len(selected) <= 12
    assert stats["selected_count"] == len(selected)
    assert 0.08 <= stats["selected_proportion"] <= 0.12

    # 2. Climax peaks must all be captured
    for climax_ch in (15, 35, 55, 75, 95):
        assert climax_ch in selected, f"Climax chapter {climax_ch} should be selected"

    # 3. Span across novel must be broad
    assert stats["min_selected_chapter"] <= 20
    assert stats["max_selected_chapter"] >= 80


def test_split_targeted_chapters_into_chunks():
    """Verify split_targeted_chapters_into_chunks processes only specified chapters."""
    toc = [
        {"title": f"第{i}章 標題{i}", "text": f"這是第{i}章的詳細內文。" * 30}
        for i in range(1, 11)
    ]
    target_indices = {3, 8}
    chunks = split_targeted_chapters_into_chunks(
        toc=toc,
        target_chapter_indices=target_indices,
        chunk_size=500,
        overlap=50,
    )

    # All produced chunks must come from chapter 3 or 8
    assert len(chunks) > 0
    produced_indices = {ch_idx for ch_idx, _ in chunks}
    assert produced_indices == {3, 8}


def test_reference_repo_chapter_and_score_persistence(tmp_path: Path):
    """Verify chapter_index and exemplar_score can be inserted and queried from reference_repo."""
    import numpy as np

    db_file = tmp_path / "test_ref.db"
    init_reference_database(db_file)
    repo = ReferenceRepository(db_file)

    src_id = repo.register_or_reset_source("test_sha", "test_novel.txt", 10)

    # Insert batch with chapter_index and exemplar_score
    dummy_vec = np.zeros(384, dtype=np.float32)
    entries = [
        {
            "description": "【宏觀戰略】：情節爆發；【微觀戰術】：感官渲染極致",
            "desc_vec": dummy_vec,
            "snippet": "這是小說的精華場景切片...",
            "tag_vectors": [("心理描寫", dummy_vec), ("宿命破局者", dummy_vec)],
            "relational_dynamic": "主體與客體的激烈博弈",
            "subject_role": "宿命枷鎖破局者",
            "object_role": "權力鬥爭者",
            "interaction_dynamic": "抗爭反制",
            "chapter_index": 77,
            "exemplar_score": 8.75,
        }
    ]

    inserted_ids = repo.insert_entries_batch(src_id, entries)
    assert len(inserted_ids) == 1

    queried, total = repo.query_entries(source_id=src_id)
    assert total == 1
    assert queried[0]["chapter_index"] == 77
    assert abs(queried[0]["exemplar_score"] - 8.75) < 1e-4
    assert queried[0]["subject_role"] == "宿命枷鎖破局者"
    assert queried[0]["object_role"] == "權力鬥爭者"
    assert queried[0]["interaction_dynamic"] == "抗爭反制"
