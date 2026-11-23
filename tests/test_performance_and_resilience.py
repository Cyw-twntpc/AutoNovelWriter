"""Unit and integration tests for performance optimizations, resilience, and caching."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from autonovel.config import RuntimeConfig
from autonovel.db.project_schema import init_project_database
from autonovel.db.reference_repo import ReferenceRepository
from autonovel.engine.hardware_guard import HardwareGuard
from autonovel.llm.llm_client import LLMClient


def test_project_schema_indexes_created(tmp_path: Path) -> None:
    """Verify that all composite and foreign-key indexes are properly initialized."""
    import sqlite3

    db_path = tmp_path / "test_project.db"
    init_project_database(db_path)

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index';"
    ).fetchall()
    index_names = {r[0] for r in rows}
    conn.close()

    expected_indexes = {
        "idx_chapters_volume_status",
        "idx_characters_faction",
        "idx_characters_location",
        "idx_characters_status_rank",
        "idx_character_history_char_num",
        "idx_foreshadowing_status_planted",
        "idx_timeline_track_elapsed",
        "idx_entity_relations_source_target",
        "idx_anomaly_logs_chapter",
    }
    for idx in expected_indexes:
        assert idx in index_names, f"Missing expected index: {idx}"


def test_reference_matrix_cache_and_invalidation(tmp_path: Path) -> None:
    """Verify that reference library in-memory matrix cache builds and invalidates on mutation."""
    db_path = tmp_path / "test_ref.db"
    repo = ReferenceRepository(db_path)

    assert repo._matrix_cache is None

    # Insert a source and an entry
    src_id = repo.register_or_reset_source("test_sha", "sample.txt", chunk_count=1)
    vec1 = np.ones(512, dtype=np.float32)
    vec1 /= np.linalg.norm(vec1)
    tag_vec = np.zeros(512, dtype=np.float32)
    tag_vec[0] = 1.0

    entry_id = repo.insert_entry_with_tags(
        source_id=src_id,
        description="劍氣縱橫三萬里",
        desc_vec=vec1,
        snippet="青石斷崖之上",
        tag_vectors=[("仙俠", tag_vec)],
    )

    # Initial search builds the cache
    res = repo.search_dual_track(
        task_vec=vec1,
        query_tag_vecs=[tag_vec],
        threshold=0.5,
        top_k=3,
    )
    assert len(res) == 1
    assert res[0]["entry_id"] == entry_id
    assert repo._matrix_cache is not None
    assert "desc_matrix" in repo._matrix_cache
    assert repo._matrix_cache["desc_matrix"].shape == (1, 512)

    # Adding a tag invalidates the cache
    new_tag_vec = np.zeros(512, dtype=np.float32)
    new_tag_vec[1] = 1.0
    repo.add_tag_to_entry(entry_id, "劍道", new_tag_vec)
    assert repo._matrix_cache is None

    # Next search rebuilds the cache with the updated tag
    res2 = repo.search_dual_track(
        task_vec=vec1,
        query_tag_vecs=[new_tag_vec],
        threshold=0.5,
        top_k=3,
    )
    assert len(res2) == 1
    assert "劍道" in res2[0]["tags"]
    assert repo._matrix_cache is not None

    # Deleting entry invalidates the cache
    repo.delete_entry(entry_id)
    assert repo._matrix_cache is None


def test_llm_client_sse_line_fault_tolerance(tmp_path: Path) -> None:
    """Verify that SSE line parser handles malformed or non-JSON data without raising exceptions."""
    cfg = RuntimeConfig(workspace_dir=tmp_path)
    client = LLMClient(cfg)

    # Normal delta line
    valid_line = 'data: {"choices": [{"delta": {"content": "你好"}}]}'
    assert client._parse_sse_line(valid_line) == "你好"

    # [DONE] line
    assert client._parse_sse_line("data: [DONE]") == ""

    # Malformed JSON or non-JSON raw log
    assert client._parse_sse_line("data: {malformed json...}") == ""
    assert client._parse_sse_line("data: [INFO] server is ready") == ""
    assert client._parse_sse_line(": ping comment") == ""
    assert client._parse_sse_line("") == ""


@pytest.mark.asyncio
async def test_llm_client_connection_pooling_and_aclose(tmp_path: Path) -> None:
    """Verify persistent AsyncClient pooling and proper closing."""
    cfg = RuntimeConfig(workspace_dir=tmp_path)
    client = LLMClient(cfg)

    c1 = client._get_http_client()
    c2 = client._get_http_client()
    assert c1 is c2
    assert not c1.is_closed

    await client.aclose()
    assert client._http_client is None


def test_hardware_guard_telemetry_caching(tmp_path: Path) -> None:
    """Verify that HardwareGuard caches telemetry results within 2.5s window."""
    cfg = RuntimeConfig(workspace_dir=tmp_path)
    guard = HardwareGuard(cfg)

    t1 = guard.get_hardware_telemetry()
    t2 = guard.get_hardware_telemetry()

    assert guard._telemetry_cache is not None
    assert t1["ram_percent"] == t2["ram_percent"]


def test_big_data_batch_ingestion_and_marginalized_pruning(tmp_path: Path) -> None:
    """Verify high-throughput batch ingestion and topological outlier self-cleansing."""
    db_path = tmp_path / "test_big_data_ref.db"
    repo = ReferenceRepository(db_path)
    src_id = repo.register_or_reset_source("big_data_sha", "big_data_novel.txt", chunk_count=100)

    # Core clustered tag vector (appears frequently)
    core_vec = np.zeros(512, dtype=np.float32)
    core_vec[0] = 1.0

    # Rare isolated outlier tag vector (orthogonal to core)
    outlier_vec = np.zeros(512, dtype=np.float32)
    outlier_vec[500] = 1.0

    batch_items = []
    for i in range(10):
        v = np.random.randn(512).astype(np.float32)
        v /= np.linalg.norm(v)
        # 10 entries with the core tag
        tags = [("仙俠:核心技法", core_vec)]
        # Only entry 0 has an accidental solitary error tag
        if i == 0:
            tags.append(("離譜噪點標籤", outlier_vec))
        batch_items.append({
            "description": f"長篇小說修仙第{i}章宏觀情節推進",
            "desc_vec": v,
            "snippet": f"青石古壇之上第{i}段文字焦點",
            "tag_vectors": tags,
            "subject_role": "求道者",
            "object_role": "天道規則",
            "interaction_dynamic": "DOMINANCE",
        })

    eids = repo.insert_entries_batch(src_id, batch_items)
    assert len(eids) == 10

    # Check tags before pruning
    tags_before = {t["tag_name"]: t["count"] for t in repo.list_tags_with_frequency()}
    assert "仙俠:核心技法" in tags_before
    assert tags_before["仙俠:核心技法"] == 10
    assert "離譜噪點標籤" in tags_before
    assert tags_before["離譜噪點標籤"] == 1

    # Verify that rare outlier tags are identified but preserved (no destructive pruning)
    prune_res = repo.prune_marginalized_tags(min_frequency=2, eps=0.45)
    assert prune_res["total_outliers_identified"] >= 1
    assert prune_res["pruned_tags_count"] == 0

    # Verify that both core cluster tags and rare idiosyncratic tags are preserved
    tags_after = {t["tag_name"]: t["count"] for t in repo.list_tags_with_frequency()}
    assert "仙俠:核心技法" in tags_after
    assert "離譜噪點標籤" in tags_after


def test_big_data_tag_confidence_weighting(tmp_path: Path) -> None:
    """Verify that rare, unique literary tags enjoy full semantic retrieval parity without frequency dampening."""
    db_path = tmp_path / "test_tag_confidence.db"
    repo = ReferenceRepository(db_path)
    src_id = repo.register_or_reset_source("confidence_sha", "novel.txt", chunk_count=20)

    query_v = np.zeros(512, dtype=np.float32)
    query_v[10] = 1.0

    # Rare solitary literary technique
    batch = [{
        "description": "奇崛罕見孤本修辭",
        "desc_vec": query_v,
        "snippet": "正文奇崛片段",
        "tag_vectors": [("奇崛罕見孤本標籤", query_v)],
    }]
    repo.insert_entries_batch(src_id, batch)

    # Query matching query_v
    hits = repo.search_dual_track(task_vec=query_v, query_tag_vecs=[query_v], threshold=0.5, top_k=5)
    assert len(hits) == 1
    # The solitary rare tag must achieve full score without logarithmic dampening
    assert hits[0]["score_tag"] == pytest.approx(1.0)
    assert "奇崛罕見孤本標籤" in hits[0]["tags"]
