"""Regression tests for P2/P3 hardening fixes (encoding, search, export, thermal, gates)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from autonovel.config import RuntimeConfig
from autonovel.db.project_repo import ProjectRepository
from autonovel.db.reference_repo import ReferenceRepository
from autonovel.engine.hardware_guard import HardwareGuard
from autonovel.export.exporter import NovelExporter
from autonovel.reference.file_parser import parse_novel_file
from autonovel.schemas.domain import ExportRequest
from autonovel.web.app import create_app


def _unit(vec: np.ndarray) -> np.ndarray:
    """Return a normalised copy of the vector."""
    return vec / np.linalg.norm(vec)


def _insert_entry(
    repo: ReferenceRepository,
    source_id: int,
    description: str,
    dim: int,
    tag_vectors: list[tuple[str, np.ndarray]] | None = None,
    desc_vec: np.ndarray | None = None,
) -> int:
    """Insert a reference entry with a deterministic description embedding."""
    if desc_vec is None:
        desc_vec = np.zeros(dim, dtype=np.float32)
        desc_vec[0] = 1.0
    return repo.insert_entry_with_tags(
        source_id=source_id,
        description=description,
        desc_vec=desc_vec,
        snippet="",
        tag_vectors=tag_vectors or [],
    )


def test_tagless_entry_does_not_inherit_full_description_score(tmp_path: Path) -> None:
    """M6: a tagless row must never be retrievable for a tag-aware query.

    Both rows match the query description perfectly, so the old ``s_tag = s_desc``
    fallback would have handed the tagless row the same score as a perfect tag hit.
    """
    repo = ReferenceRepository(tmp_path / "ref.db")
    src = repo.register_or_reset_source("sha-tags", "book.txt", chunk_count=1)
    perfect = _unit(np.ones(512, dtype=np.float32))
    tag = perfect.copy()
    tagged = _insert_entry(repo, src, "有標籤條目", 512, [("仙俠", tag)], desc_vec=perfect)
    tagless = _insert_entry(repo, src, "無標籤條目", 512, desc_vec=perfect)

    hits = repo.search_dual_track(
        task_vec=perfect,
        query_tag_vecs=[tag],
        threshold=0.72,
        top_k=5,
    )
    ids = [h["entry_id"] for h in hits]
    assert tagged in ids
    assert tagless not in ids


def test_search_tolerates_mixed_embedding_dimensions(tmp_path: Path) -> None:
    """M5: one stale row with another dimension must not break every search."""
    repo = ReferenceRepository(tmp_path / "ref.db")
    src = repo.register_or_reset_source("sha-dims", "book.txt", chunk_count=1)
    _insert_entry(repo, src, "異常維度", 8)
    _insert_entry(repo, src, "正常維度", 512)

    hits = repo.search_dual_track(
        task_vec=_unit(np.ones(512, dtype=np.float32)),
        query_tag_vecs=[],
        threshold=0.0,
        top_k=5,
    )
    assert len(hits) == 1


def test_gbk_text_decodes_as_chinese_not_mojibake(tmp_path: Path) -> None:
    """M8: a GBK-encoded book must be detected by content, not candidate order."""
    book = tmp_path / "book.txt"
    body = "第一章 劍氣如霜\n寒風凜冽，劍光一閃，少年握緊手中長劍。\n" * 40
    book.write_bytes(body.encode("gbk"))

    parsed, encoding = parse_novel_file(book)
    assert "劍氣如霜" in parsed
    assert encoding in {"gb18030", "gbk"}


def test_export_volume_scope_requires_volume_number(tmp_path: Path) -> None:
    """M23: a VOLUME export without volume_num must not silently export the whole book."""
    repo = ProjectRepository(tmp_path / "project.db")
    exporter = NovelExporter(repo, tmp_path / "exports")
    with pytest.raises(ValueError, match="volume_num"):
        exporter.export_novel(ExportRequest(scope_type="VOLUME"))


@pytest.mark.asyncio
async def test_cooldown_timeout_reports_not_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M22: cooling down past the deadline while still hot must return False."""
    guard = HardwareGuard(RuntimeConfig(workspace_dir=tmp_path))
    monkeypatch.setattr(
        guard, "get_hardware_telemetry", lambda: {"gpu_temp_c": 90}
    )

    async def _no_sleep(_seconds: float) -> None:
        """Skip the real two-second wait to keep the test fast."""
        return None

    monkeypatch.setattr("autonovel.engine.hardware_guard.asyncio.sleep", _no_sleep)
    recovered = await guard.check_and_cooldown_if_hot(resume_threshold_c=76)
    assert recovered is False


@pytest.mark.asyncio
async def test_wizard_complete_rejects_history_below_threshold(tmp_path: Path) -> None:
    """M20: the blueprint unlock threshold is enforced server-side."""
    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        llama_model_path="models/nonexistent_offline.gguf",
    )
    app = create_app(cfg)

    async def _always_ready(*_args: Any, **_kwargs: Any) -> bool:
        """Bypass the model probe; this test only exercises the threshold gate."""
        return True

    app.state.daemon.ais_llm_ready = _always_ready
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/api/wizard/complete")
        assert resp.status_code == 400
        assert "創世問答" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_manual_step_rejected_while_serialization_running(tmp_path: Path) -> None:
    """H20: a manual single-step must not interleave with the background loop."""
    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        llama_model_path="models/nonexistent_offline.gguf",
    )
    app = create_app(cfg)
    daemon = app.state.daemon
    daemon.project_repo.update_project_config(pipeline_state="RUNNING")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/api/warroom/step")
        assert resp.status_code == 409
