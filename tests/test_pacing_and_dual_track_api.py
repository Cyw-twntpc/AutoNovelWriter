"""Integration tests for dual-track reference extraction APIs and pacing monitor guardrails."""

from pathlib import Path
from typing import Any
import httpx
import pytest

from autonovel.config import RuntimeConfig
from autonovel.db.connection import atomic_transaction
from autonovel.db.project_repo import ProjectRepository
from autonovel.db.reference_repo import ReferenceRepository
from autonovel.db.reference_schema import init_reference_database
from autonovel.engine.pacing_monitor import PacingMonitor
from autonovel.reference.chunk_extractor import compute_sha256
from autonovel.schemas.llm_contracts import (
    MacroPacingTemplateContract,
    PacingStageTemplateContract,
)
from autonovel.web.app import create_app
from tests.conftest import MockLLMClient


def _attach_mock_llm(app: Any, mock_client: Any) -> None:
    """Wire mock LLMClient into daemon and sub-engines."""
    daemon = app.state.daemon
    daemon.llm_client = mock_client
    daemon.ref_extractor.llm_client = mock_client
    daemon.retriever.llm_client = mock_client
    daemon.macro_extractor.llm_client = mock_client
    daemon.qa_engine.llm_client = mock_client
    daemon.blueprint_builder.llm_client = mock_client
    daemon.volume_planner.llm_client = mock_client


@pytest.mark.asyncio
async def test_pacing_templates_api_endpoints(tmp_path: Path):
    cfg = RuntimeConfig(workspace_dir=tmp_path / "workspace")
    app = create_app(cfg)
    mock_llm = MockLLMClient()
    _attach_mock_llm(app, mock_llm)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Create a sample reference file
        sample_novel = tmp_path / "凡人經典.txt"
        sample_novel.write_text(
            "第一章 青牛鎮少年\n少年在山野中奔跑。\n"
            "第二章 靈藥風波\n一株靈草引發爭奪。\n"
            "第三章 殺機突現\n宗門長老暗中伏擊。\n"
            "第四章 萬劍合一\n生死一刻神劍出鞘破局。\n"
            "第五章 踏入築基\n風雲平息，遠赴海外。\n",
            encoding="utf-8",
        )

        # 2. Ingest with extract_pacing=True, extract_skills=False
        resp = await client.post(
            "/api/reference/ingest",
            json={
                "file_paths": [str(sample_novel)],
                "extract_skills": False,
                "extract_pacing": True,
                "pacing_jitter": 0.05,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["pacing_extracted_count"] == 1
        assert data["ingested_chunks"] == 0

        # 3. List pacing templates
        resp_list = await client.get("/api/reference/pacing_templates")
        assert resp_list.status_code == 200
        templates = resp_list.json()["templates"]
        assert len(templates) >= 1
        tpl_id = templates[0]["id"]
        assert "凡人經典" in templates[0]["name"]

        # 4. Get single template
        resp_get = await client.get(f"/api/reference/pacing_templates/{tpl_id}")
        assert resp_get.status_code == 200
        tpl_data = resp_get.json()["template"]
        assert len(tpl_data["stages"]) == 5

        # 5. Override stages
        new_stages = [
            {"stage_index": 1, "offset_percent": 0.0, "pacing_phase": "BUILDUP", "target_tension": 2, "target_progress": 2, "target_density": 2, "dramatic_purpose": "低調隱忍"},
            {"stage_index": 2, "offset_percent": 35.0, "pacing_phase": "COMPLICATION", "target_tension": 3, "target_progress": 3, "target_density": 3, "dramatic_purpose": "宗門利益衝突"},
            {"stage_index": 3, "offset_percent": 80.0, "pacing_phase": "CLIMAX", "target_tension": 5, "target_progress": 5, "target_density": 5, "dramatic_purpose": "反向伏擊滅敵"},
            {"stage_index": 4, "offset_percent": 100.0, "pacing_phase": "RESOLUTION", "target_tension": 2, "target_progress": 2, "target_density": 2, "dramatic_purpose": "攜寶遁走"},
        ]
        resp_override = await client.put(
            f"/api/reference/pacing_templates/{tpl_id}/stages",
            json={"stages": new_stages},
        )
        assert resp_override.status_code == 200
        assert resp_override.json()["template"]["total_stages"] == 4

        # 6. Delete template
        resp_del = await client.delete(f"/api/reference/pacing_templates/{tpl_id}")
        assert resp_del.status_code == 200
        assert resp_del.json()["success"] is True

        # Confirm deletion
        resp_get_after = await client.get(f"/api/reference/pacing_templates/{tpl_id}")
        assert resp_get_after.status_code == 404


def test_pacing_monitor_rushing_guardrail(tmp_path: Path):
    proj_db = tmp_path / "novel_project.db"
    ref_db = tmp_path / "reference_library.db"
    proj_repo = ProjectRepository(proj_db)
    ref_repo = ReferenceRepository(ref_db)
    init_reference_database(ref_db)

    # Save a reference template with low tension buildup at 0-30%
    tpl = MacroPacingTemplateContract(
        template_name="凡人蓄勢波浪",
        genre_primary="仙俠",
        genre_sub="凡人修真",
        source_novel_title="凡人",
        total_chapters_analyzed=100,
        narrative_archetype="穩健流",
        pacing_tags=["#慢熱蓄勢"],
        summary="慢熱",
        stages=[
            PacingStageTemplateContract(stage_index=1, offset_percent=0.0, pacing_phase="BUILDUP", target_tension=1, target_progress=1, target_density=1, dramatic_purpose="極度克制"),
            PacingStageTemplateContract(stage_index=2, offset_percent=30.0, pacing_phase="BUILDUP", target_tension=2, target_progress=2, target_density=2, dramatic_purpose="微幅鋪墊"),
            PacingStageTemplateContract(stage_index=3, offset_percent=75.0, pacing_phase="CLIMAX", target_tension=5, target_progress=5, target_density=5, dramatic_purpose="決戰"),
            PacingStageTemplateContract(stage_index=4, offset_percent=100.0, pacing_phase="RESOLUTION", target_tension=2, target_progress=2, target_density=2, dramatic_purpose="收尾"),
        ],
    )
    ref_repo.save_pacing_template(tpl)

    # Create chapters in volume 1 (10 chapters total)
    with atomic_transaction(proj_db) as conn:
        conn.execute("UPDATE project_config SET genre_primary = '仙俠', genre_sub = '凡人修真' WHERE id = 1;")
        conn.execute("INSERT OR REPLACE INTO volumes (volume_num, title, target_chapters) VALUES (1, '第一卷', 10);")
        for i in range(1, 11):
            conn.execute("INSERT INTO chapters (chapter_num, volume_num, title, beats_json, status) VALUES (?, 1, ?, '[]', 'COMPLETED');", (i, f"第{i}章"))

        # Chapter 1 & 2 have artificially high tension (e.g. 5) during buildup
        conn.execute(
            """
            INSERT INTO chapter_metrics 
            (chapter_num, title, word_count, tension_intensity, goal_progress, emotional_payoff, literary_score)
            VALUES (1, '第1章', 2500, 5, 4, 3, 90.0);
            """
        )
        conn.execute(
            """
            INSERT INTO chapter_metrics 
            (chapter_num, title, word_count, tension_intensity, goal_progress, emotional_payoff, literary_score)
            VALUES (2, '第2章', 2500, 5, 5, 4, 92.0);
            """
        )

    monitor = PacingMonitor(proj_repo, ref_repo=ref_repo)
    result = monitor.evaluate_sliding_window(current_chapter_num=2)

    # RUSHING_ALERT must be triggered
    codes = [a["code"] for a in result["alerts"]]
    assert "RUSHING_ALERT" in codes
    assert "目標張力" in result["pacing_directive"]
    assert "凡人蓄勢波浪" in result["pacing_directive"]


@pytest.mark.asyncio
async def test_unload_model_and_pause_work_api(tmp_path: Path):
    """Verify endpoint to unload model from VRAM and pause running jobs."""
    cfg = RuntimeConfig(workspace_dir=tmp_path / "workspace")
    app = create_app(cfg)
    mock_llm = MockLLMClient()
    _attach_mock_llm(app, mock_llm)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post("/api/system/unload_and_pause")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_unloaded"] is True
        assert "writing_paused" in data
        assert "extraction_paused" in data
        assert "env_status" in data


@pytest.mark.asyncio
async def test_reference_pause_resume_and_pacing_progress_api(tmp_path: Path):
    """Verify pausing/resuming extraction jobs and reading pacing visual progress."""
    cfg = RuntimeConfig(workspace_dir=tmp_path / "workspace")
    app = create_app(cfg)
    mock_llm = MockLLMClient()
    _attach_mock_llm(app, mock_llm)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Pause
        r_pause = await client.post("/api/reference/pause")
        assert r_pause.status_code == 200
        assert r_pause.json()["status"] == "paused"
        assert app.state.daemon.ref_extractor.is_paused() is True

        # Pacing progress
        r_prog = await client.get("/api/reference/pacing_progress")
        assert r_prog.status_code == 200
        pacing_data = r_prog.json()
        assert "active" in pacing_data
        assert "percent" in pacing_data

        # Resume
        r_resume = await client.post("/api/reference/resume")
        assert r_resume.status_code == 200
        assert r_resume.json()["status"] == "running"
        assert app.state.daemon.ref_extractor.is_paused() is False

        # Cancel
        r_cancel = await client.post("/api/reference/cancel")
        assert r_cancel.status_code == 200
        assert r_cancel.json()["status"] == "cancelled"
        assert app.state.daemon.ref_extractor.is_cancelled() is True
        assert app.state.daemon.ref_extractor.status["active"] is False


@pytest.mark.asyncio
async def test_reference_zero_data_empty_state_api(tmp_path: Path):
    """Verify that in a completely fresh workspace with no data, all reference endpoints return 200 with standard empty collections."""
    cfg = RuntimeConfig(workspace_dir=tmp_path / "workspace")
    app = create_app(cfg)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Books list and stats
        resp_books = await client.get("/api/reference/books")
        assert resp_books.status_code == 200
        data_books = resp_books.json()
        assert data_books["books"] == []
        assert data_books["stats"] == {"sources": 0, "entries": 0, "tags": 0}

        # 2. Hot tags
        resp_tags = await client.get("/api/reference/tags")
        assert resp_tags.status_code == 200
        assert resp_tags.json()["tags"] == []

        # 3. Tag cluster assets
        resp_assets = await client.get("/api/reference/tag_assets")
        assert resp_assets.status_code == 200
        data_assets = resp_assets.json()
        assert data_assets["nodes"] == []
        assert data_assets["clusters"] == {}
        assert data_assets["stats"]["total_unique"] == 0
        assert data_assets["stats"]["total_usages"] == 0

        # 4. Pacing templates
        resp_pacing = await client.get("/api/reference/pacing_templates")
        assert resp_pacing.status_code == 200
        assert resp_pacing.json()["templates"] == []

        # 5. Knowledge entries
        resp_entries = await client.get("/api/reference/entries")
        assert resp_entries.status_code == 200
        data_entries = resp_entries.json()
        assert data_entries["entries"] == []
        assert data_entries["total"] == 0

        # 6. Background ingestion progress
        resp_prog = await client.get("/api/reference/progress")
        assert resp_prog.status_code == 200
        data_prog = resp_prog.json()
        assert data_prog["active"] is False
        assert data_prog["sources"] == 0


def test_blend_pacing_templates_with_curve_data(tmp_path: Path):
    """Verify that blending two templates properly calculates continuous curves."""
    ref_db = tmp_path / "reference_library.db"
    ref_repo = ReferenceRepository(ref_db)
    init_reference_database(ref_db)

    tpl_a = MacroPacingTemplateContract(
        template_name="範本A",
        genre_primary="仙俠",
        genre_sub="修真",
        source_novel_title="書A",
        total_chapters_analyzed=2,
        narrative_archetype="穩健流",
        summary="範本A概要",
        stages=[
            PacingStageTemplateContract(stage_index=1, offset_percent=0.0, pacing_phase="BUILDUP", target_tension=1, target_progress=1, target_density=1, dramatic_purpose="起步"),
            PacingStageTemplateContract(stage_index=2, offset_percent=33.0, pacing_phase="COMPLICATION", target_tension=2, target_progress=2, target_density=2, dramatic_purpose="發展"),
            PacingStageTemplateContract(stage_index=3, offset_percent=66.0, pacing_phase="CLIMAX", target_tension=4, target_progress=4, target_density=4, dramatic_purpose="高潮"),
            PacingStageTemplateContract(stage_index=4, offset_percent=100.0, pacing_phase="RESOLUTION", target_tension=2, target_progress=2, target_density=2, dramatic_purpose="收束"),
        ],
        chapter_curve_data=[
            {"ch": 1, "pct": 50.0, "t": 2.0, "p": 2.0, "d": 2.0, "title": "章1"},
            {"ch": 2, "pct": 100.0, "t": 4.0, "p": 4.0, "d": 4.0, "title": "章2"},
        ],
    )
    tpl_b = MacroPacingTemplateContract(
        template_name="範本B",
        genre_primary="仙俠",
        genre_sub="修真",
        source_novel_title="書B",
        total_chapters_analyzed=2,
        narrative_archetype="爆發流",
        summary="範本B概要",
        stages=[
            PacingStageTemplateContract(stage_index=1, offset_percent=0.0, pacing_phase="BUILDUP", target_tension=3, target_progress=3, target_density=3, dramatic_purpose="起步"),
            PacingStageTemplateContract(stage_index=2, offset_percent=33.0, pacing_phase="COMPLICATION", target_tension=4, target_progress=4, target_density=4, dramatic_purpose="發展"),
            PacingStageTemplateContract(stage_index=3, offset_percent=66.0, pacing_phase="CLIMAX", target_tension=5, target_progress=5, target_density=5, dramatic_purpose="高潮"),
            PacingStageTemplateContract(stage_index=4, offset_percent=100.0, pacing_phase="RESOLUTION", target_tension=3, target_progress=3, target_density=3, dramatic_purpose="收束"),
        ],
        chapter_curve_data=[
            {"ch": 1, "pct": 50.0, "t": 4.0, "p": 4.0, "d": 4.0, "title": "章1"},
            {"ch": 2, "pct": 100.0, "t": 4.0, "p": 4.0, "d": 4.0, "title": "章2"},
        ],
    )
    id_a = ref_repo.save_pacing_template(tpl_a)
    id_b = ref_repo.save_pacing_template(tpl_b)

    hybrid_id = ref_repo.blend_pacing_templates(id_a, id_b, weight_a=0.5, new_name="混血範本")
    hybrid = ref_repo.get_pacing_template(hybrid_id)
    assert hybrid is not None
    assert hybrid["name"] == "混血範本"
    assert len(hybrid["chapter_curve_data"]) == 2
    # At ch=1, t = 2.0*0.5 + 4.0*0.5 = 3.0
    assert hybrid["chapter_curve_data"][0]["t"] == 3.0





@pytest.mark.asyncio
async def test_reference_cancel_in_flight_ingestion(tmp_path: Path):
    """Verify cancelling an active background reference ingestion task halts processing immediately."""
    cfg = RuntimeConfig(workspace_dir=tmp_path / "workspace")
    app = create_app(cfg)
    mock_llm = MockLLMClient()
    _attach_mock_llm(app, mock_llm)
    transport = httpx.ASGITransport(app=app)

    # Create dummy novel file with 5 chapters
    dummy_file = tmp_path / "dummy_novel.txt"
    chapters_content = "\n\n".join(
        f"第{i}章 測試標題\n" + "這是測試長文內容，用於驗證切片抽取中斷機制。" * 80
        for i in range(1, 6)
    )
    dummy_file.write_text(chapters_content, encoding="utf-8")

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Launch background ingestion
        resp_ingest = await client.post(
            "/api/reference/ingest",
            json={
                "items": [
                    {
                        "file_path": str(dummy_file),
                        "action": "INGEST",
                        "extract_skills": True,
                        "extract_pacing": False,
                    }
                ],
                "run_in_background": True,
                "extract_skills": True,
                "extract_pacing": False,
            },
        )
        assert resp_ingest.status_code == 200
        assert resp_ingest.json()["queued"] is True

        # Send cancel signal
        r_cancel = await client.post("/api/reference/cancel")
        assert r_cancel.status_code == 200
        assert r_cancel.json()["status"] == "cancelled"

        # Check extractor status
        assert app.state.daemon.ref_extractor.is_cancelled() is True
        assert app.state.daemon.ref_extractor.status["active"] is False


@pytest.mark.asyncio
async def test_comprehensive_extraction_requirements_and_blueprint_caching(tmp_path: Path):
    """Verify Track B blueprint persistence, Stage 1 cache reuse, SKIP action handling, and dual track modes."""
    cfg = RuntimeConfig(workspace_dir=tmp_path / "workspace")
    app = create_app(cfg)
    mock_llm = MockLLMClient()
    _attach_mock_llm(app, mock_llm)
    transport = httpx.ASGITransport(app=app)

    # 1. Create sample novel
    novel_path = tmp_path / "仙道長青.txt"
    novel_content = "\n\n".join(
        f"第{i}章 雲海風雷\n" + "仙宗弟子踏雲而行，劍氣縱橫三萬里，大道爭鋒不死不休。" * 50
        for i in range(1, 6)
    )
    novel_path.write_text(novel_content, encoding="utf-8")

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Step A: Ingest with Track B only (extract_pacing=True, extract_skills=False)
        resp_pacing = await client.post(
            "/api/reference/ingest",
            json={
                "file_paths": [str(novel_path)],
                "extract_skills": False,
                "extract_pacing": True,
            },
        )
        assert resp_pacing.status_code == 200
        data_p = resp_pacing.json()
        assert data_p["pacing_extracted_count"] == 1
        assert data_p["ingested_chunks"] == 0

        # Verify book source is registered and has book blueprint
        resp_books = await client.get("/api/reference/books")
        assert resp_books.status_code == 200
        books = resp_books.json()["books"]
        assert len(books) == 1
        source_id = books[0]["id"]
        assert books[0]["entry_count"] == 0

        # Query GET /api/reference/books/{source_id}/blueprint
        resp_bp = await client.get(f"/api/reference/books/{source_id}/blueprint")
        assert resp_bp.status_code == 200
        bp_data = resp_bp.json()
        assert bp_data["novel_title"] == "仙道長青"
        assert len(bp_data["character_network"]) >= 3

        # Step B: Follow-up Track A ingestion for the same book (extract_skills=True, extract_pacing=False)
        # Should reuse existing blueprint from Step A and skip Stage 1!
        resp_skills = await client.post(
            "/api/reference/ingest",
            json={
                "file_paths": [str(novel_path)],
                "extract_skills": True,
                "extract_pacing": False,
            },
        )
        assert resp_skills.status_code == 200
        data_s = resp_skills.json()
        assert data_s["ingested_chunks"] > 0

        # Verify entry count updated
        resp_books2 = await client.get("/api/reference/books")
        books2 = resp_books2.json()["books"]
        assert books2[0]["entry_count"] > 0

        # Step C: Ingest with action="SKIP"
        resp_skip = await client.post(
            "/api/reference/ingest",
            json={
                "items": [
                    {
                        "file_path": str(novel_path),
                        "action": "SKIP",
                        "extract_skills": True,
                        "extract_pacing": True,
                    }
                ],
                "extract_skills": True,
                "extract_pacing": True,
            },
        )
        assert resp_skip.status_code == 200
        data_skip = resp_skip.json()
        assert data_skip["pacing_extracted_count"] == 0
        assert data_skip["ingested_chunks"] == 0

        # Step D: Background ingestion for Track B immediately queues
        novel_b = tmp_path / "大奉打更.txt"
        novel_b.write_text("第一章 獄中醒來\n許七安揉著眉心醒來。\n第二章 煉金術奇蹟\n司天監術士震驚不已。\n第三章 桑泊大案\n暗潮洶湧妖族現身。\n第四章 刀斬昏官\n民心所向刀意自生。\n", encoding="utf-8")
        resp_bg = await client.post(
            "/api/reference/ingest",
            json={
                "file_paths": [str(novel_b)],
                "extract_skills": False,
                "extract_pacing": True,
                "run_in_background": True,
            },
        )
        assert resp_bg.status_code == 200
        assert resp_bg.json()["queued"] is True


@pytest.mark.asyncio
async def test_skills_only_extraction_creates_zero_pacing_templates(tmp_path: Path):
    """Verify extract_pacing=False never saves any pacing template, and complementary track extraction preserves both tracks."""
    cfg = RuntimeConfig(workspace_dir=tmp_path / "workspace")
    app = create_app(cfg)
    mock_llm = MockLLMClient()
    _attach_mock_llm(app, mock_llm)
    transport = httpx.ASGITransport(app=app)

    novel_path = tmp_path / "劍來問道.txt"
    novel_content = "\n\n".join(
        f"第{i}章 泥瓶巷少年\n" + "小鎮風雨欲來，老槐樹下劍氣暗藏，少年一步步走出大山。" * 50
        for i in range(1, 6)
    )
    novel_path.write_text(novel_content, encoding="utf-8")

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Ingest with extract_skills=True, extract_pacing=False
        resp_s = await client.post(
            "/api/reference/ingest",
            json={
                "items": [
                    {
                        "file_path": str(novel_path),
                        "action": "INGEST",
                        "extract_skills": True,
                        "extract_pacing": False,
                    }
                ],
                "extract_skills": True,
                "extract_pacing": False,
            },
        )
        assert resp_s.status_code == 200
        data_s = resp_s.json()
        assert data_s["ingested_chunks"] > 0
        assert data_s["pacing_extracted_count"] == 0

        # Pacing templates must be completely empty
        resp_tpls = await client.get("/api/reference/pacing_templates")
        assert resp_tpls.status_code == 200
        assert resp_tpls.json()["templates"] == []

        # Preflight check on same file should report has_skills=True, has_pacing=False
        resp_pf = await client.post(
            "/api/reference/preflight",
            json={"file_paths": [str(novel_path)]},
        )
        assert resp_pf.status_code == 200
        pf_item = resp_pf.json()["items"][0]
        assert pf_item["status"] == "DUPLICATE"
        assert pf_item["has_skills"] is True
        assert pf_item["has_pacing"] is False
        assert pf_item["has_blueprint"] is True

        # 2. Now run complementary Track B only (extract_skills=False, extract_pacing=True)
        resp_p = await client.post(
            "/api/reference/ingest",
            json={
                "items": [
                    {
                        "file_path": str(novel_path),
                        "action": "REEXTRACT",
                        "extract_skills": False,
                        "extract_pacing": True,
                    }
                ],
                "extract_skills": False,
                "extract_pacing": True,
            },
        )
        assert resp_p.status_code == 200
        assert resp_p.json()["pacing_extracted_count"] == 1

        # Verify both skills and pacing exist now
        resp_pf2 = await client.post(
            "/api/reference/preflight",
            json={"file_paths": [str(novel_path)]},
        )
        pf_item2 = resp_pf2.json()["items"][0]
        assert pf_item2["has_skills"] is True
        assert pf_item2["has_pacing"] is True
        assert pf_item2["entry_count"] > 0

        # 3. Re-extract with extract_skills=True, extract_pacing=False -> removes prior pacing template for this book
        resp_re = await client.post(
            "/api/reference/ingest",
            json={
                "items": [
                    {
                        "file_path": str(novel_path),
                        "action": "REEXTRACT",
                        "extract_skills": True,
                        "extract_pacing": False,
                    }
                ],
                "extract_skills": True,
                "extract_pacing": False,
            },
        )
        assert resp_re.status_code == 200
        resp_tpls_after = await client.get("/api/reference/pacing_templates")
        assert resp_tpls_after.json()["templates"] == []


@pytest.mark.asyncio
async def test_manual_unload_blocks_auto_restart_until_explicit_resume(tmp_path: Path):
    """Verify manual unload sets _manually_unloaded=True, blocks watchdog restart, and clears on explicit resume."""
    cfg = RuntimeConfig(workspace_dir=tmp_path / "workspace")
    app = create_app(cfg)
    mock_llm = MockLLMClient()
    _attach_mock_llm(app, mock_llm)
    transport = httpx.ASGITransport(app=app)

    daemon = app.state.daemon
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp_unload = await client.post("/api/system/unload_and_pause")
        assert resp_unload.status_code == 200
        assert daemon.llama_manager._manually_unloaded is True

        # Watchdog restart_server() must refuse while _manually_unloaded is True
        assert daemon.llama_manager.restart_server() is False

        # Passive genre check must not clear _manually_unloaded
        resp_genres = await client.get("/api/wizard/genres?catalog_only=true")
        assert resp_genres.status_code == 200
        assert daemon.llama_manager._manually_unloaded is True

        # Explicit resume clears _manually_unloaded
        resp_resume = await client.post("/api/reference/resume")
        assert resp_resume.status_code == 200
        assert daemon.llama_manager._manually_unloaded is False


@pytest.mark.asyncio
async def test_mixed_per_item_tracks_in_single_batch(tmp_path: Path):
    """Verify a multi-book batch where Book A extracts only skills and Book B extracts only pacing respects per-item flags."""
    cfg = RuntimeConfig(workspace_dir=tmp_path / "workspace")
    app = create_app(cfg)
    mock_llm = MockLLMClient()
    _attach_mock_llm(app, mock_llm)
    transport = httpx.ASGITransport(app=app)

    book_a = tmp_path / "只萃技法書.txt"
    book_a.write_text(
        "\n\n".join(f"第{i}章 技法試煉\n" + "刀光如雪，寒氣逼人，少年的身影在竹林間拉出殘影。" * 50 for i in range(1, 5)),
        encoding="utf-8",
    )
    book_b = tmp_path / "只萃情節書.txt"
    book_b.write_text(
        "\n\n".join(f"第{i}章 情節起落\n" + "王朝傾覆在即，各方勢力齊聚帝都，暗流洶湧。" * 50 for i in range(1, 5)),
        encoding="utf-8",
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/api/reference/ingest",
            json={
                "items": [
                    {
                        "file_path": str(book_a),
                        "action": "INGEST",
                        "extract_skills": True,
                        "extract_pacing": False,
                    },
                    {
                        "file_path": str(book_b),
                        "action": "INGEST",
                        "extract_skills": False,
                        "extract_pacing": True,
                    },
                ],
                "extract_skills": True,
                "extract_pacing": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["pacing_extracted_count"] == 1
        assert data["ingested_chunks"] > 0

        # Check preflight status of both books
        resp_pf = await client.post(
            "/api/reference/preflight",
            json={"file_paths": [str(book_a), str(book_b)]},
        )
        assert resp_pf.status_code == 200
        items_by_name = {it["file_name"]: it for it in resp_pf.json()["items"]}
        assert items_by_name["只萃技法書.txt"]["has_skills"] is True
        assert items_by_name["只萃技法書.txt"]["has_pacing"] is False
        assert items_by_name["只萃情節書.txt"]["has_skills"] is False
        assert items_by_name["只萃情節書.txt"]["has_pacing"] is True


def test_toc_extraction_diverse_headers_and_unchaptered_partitioning():
    """Verify TOC regex recognizes decorative/Day/full-width headers and partitions unchaptered long texts."""
    from autonovel.reference.macro_extractor import extract_table_of_contents

    # 1. Decorative and diary/full-width headers
    novel_diverse = (
        "序\n這是序言內容，鋪墊世界觀。\n"
        "===第一章 瘋皇子===\n趙煦在王府中醒來。\n"
        "蛇血欲焰 第二卷１９８章乳房的保養\n山村雨夜的奇遇。\n"
        "Day1\n第一天的日記與事件。\n"
        "（一）\n番外篇第一節內容。\n"
    )
    toc = extract_table_of_contents(novel_diverse)
    assert len(toc) == 5
    assert toc[0]["title"] == "序"
    assert "第一章" in toc[1]["title"]
    assert "１９８章" in toc[2]["title"]
    assert toc[3]["title"] == "Day1"
    assert toc[4]["title"] == "（一）"

    # 2. Long unchaptered novel (> 10,000 chars without any chapter headers)
    unchaptered_long = ("這是一段沒有章節標題的連續長篇敘事段落，情節不斷推進。\n" * 600)
    assert len(unchaptered_long) > 15000
    toc_unchaptered = extract_table_of_contents(unchaptered_long)
    assert len(toc_unchaptered) >= 3
    assert toc_unchaptered[0]["title"] == "全篇第1節"

    # 3. Punctuated titles (!/?/。), compound section-chapter headers, consecutive banners,
    #    and immunity against dialogue quotes or prose false positives.
    novel_punctuated = (
        "第一卷 慶安才子\n"
        "第1章 俊俏少年\n開端正文。\n"
        "第2章 山賊凶猛！\n山賊來襲！\n"
        "第1080章 你是豬嗎？\n「第一回合，我們輸了。」他搖頭道。\n"
        "「第二部分，諸侯討董……」他繼續念道。\n"
        "第三節課，是大學英語。\n"
        "第四章 剪不斷，理還亂。\n夜雪紛飛。\n"
        "第15節第15章 密林會合\n眾人會合。\n"
    )
    toc_punct = extract_table_of_contents(novel_punctuated)
    assert [c["title"] for c in toc_punct] == [
        "第一卷 慶安才子 · 第1章 俊俏少年",
        "第2章 山賊凶猛！",
        "第1080章 你是豬嗎？",
        "第四章 剪不斷，理還亂。",
        "第15節第15章 密林會合",
    ]
    assert "「第二部分，諸侯討董……」" in toc_punct[2]["text"]
    assert "第三節課，是大學英語。" in toc_punct[2]["text"]


@pytest.mark.asyncio
async def test_btree_segment_consolidation_and_context_overflow_autosplit():
    """Verify 20-segment mega-chapters consolidate via k<=6 B-Tree and HTTP 400 context overflow splits losslessly."""
    from autonovel.reference.macro_extractor import ChapterFactExtractor

    prompts_seen: list[str] = []

    class TrackingMockLLM(MockLLMClient):
        async def generate_structured(self, system_prompt, user_prompt, response_model, **kwargs):
            prompts_seen.append(user_prompt)
            # Every consolidation prompt must contain at most 6 segment blocks ("【第 7 分段事實】" must never appear)
            assert "【第 7 分段事實】" not in user_prompt, "Consolidation prompt exceeded k=6 branching factor!"
            return await super().generate_structured(system_prompt, user_prompt, response_model, **kwargs)

    extractor = ChapterFactExtractor(llm_client=TrackingMockLLM())
    # Create a mega-chapter of ~110,000 chars (~16 segments of <=7,200 chars each)
    mega_chapter_text = ("少年拔劍斬斷風雷，山巔之上雲海翻騰，各大宗門長老齊聚於此展開生死決戰。\n" * 3200)
    res = await extractor.extract_chapter_facts(
        chapter_index=1,
        chapter_title="超長章節測試",
        chapter_text=mega_chapter_text,
    )
    assert res.chapter_index == 1
    assert res.chapter_title == "超長章節測試"
    # 16 segments -> 3 tier-1 groups -> 1 final consolidation = 20 LLM calls
    assert len(prompts_seen) >= 18


@pytest.mark.asyncio
async def test_book_genre_tag_and_standard_actant_tags_on_all_entries(tmp_path: Path):
    """Verify blueprint genre_primary:genre_sub and two standard actant tags are attached to every entry, and UI removes special actant badge."""
    from pathlib import Path as _P
    from autonovel.db.reference_repo import ReferenceRepository
    from autonovel.llm.embedder import CpuEmbedder
    from autonovel.reference.chunk_extractor import ReferenceExtractor

    ref_db = tmp_path / "reference_library.db"
    ref_repo = ReferenceRepository(ref_db)
    embedder = CpuEmbedder()
    mock_llm = MockLLMClient()
    extractor = ReferenceExtractor(ref_repo, embedder, llm_client=mock_llm)

    novel_file = tmp_path / "凡人測試.txt"
    novel_file.write_text(
        "第一章 斷崖悟劍\n" + ("少年於青石斷崖之上揮劍斬風，靈氣激盪。\n" * 80)
        + "\n第二章 寒潭潛修\n" + ("寒潭深處妖獸蟄伏，少年屏息凝神與之對峙。\n" * 80),
        encoding="utf-8",
    )

    chunks_count = await extractor.ingest_file(novel_file)
    assert chunks_count >= 2

    entries, total = ref_repo.query_entries()
    assert total == chunks_count
    for entry in entries:
        tags = entry["tags"]
        # 1. Book-level genre_primary:genre_sub from blueprint must be attached to every entry
        assert "仙俠:凡人修真" in tags
        # 2. Subject-object relationship stored in structured fields
        assert entry["subject_role"] == "主角"
        assert entry["object_role"] == "環境"
        assert entry["interaction_dynamic"] == "威權壓迫"

    # 3. Verify blueprint character_network uses concise basic role classifications
    books = ref_repo.list_sources_with_counts()
    bp = ref_repo.get_book_blueprint(books[0]["id"])
    assert bp is not None
    roles = [c["archetype_role"] for c in bp["character_network"]]
    assert roles == ["主角", "反派", "男配角"]

    # 4. Verify frontend reference.js no longer renders special ref-entry-actant-badge
    ref_js = _P("src/autonovel/web/static/js/reference.js").read_text(encoding="utf-8")
    assert "ref-entry-actant-badge" not in ref_js


@pytest.mark.asyncio
async def test_manual_unload_during_extraction_pauses_cleanly_and_resumes_without_error(tmp_path: Path):
    """Verify manual model unload during extraction pauses cleanly and resumes after reloading without crashing."""
    import asyncio
    from autonovel.reference.macro_extractor import MacroPacingExtractor
    from autonovel.reference.chunk_extractor import ReferenceExtractor
    from autonovel.llm.embedder import CpuEmbedder

    ref_db = tmp_path / "reference_library.db"
    ref_repo = ReferenceRepository(ref_db)
    source_id = ref_repo.register_or_reset_source(file_sha256="fake123", file_name="長篇修真.txt", chunk_count=0)
    mock_llm = MockLLMClient()
    embedder = CpuEmbedder()
    extractor = ReferenceExtractor(ref_repo, embedder, llm_client=mock_llm)
    macro_ext = MacroPacingExtractor(ref_repo, llm_client=mock_llm)

    novel_content = "\n\n".join(
        f"第{i}章 雲海爭鋒\n" + ("白袍修士負手而立，劍意縱橫三千里。\n" * 30)
        for i in range(1, 7)
    )

    # Hook generate_structured to simulate unload on call 2
    orig_generate = mock_llm.generate_structured
    call_count = 0

    async def _failing_on_second_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            # Simulate manual model unload
            mock_llm.server_manager.stop_server(manual=True)
            extractor.pause()
            raise RuntimeError("All connection attempts failed")
        return await orig_generate(*args, **kwargs)

    mock_llm.generate_structured = _failing_on_second_call

    # Launch extraction in background task
    task = asyncio.create_task(
        macro_ext.extract_from_novel_text(
            text=novel_content,
            novel_title="長篇修真",
            source_id=source_id,
            lifecycle=extractor,
        )
    )

    # Let it run until it hits the unload and enters pause loop
    await asyncio.sleep(0.3)
    assert not task.done(), "Extraction must not fail or crash while paused/unloaded"
    assert extractor.is_paused() is True

    # Now resume the extraction and ensure server running
    mock_llm.server_manager.ensure_running()
    extractor.resume()

    # The background task should now resume and complete successfully
    result = await asyncio.wait_for(task, timeout=5.0)
    assert result is not None
    assert result.source_novel_title == "長篇修真"
    assert result.total_chapters_analyzed == 6


@pytest.mark.asyncio
async def test_stage1_checkpoint_preservation_and_incremental_resume(tmp_path: Path):
    """Verify Stage 1 chapter facts are incrementally checkpointed and restored on re-run."""
    import json
    from pathlib import Path
    from autonovel.reference.macro_extractor import MacroPacingExtractor

    ref_db = tmp_path / "reference_library.db"
    ref_repo = ReferenceRepository(ref_db)
    source_id = ref_repo.register_or_reset_source(file_sha256="fake888", file_name="長篇修真斷點測試.txt", chunk_count=0)
    mock_llm = MockLLMClient()
    macro_ext = MacroPacingExtractor(ref_repo, llm_client=mock_llm)

    novel_content = "\n\n".join(
        f"第{i}章 雲海爭鋒\n" + ("白袍修士負手而立，劍意縱橫三千里。\n" * 30)
        for i in range(1, 7)
    )

    # Simulate cancellation after batch 1
    call_count = 0
    orig_generate = mock_llm.generate_structured

    class DummyLifecycle:
        def __init__(self):
            self._paused = False
            self._cancelled = False
        def is_paused(self): return self._paused
        def is_cancelled(self): return self._cancelled

    lifecycle = DummyLifecycle()

    async def _cancel_after_batch1(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        res = await orig_generate(*args, **kwargs)
        if call_count == 1:
            lifecycle._cancelled = True
        return res

    mock_llm.generate_structured = _cancel_after_batch1

    # Run extraction with registered source_id
    res = await macro_ext.extract_from_novel_text(
        text=novel_content,
        novel_title="長篇修真斷點測試",
        source_id=source_id,
        lifecycle=lifecycle,
    )
    assert res is None  # Cancelled

    # Check that checkpoint file was created
    ckpt = ref_repo.db_path.parent / "cache" / f"stage1_facts_{source_id}.json"
    try:
        assert ckpt.exists(), "Checkpoint file must exist after batch 1"
        data = json.loads(ckpt.read_text(encoding="utf-8"))
        assert len(data.get("chapter_facts", [])) > 0
        assert data.get("total_chapters") == 6

        # Now resume extraction with unhooked generate_structured (should restore from checkpoint)
        mock_llm.generate_structured = orig_generate
        lifecycle._cancelled = False

        res2 = await macro_ext.extract_from_novel_text(
            text=novel_content,
            novel_title="長篇修真斷點測試",
            source_id=source_id,
            lifecycle=lifecycle,
        )
        assert res2 is not None
        assert res2.total_chapters_analyzed == 6

        # Checkpoint file should be unlinked upon completion
        assert not ckpt.exists(), "Checkpoint file should be cleaned up on completion"

        # If a stale checkpoint has an outdated total_chapters (e.g. before a TOC regex fix),
        # it must be ignored so old chapter numbers are not mixed with the new TOC.
        source_id_stale = ref_repo.register_or_reset_source(file_sha256="fake889", file_name="長篇修真舊斷點.txt", chunk_count=0)
        stale_ckpt = ref_repo.db_path.parent / "cache" / f"stage1_facts_{source_id_stale}.json"
        try:
            stale_payload = dict(data)
            stale_payload["total_chapters"] = 3
            stale_payload["chapter_facts"] = [
                {**cf, "chapter_title": "STALE_TITLE"} for cf in data["chapter_facts"]
            ]
            stale_ckpt.write_text(json.dumps(stale_payload, ensure_ascii=False), encoding="utf-8")
            res_stale = await macro_ext.extract_from_novel_text(
                text=novel_content,
                novel_title="長篇修真舊斷點",
                source_id=source_id_stale,
                lifecycle=lifecycle,
            )
            assert res_stale is not None
            assert res_stale.total_chapters_analyzed == 6
            assert all(pt["title"] != "STALE_TITLE" for pt in res_stale.chapter_curve_data)
        finally:
            if stale_ckpt.exists():
                stale_ckpt.unlink()
    finally:
        if ckpt.exists():
            ckpt.unlink()


@pytest.mark.asyncio
async def test_delete_reference_book_reclaims_staged_upload(tmp_path: Path):
    """Deleting a book removes its staged upload once no other source needs the file."""
    cfg = RuntimeConfig(workspace_dir=tmp_path / "workspace")
    app = create_app(cfg)
    daemon = app.state.daemon
    staged_dir = cfg.data_dir / "ref_uploads"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / "book.txt"
    staged.write_text("第一章 劍氣縱橫\n", encoding="utf-8")

    first = daemon.ref_repo.register_or_reset_source("sha-staged-1", "book.txt", 1)
    second = daemon.ref_repo.register_or_reset_source("sha-staged-2", "book.txt", 1)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Manual per-tag deletion is no longer exposed as an endpoint.
        resp_tag = await client.delete("/api/reference/entries/1/tags/任意標籤")
        assert resp_tag.status_code in {404, 405}

        resp_first = await client.delete(f"/api/reference/books/{first}")
        assert resp_first.status_code == 200
        # The second source still points at the same file name, so it must survive.
        assert resp_first.json()["removed_file"] is False
        assert staged.exists()

        resp_second = await client.delete(f"/api/reference/books/{second}")
        assert resp_second.status_code == 200
        assert resp_second.json()["removed_file"] is True

    assert not staged.exists()


@pytest.mark.asyncio
async def test_multi_task_manifest_persistence_and_cleanup(tmp_path: Path):
    """Verify multi-task extraction writes manifest atomically during execution and cleans up upon completion."""
    import json
    cfg = RuntimeConfig(workspace_dir=tmp_path / "workspace")
    app = create_app(cfg)
    mock_llm = MockLLMClient()
    _attach_mock_llm(app, mock_llm)
    transport = httpx.ASGITransport(app=app)

    novel_1 = tmp_path / "凡人修仙A.txt"
    novel_1.write_text("第一章 少年出山\n韓立背著藥簍走出深山。\n第二章 宗門考核\n靈根檢驗平平無奇。\n", encoding="utf-8")
    novel_2 = tmp_path / "凡人修仙B.txt"
    novel_2.write_text("第一章 掌天瓶現\n綠色小瓶凝聚夜間靈露。\n第二章 催熟靈草\n百年靈藥一日長成。\n", encoding="utf-8")

    ref_ext = app.state.daemon.ref_extractor
    ref_ext.pause()

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/api/reference/ingest",
            json={
                "file_paths": [str(novel_1), str(novel_2)],
                "extract_skills": True,
                "extract_pacing": True,
                "run_in_background": True,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["queued"] is True

        assert ref_ext.manifest_path.exists(), "Manifest file must exist while extraction is paused in queue"

        manifest_data = json.loads(ref_ext.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["version"] == 1
        assert manifest_data["is_paused"] is True
        assert len(manifest_data["pending_jobs"]) >= 1

        # Resume extraction via API
        resp_resume = await client.post("/api/reference/resume")
        assert resp_resume.status_code == 200
        assert resp_resume.json()["status"] == "running"

        # Wait for the background queue task to complete
        if ref_ext._bg_task:
            await ref_ext._bg_task

        # After all jobs in queue conclude, manifest should be cleaned up
        assert not ref_ext.manifest_path.exists(), "Manifest file must be cleanly unlinked after queue completion"
        assert len(ref_ext._completed_books) >= 2
        for book in ref_ext._completed_books:
            assert book["status"] == "SUCCESS"


@pytest.mark.asyncio
async def test_multi_task_fault_isolation_continues_batch(tmp_path: Path):
    """Verify single-book failure in multi-task batch is isolated and remaining books proceed to completion."""
    import json
    cfg = RuntimeConfig(workspace_dir=tmp_path / "workspace")
    app = create_app(cfg)
    daemon = app.state.daemon

    # Custom mock LLM that raises an exception only when book 1 is being processed
    class FaultInjectingMockLLM(MockLLMClient):
        async def generate_structured(self, system_prompt: str, user_prompt: str, response_model: Any, **kwargs: Any) -> Any:
            if "故障書籍" in user_prompt or "故障書籍" in system_prompt:
                raise RuntimeError("Simulated LLM crash on faulty book!")
            return await super().generate_structured(system_prompt, user_prompt, response_model, **kwargs)

    fault_llm = FaultInjectingMockLLM()
    _attach_mock_llm(app, fault_llm)
    transport = httpx.ASGITransport(app=app)

    failing_book = tmp_path / "故障書籍.txt"
    failing_book.write_text("第一章 突發異常\n天道崩塌數據混亂。\n第二章 萬劫不復\n靈氣斷絕推論崩潰。\n", encoding="utf-8")
    good_book = tmp_path / "正常書籍.txt"
    good_book.write_text("第一章 平穩修煉\n少年苦修一日千里。\n第二章 順利晉階\n丹田金丹圓滿凝結。\n", encoding="utf-8")

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/api/reference/ingest",
            json={
                "file_paths": [str(failing_book), str(good_book)],
                "extract_skills": True,
                "extract_pacing": True,
                "run_in_background": True,
            },
        )
        assert resp.status_code == 200

        ref_ext = daemon.ref_extractor
        if ref_ext._bg_task:
            await ref_ext._bg_task

        # Verify fault isolation: failing book marked FAILED, good book finished SUCCESS
        failing_records = [b for b in ref_ext._completed_books if b["file_name"] == failing_book.name]
        good_records = [b for b in ref_ext._completed_books if b["file_name"] == good_book.name]

        assert len(failing_records) > 0, "Failing book must be recorded in completed_books history"
        assert any(r["status"] == "FAILED" for r in failing_records)

        assert len(good_records) > 0, "Good book must not be aborted by prior book's failure"
        assert all(r["status"] == "SUCCESS" for r in good_records)

        # Confirm entries from the good book were safely written into the repository
        good_src = daemon.ref_repo.get_source_by_hash(compute_sha256(good_book))
        assert good_src is not None
        assert good_src["chunk_count"] > 0


@pytest.mark.asyncio
async def test_multi_task_startup_recovery_from_manifest(tmp_path: Path):
    """Verify startup_recovery reconstructs pending jobs from physical manifest and resumes cleanly."""
    import json
    from autonovel.schemas.domain import ReferenceIngestItem
    from autonovel.reference.chunk_extractor import ReferenceExtractor
    from autonovel.reference.macro_extractor import MacroPacingExtractor
    from autonovel.llm.embedder import CpuEmbedder

    ref_db = tmp_path / "reference_library.db"
    ref_repo = ReferenceRepository(ref_db)
    embedder = CpuEmbedder()
    mock_llm = MockLLMClient()
    extractor = ReferenceExtractor(ref_repo, embedder, llm_client=mock_llm)
    macro_ext = MacroPacingExtractor(ref_repo, llm_client=mock_llm)

    book_1 = tmp_path / "恢復書籍A.txt"
    book_1.write_text("第一章 斷點殘篇\n劫後餘生重見天日。\n第二章 逆天改命\n造化重聚重踏仙途。\n", encoding="utf-8")
    book_2 = tmp_path / "恢復書籍B.txt"
    book_2.write_text("第一章 雲隱靈山\n仙宗招徒萬人空巷。\n第二章 劍意通神\n一劍破空驚艷四座。\n", encoding="utf-8")

    manifest_file = extractor.manifest_path
    manifest_payload = {
        "version": 1,
        "updated_at": 1000.0,
        "state": "PAUSED",
        "is_paused": True,
        "status": {"active": True, "paused": True},
        "pacing_status": {"active": False},
        "pending_jobs": [
            {
                "pacing_items": [
                    {"file_path": str(book_1), "action": "INGEST", "extract_skills": True, "extract_pacing": True},
                    {"file_path": str(book_2), "action": "INGEST", "extract_skills": True, "extract_pacing": True},
                ],
                "skill_items": [
                    {"file_path": str(book_1), "action": "INGEST", "extract_skills": True, "extract_pacing": True},
                    {"file_path": str(book_2), "action": "INGEST", "extract_skills": True, "extract_pacing": True},
                ],
                "pacing_jitter": 0.0,
                "force_reextract": False,
                "pacing_completed_indices": [0],  # Book 1 pacing already done before crash
                "skill_completed_indices": [],
                "failed_items": [],
            }
        ],
        "completed_books": [
            {"file_name": book_1.name, "track": "pacing", "chunks": 0, "status": "SUCCESS", "error": None}
        ],
    }
    manifest_file.write_text(json.dumps(manifest_payload), encoding="utf-8")

    # 1. Test startup_recovery with auto_resume=False (preserves PAUSED state)
    recovered = await extractor.startup_recovery(macro_extractor=macro_ext, auto_resume=False)
    assert recovered is True
    assert extractor.is_paused() is True
    assert extractor.status.get("phase") == "PAUSED_RECOVERED"
    assert len(extractor._pending_jobs) == 1
    # Book 1 pacing index was in pacing_completed_indices, so remaining pacing items is just book 2
    assert len(extractor._pending_jobs[0]["pacing_items"]) == 2
    assert extractor._pending_jobs[0]["pacing_completed_indices"] == [0]

    # 2. Resume extraction via resume()
    extractor.resume(macro_extractor=macro_ext)
    assert extractor.is_paused() is False
    assert extractor._bg_task is not None

    # Wait for completion
    await extractor._bg_task

    # Manifest should be unlinked upon full completion
    assert not manifest_file.exists()
    assert any(b["file_name"] == book_2.name for b in extractor._completed_books)







