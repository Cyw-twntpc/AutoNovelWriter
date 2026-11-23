"""End-to-end tests for FastAPI wizard/review/warroom routes, deep organizer, and 5-format exporter."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from autonovel.config import RuntimeConfig
from autonovel.web.app import create_app


def _attach_mock_llm(app: Any, mock_client: Any) -> None:
    """Wire mock LLMClient into the daemon and its dependent sub-engines."""
    daemon = app.state.daemon
    daemon.llm_client = mock_client
    daemon.qa_engine.llm_client = mock_client
    daemon.blueprint_builder.llm_client = mock_client
    daemon.chapter_pipeline.llm_client = mock_client
    daemon.quality_gate.llm_client = mock_client
    daemon.volume_planner.llm_client = mock_client
    daemon.ref_extractor.llm_client = mock_client
    daemon.retriever.llm_client = mock_client
    daemon.scanner.llm_client = mock_client


def _seed_wizard_history(app: Any) -> None:
    """Append dummy Q&A turns until the blueprint unlock threshold is satisfied."""
    daemon = app.state.daemon
    threshold = daemon.qa_engine.get_unlock_threshold()
    history = daemon.qa_engine.get_qa_history()
    for step in range(len(history) + 1, threshold + 1):
        history.append(
            {
                "step_index": step,
                "question": f"測試提問{step}",
                "aspect": "世界觀",
                "answer": f"測試回答{step}",
            }
        )
    daemon.project_repo.update_project_config(
        qa_history_json=json.dumps(history, ensure_ascii=False)
    )


@pytest.mark.asyncio
async def test_end_to_end_wizard_review_warroom_and_5_format_export(
    tmp_path: Path, mock_llm_client: Any
) -> None:
    """Verify full workflow: Wizard Q&A -> Review Gate -> Serialization -> Deep Organize -> 5-Format Export."""
    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        llama_model_path="models/nonexistent_offline.gguf",
    )
    app = create_app(cfg)
    _attach_mock_llm(app, mock_llm_client)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Setup genre and minimum word count
        r_setup = await client.post(
            "/api/wizard/setup",
            json={
                "title": "《北境劍仙傳》",
                "genre_primary": "仙俠",
                "genre_sub": "古典修仙",
                "min_word_count": 500000,
            },
        )
        assert r_setup.status_code == 200

        # 2. Submit valid Q&A answer, then test contradiction detection
        r_ans1 = await client.post(
            "/api/wizard/answer",
            json={
                "question_title": "第一層基調",
                "selected_answers": ["無魔寫實純歷史凡人無超凡"],
                "custom_note": "",
            },
        )
        assert r_ans1.json()["accepted"] is True

        r_conflict = await client.post(
            "/api/wizard/answer",
            json={
                "question_title": "第二層能力",
                "selected_answers": ["御劍飛行施展修仙法術與天地靈氣"],
                "custom_note": "",
            },
        )
        assert r_conflict.json()["accepted"] is False
        assert r_conflict.json()["contradiction"]["has_contradiction"] is True

        # 3. Complete blueprint -> transition to REVIEW_GATE
        _seed_wizard_history(app)
        r_comp = await client.post("/api/wizard/complete")
        assert r_comp.status_code == 200
        assert r_comp.json()["project_config"]["pipeline_state"] == "REVIEW_GATE"

        # 4. Update Review Gate settings and lock to start
        r_upd = await client.post(
            "/api/review/update",
            json={"title": "《太初劍典》"},
        )
        assert r_upd.json()["project_config"]["title"] == "《太初劍典》"

        # 5. Execute 2 chapters via /api/warroom/step
        for expected_ch in (1, 2):
            r_step = await client.post("/api/warroom/step")
            assert r_step.status_code == 200
            res_data = r_step.json()["result"]
            assert res_data["chapter_num"] == expected_ch
            assert res_data["word_count"] >= 1800
            assert res_data["literary_score"] >= 80.0

        # 6. Verify dashboard and chapter reader (no summary)
        r_dash = await client.get("/api/warroom/dashboard")
        assert r_dash.status_code == 200
        dash = r_dash.json()
        assert dash["summary_stats"]["completed_chapters"] == 2
        assert len(dash["metrics_30"]) == 2

        r_ch1 = await client.get("/api/warroom/chapter/1")
        assert r_ch1.status_code == 200
        assert "林墨" in r_ch1.json()["chapter"]["final_content"]

        # 7. Test latest chapter rewrite
        r_rew = await client.post("/api/warroom/rewrite_latest")
        assert r_rew.status_code == 200
        assert r_rew.json()["chapter_num"] == 2

        # 8. Test full-series 5-format export (MD, TXT, HTML, EPUB, DOCX) in SINGLE_FILE and PER_VOLUME modes
        r_exp_single = await client.post(
            "/api/export",
            json={
                "scope_type": "ALL",
                "formats": ["md", "txt", "html", "epub", "docx"],
                "merge_mode": "SINGLE_FILE",
            },
        )
        assert r_exp_single.status_code == 200
        files_single = [Path(p) for p in r_exp_single.json()["files"]]
        assert len(files_single) == 5
        for fpath in files_single:
            assert fpath.exists() and fpath.stat().st_size > 0
            if fpath.suffix in (".epub", ".docx"):
                assert zipfile.is_zipfile(fpath)

        r_exp_vol = await client.post(
            "/api/export",
            json={
                "scope_type": "VOLUME",
                "volume_num": 1,
                "formats": ["md", "epub"],
                "merge_mode": "PER_VOLUME",
            },
        )
        assert r_exp_vol.status_code == 200
        assert len(r_exp_vol.json()["files"]) == 2


def test_deep_organizer_archives_inactive_minor_characters(tmp_path: Path) -> None:
    """Verify DeepOrganizer archives minor NPCs unseen for >=30 chapters without pending foreshadowing."""
    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        llama_model_path="models/nonexistent_offline.gguf",
    )
    app = create_app(cfg)
    daemon = app.state.daemon
    repo = daemon.project_repo

    repo.fetch_all(
        """
        INSERT INTO characters (name, aliases, role_type, status, last_appeared_chapter)
        VALUES
            ('路人甲', '["甲兄", "甲兄"]', 'SUPPORTING', 'ALIVE', 1),
            ('重要伏筆人', '[]', 'SUPPORTING', 'ALIVE', 1),
            ('林墨', '[]', 'PROTAGONIST', 'ALIVE', 1);
        """
    )
    repo.fetch_all(
        """
        INSERT INTO foreshadowing (description, scope, planted_chapter, expected_span, status, related_character_ids)
        VALUES ('重要伏筆人的身世', 'MAIN', 1, 100, 'PENDING', '["重要伏筆人"]');
        """
    )
    maint = daemon.deep_organizer.run_maintenance(current_chapter_num=35)
    assert "路人甲" in maint["archived_characters"]
    assert "重要伏筆人" not in maint["archived_characters"]
    assert "林墨" not in maint["archived_characters"]


@pytest.mark.asyncio
async def test_review_gate_full_edit_regenerate_acceptance_and_background_ingest(
    tmp_path: Path, mock_llm_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify 5-tab Review Gate editing, AI section regeneration, acceptance report, and per-file ingest."""
    from scripts.setup_environment import inspect_environment

    monkeypatch.setattr(
        "scripts.setup_environment.download_bge_small_onnx",
        lambda *args, **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        "scripts.setup_environment.download_gguf_model",
        lambda *args, **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        "scripts.download_llama_server.download_and_extract_llama_server",
        lambda *args, **kwargs: Path("bin/llama-server.exe"),
    )

    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        llama_model_path="models/nonexistent_offline.gguf",
    )
    env_info = inspect_environment(cfg)
    assert "bge_onnx_ready" in env_info

    app = create_app(cfg)
    _attach_mock_llm(app, mock_llm_client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        _seed_wizard_history(app)
        await client.post("/api/wizard/complete")

        # 1. Test full 5-tab editing in Review Gate
        r_upd = await client.post(
            "/api/review/update",
            json={
                "hierarchy_levels": [
                    {"level_num": 1, "level_name": "第一階·鍛體", "description": "淬鍊筋骨"},
                    {"level_num": 2, "level_name": "第二階·凝氣", "description": "氣海初開"},
                ],
                "characters": [
                    {
                        "name": "顧長歌",
                        "role_type": "PROTAGONIST",
                        "rank_level": 1,
                        "faction_name": "天劍閣",
                        "location_name": "東荒",
                        "description": "劍道奇才",
                    }
                ],
                "factions": ["天劍閣", "血魔宗"],
                "locations": ["東荒", "北冥"],
                "chapters": [
                    {"chapter_num": 1, "title": "第1章 劍鳴東荒", "beats_json": "[]"}
                ],
            },
        )
        assert r_upd.status_code == 200
        upd_data = r_upd.json()
        assert upd_data["hierarchy_levels"][0]["level_name"] == "第一階·鍛體"
        assert upd_data["characters"][0]["name"] == "顧長歌"
        assert upd_data["chapters"][0]["title"] == "第1章 劍鳴東荒"

        # 2. Test partial regeneration of a single section ("world") without overwriting edited characters
        r_regen = await client.post("/api/review/regenerate", json={"section": "world"})
        assert r_regen.status_code == 200
        regen_data = r_regen.json()
        assert len(regen_data["hierarchy_levels"]) >= 4
        assert regen_data["characters"][0]["name"] == "顧長歌"

        # 3. Test per-file action ("INGEST" vs "SKIP") in reference ingestion
        book_a = tmp_path / "book_a.txt"
        book_b = tmp_path / "book_b.txt"
        book_a.write_text("第一章 劍氣如霜\n寒風凜冽，劍光一閃。\n" * 60, encoding="utf-8")
        book_b.write_text("第一章 略過之書\n此書應被跳過不入庫。\n" * 60, encoding="utf-8")
        r_ing = await client.post(
            "/api/reference/ingest",
            json={
                "items": [
                    {"file_path": str(book_a), "action": "INGEST"},
                    {"file_path": str(book_b), "action": "SKIP"},
                ],
                "run_in_background": False,
            },
        )
        assert r_ing.status_code == 200
        assert r_ing.json()["ingested_chunks"] >= 1
        assert r_ing.json()["library_stats"]["sources"] == 1

        r_gen = await client.get("/api/wizard/genres")
        assert r_gen.status_code == 200
        dyn_xianxia = r_gen.json()["genres"].get("仙俠", [])
        assert "劍道殺伐" in dyn_xianxia
        assert "凡人修真" in dyn_xianxia

        r_regen_q = await client.post("/api/wizard/regenerate_question")
        assert r_regen_q.status_code == 200
        assert len(r_regen_q.json()["next_question"]["options"]) == 4

        # 4. Run 1 chapter and check acceptance report + telemetry
        await client.post("/api/warroom/step")
        r_acc = await client.get("/api/warroom/acceptance_report")
        assert r_acc.status_code == 200
        acc = r_acc.json()
        assert acc["hard_conflict_rate"] == 0.0
        assert acc["timeline_consistency_rate"] == 1.0
        assert acc["mean_literary_score"] >= 80.0

        r_dash = await client.get("/api/warroom/dashboard")
        stats = r_dash.json()["summary_stats"]
        assert stats["tokens_per_sec"] > 0
        assert stats["last_chapter_duration_sec"] > 0

        # 5. Test REVISING status rollback on crash recovery
        app.state.daemon.project_repo.save_staging_chapter(
            2, "第2章", "修訂中草稿", status="REVISING"
        )
        rolled = await app.state.daemon.startup_recovery(auto_resume=False)
        assert 2 in rolled

        # 6. Test UI Environment Check & Model Download API endpoints
        r_env = await client.get("/api/system/env_status")
        assert r_env.status_code == 200
        env_payload = r_env.json()
        assert "embedder_mode" in env_payload
        assert "download_state" in env_payload

        r_deploy = await client.post(
            "/api/system/deploy_all",
            json={"run_in_background": True},
        )
        assert r_deploy.status_code == 200
        deploy_payload = r_deploy.json()
        assert "download_state" in deploy_payload
        assert "env_status" in deploy_payload

        r_llama = await client.post("/api/system/start_llama")
        assert r_llama.status_code == 200
        assert "env_status" in r_llama.json()

        r_bad_dl = await client.post(
            "/api/system/download_model",
            json={"target": "invalid_model", "run_in_background": False},
        )
        assert r_bad_dl.status_code == 400


@pytest.mark.asyncio
async def test_frontend_dom_ids_and_event_handlers_integrity(
    tmp_path: Path, mock_llm_client: Any
) -> None:
    """Verify every getElementById in app.js exists in index.html and every onclick handler is defined."""
    import re

    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        llama_model_path="models/nonexistent_offline.gguf",
    )
    app = create_app(cfg)
    _attach_mock_llm(app, mock_llm_client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        r_html = await client.get("/")
        r_css = await client.get("/static/styles.css")
        assert r_html.status_code == 200
        assert r_css.status_code == 200

        html_text = r_html.text

        # Verify all modular JS files are served correctly and loaded in index.html
        modular_files = [
            "state.js", "projects.js", "wizard.js", "review_workbench.js",
            "warroom.js", "environment.js", "reference.js", "main.js"
        ]
        combined_modular_text = ""
        for m_file in modular_files:
            assert f"/static/js/{m_file}" in html_text, f"Modular script not included in index.html: {m_file}"
            r_mod = await client.get(f"/static/js/{m_file}")
            assert r_mod.status_code == 200, f"Failed to fetch modular script: {m_file}"
            assert len(r_mod.text) > 0
            combined_modular_text += "\n" + r_mod.text

        html_ids = set(re.findall(r'id="([^"]+)"', html_text))
        js_static_ids = set(re.findall(r'getElementById\("([^"$]+)"\)', combined_modular_text))
        missing_ids = js_static_ids - html_ids
        assert not missing_ids, f"Missing DOM IDs in index.html: {missing_ids}"

        inline_calls = set(
            re.findall(r'on(?:click|change)="([a-zA-Z0-9_]+)\(', html_text)
        )
        modular_funcs = set(re.findall(r"function\s+([a-zA-Z0-9_]+)\s*\(", combined_modular_text))
        missing_mod_funcs = inline_calls - modular_funcs
        assert not missing_mod_funcs, f"Missing JS functions in modular scripts: {missing_mod_funcs}"

        # Verify studio sub-panes are direct sibling divs and not nested inside each other
        wiz_idx = html_text.find('id="view-wizard"')
        rev_idx = html_text.find('id="view-review"')
        war_idx = html_text.find('id="view-warroom"')
        assert wiz_idx != -1 and rev_idx != -1 and war_idx != -1
        assert wiz_idx < rev_idx < war_idx
        # Check that view-wizard has its closing tag before view-review begins
        between_wiz_rev = html_text[wiz_idx:rev_idx]
        assert between_wiz_rev.count("<div") == between_wiz_rev.count("</div>")
        between_rev_war = html_text[rev_idx:war_idx]
        assert between_rev_war.count("<div") == between_rev_war.count("</div>")

        _seed_wizard_history(app)
        await client.post("/api/wizard/complete")
        r_rev = await client.get("/api/review/state")
        chars = r_rev.json()["characters"]
        assert chars and chars[0]["faction_name"] == "青雲劍宗"
        assert chars[0]["location_name"] == "北境寒淵"


@pytest.mark.asyncio
async def test_no_fallback_mode_when_offline_requires_env_setup(tmp_path: Path) -> None:
    """Verify system strictly rejects virtual fallback questions when AI model is offline."""
    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        llama_model_path="models/nonexistent_offline.gguf",
    )
    app = create_app(cfg)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. GET /api/wizard/genres reports OFFLINE mode and no questions generated
        r_init = await client.get("/api/wizard/genres")
        assert r_init.status_code == 200
        init_data = r_init.json()
        assert init_data["ai_mode"] == "OFFLINE"
        assert init_data["next_question"] is None

        # 2. POST /api/wizard/setup strictly rejects virtual generation with 400 & needs_env_setup
        r_setup = await client.post(
            "/api/wizard/setup",
            json={
                "title": "《無模型測試》",
                "genre_primary": "仙俠",
                "genre_sub": "古典修仙",
                "min_word_count": 5000000,
            },
        )
        assert r_setup.status_code == 400
        setup_data = r_setup.json()
        assert setup_data["error"] == "AI_MODEL_UNAVAILABLE"
        assert setup_data["needs_env_setup"] is True
        assert setup_data["ai_mode"] == "OFFLINE"
        assert "環境設置" in setup_data["detail"]

        # 3. POST /api/wizard/regenerate_question also rejects virtual regeneration
        r_regen = await client.post("/api/wizard/regenerate_question")
        assert r_regen.status_code == 400
        assert r_regen.json()["needs_env_setup"] is True

        # 4. POST /api/wizard/answer rejects virtual next question
        r_ans = await client.post(
            "/api/wizard/answer",
            json={
                "question_title": "測試標題",
                "selected_answers": ["選項A"],
                "custom_note": "",
            },
        )
        assert r_ans.status_code == 400
        assert r_ans.json()["needs_env_setup"] is True

        # 5. POST /api/wizard/complete rejects virtual blueprint
        r_comp = await client.post("/api/wizard/complete")
        assert r_comp.status_code == 400
        assert r_comp.json()["needs_env_setup"] is True

        # 6. POST /api/review/lock_and_start rejects starting serialization without AI model
        app.state.daemon.project_repo.update_project_config(style_contract_text="已有契約書")
        r_lock = await client.post("/api/review/lock_and_start")
        assert r_lock.status_code == 400
        assert r_lock.json()["needs_env_setup"] is True

        # 7. POST /api/warroom/resume rejects starting serialization without AI model
        r_resume = await client.post("/api/warroom/resume")
        assert r_resume.status_code == 400
        assert r_resume.json()["needs_env_setup"] is True

        # 8. POST /api/warroom/step rejects generating single chapter without AI model
        r_step = await client.post("/api/warroom/step")
        assert r_step.status_code == 400
        assert r_step.json()["needs_env_setup"] is True

        # 9. POST /api/warroom/rewrite_latest rejects rewriting chapter without AI model
        r_rew = await client.post("/api/warroom/rewrite_latest")
        assert r_rew.status_code == 400
        assert r_rew.json()["needs_env_setup"] is True

        # 10. POST /api/reference/ingest rejects chunk extraction without AI model
        r_ing = await client.post(
            "/api/reference/ingest",
            json={"items": [{"file_path": "sample.txt", "action": "INGEST"}]},
        )
        assert r_ing.status_code == 400
        assert r_ing.json()["needs_env_setup"] is True


@pytest.mark.asyncio
async def test_manual_snapshot_guard_in_wizard_stage(
    tmp_path: Path, mock_llm_client: Any
) -> None:
    """Verify manual snapshot endpoint blocks creation during WIZARD stage and succeeds once locked."""
    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        llama_model_path="models/nonexistent_offline.gguf",
    )
    app = create_app(cfg)
    _attach_mock_llm(app, mock_llm_client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # In initial WIZARD stage: POST /api/warroom/snapshot/create should return 400
        r_blocked = await client.post("/api/warroom/snapshot/create")
        assert r_blocked.status_code == 400
        assert "創世問答階段" in r_blocked.json()["detail"]

        # Complete wizard and advance to REVIEW_GATE
        _seed_wizard_history(app)
        await client.post("/api/wizard/complete")
        r_allowed = await client.post("/api/warroom/snapshot/create")
        assert r_allowed.status_code == 200
        assert "snapshot_id" in r_allowed.json()
