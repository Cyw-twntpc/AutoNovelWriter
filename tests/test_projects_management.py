"""Tests for multi-project management, isolation, hot switching, and REST endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from autonovel.config import RuntimeConfig
from autonovel.db.project_manager import ProjectManager
from autonovel.web.app import create_app


def test_project_manager_crud_and_isolation(tmp_path: Path) -> None:
    """Verify ProjectManager handles creation, switching, renaming, deletion, and directory isolation."""
    cfg = RuntimeConfig(workspace_dir=tmp_path)
    pm = ProjectManager(cfg)

    # 1. Initial setup creates default project
    init_active = pm.ensure_initialized()
    assert init_active.startswith("proj_")
    projects = pm.list_projects()
    assert len(projects) == 1
    assert projects[0]["project_id"] == init_active
    assert projects[0]["is_active"] is True
    assert "volumes_count" in projects[0]
    assert "chapters_count" in projects[0]
    assert "current_words" in projects[0]
    assert "current_stage" in projects[0]
    import re
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", projects[0]["created_at"])
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", projects[0]["updated_at"])

    # 2. Create second novel project
    p2 = pm.create_project(
        title="《星際巡航記》",
        genre_primary="科幻",
        genre_sub="星際機甲",
        min_word_count=2000000,
    )
    p2_id = p2["project_id"]
    assert pm.get_active_project_id() == p2_id
    p2_paths = pm.get_project_paths(p2_id)
    assert p2_paths["project_db_path"].exists()
    assert p2_paths["snapshot_dir"].exists()

    projects = pm.list_projects()
    assert len(projects) == 2
    active_entry = next(p for p in projects if p["is_active"])
    assert active_entry["project_id"] == p2_id
    assert active_entry["title"] == "《星際巡航記》"

    # 3. Rename second project
    renamed = pm.rename_project(p2_id, "《銀河遠征史詩》")
    assert renamed["title"] == "《銀河遠征史詩》"
    projects = pm.list_projects()
    p2_entry = next(p for p in projects if p["project_id"] == p2_id)
    assert p2_entry["title"] == "《銀河遠征史詩》"

    # 4. Switch back to first project
    pm.switch_project(init_active)
    assert pm.get_active_project_id() == init_active
    projects = pm.list_projects()
    assert projects[0]["project_id"] == init_active
    assert projects[0]["is_active"] is True

    # 5. Delete second project
    remaining_active = pm.delete_project(p2_id)
    assert remaining_active == init_active
    assert not p2_paths["project_dir"].exists()
    projects = pm.list_projects()
    assert len(projects) == 1
    assert projects[0]["project_id"] == init_active

    # 6. Delete the last remaining novel project down to 0 (True Empty State)
    last_active = pm.delete_project(init_active)
    assert last_active == ""
    assert pm.get_active_project_id() == ""
    assert pm.list_projects() == []

    # 7. Create a new novel project from zero-state
    p3 = pm.create_project(
        title="《重生之神級系統》",
        genre_primary="都市",
        genre_sub="系統爽文",
        min_word_count=1000000,
    )
    assert p3["project_id"] == pm.get_active_project_id()
    assert len(pm.list_projects()) == 1


@pytest.mark.asyncio
async def test_projects_api_routes(tmp_path: Path, mock_llm_client: Any) -> None:
    """Verify REST endpoints for /api/projects (list, create, switch, rename, delete)."""
    cfg = RuntimeConfig(workspace_dir=tmp_path)
    app = create_app(cfg)
    app.state.daemon.llm_client = mock_llm_client

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Initial list has 1 default project
        r_list = await client.get("/api/projects")
        assert r_list.status_code == 200
        data = r_list.json()
        assert data["total_projects"] == 1
        default_id = data["active_project_id"]

        # 2. Create new project via API
        r_create = await client.post(
            "/api/projects/create",
            json={
                "title": "《大荒古經》",
                "genre_primary": "仙俠",
                "genre_sub": "凡人修真",
                "min_word_count": 3000000,
            },
        )
        assert r_create.status_code == 200
        create_data = r_create.json()
        new_id = create_data["project"]["project_id"]
        assert app.state.daemon.active_project_id == new_id

        # 3. Check list shows new project active
        r_list2 = await client.get("/api/projects")
        assert r_list2.json()["total_projects"] == 2
        assert r_list2.json()["active_project_id"] == new_id

        # 4. Rename new project
        r_rename = await client.post(
            "/api/projects/rename",
            json={"project_id": new_id, "new_title": "《大荒天尊傳》"},
        )
        assert r_rename.status_code == 200
        assert r_rename.json()["title"] == "《大荒天尊傳》"

        # 5. Switch back to default project
        r_switch = await client.post(
            "/api/projects/switch",
            json={"project_id": default_id},
        )
        assert r_switch.status_code == 200
        assert app.state.daemon.active_project_id == default_id

        # 6. Delete created project
        r_del = await client.delete(f"/api/projects/{new_id}")
        assert r_del.status_code == 200
        assert r_del.json()["deleted_project_id"] == new_id
        assert r_del.json()["active_project_id"] == default_id

        # Verify only 1 project remains
        r_list3 = await client.get("/api/projects")
        assert r_list3.json()["total_projects"] == 1

        # 7. Delete the last remaining project (Zero-State verification)
        r_del_last = await client.delete(f"/api/projects/{default_id}")
        assert r_del_last.status_code == 200
        assert r_del_last.json()["deleted_project_id"] == default_id
        assert r_del_last.json()["active_project_id"] == ""
        assert app.state.daemon.active_project_id == ""
        assert app.state.daemon.has_active_project() is False

        # 8. Verify list endpoint in Zero-State
        r_list_empty = await client.get("/api/projects")
        assert r_list_empty.status_code == 200
        assert r_list_empty.json()["total_projects"] == 0
        assert r_list_empty.json()["active_project_id"] == ""
        assert r_list_empty.json()["projects"] == []

        # 9. Verify warroom dashboard endpoint in Zero-State
        r_dashboard = await client.get("/api/warroom/dashboard")
        assert r_dashboard.status_code == 200
        dash_data = r_dashboard.json()
        assert dash_data["has_project"] is False
        assert dash_data["volumes"] == []
        assert dash_data["chapters"] == []

        # 10. Verify wizard genres in Zero-State
        r_genres = await client.get("/api/wizard/genres")
        assert r_genres.status_code == 200
        assert r_genres.json()["has_project"] is False
        assert r_genres.json()["qa_history"] == []

        # 11. Verify review state in Zero-State
        r_review = await client.get("/api/review/state")
        assert r_review.status_code == 200
        assert r_review.json()["has_project"] is False

        # 12. Create a project from Zero-State via API
        r_create_from_zero = await client.post(
            "/api/projects/create",
            json={
                "title": "《全新修真路》",
                "genre_primary": "仙俠",
                "genre_sub": "古典修仙",
                "min_word_count": 5000000,
            },
        )
        assert r_create_from_zero.status_code == 200
        p_from_zero_id = r_create_from_zero.json()["project"]["project_id"]
        assert app.state.daemon.active_project_id == p_from_zero_id
        assert app.state.daemon.has_active_project() is True
