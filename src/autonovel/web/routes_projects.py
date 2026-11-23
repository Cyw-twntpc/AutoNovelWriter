"""REST routes for multi-novel project management (Create, List, Switch, Rename, Delete, Export)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from autonovel.export.exporter import NovelExporter
from autonovel.db.project_repo import ProjectRepository
from autonovel.schemas.domain import ExportRequest

logger = get_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateProjectPayload(BaseModel):
    """Payload to create a new independent novel project."""

    title: str = Field(default="未命名長篇小說")
    genre_primary: str = Field(default="奇幻")
    genre_sub: str = Field(default="古典修仙")
    min_word_count: int = Field(default=5000000, ge=100000)


class SwitchProjectPayload(BaseModel):
    """Payload to switch the active novel project."""

    project_id: str


class RenameProjectPayload(BaseModel):
    """Payload to rename an existing novel project."""

    project_id: str
    new_title: str


@router.get("")
async def list_novel_projects(request: Request) -> dict[str, Any]:
    """Return all created novel projects with live metrics and active project ID."""
    daemon = request.app.state.daemon
    pm = daemon.project_manager
    projects = pm.list_projects()
    active_id = pm.get_active_project_id()
    return {
        "projects": projects,
        "active_project_id": active_id,
        "total_projects": len(projects),
    }


@router.post("/create")
async def create_novel_project(
    payload: CreateProjectPayload, request: Request
) -> dict[str, Any]:
    """Create a new novel project sandbox, initialize database, and switch daemon to it."""
    daemon = request.app.state.daemon
    pm = daemon.project_manager
    new_meta = pm.create_project(
        title=payload.title,
        genre_primary=payload.genre_primary,
        genre_sub=payload.genre_sub,
        min_word_count=payload.min_word_count,
    )
    switched = await daemon.switch_active_project(str(new_meta["project_id"]))
    return {
        "project": new_meta,
        "switched": switched,
    }


@router.post("/switch")
async def switch_novel_project(
    payload: SwitchProjectPayload, request: Request
) -> dict[str, Any]:
    """Safely pause current serialization, rebind daemon to target novel project, and return status."""
    daemon = request.app.state.daemon
    try:
        res = await daemon.switch_active_project(payload.project_id)
        return res
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/rename")
async def rename_novel_project(
    payload: RenameProjectPayload, request: Request
) -> dict[str, Any]:
    """Rename a novel project in metadata and inside its SQLite database."""
    daemon = request.app.state.daemon
    try:
        updated = daemon.project_manager.rename_project(payload.project_id, payload.new_title)
        # If renamed project is current active project, broadcast state change
        if payload.project_id == daemon.active_project_id:
            daemon.broadcast_event("project_renamed", {"project_id": payload.project_id, "title": payload.new_title})
        return updated
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{project_id}")
async def delete_novel_project(
    project_id: str, request: Request
) -> dict[str, Any]:
    """Delete a novel project directory and database permanently."""
    daemon = request.app.state.daemon
    was_active = (project_id == daemon.active_project_id)
    try:
        new_active = daemon.project_manager.delete_project(project_id)
        if was_active:
            if new_active:
                await daemon.switch_active_project(new_active)
            else:
                await daemon.unload_active_project()
        return {
            "deleted_project_id": project_id,
            "active_project_id": daemon.active_project_id,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        # Metadata is intentionally kept when the directory could not be removed.
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{project_id}/export")
async def export_specific_novel_project(
    project_id: str, payload: ExportRequest, request: Request
) -> dict[str, Any]:
    """Export completed chapters for a specific novel project."""
    daemon = request.app.state.daemon
    try:
        paths = daemon.project_manager.get_project_paths(project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not paths["project_db_path"].exists():
        raise HTTPException(status_code=404, detail="該小說專案不存在。")

    repo = ProjectRepository(paths["project_db_path"])
    exporter = NovelExporter(repo, paths["export_dir"])
    try:
        return exporter.export_novel(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
