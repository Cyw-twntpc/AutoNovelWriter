"""REST and SSE routes for View 3 (3-Column War Room, Snapshots, Reference Library, Export)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from autonovel.config import RuntimeConfig
from autonovel.export.exporter import NovelExporter
from autonovel.llm.llm_client import LLMResponseFormatError
from autonovel.reference.file_parser import (
    compute_sha256,
    parse_novel_file,
    preflight_check_files,
)
from autonovel.schemas.domain import ExportRequest, ReferenceIngestItem
from autonovel.wizard.qa_engine import AIModelUnavailableError
from scripts.setup_environment import (
    ENV_DOWNLOAD_MANAGER,
    format_friendly_error_message,
    inspect_environment,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["warroom"])


class SnapshotRestorePayload(BaseModel):
    """Payload to restore a specific snapshot file."""

    snapshot_id: str


class ModelDownloadPayload(BaseModel):
    """Payload to download BGE ONNX or Qwen2.5 GGUF model from Hugging Face Hub."""

    target: str = "bge"  # "bge" | "gguf"
    repo_id: str = ""
    filename: str = ""
    run_in_background: bool = True


class DeployAllPayload(BaseModel):
    """Payload to check and deploy all missing environment components."""

    repo_id: str = ""
    filename: str = ""
    run_in_background: bool = True


class ReferencePathsPayload(BaseModel):
    """Payload with local file paths or per-file action items for reference ingestion."""

    file_paths: list[str] = Field(default_factory=list)
    force_reextract: bool = False
    items: list[ReferenceIngestItem] = Field(default_factory=list)
    run_in_background: bool = False
    extract_skills: bool = True
    extract_pacing: bool = False
    pacing_jitter: float = 0.0


class PacingStageOverridePayload(BaseModel):
    """Payload to override/drag stages for a pacing template."""

    stages: list[dict[str, Any]] = Field(default_factory=list)


class PacingBlendPayload(BaseModel):
    """Payload to blend two pacing templates into a hybrid archetype."""

    template_id_a: int
    template_id_b: int
    weight_a: float = 0.5
    new_name: str = ""


class AddTagPayload(BaseModel):
    """Payload to add a semantic tag to a reference entry."""

    tag_name: str


class RenameTagPayload(BaseModel):
    """Payload to rename a tag globally."""

    old_tag: str
    new_tag: str


class PruneMarginalizedPayload(BaseModel):
    """Payload to prune isolated long-tail outlier tags."""

    min_frequency: int = 2
    eps: float = 0.45


@router.get("/warroom/dashboard")
async def get_warroom_dashboard(request: Request) -> dict[str, Any]:
    """Assemble full 3-column war room state: outline tree, 3D ECG metrics, hardware, anomalies."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        return {
            "has_project": False,
            "project_config": {},
            "volumes": [],
            "chapters": [],
            "metrics_30": [],
            "anomalies": [],
            "summary_stats": {
                "completed_chapters": 0,
                "total_words": 0,
                "avg_literary_score": 0.0,
                "current_writing_chapter": None,
                "last_chapter_duration_sec": 0.0,
                "tokens_per_sec": 0.0,
            },
            "hardware": daemon.hardware_guard.get_hardware_telemetry(),
            "reference_status": {
                **daemon.ref_extractor.status,
                **daemon.ref_repo.get_library_stats(),
            },
        }
    repo = daemon.project_repo
    cfg = repo.get_project_config()
    volumes = repo.fetch_all("SELECT * FROM volumes ORDER BY volume_num ASC;")
    chapters = repo.fetch_all(
        "SELECT chapter_num, volume_num, title, status, updated_at FROM chapters ORDER BY chapter_num ASC;"
    )
    metrics_30 = repo.fetch_all(
        "SELECT * FROM chapter_metrics ORDER BY chapter_num DESC LIMIT 30;"
    )
    anomalies = repo.fetch_all(
        "SELECT * FROM anomaly_logs ORDER BY id DESC LIMIT 50;"
    )
    totals = repo.fetch_all(
        "SELECT COUNT(*) AS ch_done, COALESCE(SUM(word_count), 0) AS total_words, "
        "COALESCE(AVG(literary_score), 0.0) AS avg_score FROM chapter_metrics;"
    )[0]
    return {
        "has_project": True,
        "project_config": cfg,
        "volumes": _enrich_volumes_with_progress(volumes, chapters),
        "chapters": chapters,
        "metrics_30": list(reversed(metrics_30)),
        "anomalies": anomalies,
        "summary_stats": {
            "completed_chapters": int(totals["ch_done"]),
            "total_words": int(totals["total_words"]),
            "avg_literary_score": round(float(totals["avg_score"]), 2),
            "current_writing_chapter": daemon.current_writing_chapter,
            "last_chapter_duration_sec": daemon.last_chapter_duration_sec,
            "tokens_per_sec": daemon.last_tokens_per_sec,
        },
        "hardware": daemon.hardware_guard.get_hardware_telemetry(),
        "reference_status": {
            **daemon.ref_extractor.status,
            **daemon.ref_repo.get_library_stats(),
        },
    }


def _enrich_volumes_with_progress(
    volumes: list[dict[str, Any]], chapters: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compute completed chapter count and percentage per volume for the left tree."""
    enriched: list[dict[str, Any]] = []
    for vol in volumes:
        vnum = int(vol["volume_num"])
        vol_chs = [c for c in chapters if int(c["volume_num"]) == vnum]
        done_chs = [
            c for c in vol_chs if c["status"] in ("COMPLETED", "REVISED")
        ]
        total_cnt = max(1, len(vol_chs) or int(vol["target_chapters"]))
        pct = round((len(done_chs) / total_cnt) * 100.0, 1)
        enriched.append(
            {
                **vol,
                "completed_count": len(done_chs),
                "total_count": len(vol_chs) or int(vol["target_chapters"]),
                "progress_percent": pct,
            }
        )
    return enriched


@router.get("/warroom/chapter/{chapter_num}")
async def get_chapter_reader_detail(
    chapter_num: int, request: Request
) -> dict[str, Any]:
    """Return full chapter text and Rubric score card without any chapter summary."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        raise HTTPException(status_code=404, detail="目前無作用中的小說專案。")
    ch_rows = daemon.project_repo.fetch_all(
        "SELECT chapter_num, volume_num, title, final_content, staging_content, status, updated_at "
        "FROM chapters WHERE chapter_num = ?;",
        (chapter_num,),
    )
    if not ch_rows:
        raise HTTPException(status_code=404, detail=f"找不到第 {chapter_num} 章。")
    m_rows = daemon.project_repo.fetch_all(
        "SELECT * FROM chapter_metrics WHERE chapter_num = ?;", (chapter_num,)
    )
    metric = m_rows[0] if m_rows else None
    if metric and metric.get("literary_rubric_json"):
        metric["literary_rubric"] = json.loads(metric["literary_rubric_json"])
    return {"chapter": ch_rows[0], "metrics": metric}


@router.post("/warroom/pause")
async def pause_serialization(request: Request) -> dict[str, Any]:
    """Request graceful pause after current chapter atomic commit completes."""
    return request.app.state.daemon.request_graceful_pause()


@router.post("/warroom/resume")
async def resume_serialization(request: Request) -> dict[str, Any]:
    """Resume background serialization loop."""
    daemon = request.app.state.daemon
    try:
        return await daemon.start_serialization()
    except AIModelUnavailableError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": str(exc),
            },
        )


@router.post("/warroom/step")
async def execute_single_chapter_step(request: Request) -> dict[str, Any]:
    """Execute one chapter synchronously (useful for controlled testing or manual step)."""
    daemon = request.app.state.daemon
    # Generating a chapter here while the background loop is mid-chapter would
    # build the next chapter from stale context and violate Gate A coherency.
    if daemon.is_serialization_active():
        raise HTTPException(
            status_code=409,
            detail="背景連載正在進行中，請先暫停連載，待當前章完成後再使用單章生成。",
        )
    try:
        res = await daemon.run_single_step()
        return {"result": res}
    except AIModelUnavailableError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": str(exc),
            },
        )


@router.post("/warroom/rewrite_latest")
async def rewrite_latest_completed_chapter(request: Request) -> dict[str, Any]:
    """Re-roll random seed and rewrite only the latest completed chapter."""
    daemon = request.app.state.daemon
    try:
        return await daemon.rewrite_latest_chapter()
    except AIModelUnavailableError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": str(exc),
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/warroom/snapshot/create")
async def create_manual_snapshot(request: Request) -> dict[str, Any]:
    """Create a permanent manual milestone snapshot."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project() or daemon.snapshots is None:
        raise HTTPException(
            status_code=400,
            detail="目前無作用中的小說專案，無法建立快照。",
        )
    cfg = daemon.project_repo.get_project_config()
    state = cfg.get("pipeline_state", "WIZARD")
    if state == "WIZARD":
        raise HTTPException(
            status_code=400,
            detail="目前處於創世問答階段，尚未生成小說藍圖與正文，無法建立快照。請先完成創世問答並鎖定風格契約書。",
        )
    rows = daemon.project_repo.fetch_all(
        "SELECT COALESCE(MAX(chapter_num), 0) AS c FROM chapter_metrics;"
    )
    cur_ch = int(rows[0]["c"]) if rows else 0
    return daemon.snapshots.create_snapshot("manual", chapter_num=cur_ch, label="user")


@router.get("/warroom/snapshot/list")
async def list_all_snapshots(request: Request) -> dict[str, Any]:
    """Return list of all periodic, milestone, and manual snapshots."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project() or daemon.snapshots is None:
        return {"snapshots": []}
    return {"snapshots": daemon.snapshots.list_snapshots()}


@router.post("/warroom/snapshot/restore")
async def restore_selected_snapshot(
    payload: SnapshotRestorePayload, request: Request
) -> dict[str, Any]:
    """Restore selected snapshot when serialization is paused."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project() or daemon.snapshots is None:
        raise HTTPException(status_code=400, detail="目前無作用中的小說專案，無法還原快照。")
    cfg = daemon.project_repo.get_project_config()
    # A chapter freeze window is not enough: the worker keeps looping between
    # chapters and would immediately write into the restored database.
    if cfg.get("pipeline_state") == "RUNNING":
        raise HTTPException(status_code=400, detail="請先暫停連載並等待當前章完成後再還原快照。")
    try:
        return daemon.snapshots.restore_snapshot(payload.snapshot_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/reference/upload_preflight")
async def upload_and_preflight_reference_files(
    files: list[UploadFile], request: Request
) -> dict[str, Any]:
    """Save uploaded reference files to staging folder and run 1s SHA-256 pre-flight check."""
    daemon = request.app.state.daemon
    upload_dir = daemon.config.data_dir / "ref_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for uf in files:
        safe_name = Path(uf.filename or "book.txt").name
        dest = upload_dir / safe_name
        content = await uf.read()
        dest.write_bytes(content)
        saved_paths.append(dest)
    items = preflight_check_files(saved_paths, daemon.ref_repo)
    return {"items": [i.model_dump() for i in items]}


@router.post("/reference/preflight")
async def preflight_local_reference_files(
    payload: ReferencePathsPayload, request: Request
) -> dict[str, Any]:
    """Run 1s static pre-flight check on local file paths."""
    daemon = request.app.state.daemon
    paths = [Path(p) for p in payload.file_paths]
    items = preflight_check_files(paths, daemon.ref_repo)
    return {"items": [i.model_dump() for i in items]}


@router.post("/reference/ingest")
async def ingest_confirmed_reference_files(
    payload: ReferencePathsPayload, request: Request
) -> dict[str, Any]:
    """Ingest confirmed reference files into reference_library.db using two-stage pipeline (Blueprint + Pattern)."""
    daemon = request.app.state.daemon
    if not await daemon.ais_llm_ready():
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": "本地 AI 模型未就緒或推論服務離線，無法進行書庫語義條目抽取。請先至環境設置完成模型下載與推論服務設定。",
            },
        )
    ingest_items = list(payload.items)
    if not ingest_items and payload.file_paths:
        act = "REEXTRACT" if payload.force_reextract else "INGEST"
        ingest_items = [
            ReferenceIngestItem(
                file_path=p,
                action=act,
                extract_skills=payload.extract_skills,
                extract_pacing=payload.extract_pacing,
            )
            for p in payload.file_paths
        ]

    def _on_ref_progress(st: dict[str, Any]) -> None:
        daemon.broadcast_event(
            "ref_progress",
            {**st, **daemon.ref_repo.get_library_stats()},
        )

    def _on_pacing_progress(st: dict[str, Any]) -> None:
        daemon.ref_extractor.update_pacing_status(**st)
        daemon.broadcast_event("ref_pacing_progress", st)

    # Normalize per-item track flags (respect explicit item flags; fall back to payload flags if omitted)
    for it in ingest_items:
        fields_set = getattr(it, "model_fields_set", set())
        if "extract_pacing" not in fields_set:
            it.extract_pacing = bool(payload.extract_pacing)
        if "extract_skills" not in fields_set:
            it.extract_skills = bool(payload.extract_skills)

    # Separate items into Track B (Macro Pacing) and Track A (Micro Skills)
    pacing_items = [
        it for it in ingest_items
        if bool(getattr(it, "extract_pacing", False))
        and it.action.upper() != "SKIP"
        and Path(it.file_path).exists()
    ]
    skill_items = [
        it for it in ingest_items
        if bool(getattr(it, "extract_skills", True))
        and it.action.upper() != "SKIP"
        and Path(it.file_path).exists()
    ]

    if not pacing_items and not daemon.ref_extractor.pacing_status.get("active", False):
        daemon.ref_extractor.update_pacing_status(
            active=False,
            novel_title="",
            percent=0.0,
            phase="IDLE",
            detail="",
            batch_index=0,
            batch_total=0,
        )
        daemon.broadcast_event("ref_pacing_progress", daemon.ref_extractor.get_pacing_status())

    # Handle asynchronous background execution
    if payload.run_in_background and (pacing_items or skill_items):
        was_already_active = bool(
            daemon.ref_extractor._bg_task and not daemon.ref_extractor._bg_task.done()
        )
        daemon.ref_extractor.enqueue_pipeline(
            pacing_items=pacing_items,
            skill_items=skill_items,
            pacing_jitter=payload.pacing_jitter,
            force_reextract=payload.force_reextract,
            macro_extractor=daemon.macro_extractor,
            on_skills_progress=_on_ref_progress,
            on_pacing_progress=_on_pacing_progress,
        )
        return {
            "queued": True,
            "appended_to_active_queue": was_already_active,
            "queued_batches": len(daemon.ref_extractor._pending_jobs),
            "item_count": len(pacing_items) + len(skill_items),
            "pacing_extracted_count": 0,
            "library_stats": daemon.ref_repo.get_library_stats(),
        }

    # Synchronous execution
    pacing_results: list[dict[str, Any]] = []
    fresh_bp_paths: set[str] = set()
    for item in pacing_items:
        p = Path(item.file_path)
        sha = compute_sha256(p)
        act = item.action.upper()
        is_force = (act == "REEXTRACT") or payload.force_reextract

        existing_src = daemon.ref_repo.get_source_by_hash(sha)
        if existing_src:
            source_id = existing_src["id"]
        else:
            source_id = daemon.ref_repo.register_or_reset_source(
                sha, p.name, 0, reset_entries=False
            )

        raw_text, _ = parse_novel_file(p)
        title = p.stem
        tpl = await daemon.macro_extractor.extract_from_novel_text(
            text=raw_text,
            novel_title=title,
            source_id=source_id,
            jitter_ratio=payload.pacing_jitter,
            hardware_guard=daemon.hardware_guard,
            lifecycle=daemon.ref_extractor,
            force_reextract=is_force,
            on_progress=_on_pacing_progress,
        )
        fresh_bp_paths.add(str(p))
        fresh_bp_paths.add(str(p.resolve()))
        pacing_results.append(tpl.model_dump())

    total_chunks = 0
    if skill_items:
        total_chunks = await daemon.ref_extractor.ingest_items(
            skill_items,
            on_progress=_on_ref_progress,
            fresh_blueprint_paths=fresh_bp_paths,
        )

    return {
        "queued": False,
        "ingested_chunks": total_chunks,
        "pacing_extracted_count": len(pacing_results),
        "library_stats": daemon.ref_repo.get_library_stats(),
    }


@router.get("/reference/books/{source_id}/blueprint")
async def get_reference_book_blueprint(source_id: int, request: Request) -> dict[str, Any]:
    """Fetch permanent factual book blueprint (macro outline, character network, faction geopolitics, hierarchy rules)."""
    daemon = request.app.state.daemon
    bp = daemon.ref_repo.get_book_blueprint(source_id)
    if not bp:
        return JSONResponse(
            status_code=404,
            content={"error": "BLUEPRINT_NOT_FOUND", "detail": f"書籍 ID {source_id} 尚未建構事實藍圖。"},
        )
    return bp


@router.get("/reference/books")
async def get_reference_books(request: Request) -> dict[str, Any]:
    """List all ingested books and library summary statistics."""
    daemon = request.app.state.daemon
    books = daemon.ref_repo.list_sources_with_counts()
    stats = daemon.ref_repo.get_library_stats()
    return {"books": books, "stats": stats}


@router.delete("/reference/books/{source_id}")
async def delete_reference_book(source_id: int, request: Request) -> dict[str, Any]:
    """Delete a reference book, its entries, and the staged upload file it came from."""
    daemon = request.app.state.daemon
    if hasattr(daemon, "ref_extractor") and daemon.ref_extractor.is_source_in_use(source_id):
        raise HTTPException(
            status_code=400,
            detail="無法刪除正在萃取中或排隊中的書籍，請先取消萃取或等待其完成。",
        )
    source = daemon.ref_repo.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="找不到指定書籍")
    deleted = daemon.ref_repo.delete_source(source_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="找不到指定書籍")
    removed_file = _remove_staged_upload(daemon, str(source["file_name"]), source_id)
    stats = daemon.ref_repo.get_library_stats()
    daemon.broadcast_event("ref_progress", {"active": False, **stats})
    return {
        "deleted": True,
        "source_id": source_id,
        "removed_file": removed_file,
        "stats": stats,
    }


def _remove_staged_upload(daemon: Any, file_name: str, source_id: int) -> bool:
    """Delete the staged copy of an uploaded book when no other source still needs it."""
    # Two books may share a file name (different content), and books can be ingested
    # from arbitrary local paths that were never copied into ref_uploads.
    if daemon.ref_repo.count_sources_by_file_name(file_name, source_id) > 0:
        return False
    staged = daemon.config.data_dir / "ref_uploads" / Path(file_name).name
    try:
        if staged.is_file():
            staged.unlink()
            return True
    except OSError as exc:
        logger.warning("Failed to remove staged reference upload %s: %s", staged, exc)
    return False


@router.get("/reference/entries")
async def get_reference_entries(
    request: Request,
    source_id: int | None = None,
    keyword: str = "",
    tag: str = "",
    limit: int = 30,
    offset: int = 0,
) -> dict[str, Any]:
    """Query reference entries with filtering and pagination."""
    daemon = request.app.state.daemon
    entries, total = daemon.ref_repo.query_entries(
        source_id=source_id,
        keyword=keyword,
        tag=tag,
        limit=min(100, max(1, limit)),
        offset=max(0, offset),
    )
    return {
        "entries": entries,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.delete("/reference/entries/{entry_id}")
async def delete_reference_entry(entry_id: int, request: Request) -> dict[str, Any]:
    """Delete a single knowledge entry."""
    daemon = request.app.state.daemon
    deleted = daemon.ref_repo.delete_entry(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="找不到指定條目")
    stats = daemon.ref_repo.get_library_stats()
    return {"deleted": True, "entry_id": entry_id, "stats": stats}


@router.post("/reference/entries/{entry_id}/tags")
async def add_tag_to_entry(
    entry_id: int, payload: AddTagPayload, request: Request
) -> dict[str, Any]:
    """Add a semantic tag to an entry and calculate its vector embedding."""
    daemon = request.app.state.daemon
    tag_name = payload.tag_name.strip()
    if not tag_name:
        raise HTTPException(status_code=400, detail="標籤名稱不可為空")
    tag_vec = daemon.ref_extractor.embedder.embed_text(tag_name)
    success = daemon.ref_repo.add_tag_to_entry(entry_id, tag_name, tag_vec)
    if not success:
        raise HTTPException(status_code=404, detail="條目不存在或新增標籤失敗")
    return {"success": True, "entry_id": entry_id, "tag_name": tag_name}


@router.get("/reference/tags")
async def list_reference_tags(request: Request) -> dict[str, Any]:
    """List all unique tags and their occurrence counts."""
    daemon = request.app.state.daemon
    tags = daemon.ref_repo.list_tags_with_frequency()
    return {"tags": tags}


@router.put("/reference/tags/rename")
async def rename_reference_tag(
    payload: RenameTagPayload, request: Request
) -> dict[str, Any]:
    """Rename a tag globally across all entries and update embedding."""
    daemon = request.app.state.daemon
    old_tag = payload.old_tag.strip()
    new_tag = payload.new_tag.strip()
    if not old_tag or not new_tag:
        raise HTTPException(status_code=400, detail="新舊標籤名稱皆不可為空")
    new_vec = daemon.ref_extractor.embedder.embed_text(new_tag)
    updated_count = daemon.ref_repo.rename_tag_globally(old_tag, new_tag, new_vec)
    return {"updated_count": updated_count, "old_tag": old_tag, "new_tag": new_tag}


@router.post("/reference/tags/auto_clean")
async def auto_clean_reference_tags(request: Request) -> dict[str, Any]:
    """Execute AI-driven unified scanning, normalization, and deduplication of tags."""
    daemon = request.app.state.daemon
    if not await daemon.ais_llm_ready():
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": "本地 AI 模型未就緒或推論服務離線，無法進行全庫標籤智慧整頓。請先下載模型並啟動推論服務。",
            },
        )
    try:
        stats = await daemon.ref_extractor.auto_clean_tags()
        lib_stats = daemon.ref_repo.get_library_stats()
        daemon.broadcast_event("ref_progress", {"active": False, **lib_stats})
        return {"success": True, "stats": stats, "library_stats": lib_stats}
    except AIModelUnavailableError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": str(exc),
            },
        )
    except (LLMResponseFormatError, Exception) as exc:  # noqa: BLE001
        logger.warning("AI tag auto-clean failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "TAG_CLEAN_FAILED",
                "detail": f"AI 標籤智慧整頓失敗：{exc}",
            },
        )


@router.post("/reference/tags/prune_marginalized")
async def prune_marginalized_tags_endpoint(
    request: Request, payload: PruneMarginalizedPayload | None = None
) -> dict[str, Any]:
    """Big Data statistical self-cleansing: prune isolated, unreinforced long-tail outlier tags."""
    daemon = request.app.state.daemon
    p = payload or PruneMarginalizedPayload()
    result = await asyncio.to_thread(
        daemon.ref_repo.prune_marginalized_tags, p.min_frequency, p.eps
    )
    lib_stats = daemon.ref_repo.get_library_stats()
    daemon.broadcast_event("ref_progress", {"active": False, **lib_stats})
    return {"success": True, "result": result, "library_stats": lib_stats}


@router.get("/reference/progress")
async def get_reference_progress(request: Request) -> dict[str, Any]:
    """Get the active background ingestion progress and stats."""
    daemon = request.app.state.daemon
    status = daemon.ref_extractor.get_current_status()
    stats = daemon.ref_repo.get_library_stats()
    return {**status, **stats}


@router.post("/reference/pause")
async def pause_reference_extraction(request: Request) -> dict[str, Any]:
    """Pause currently running reference extraction."""
    daemon = request.app.state.daemon
    ok = daemon.ref_extractor.pause()
    status = daemon.ref_extractor.get_current_status()
    daemon.broadcast_event("ref_progress", status)
    return {"status": "paused", "success": ok, "ref_status": status}


@router.post("/reference/resume")
async def resume_reference_extraction(request: Request) -> dict[str, Any]:
    """Resume paused reference extraction and ensure local LLM is ready if previously unloaded."""
    daemon = request.app.state.daemon
    if getattr(daemon.llama_manager, "_manually_unloaded", False):
        await daemon.ais_llm_ready(clear_manual_unload=True)
    ok = daemon.ref_extractor.resume(
        macro_extractor=daemon.macro_extractor,
        on_skills_progress=daemon._broadcast_ref_progress,
        on_pacing_progress=daemon._broadcast_pacing_progress,
    )
    status = daemon.ref_extractor.get_current_status()
    daemon.broadcast_event("ref_progress", status)
    return {"status": "running", "success": ok, "ref_status": status}


@router.post("/reference/cancel")
async def cancel_reference_extraction(request: Request) -> dict[str, Any]:
    """Cancel currently running or paused reference extraction."""
    daemon = request.app.state.daemon
    ok = daemon.ref_extractor.cancel()
    status = daemon.ref_extractor.get_current_status()
    pacing_status = daemon.ref_extractor.get_pacing_status()
    lib_stats = daemon.ref_repo.get_library_stats()
    daemon.broadcast_event("ref_progress", {**status, **lib_stats})
    daemon.broadcast_event("ref_pacing_progress", pacing_status)
    return {"status": "cancelled", "success": ok, "ref_status": status}


@router.get("/reference/pacing_progress")
async def get_pacing_progress(request: Request) -> dict[str, Any]:
    """Fetch visual progress of ongoing macro pacing analysis."""
    daemon = request.app.state.daemon
    return daemon.ref_extractor.get_pacing_status()


@router.get("/reference/tag_assets")
async def get_reference_tag_assets(request: Request) -> dict[str, Any]:
    """Return 2D PCA cluster coordinates, frequency statistics, and cluster groupings for all tags."""
    daemon = request.app.state.daemon
    assets = await asyncio.to_thread(daemon.ref_repo.get_tag_clusters_and_stats)
    return assets


@router.get("/reference/pacing_templates")
async def list_pacing_templates(request: Request) -> dict[str, Any]:
    """List all extracted and synthesized macro pacing wave archetypes."""
    daemon = request.app.state.daemon
    templates = daemon.ref_repo.list_pacing_templates()
    return {"templates": templates}


@router.get("/reference/pacing_templates/{template_id}")
async def get_pacing_template(template_id: int, request: Request) -> dict[str, Any]:
    """Retrieve full details and stage milestones of a specific pacing template."""
    daemon = request.app.state.daemon
    template = daemon.ref_repo.get_pacing_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="情節規劃範本不存在。")
    return {"template": template}


@router.delete("/reference/pacing_templates/{template_id}")
async def delete_pacing_template(template_id: int, request: Request) -> dict[str, Any]:
    """Delete a macro pacing template and its stages."""
    daemon = request.app.state.daemon
    success = daemon.ref_repo.delete_pacing_template(template_id)
    if not success:
        raise HTTPException(status_code=404, detail="情節規劃範本不存在或刪除失敗。")
    return {"success": True}


@router.put("/reference/pacing_templates/{template_id}/stages")
async def override_pacing_template_stages(
    template_id: int, payload: PacingStageOverridePayload, request: Request
) -> dict[str, Any]:
    """Allow author to override/drag stage points and subvert regular pacing."""
    daemon = request.app.state.daemon
    success = daemon.ref_repo.override_pacing_stages(template_id, payload.stages)
    if not success:
        raise HTTPException(status_code=404, detail="情節規劃範本不存在或更新失敗。")
    updated = daemon.ref_repo.get_pacing_template(template_id)
    return {"success": True, "template": updated}


@router.post("/reference/pacing_templates/blend")
async def blend_pacing_templates_route(
    payload: PacingBlendPayload, request: Request
) -> dict[str, Any]:
    """Blend two pacing templates into an original hybrid archetype."""
    daemon = request.app.state.daemon
    try:
        new_id = daemon.ref_repo.blend_pacing_templates(
            template_id_a=payload.template_id_a,
            template_id_b=payload.template_id_b,
            weight_a=payload.weight_a,
            new_name=payload.new_name,
        )
        created = daemon.ref_repo.get_pacing_template(new_id)
        return {"success": True, "template": created}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/warroom/acceptance_report")
async def get_acceptance_report(
    request: Request, volume_num: int | None = None
) -> dict[str, Any]:
    """Compute and return the 5 perfectionist quantitative acceptance metrics."""
    daemon = request.app.state.daemon
    report = daemon.volume_planner.evaluate_acceptance_report(volume_num=volume_num)
    return report.model_dump()


@router.get("/system/logs")
async def get_system_logs(
    request: Request,
    lines: int = 100,
) -> dict[str, Any]:
    """Retrieve the tail of the persistent rotating log file."""
    cfg: RuntimeConfig = request.app.state.config
    log_file = cfg.log_file_path
    if not log_file.exists():
        return {"logs": [], "total_lines": 0, "log_path": str(log_file)}

    try:
        with open(log_file, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            tail_lines = all_lines[-lines:] if lines > 0 else all_lines
        return {
            "logs": [line.rstrip("\r\n") for line in tail_lines],
            "total_lines": len(all_lines),
            "log_path": str(log_file),
        }
    except Exception as exc:
        logger.error("Failed to read system log file: %s", exc)
        return {"logs": [f"讀取日誌失敗: {exc}"], "total_lines": 0, "log_path": str(log_file)}


@router.get("/system/env_status")
async def get_system_environment_status(request: Request) -> dict[str, Any]:
    """Inspect 8GB VRAM hardware, GGUF model, BGE ONNX weights, and llama-server status."""
    daemon = request.app.state.daemon
    status = inspect_environment(daemon.config, daemon=daemon)
    status["download_state"] = dict(ENV_DOWNLOAD_MANAGER.state)
    return status


@router.post("/system/download_model")
async def trigger_model_download(
    payload: ModelDownloadPayload, request: Request
) -> dict[str, Any]:
    """Download BGE ONNX or GGUF model weights and hot-reload into the running daemon."""
    if payload.target not in ("bge", "gguf", "llama"):
        raise HTTPException(status_code=400, detail=f"不支援的模型下載目標：{payload.target}")
    daemon = request.app.state.daemon

    def _on_dl_event(state: dict[str, Any]) -> None:
        daemon.broadcast_event("env_download", state)

    if payload.run_in_background:
        ENV_DOWNLOAD_MANAGER.start_background_download(
            daemon,
            target=payload.target,
            repo_id=payload.repo_id.strip(),
            filename=payload.filename.strip(),
            on_event=_on_dl_event,
        )
        return {
            "queued": True,
            "download_state": dict(ENV_DOWNLOAD_MANAGER.state),
        }
    try:
        res = await ENV_DOWNLOAD_MANAGER.execute_download(
            daemon,
            target=payload.target,
            repo_id=payload.repo_id.strip(),
            filename=payload.filename.strip(),
            on_event=_on_dl_event,
        )
    except Exception as exc:
        logger.exception("Sequential deployment failed: %s", exc)
        friendly_msg = format_friendly_error_message(exc, prefix="模型部署失敗")
        raise HTTPException(status_code=500, detail=friendly_msg) from exc
    return {
        "queued": False,
        "result": res,
        "env_status": inspect_environment(daemon.config, daemon=daemon),
    }


@router.post("/system/heartbeat")
async def register_frontend_heartbeat(request: Request) -> dict[str, Any]:
    """Register frontend user activity heartbeat to reset the idle countdown."""
    daemon = request.app.state.daemon
    daemon.record_user_activity()
    return {
        "status": "ok",
        "idle_seconds": round(daemon.get_idle_seconds(), 1),
        "idle_threshold_seconds": daemon.config.model_idle_timeout_sec,
        "model_in_use": daemon.is_model_in_use(),
        "llama_server_healthy": daemon.llama_manager.is_healthy(),
    }


@router.post("/system/start_llama")
async def start_or_probe_llama_server(request: Request) -> dict[str, Any]:
    """Attempt to launch or probe local llama-server and return updated environment status."""
    daemon = request.app.state.daemon
    started = await asyncio.to_thread(daemon.llama_manager.ensure_running)
    return {
        "started": started,
        "env_status": inspect_environment(daemon.config, daemon=daemon),
    }


@router.post("/system/deploy_all")
async def deploy_all_environments(
    payload: DeployAllPayload, request: Request
) -> dict[str, Any]:
    """Check all environment requirements, sequentially download missing models, and start llama-server."""
    daemon = request.app.state.daemon

    def _on_dl_event(state: dict[str, Any]) -> None:
        daemon.broadcast_event("env_download", state)

    if ENV_DOWNLOAD_MANAGER.state.get("active"):
        return {
            "queued": False,
            "message": "已有部署或下載任務正在執行中",
            "download_state": dict(ENV_DOWNLOAD_MANAGER.state),
            "env_status": inspect_environment(daemon.config, daemon=daemon),
        }

    if payload.run_in_background:
        ENV_DOWNLOAD_MANAGER.start_background_deploy_all(
            daemon,
            repo_id=payload.repo_id.strip(),
            filename=payload.filename.strip(),
            on_event=_on_dl_event,
        )
        return {
            "queued": True,
            "download_state": dict(ENV_DOWNLOAD_MANAGER.state),
            "env_status": inspect_environment(daemon.config, daemon=daemon),
        }

    try:
        env_res = await ENV_DOWNLOAD_MANAGER.execute_deploy_all(
            daemon,
            repo_id=payload.repo_id.strip(),
            filename=payload.filename.strip(),
            on_event=_on_dl_event,
        )
    except Exception as exc:
        logger.exception("Sequential deployment failed: %s", exc)
        friendly_msg = format_friendly_error_message(exc, prefix="模型部署失敗")
        raise HTTPException(status_code=500, detail=friendly_msg) from exc
    return {
        "queued": False,
        "env_status": env_res,
        "download_state": dict(ENV_DOWNLOAD_MANAGER.state),
    }


@router.post("/system/unload_and_pause")
async def unload_model_and_pause_work(request: Request) -> dict[str, Any]:
    """Unload model from GPU VRAM and pause all running writing and reference extraction jobs."""
    daemon = request.app.state.daemon
    write_pause = daemon.request_graceful_pause()
    if daemon._worker_task and not daemon._worker_task.done():
        daemon._worker_task.cancel()
        daemon._worker_task = None
        daemon.current_writing_chapter = None
        daemon.hardware_guard.release_anti_sleep_lock()
        if daemon.project_repo:
            daemon.project_repo.rollback_staging_chapters()
            daemon.project_repo.update_project_config(pipeline_state="PAUSED")
        daemon.broadcast_event("state_change", {"pipeline_state": "PAUSED"})
        write_pause = {"pipeline_state": "PAUSED", "graceful_pending": False}

    ref_pause = daemon.ref_extractor.pause() if hasattr(daemon, "ref_extractor") else False
    if hasattr(daemon, "ref_extractor"):
        daemon.broadcast_event(
            "ref_progress",
            {**daemon.ref_extractor.get_current_status(), **daemon.ref_repo.get_library_stats()},
        )

    if daemon.llm_client and hasattr(daemon.llm_client, "aclose"):
        try:
            await daemon.llm_client.aclose()
        except Exception:
            pass

    try:
        daemon.llama_manager.stop_server(manual=True)
    except TypeError:
        daemon.llama_manager.stop_server()

    if daemon.llm_client and getattr(daemon.llm_client, "server_manager", None) is not None:
        sm = daemon.llm_client.server_manager
        sm._manually_unloaded = True
        if sm is not daemon.llama_manager and hasattr(sm, "stop_server"):
            try:
                sm.stop_server(manual=True)
            except TypeError:
                sm.stop_server()

    daemon.broadcast_event(
        "env_status_changed",
        {"llama_server_healthy": False, "reason": "USER_MANUAL_UNLOAD"},
    )
    env_status = inspect_environment(daemon.config, daemon=daemon)
    return {
        "status": "ok",
        "writing_paused": write_pause,
        "extraction_paused": ref_pause,
        "model_unloaded": True,
        "env_status": env_status,
    }


@router.post("/export")
async def export_novel_packages(
    payload: ExportRequest, request: Request
) -> dict[str, Any]:
    """Export novel chapters in selected formats (.md/.txt/.html/.epub/.docx)."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        raise HTTPException(status_code=400, detail="目前無作用中的小說專案，無法匯出。")
    exporter = NovelExporter(daemon.project_repo, daemon.config.export_dir)
    try:
        return exporter.export_novel(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/warroom/stream")
async def stream_warroom_events(request: Request) -> StreamingResponse:
    """SSE endpoint pushing real-time generated tokens and chapter completion events."""
    daemon = request.app.state.daemon
    queue = daemon.subscribe_events()

    async def _event_generator() -> AsyncIterator[str]:
        try:
            yield f"data: {json.dumps({'event': 'connected', 'data': {}}, ensure_ascii=False)}\n\n"
            idle_seconds = 0.0
            while True:
                try:
                    # Wake every second so a closed tab releases its subscription
                    # promptly instead of lingering until the 15 s keep-alive.
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    idle_seconds += 1.0
                    if await request.is_disconnected():
                        break
                    if idle_seconds >= 15.0:
                        idle_seconds = 0.0
                        yield ": keep-alive\n\n"
                    continue
                if await request.is_disconnected():
                    break
                idle_seconds = 0.0
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        finally:
            daemon.unsubscribe_events(queue)

    return StreamingResponse(_event_generator(), media_type="text/event-stream")
