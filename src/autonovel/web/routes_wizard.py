"""REST API routes for View 1 (Story Q&A Wizard) and View 2 (Review Gate Workbench)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from autonovel.wizard.blueprint_builder import BlueprintLockedError
from autonovel.wizard.genre_catalog import GENRE_HIERARCHY
from autonovel.wizard.qa_engine import AIModelUnavailableError

router = APIRouter(prefix="/api", tags=["wizard_and_review"])


class GenreSetupPayload(BaseModel):
    """Initial project genre and minimum word count setup request (supports multi-sub-genre)."""

    title: str = "未命名長篇小說"
    genre_primary: str = "仙俠"
    genre_sub: str = "古典修仙"
    genre_subs: list[str] = Field(default_factory=list)
    min_word_count: int = Field(default=5_000_000, ge=10_000)


class AnswerSubmitPayload(BaseModel):
    """User answer submission in the infinite-depth Q&A wizard."""

    question_title: str
    selected_answers: list[str]
    custom_note: str = ""
    question_aspect: str = ""


class AutoStepPayload(BaseModel):
    """Payload for a single interactive turn of AI autocomplete."""

    question_title: str
    question_aspect: str = ""
    options: list[str]


class RollbackPayload(BaseModel):
    """Request to roll back Q&A history to an earlier question step."""

    step_index: int = Field(..., ge=1)


class RegenerateSectionPayload(BaseModel):
    """Request to regenerate a single section in the Review Gate workbench."""

    section: str  # "world" | "characters" | "outline" | "style" | "blacklist"


def _detect_ai_mode(daemon: Any) -> str:
    """Detect whether live llama-server LLM is active or offline."""
    if daemon.llm_client and daemon.llm_client.server_manager.is_healthy():
        return "LLM"
    return "OFFLINE"


@router.get("/wizard/genres")
async def get_genre_catalog(
    request: Request, catalog_only: bool = False
) -> dict[str, Any]:
    """Return built-in + reference-discovered genre hierarchy, current config, and Q&A history."""
    daemon = request.app.state.daemon
    dynamic_genres = daemon.ref_repo.discover_dynamic_genres(GENRE_HIERARCHY)
    if catalog_only:
        return {
            "has_project": daemon.has_active_project(),
            "genres": dynamic_genres,
            "ai_mode": _detect_ai_mode(daemon),
        }
    if not daemon.has_active_project():
        return {
            "has_project": False,
            "genres": dynamic_genres,
            "project_config": {},
            "qa_history": [],
            "next_question": None,
            "ai_mode": "IDLE",
        }
    cfg = daemon.project_repo.get_project_config()
    history = daemon.qa_engine.get_qa_history()
    # A GET must be side-effect free: generating a question here would spend GPU
    # time (and mutate variation counters) on every page load or prefetch.
    next_q_data = None
    threshold = daemon.qa_engine.get_unlock_threshold()
    return {
        "has_project": True,
        "genres": dynamic_genres,
        "project_config": cfg,
        "qa_history": history,
        "next_question": next_q_data,
        "qa_threshold": threshold,
        "can_build_blueprint": len(history) >= threshold,
        "ai_mode": _detect_ai_mode(daemon),
    }


@router.post("/wizard/setup")
async def setup_project_genre(
    payload: GenreSetupPayload, request: Request
) -> dict[str, Any]:
    """Save initial genre selection and generate question 1 via live AI."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        raise HTTPException(status_code=400, detail="目前無作用中的小說專案，請先建立新小說。")
    merged_subs = [s.strip() for s in payload.genre_subs if s.strip()]
    if payload.genre_sub.strip():
        for part in payload.genre_sub.replace("，", "、").replace(",", "、").split("、"):
            clean_part = part.strip()
            if clean_part and clean_part not in merged_subs:
                merged_subs.append(clean_part)
    final_sub = "、".join(merged_subs) if merged_subs else "古典修仙"
    daemon.project_repo.update_project_config(
        title=payload.title.strip() or "未命名長篇小說",
        genre_primary=payload.genre_primary.strip(),
        genre_sub=final_sub,
        min_word_count=payload.min_word_count,
        qa_history_json="[]",
        pipeline_state="WIZARD",
    )
    try:
        first_q = await daemon.qa_engine.generate_next_question([])
        threshold = daemon.qa_engine.get_unlock_threshold()
        return {
            "project_config": daemon.project_repo.get_project_config(),
            "next_question": first_q.model_dump(),
            "qa_threshold": threshold,
            "can_build_blueprint": False,
            "ai_mode": "LLM",
        }
    except AIModelUnavailableError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": str(exc),
                "ai_mode": "OFFLINE",
                "project_config": daemon.project_repo.get_project_config(),
            },
        )


@router.post("/wizard/next_question")
async def fetch_current_wizard_question(request: Request) -> dict[str, Any]:
    """Return the deterministic next question for the current history step.

    Page reloads need the pending question, but fetching it must be an explicit
    POST so it cannot be triggered by crawlers or prefetching.
    """
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        raise HTTPException(status_code=400, detail="目前無作用中的小說專案。")
    history = daemon.qa_engine.get_qa_history()
    try:
        next_q = await daemon.qa_engine.generate_next_question(history)
        return {"next_question": next_q.model_dump(), "ai_mode": "LLM"}
    except AIModelUnavailableError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": str(exc),
                "ai_mode": "OFFLINE",
            },
        )


@router.post("/wizard/regenerate_question")
async def regenerate_wizard_question(request: Request) -> dict[str, Any]:
    """Re-query AI options for the current Q&A layer."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        raise HTTPException(status_code=400, detail="目前無作用中的小說專案。")
    history = daemon.qa_engine.get_qa_history()
    try:
        next_q = await daemon.qa_engine.generate_next_question(history, force_variation=True)
        return {
            "history": history,
            "next_question": next_q.model_dump(),
            "ai_mode": "LLM",
        }
    except AIModelUnavailableError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": str(exc),
                "ai_mode": "OFFLINE",
            },
        )


@router.post("/wizard/answer")
async def submit_wizard_answer(
    payload: AnswerSubmitPayload, request: Request
) -> dict[str, Any]:
    """Validate user answer for contradictions and return the next deeper question."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        raise HTTPException(status_code=400, detail="目前無作用中的小說專案。")
    try:
        result = await daemon.qa_engine.submit_answer_and_next(
            question_title=payload.question_title,
            selected_answers=payload.selected_answers,
            custom_note=payload.custom_note,
            question_aspect=payload.question_aspect,
        )
        result["ai_mode"] = _detect_ai_mode(daemon)
        return result
    except AIModelUnavailableError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": str(exc),
                "ai_mode": "OFFLINE",
            },
        )


@router.post("/wizard/autocomplete_step")
async def autocomplete_wizard_step(
    payload: AutoStepPayload, request: Request
) -> dict[str, Any]:
    """Execute a single interactive turn: Role B answers -> Role A generates next balanced question."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        raise HTTPException(status_code=400, detail="目前無作用中的小說專案。")
    try:
        result = await daemon.qa_engine.step_auto_turn(
            question_title=payload.question_title,
            question_aspect=payload.question_aspect,
            options=payload.options,
        )
        result["ai_mode"] = _detect_ai_mode(daemon)
        return result
    except AIModelUnavailableError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": str(exc),
                "ai_mode": "OFFLINE",
            },
        )


@router.post("/wizard/rollback")
async def rollback_wizard_step(
    payload: RollbackPayload, request: Request
) -> dict[str, Any]:
    """Roll back Q&A history to step_index and regenerate that question."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        raise HTTPException(status_code=400, detail="目前無作用中的小說專案。")
    trimmed = daemon.qa_engine.rollback_to_step(payload.step_index)
    try:
        next_q = await daemon.qa_engine.generate_next_question(trimmed)
        threshold = daemon.qa_engine.get_unlock_threshold()
        return {
            "history": trimmed,
            "next_question": next_q.model_dump(),
            "qa_threshold": threshold,
            "can_build_blueprint": len(trimmed) >= threshold,
            "ai_mode": "LLM",
        }
    except AIModelUnavailableError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": str(exc),
                "ai_mode": "OFFLINE",
            },
        )


@router.post("/wizard/complete")
async def complete_wizard_and_build_blueprint(request: Request) -> dict[str, Any]:
    """Finish Q&A, trigger AI blueprint completion, and enter REVIEW_GATE."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        raise HTTPException(status_code=400, detail="目前無作用中的小說專案。")
    if not await daemon.ais_llm_ready():
        # Report a missing model before an unfinished Q&A: env setup is the action
        # the UI must offer, and it is the more actionable failure of the two.
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": (
                    "本地 AI 模型未就緒或推論服務離線，無法進行 AI 藍圖自動補全。"
                    "請先至環境設置完成模型下載與推論服務設定。"
                ),
                "ai_mode": "OFFLINE",
            },
        )
    # The threshold was only echoed to the UI; enforce it server-side so the
    # blueprint cannot be generated from one or two answers.
    threshold = daemon.qa_engine.get_unlock_threshold()
    answered = len(daemon.qa_engine.get_qa_history())
    if answered < threshold:
        raise HTTPException(
            status_code=400,
            detail=f"創世問答尚未完成（已答 {answered} / 需 {threshold} 題），無法生成藍圖。",
        )
    try:
        return await daemon.blueprint_builder.complete_blueprint()
    except BlueprintLockedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AIModelUnavailableError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": str(exc),
                "ai_mode": "OFFLINE",
            },
        )


@router.get("/review/state")
async def get_review_gate_state(request: Request) -> dict[str, Any]:
    """Return all 5 structured tabs for the Step 4 Review Gate Workbench."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        return {
            "has_project": False,
            "world": {},
            "characters": [],
            "outline": [],
            "style": {},
            "blacklist": {},
        }
    res = daemon.blueprint_builder.get_review_gate_payload()
    res["has_project"] = True
    return res


@router.post("/review/update")
async def update_review_gate_settings(
    payload: dict[str, Any], request: Request
) -> dict[str, Any]:
    """Apply user edits to world rules, hierarchy, characters, outline, style, or blacklist."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        raise HTTPException(status_code=400, detail="目前無作用中的小說專案。")
    try:
        return daemon.blueprint_builder.update_review_gate_data(payload)
    except BlueprintLockedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/review/regenerate")
async def regenerate_review_gate_section(
    payload: RegenerateSectionPayload, request: Request
) -> dict[str, Any]:
    """Regenerate a single section of the Review Gate workbench via AI or blueprint engine."""
    valid_sections = {"world", "characters", "outline", "style", "blacklist"}
    if payload.section not in valid_sections:
        raise HTTPException(status_code=400, detail=f"不支援的局部重抽分頁：{payload.section}")
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        raise HTTPException(status_code=400, detail="目前無作用中的小說專案。")
    try:
        return await daemon.blueprint_builder.regenerate_review_section(payload.section)
    except BlueprintLockedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AIModelUnavailableError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": str(exc),
                "ai_mode": "OFFLINE",
            },
        )


@router.post("/review/lock_and_start")
async def lock_review_gate_and_start(request: Request) -> dict[str, Any]:
    """Lock the style contract and blueprint, then launch autonomous serialization."""
    daemon = request.app.state.daemon
    if not daemon.has_active_project():
        raise HTTPException(status_code=400, detail="目前無作用中的小說專案。")
    cfg = daemon.project_repo.get_project_config()
    if not cfg.get("style_contract_text"):
        raise HTTPException(status_code=400, detail="風格契約書尚未生成，無法啟動連載。")
    try:
        return await daemon.start_serialization()
    except AIModelUnavailableError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "AI_MODEL_UNAVAILABLE",
                "needs_env_setup": True,
                "detail": str(exc),
                "ai_mode": "OFFLINE",
            },
        )
