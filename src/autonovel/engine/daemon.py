"""Decoupled background writing daemon with graceful pause, SSE pub/sub, and watchdog."""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path
from typing import Any

from autonovel.config import RuntimeConfig, get_logger
from autonovel.db.project_manager import ProjectManager
from autonovel.db.project_repo import ProjectRepository
from autonovel.db.reference_repo import ReferenceRepository
from autonovel.db.snapshot_manager import SnapshotManager
from autonovel.engine.blacklist_scanner import BlacklistScanner
from autonovel.engine.chapter_pipeline import ChapterPipeline
from autonovel.engine.context_router import ContextRouter
from autonovel.engine.deep_organizer import DeepOrganizer
from autonovel.engine.entity_resolver import EntityResolver
from autonovel.engine.hardware_guard import HardwareGuard
from autonovel.engine.pacing_monitor import PacingMonitor
from autonovel.engine.quality_gate import QualityGateEngine
from autonovel.engine.volume_planner import VolumePlanner
from autonovel.llm.embedder import CpuEmbedder
from autonovel.llm.llama_manager import LlamaServerManager
from autonovel.llm.llm_client import LLMClient
from autonovel.reference.chunk_extractor import ReferenceExtractor
from autonovel.reference.macro_extractor import MacroPacingExtractor
from autonovel.reference.retriever import ReferenceRetriever
from autonovel.schemas.domain import AnomalyType, PipelineState, VolumeStatus
from autonovel.wizard.blueprint_builder import BlueprintBuilder
from autonovel.wizard.qa_engine import AIModelUnavailableError, WizardQAEngine

logger = get_logger(__name__)


class NovelWritingDaemon:
    """Singleton background serialization engine decoupled from browser lifecycle."""

    def __init__(
        self,
        config: RuntimeConfig,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Assemble all repositories, engines, hardware guards, and event queues."""
        self.config = config
        self.project_manager = ProjectManager(config)
        self.active_project_id = self.project_manager.get_active_project_id()
        if self.active_project_id:
            proj_paths = self.project_manager.get_project_paths(self.active_project_id)
            self.project_repo = ProjectRepository(proj_paths["project_db_path"])
        else:
            self.project_repo = None
        self.ref_repo = ReferenceRepository(config.reference_db_path)
        self.snapshots = (
            SnapshotManager(
                proj_paths["project_db_path"],
                proj_paths["snapshot_dir"],
                config.max_periodic_snapshots,
            )
            if self.active_project_id
            else None
        )
        self.hardware_guard = HardwareGuard(config)
        self.embedder = CpuEmbedder(config.embedding_model_name, config.embedding_dim)
        # Genre weighting embeds the book's "主:子" label; borrow the CPU embedder.
        self.ref_repo.set_embed_fn(self.embedder.embed_text)
        self.llama_manager = LlamaServerManager(config)
        self.llm_client = llm_client or LLMClient(config, self.llama_manager)
        self.ref_extractor = ReferenceExtractor(
            self.ref_repo, self.embedder, self.llm_client, hardware_guard=self.hardware_guard
        )
        self.macro_extractor = MacroPacingExtractor(
            self.ref_repo, self.llm_client
        )
        self.retriever = ReferenceRetriever(
            self.ref_repo,
            self.embedder,
            self.llm_client,
            config.quality_thresholds.reference_similarity_threshold,
            config.quality_thresholds.reference_top_k,
            genre_provider=self._project_genre_label,
        )
        self.qa_engine = WizardQAEngine(
            self.project_repo, self.llm_client, self.ref_repo
        )
        self.blueprint_builder = BlueprintBuilder(
            self.project_repo, self.retriever, self.llm_client
        )
        self.router = ContextRouter(self.project_repo, config.context_budget)
        self.resolver = EntityResolver(self.project_repo)
        self.scanner = BlacklistScanner(self.project_repo, self.llm_client)
        self.quality_gate = QualityGateEngine(
            self.project_repo,
            self.scanner,
            self.resolver,
            config.quality_thresholds,
            self.llm_client,
        )
        self.pacing = PacingMonitor(self.project_repo, self.ref_repo)
        self.deep_organizer = DeepOrganizer(self.project_repo)
        self.volume_planner = VolumePlanner(
            self.project_repo, self.retriever, config.quality_thresholds, self.llm_client
        )
        self.chapter_pipeline = ChapterPipeline(
            config,
            self.project_repo,
            self.router,
            self.retriever,
            self.pacing,
            self.quality_gate,
            self.llm_client,
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._pause_requested = False
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self.current_stream_buffer = ""
        self.current_writing_chapter: int | None = None
        self.last_chapter_duration_sec: float = 0.0
        self.last_tokens_per_sec: float = 0.0
        self._last_user_activity_ts: float = time.time()
        self._idle_watchdog_task: asyncio.Task[None] | None = None

    def record_user_activity(self) -> None:
        """Mark user interaction on the frontend or API to prevent idle model unloading."""
        self._last_user_activity_ts = time.time()

    def get_idle_seconds(self) -> float:
        """Return elapsed seconds since the last frontend user interaction."""
        return max(0.0, time.time() - self._last_user_activity_ts)

    def is_model_in_use(self) -> bool:
        """Check if LLM model is actively generating text, serializing, or extracting references."""
        if self._worker_task is not None and not self._worker_task.done():
            return True
        if (
            hasattr(self, "ref_extractor")
            and getattr(self.ref_extractor, "_bg_task", None) is not None
            and not self.ref_extractor._bg_task.done()
        ):
            return True
        if self.llm_client and self.llm_client.is_busy():
            return True
        if hasattr(self, "env_download_manager") and getattr(self.env_download_manager, "state", {}).get("active", False):
            return True
        return False

    def start_idle_watchdog(self, check_interval_sec: float = 10.0) -> asyncio.Task[None]:
        """Start background watchdog loop monitoring frontend idle time and unloading model when appropriate."""
        if self._idle_watchdog_task and not self._idle_watchdog_task.done():
            return self._idle_watchdog_task

        async def _idle_loop() -> None:
            logger.info(
                "Started model idle watchdog (threshold: %.1f s, check interval: %.1f s).",
                self.config.model_idle_timeout_sec,
                check_interval_sec,
            )
            while True:
                try:
                    await asyncio.sleep(check_interval_sec)
                    idle_sec = self.get_idle_seconds()
                    if idle_sec >= self.config.model_idle_timeout_sec:
                        if self.llama_manager.is_healthy() and not self.is_model_in_use():
                            logger.info(
                                "Frontend inactive for %.1f s (>= %.1f s threshold) and model is idle. Automatically stopping llama-server to release GPU VRAM.",
                                idle_sec,
                                self.config.model_idle_timeout_sec,
                            )
                            self.llama_manager.stop_server()
                            self.broadcast_event(
                                "env_status_changed",
                                {
                                    "llama_server_healthy": False,
                                    "reason": "IDLE_AUTO_RELEASE",
                                },
                            )
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.warning("Idle watchdog encountered error: %s", exc)

        self._idle_watchdog_task = asyncio.create_task(_idle_loop())
        return self._idle_watchdog_task

    def stop_idle_watchdog(self) -> None:
        """Cancel idle watchdog background task on daemon shutdown."""
        if self._idle_watchdog_task and not self._idle_watchdog_task.done():
            self._idle_watchdog_task.cancel()
            self._idle_watchdog_task = None

    async def shutdown(self) -> None:
        """Cleanly terminate daemon worker, idle watchdog, hardware locks, and LLM client sessions."""
        self.stop_idle_watchdog()
        self.request_graceful_pause()
        self.hardware_guard.release_anti_sleep_lock()
        self.llama_manager.stop_server()
        if self.llm_client and hasattr(self.llm_client, "aclose"):
            await self.llm_client.aclose()
        logger.info("Daemon shutdown completed cleanly.")

    def has_active_project(self) -> bool:
        """Check if an active novel project is currently loaded."""
        return bool(self.active_project_id and self.project_repo is not None)

    def _project_genre_label(self) -> str:
        """Return the active project's "主:子" genre label used for genre-weighted retrieval."""
        if self.project_repo is None:
            return ""
        cfg = self.project_repo.get_project_config()
        primary = str(cfg.get("genre_primary") or "").strip()
        sub = str(cfg.get("genre_sub") or "").strip()
        return f"{primary}:{sub}" if (primary and sub) else (primary or sub)

    async def _pause_worker(self) -> None:
        """Cancel the serialization worker and wait until it stops touching the old repo.

        A cancelled task keeps running synchronously between awaits, so rebinding
        repositories before it actually finishes would let stale commits land in
        the new project. Callers must await this before detaching ``project_repo``.
        """
        self._pause_requested = True
        task = self._worker_task
        old_repo = self.project_repo
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=30.0)
            except asyncio.CancelledError:
                # Only swallow the cancellation raised by the worker task itself;
                # a cancellation of the caller must still propagate.
                if not task.cancelled():
                    raise
                logger.info("Serialization worker stopped after cancellation.")
            except asyncio.TimeoutError:
                logger.warning(
                    "Serialization worker did not stop within 30s; detaching anyway."
                )
        self._worker_task = None
        # The whole project directory may have just been deleted (delete route), in
        # which case any SQL would recreate an empty file without the schema.
        if old_repo is not None and Path(old_repo.db_path).exists():
            rolled = old_repo.rollback_staging_chapters()
            if rolled:
                logger.info(
                    "Rolled back STAGING chapters before detaching project: %s", rolled
                )
            old_repo.update_project_config(pipeline_state=PipelineState.PAUSED.value)
        self.hardware_guard.release_anti_sleep_lock()

    async def unload_active_project(self) -> None:
        """Safely detach active project database and enter standby mode."""
        # Clear RUNNING/STAGING first, otherwise startup_recovery(auto_resume=True)
        # would silently resume writing next launch.
        await self._pause_worker()
        self.active_project_id = ""
        self.project_repo = None
        self.snapshots = None
        self.current_writing_chapter = None
        self.current_stream_buffer = ""
        self.qa_engine.repo = None
        self.blueprint_builder.repo = None
        self.router.repo = None
        self.resolver.repo = None
        self.scanner.repo = None
        self.quality_gate.repo = None
        self.pacing.repo = None
        self.deep_organizer.repo = None
        self.volume_planner.repo = None
        self.chapter_pipeline.repo = None
        self.broadcast_event(
            "project_unloaded",
            {"active_project_id": ""},
        )
        logger.info("Unloaded active novel project; daemon entered zero-state standby.")

    async def startup_recovery(self, auto_resume: bool = False) -> list[int]:
        """Roll back dirty STAGING chapters and auto-resume if pipeline_state was RUNNING."""
        try:
            await self.ref_extractor.startup_recovery(
                macro_extractor=self.macro_extractor,
                auto_resume=auto_resume,
                on_skills_progress=self._broadcast_ref_progress,
                on_pacing_progress=self._broadcast_pacing_progress,
            )
        except Exception as exc:
            logger.warning("Reference extraction startup recovery failed: %s", exc)

        if not self.has_active_project():
            return []
        rolled = self.project_repo.rollback_staging_chapters()
        if rolled:
            logger.info("Startup recovery rolled back STAGING chapters: %s", rolled)
        if auto_resume:
            cfg = self.project_repo.get_project_config()
            if cfg.get("pipeline_state") == PipelineState.RUNNING.value:
                try:
                    await self.start_serialization()
                    logger.info("Auto-resumed background serialization after device restart.")
                except (AIModelUnavailableError, RuntimeError) as exc:
                    logger.warning("Auto-resume skipped: %s", exc)
        return rolled

    async def switch_active_project(self, project_id: str) -> dict[str, Any]:
        """Switch the daemon's active project database and rebind all sub-engines."""
        if self._worker_task and not self._worker_task.done():
            logger.info("Pausing active serialization before switching project...")
        await self._pause_worker()

        proj_paths = self.project_manager.switch_project(project_id)
        self.active_project_id = project_id
        self.project_repo = ProjectRepository(proj_paths["project_db_path"])
        self.snapshots = SnapshotManager(
            proj_paths["project_db_path"],
            proj_paths["snapshot_dir"],
            self.config.max_periodic_snapshots,
        )

        self.qa_engine.repo = self.project_repo
        self.blueprint_builder.repo = self.project_repo
        self.router.repo = self.project_repo
        self.resolver.repo = self.project_repo
        self.scanner.repo = self.project_repo
        self.quality_gate.repo = self.project_repo
        self.pacing.repo = self.project_repo
        self.deep_organizer.repo = self.project_repo
        self.volume_planner.repo = self.project_repo
        self.chapter_pipeline.repo = self.project_repo

        self.current_stream_buffer = ""
        self.current_writing_chapter = None
        self._pause_requested = False
        await self.startup_recovery(auto_resume=False)

        cfg = self.project_repo.get_project_config()
        self.broadcast_event(
            "project_switched",
            {"project_id": project_id, "project_config": cfg},
        )
        return {
            "active_project_id": project_id,
            "project_config": cfg,
            "paths": {k: str(v) for k, v in proj_paths.items()},
        }

    def subscribe_events(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a browser SSE subscriber queue."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._subscribers.add(q)
        return q

    def unsubscribe_events(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a closed browser SSE subscriber queue."""
        self._subscribers.discard(q)

    def broadcast_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Push an event payload to all active SSE subscribers."""
        payload = {"event": event_type, "data": data}
        for q in list(self._subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(payload)

    def _broadcast_ref_progress(self, status: dict[str, Any]) -> None:
        """Push unified skills reference extraction progress to subscribers."""
        self.broadcast_event("ref_progress", status)

    def _broadcast_pacing_progress(self, status: dict[str, Any]) -> None:
        """Push macro pacing extraction progress to subscribers."""
        self.broadcast_event("ref_pacing_progress", status)

    async def ais_llm_ready(self, clear_manual_unload: bool = False) -> bool:
        """Async readiness probe: server start-up polling must run off the event loop.

        A passive probe (chapter loop, Q&A generation, polling endpoints) must never
        resurrect a model the user explicitly unloaded; only an explicit resume passes
        ``clear_manual_unload=True``.
        """
        if not self.llm_client:
            return False
        manager = self.llm_client.server_manager
        unloaded = bool(
            getattr(self.llama_manager, "_manually_unloaded", False)
            or getattr(manager, "_manually_unloaded", False)
        )
        if unloaded:
            if not clear_manual_unload:
                return False
            self.llama_manager._manually_unloaded = False
            manager._manually_unloaded = False
        if manager.is_healthy():
            return True
        await manager.ensure_running_async()
        return manager.is_healthy()

    async def start_serialization(self) -> dict[str, Any]:
        """Transition state to RUNNING, lock anti-sleep, and launch background task."""
        if not self.has_active_project():
            raise RuntimeError("目前尚無作用中的小說專案，請先建立或切換作品。")
        if not await self.ais_llm_ready(clear_manual_unload=True):
            raise AIModelUnavailableError(
                "本地 AI 模型未就緒或推論服務離線，無法啟動連載寫作。請先至環境設置完成模型下載與推論服務設定。"
            )
        self._pause_requested = False
        self.project_repo.update_project_config(pipeline_state=PipelineState.RUNNING.value)
        self.hardware_guard.acquire_anti_sleep_lock()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._serialization_loop())
        self.broadcast_event("state_change", {"pipeline_state": PipelineState.RUNNING.value})
        return {"pipeline_state": PipelineState.RUNNING.value}

    def request_graceful_pause(self) -> dict[str, Any]:
        """Signal the daemon to pause safely after the current chapter finishes atomic commit."""
        self._pause_requested = True
        if not self.has_active_project():
            return {"pipeline_state": "IDLE", "graceful_pending": False}
        cfg = self.project_repo.get_project_config()
        if self.current_writing_chapter is None:
            self.project_repo.update_project_config(pipeline_state=PipelineState.PAUSED.value)
            self.hardware_guard.release_anti_sleep_lock()
            return {"pipeline_state": PipelineState.PAUSED.value, "graceful_pending": False}
        return {"pipeline_state": cfg.get("pipeline_state"), "graceful_pending": True}

    def is_serialization_active(self) -> bool:
        """Return True while the background loop owns the chapter pipeline.

        ``pipeline_state`` is written before the worker task starts, so checking both
        the task and the state closes the window where a manual step could interleave.
        """
        if self._worker_task is not None and not self._worker_task.done():
            return True
        if not self.has_active_project():
            return False
        return (
            self.project_repo.get_project_config().get("pipeline_state")
            == PipelineState.RUNNING.value
        )

    async def run_single_step(self) -> dict[str, Any] | None:
        """Execute the next unwritten chapter and any triggered volume/maintenance hooks."""
        if not self.has_active_project():
            return None
        if not await self.ais_llm_ready():
            raise AIModelUnavailableError(
                "本地 AI 模型未就緒或推論服務離線，無法執行單章生成。請先至環境設置完成模型下載與推論服務設定。"
            )
        unwritten = self.project_repo.fetch_all(
            "SELECT chapter_num, volume_num FROM chapters WHERE status = 'UNWRITTEN' ORDER BY chapter_num ASC LIMIT 1;"
        )
        if not unwritten:
            return await self._handle_volume_completion()
        ch_num = int(unwritten[0]["chapter_num"])
        vol_num = int(unwritten[0]["volume_num"])
        self.project_repo.set_volume_status(vol_num, VolumeStatus.WRITING.value)
        self.current_writing_chapter = ch_num
        self.current_stream_buffer = ""
        self.broadcast_event("chapter_start", {"chapter_num": ch_num, "volume_num": vol_num})

        def _on_token(tok: str) -> None:
            self.current_stream_buffer += tok
            self.broadcast_event("token", {"chapter_num": ch_num, "token": tok})

        def _on_status(st: str) -> None:
            self.broadcast_event(
                "chapter_status", {"chapter_num": ch_num, "status": st}
            )

        try:
            result = await self.chapter_pipeline.execute_chapter(
                ch_num, on_token=_on_token, on_status=_on_status
            )
            self.last_chapter_duration_sec = float(result.get("duration_sec", 0.0))
            self.last_tokens_per_sec = float(result.get("tokens_per_sec", 0.0))
            self.broadcast_event("chapter_complete", result)
            await self._run_post_chapter_hooks(ch_num)
            return result
        finally:
            self.current_writing_chapter = None

    async def _run_post_chapter_hooks(self, ch_num: int) -> None:
        """Run every-10-chapter sequential deep maintenance, snapshots, and thermal guard."""
        if ch_num % self.config.deep_organize_interval_chapters == 0:
            maint = self.deep_organizer.run_maintenance(ch_num)
            self.broadcast_event("deep_organize", maint)
        if ch_num % self.config.periodic_snapshot_interval_chapters == 0:
            snap = self.snapshots.create_snapshot("periodic", chapter_num=ch_num)
            self.broadcast_event("snapshot_created", snap)
        await self.hardware_guard.check_and_cooldown_if_hot()

    async def _handle_volume_completion(self) -> dict[str, Any] | None:
        """Execute end-of-volume total review, milestone snapshot, and next volume rollout."""
        vols = self.project_repo.fetch_all(
            "SELECT * FROM volumes ORDER BY volume_num DESC LIMIT 1;"
        )
        if not vols:
            return None
        cur_vol = int(vols[0]["volume_num"])
        report = self.volume_planner.evaluate_acceptance_report(cur_vol)
        if not report.passed:
            low_chs = report.details.get("bottom_20_percent_chapters", [])
            for l_ch in low_chs:
                try:
                    await self.chapter_pipeline.execute_chapter(
                        int(l_ch), seed=random.randint(1, 999999)
                    )
                except Exception as exc:  # noqa: BLE001 - one bad chapter must not abort the batch
                    logger.error(
                        "Volume acceptance rewrite failed for chapter %s: %s", l_ch, exc
                    )
                    self.project_repo.log_anomaly(
                        chapter_num=int(l_ch),
                        anomaly_type=AnomalyType.WATCHDOG_RESTART.value,
                        severity="ERROR",
                        resolution_method=f"分卷驗收重寫失敗，已跳過該章：{exc}",
                        affected_chapters=[int(l_ch)],
                    )
            report = self.volume_planner.evaluate_acceptance_report(cur_vol)
        if not report.passed:
            # Never mark a volume COMPLETED on a failed report; otherwise every later
            # volume inherits the quality debt and the gate becomes meaningless.
            logger.warning(
                "Volume %s acceptance still failing after rewrites: %s",
                cur_vol,
                report.failed_reasons,
            )
            self.project_repo.log_anomaly(
                chapter_num=self.current_writing_chapter or 0,
                anomaly_type=AnomalyType.PACING_ALERT.value,
                severity="CRITICAL",
                resolution_method="分卷驗收未通過，暫停連載等待人工介入：" + "；".join(report.failed_reasons),
                affected_chapters=[cur_vol * 10],
                detail_diff=report.model_dump(),
            )
            self.broadcast_event("volume_acceptance_failed", report.model_dump())
            # Returning None stops the loop; _serialization_loop then persists PAUSED.
            return None
        self.project_repo.set_volume_status(cur_vol, VolumeStatus.COMPLETED.value)
        # Dynamically query the actual maximum completed chapter for this volume to prevent snapshot misnumbering
        max_ch_rows = self.project_repo.fetch_all(
            "SELECT MAX(chapter_num) AS max_c FROM chapters WHERE volume_num = ?;",
            (cur_vol,),
        )
        actual_ch = (
            int(max_ch_rows[0]["max_c"])
            if max_ch_rows and max_ch_rows[0]["max_c"] is not None
            else cur_vol * 10
        )
        if self.snapshots:
            self.snapshots.create_snapshot(
                "milestone", chapter_num=actual_ch, label=f"vol{cur_vol}"
            )
        if self.volume_planner.should_conclude_novel():
            self.project_repo.update_project_config(pipeline_state=PipelineState.COMPLETED.value)
            self.hardware_guard.release_anti_sleep_lock()
            self.broadcast_event("novel_completed", report.model_dump())
            return None
        next_vol = await self.volume_planner.plan_next_volume(cur_vol + 1)
        self.broadcast_event("volume_planned", next_vol)
        return next_vol

    async def _serialization_loop(self) -> None:
        """Continuous zero-wait serialization loop until paused, concluded, or failed."""
        try:
            while not self._pause_requested:
                res = await self.run_single_step()
                if res is None:
                    break
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            # Cancellation is a normal pause path; the finally block persists PAUSED.
            raise
        except Exception as exc:
            logger.error("Daemon loop caught error: %s", exc)
            repo = self.project_repo
            if repo is not None:
                repo.log_anomaly(
                    chapter_num=self.current_writing_chapter or 0,
                    anomaly_type=AnomalyType.WATCHDOG_RESTART.value,
                    severity="CRITICAL",
                    resolution_method=f"看門狗捕獲異常並回滾暫存章節：{exc}",
                    affected_chapters=[self.current_writing_chapter or 0],
                )
                repo.rollback_staging_chapters()
            self.broadcast_event("daemon_error", {"detail": str(exc)})
        finally:
            # An abnormal stop must never leave pipeline_state=RUNNING behind, or
            # startup_recovery(auto_resume=True) would silently restart writing while
            # the UI still reports the daemon as dead. COMPLETED/PAUSED are kept as-is.
            repo = self.project_repo
            if repo is not None:
                state = repo.get_project_config().get("pipeline_state")
                if state == PipelineState.RUNNING.value:
                    repo.update_project_config(
                        pipeline_state=PipelineState.PAUSED.value
                    )
                    self.broadcast_event(
                        "state_change", {"pipeline_state": PipelineState.PAUSED.value}
                    )
            self.hardware_guard.release_anti_sleep_lock()

    async def rewrite_latest_chapter(self) -> dict[str, Any]:
        """Re-roll random seed and rewrite only the latest completed chapter."""
        if not self.has_active_project():
            raise ValueError("目前尚無作用中的小說專案。")
        if not await self.ais_llm_ready():
            raise AIModelUnavailableError(
                "本地 AI 模型未就緒或推論服務離線，無法重寫章節。請先至環境設置完成模型下載與推論服務設定。"
            )
        rows = self.project_repo.fetch_all(
            "SELECT MAX(chapter_num) AS max_c FROM chapters WHERE status IN ('COMPLETED', 'REVISED');"
        )
        latest = int(rows[0]["max_c"] or 0) if rows else 0
        if latest <= 0:
            raise ValueError("尚無已完成章節可供重寫。")
        new_seed = random.randint(1, 2_000_000_000)
        res = await self.chapter_pipeline.execute_chapter(latest, seed=new_seed)
        self.broadcast_event("chapter_rewritten", res)
        return res
