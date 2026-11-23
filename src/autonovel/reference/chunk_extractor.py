"""Chunking (2,500 chars with 250 overlap) and dual-stage extraction pipeline for reference books."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import os
from pathlib import Path
import time
from typing import Any

from autonovel.config import env_int, get_logger
from autonovel.db.connection import atomic_transaction
from autonovel.db.reference_repo import ReferenceRepository, _clean_rule_tag_name
from autonovel.engine.prose_sanitizer import (
    convert_simplified_to_traditional,
    normalize_quotes_to_traditional,
    strip_web_novel_noise,
)
from autonovel.llm.embedder import CpuEmbedder
from autonovel.llm.llm_client import LLMClient, LLMResponseFormatError
from autonovel.reference.file_parser import compute_sha256, parse_novel_file
from autonovel.reference.macro_extractor import (
    CHAPTER_HEADER_RE,
    ChapterFactExtractor,
    build_factual_pacing_curve,
    extract_table_of_contents,
    pack_chapters_dynamically,
    reconstruct_chapter_facts_from_curve,
    select_high_value_chapters_zscore,
)
from autonovel.schemas.domain import ReferenceIngestItem
from autonovel.schemas.llm_contracts import (
    BookBlueprintSynthesisContract,
    ChapterFactExtractContract,
    CharacterArchetypeContract,
    FactionGeopoliticsContract,
    GroupEntityDeltas,
    PowerHierarchyContract,
    OrthogonalTagsContract,
    ReferenceChunkExtractionContract,
    TagMaintenanceContract,
)
from autonovel.wizard.qa_engine import AIModelUnavailableError

logger = get_logger(__name__)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temp file + atomic replace so a crash never truncates it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)

PUNCT_TERMINATORS: set[str] = {"\n", "。", "！", "？", "!", "?"}
PUNCT_CLOSING: set[str] = {"”", "’", "」", "』", '"', "'", "）", ")"}
PUNCT_SECONDARY: set[str] = {"；", ";", "…"}


def _find_snapped_end(
    sec: str, start: int, target_end: int, max_lookback: int = 250
) -> int:
    """Find the nearest punctuation boundary at or before target_end to prevent cutting sentences."""
    if target_end >= len(sec):
        return len(sec)
    search_start = max(start + 1, target_end - max_lookback)
    for i in range(target_end - 1, search_start - 1, -1):
        if sec[i] in PUNCT_TERMINATORS:
            end_pos = i + 1
            while end_pos < len(sec) and sec[end_pos] in PUNCT_CLOSING:
                end_pos += 1
            if end_pos < len(sec) and sec[end_pos] == "\n":
                end_pos += 1
            if end_pos <= target_end:
                return end_pos
    for i in range(target_end - 1, search_start - 1, -1):
        if sec[i] in PUNCT_SECONDARY:
            end_pos = i + 1
            while end_pos < len(sec) and sec[end_pos] in PUNCT_CLOSING:
                end_pos += 1
            if end_pos <= target_end:
                return end_pos
    return target_end


def _find_snapped_start(
    sec: str, raw_start: int, bound_end: int, max_lookahead: int = 100
) -> int:
    """Snap next chunk start forward to the beginning of a sentence within the overlap zone."""
    if raw_start <= 0:
        return 0
    limit = min(bound_end - 50, raw_start + max_lookahead)
    for i in range(raw_start, limit):
        if sec[i] in PUNCT_TERMINATORS or sec[i] in PUNCT_SECONDARY:
            pos = i + 1
            while pos < limit and sec[pos] in PUNCT_CLOSING:
                pos += 1
            while pos < limit and sec[pos] in (" ", "\t", "\n", "\r"):
                pos += 1
            return pos
    return raw_start


def split_novel_into_chunks(
    text: str, chunk_size: int = 2500, overlap: int = 250
) -> list[str]:
    """Split novel text by chapter boundaries and 2,500-char windows with 250-char overlap, stripping web noise."""
    cleaned = strip_web_novel_noise(text.strip()).strip()
    if not cleaned:
        return []
    parts = CHAPTER_HEADER_RE.split(cleaned)
    raw_sections: list[str] = []
    if len(parts) <= 1:
        raw_sections = [cleaned]
    else:
        if parts[0].strip():
            raw_sections.append(parts[0].strip())
        for idx in range(1, len(parts), 2):
            header = parts[idx].strip()
            body = parts[idx + 1].strip() if idx + 1 < len(parts) else ""
            raw_sections.append(f"{header}\n{body}".strip())
    return _window_sections(raw_sections, chunk_size, overlap)


def _window_sections(
    sections: list[str], chunk_size: int, overlap: int
) -> list[str]:
    """Slice sections exceeding chunk_size into overlapping windows with punctuation snapping."""
    chunks: list[str] = []
    for sec in sections:
        if len(sec) <= chunk_size:
            cleaned = sec.strip()
            if cleaned:
                chunks.append(cleaned)
            continue
        start = 0
        while start < len(sec):
            target_end = min(len(sec), start + chunk_size)
            end = _find_snapped_end(sec, start, target_end, max_lookback=overlap)
            if end <= start:
                end = target_end
            piece = sec[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(sec):
                break
            raw_next_start = max(start + 1, end - overlap)
            start = _find_snapped_start(
                sec, raw_next_start, end, max_lookahead=min(100, overlap // 2)
            )
            if start <= 0 or start >= end:
                start = raw_next_start
    return chunks


def split_targeted_chapters_into_chunks(
    toc: list[dict[str, Any]],
    target_chapter_indices: set[int],
    chunk_size: int = 2500,
    overlap: int = 250,
) -> list[tuple[int, str]]:
    """Slice only selected high-value chapters into chunks with punctuation snapping.

    Returns a list of (chapter_index, chunk_text) tuples.
    """
    results: list[tuple[int, str]] = []
    for idx, ch in enumerate(toc, start=1):
        if idx in target_chapter_indices:
            text = ch.get("text", "").strip()
            title = ch.get("title", f"第{idx}章").strip()
            full_sec = f"{title}\n{text}".strip()
            cleaned = strip_web_novel_noise(full_sec).strip()
            if cleaned:
                ch_chunks = _window_sections([cleaned], chunk_size, overlap)
                for piece in ch_chunks:
                    results.append((idx, piece))
    return results


def _build_density_hints(
    dense_clusters: list[dict[str, Any]], outliers: list[dict[str, Any]]
) -> str:
    """Format DBSCAN clusters and outlier tags as informative prompt hints in Traditional Chinese."""
    lines: list[str] = []
    if dense_clusters:
        lines.append("【高維語義聚類核心密集區（同義或相近概念叢集）】：")
        for c in dense_clusters[:12]:
            tags_str = "、".join(c["tags"][:8])
            lines.append(f"- 聚類簇 #{c['cluster_id']}: [{tags_str}]")
    if outliers:
        outlier_names = [o["tag_name"] for o in outliers if o["count"] <= 2][:25]
        if outlier_names:
            lines.append(
                "【孤立低頻長尾孤島候選（可能為偶發噪點或異體字）】： "
                + "、".join(outlier_names)
            )
    return "\n".join(lines)


def normalize_extracted_tag(tag: str) -> str:
    """Normalize extracted tag by stripping surrounding quotes, brackets, and whitespace."""
    return tag.strip().strip("「」\"'【】[] ")


class ReferenceExtractor:
    """Dual-stage reference novel extractor: fact blueprint synthesis followed by concurrent pattern extraction."""

    def __init__(
        self,
        repo: ReferenceRepository,
        embedder: CpuEmbedder,
        llm_client: LLMClient | None = None,
        hardware_guard: Any | None = None,
    ) -> None:
        """Initialize extractor with repository, CPU embedder, LLM client, and thermal hardware guard."""
        self.repo = repo
        self.embedder = embedder
        self._llm_client = llm_client
        self.hardware_guard = hardware_guard
        self.fact_extractor = ChapterFactExtractor(llm_client=llm_client, lifecycle=self)
        self.status: dict[str, Any] = {
            "active": False,
            "paused": False,
            "current_file": "",
            "batch_total": 0,
            "batch_index": 0,
            "processed_chunks": 0,
            "total_chunks": 0,
            "percent": 0.0,
            "phase": "IDLE",
        }
        self.pacing_status: dict[str, Any] = {
            "active": False,
            "novel_title": "",
            "percent": 0.0,
            "phase": "IDLE",
            "detail": "",
            "batch_index": 0,
            "batch_total": 0,
        }
        self._is_paused: bool = False
        self._is_cancelled: bool = False
        self._bg_task: asyncio.Task[Any] | None = None
        self._pending_jobs: list[dict[str, Any]] = []
        self._completed_books: list[dict[str, Any]] = []

    @property
    def llm_client(self) -> LLMClient | None:
        return self._llm_client

    @llm_client.setter
    def llm_client(self, client: LLMClient | None) -> None:
        self._llm_client = client
        if hasattr(self, "fact_extractor") and self.fact_extractor:
            self.fact_extractor.llm_client = client

    @property
    def manifest_path(self) -> Path:
        """Physical manifest file path recording multi-task queue and resume state."""
        cache_dir = self.repo.db_path.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "extraction_manifest.json"

    def _save_manifest(self) -> None:
        """Atomically persist current extraction queue and execution state to physical manifest."""
        try:
            payload = {
                "version": 1,
                "updated_at": time.time(),
                "state": (
                    "PAUSED"
                    if self._is_paused
                    else ("ACTIVE" if self.status.get("active") else ("IDLE" if not self._pending_jobs else "QUEUED"))
                ),
                "is_paused": self._is_paused,
                "status": dict(self.status),
                "pacing_status": dict(self.pacing_status),
                "pending_jobs": [
                    {
                        "pacing_items": [
                            it.model_dump() if hasattr(it, "model_dump") else dict(it)
                            for it in job.get("pacing_items", [])
                        ],
                        "skill_items": [
                            it.model_dump() if hasattr(it, "model_dump") else dict(it)
                            for it in job.get("skill_items", [])
                        ],
                        "pacing_jitter": float(job.get("pacing_jitter", 0.0)),
                        "force_reextract": bool(job.get("force_reextract", False)),
                        "pacing_completed_indices": list(job.get("pacing_completed_indices", [])),
                        "skill_completed_indices": list(job.get("skill_completed_indices", [])),
                        "failed_items": list(job.get("failed_items", [])),
                    }
                    for job in self._pending_jobs
                ],
                "completed_books": list(self._completed_books[-50:]),
            }
            _write_json_atomic(self.manifest_path, payload)
        except Exception as exc:
            logger.warning("Failed to save extraction manifest: %s", exc)

    def _clear_manifest(self) -> None:
        """Safely clean up or unlink the manifest file when all jobs conclude or cancel."""
        try:
            if self.manifest_path.exists():
                self.manifest_path.unlink()
        except Exception as exc:
            logger.warning("Failed to clear extraction manifest: %s", exc)

    def pause(self) -> bool:
        """Pause ongoing chunk ingestion."""
        self._is_paused = True
        self.status["paused"] = True
        self.pacing_status["paused"] = True
        self._save_manifest()
        return True

    def resume(
        self,
        macro_extractor: Any = None,
        on_skills_progress: Callable[[dict[str, Any]], None] | None = None,
        on_pacing_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> bool:
        """Resume paused chunk ingestion."""
        self._is_paused = False
        self.status["paused"] = False
        self.pacing_status["paused"] = False
        self._save_manifest()
        if (self._bg_task is None or self._bg_task.done()) and self._pending_jobs:
            if macro_extractor:
                for j in self._pending_jobs:
                    if not j.get("macro_extractor"):
                        j["macro_extractor"] = macro_extractor
            self._start_queue_task(on_skills_progress, on_pacing_progress)
        return True

    def cancel(self) -> bool:
        """Cancel ongoing chunk ingestion and macro pacing extraction."""
        self._is_cancelled = True
        self._is_paused = False
        self._pending_jobs.clear()
        if self._bg_task and not self._bg_task.done():
            self._bg_task.cancel()
        self.status["active"] = False
        self.status["paused"] = False
        self.pacing_status["active"] = False
        self.pacing_status["phase"] = "IDLE"
        self._clear_manifest()
        return True

    def is_paused(self) -> bool:
        """Check if extraction is currently paused."""
        return self._is_paused

    def is_cancelled(self) -> bool:
        """Check if extraction has been requested to cancel."""
        return self._is_cancelled

    async def _wait_if_paused_or_unloaded(
        self,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> bool:
        """Pause loop: wait while extraction is paused or model is manually unloaded. Returns False if cancelled."""
        paused = False
        while True:
            if self._is_cancelled:
                return False
            is_u = getattr(getattr(self.llm_client, "server_manager", None), "_manually_unloaded", False) is True
            if not self._is_paused and not is_u:
                break
            paused = True
            self.status["paused"] = True
            self.pacing_status["paused"] = True
            if on_progress:
                on_progress(dict(self.status))
            await asyncio.sleep(0.5)

        if paused:
            # Resumed: wait until server is healthy before continuing
            while True:
                if self._is_cancelled:
                    return False
                sm = getattr(self.llm_client, "server_manager", None)
                if sm is None:
                    # No LLM client configured (e.g. offline extraction): nothing to wait for.
                    break
                if bool(sm.is_healthy()):
                    break
                await asyncio.sleep(0.5)
            self.status["paused"] = False
            self.pacing_status["paused"] = False
            if on_progress:
                on_progress(dict(self.status))
        return True

    def is_source_in_use(self, source_id: int) -> bool:
        """Check if a source is currently being extracted or queued in pending jobs."""
        if not self.status.get("active") and not self.pacing_status.get("active") and not self._pending_jobs:
            return False
        cur_file = self.status.get("current_file") or self.pacing_status.get("novel_title")
        if cur_file:
            src = self.repo.get_source(source_id)
            if src and (src["file_name"] == cur_file or Path(src["file_name"]).stem == cur_file):
                return True
        for job in self._pending_jobs:
            for item in job.get("pacing_items", []) + job.get("skill_items", []):
                fp = getattr(item, "file_path", None) or (item.get("file_path") if isinstance(item, dict) else None)
                if fp:
                    p = Path(fp)
                    if p.exists():
                        sha = compute_sha256(p)
                        existing = self.repo.get_source_by_hash(sha)
                        if existing and existing["id"] == source_id:
                            return True
        return False

    def get_pacing_status(self) -> dict[str, Any]:
        """Return the current macro pacing extraction visual progress."""
        return dict(self.pacing_status)

    def update_pacing_status(self, **kwargs: Any) -> None:
        """Update macro pacing extraction progress attributes."""
        self.pacing_status.update(kwargs)

    def get_current_status(self) -> dict[str, Any]:
        """Return the current extraction status snapshot including pacing, queue and recovery metadata."""
        snap = dict(self.status)
        snap["pacing_status"] = dict(self.pacing_status)
        snap["queued_batches"] = len(self._pending_jobs)
        total_pending = sum(
            len(j.get("pacing_items", [])) + len(j.get("skill_items", []))
            for j in self._pending_jobs
        )
        snap["queued_items_count"] = total_pending
        snap["completed_books"] = list(self._completed_books[-50:])
        snap["manifest_active"] = self.manifest_path.exists()
        return snap

    async def ingest_file(
        self,
        file_path: Path,
        force_reextract: bool = False,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        batch_index: int = 1,
        batch_total: int = 1,
        target_chapter_indices: set[int] | None = None,
        chapter_scores_map: dict[int, float] | None = None,
        reuse_fresh_blueprint: bool = False,
        pre_parsed_novel: tuple[str, list[dict[str, Any]]] | None = None,
    ) -> int:
        """Execute two-stage full-text ingestion: Chapter fact blueprint synthesis -> Concurrent pattern extraction."""
        sha = compute_sha256(file_path)
        novel_title = file_path.stem

        cache_dir = self.repo.db_path.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        existing_src = self.repo.get_source_by_hash(sha)
        src_id_for_check = existing_src["id"] if existing_src else None
        s1_ckpt = cache_dir / f"stage1_facts_{src_id_for_check}.json" if src_id_for_check else None
        s2_ckpt = cache_dir / f"stage2_chunks_{src_id_for_check}.json" if src_id_for_check else None
        has_bp = bool(src_id_for_check and self.repo.has_book_blueprint(src_id_for_check))

        # A partially extracted book must stay resumable; only skip the file when the
        # previous run already produced one entry for every recorded chunk.
        resuming = bool(
            existing_src
            and not force_reextract
            and (
                int(existing_src["entry_count"]) > 0
                or (s1_ckpt and s1_ckpt.exists())
                or (s2_ckpt and s2_ckpt.exists())
                or has_bp
            )
        )
        if (
            resuming
            and int(existing_src["chunk_count"]) > 0
            and int(existing_src["entry_count"]) >= int(existing_src["chunk_count"])
        ):
            return 0

        if pre_parsed_novel:
            raw_text, toc = pre_parsed_novel
        else:
            raw_text, _ = parse_novel_file(file_path)
            toc = extract_table_of_contents(raw_text)
        total_chapters = max(1, len(toc))
        source_id = self.repo.register_or_reset_source(
            sha, file_path.name, 0, reset_entries=not resuming
        )

        self.status = {
            "active": True,
            "paused": self._is_paused,
            "current_file": file_path.name,
            "batch_index": batch_index,
            "batch_total": batch_total,
            "processed_chunks": 0,
            "total_chunks": 0,
            "percent": 0.0,
            "phase": "STAGE_1_FACTS",
        }
        if on_progress:
            on_progress(dict(self.status))

        # Check if permanent blueprint already exists (e.g. from Track B or prior extraction)
        blueprint = None
        if (not force_reextract or reuse_fresh_blueprint) and self.repo.has_book_blueprint(source_id):
            existing_bp = self.repo.get_book_blueprint(source_id)
            if existing_bp:
                blueprint = BookBlueprintSynthesisContract(
                    novel_title=existing_bp["novel_title"],
                    genre_primary=existing_bp["genre_primary"],
                    genre_sub=existing_bp["genre_sub"],
                    macro_outline=existing_bp["macro_outline"]
                    if isinstance(existing_bp["macro_outline"], str)
                    else json.dumps(existing_bp["macro_outline"], ensure_ascii=False),
                    character_network=[
                        CharacterArchetypeContract(**c)
                        for c in existing_bp.get("character_network", [])
                    ],
                    faction_geopolitics=[
                        FactionGeopoliticsContract(**f)
                        for f in existing_bp.get("faction_geopolitics", [])
                    ],
                    hierarchy_rules=[
                        PowerHierarchyContract(**h)
                        for h in existing_bp.get("hierarchy_rules", [])
                    ],
                    world_laws=existing_bp.get("world_laws", []),
                )
                cached_curve = existing_bp.get("factual_pacing_curve", [])
                if target_chapter_indices is None and cached_curve:
                    cached_facts = reconstruct_chapter_facts_from_curve(cached_curve)
                    if len(cached_facts) >= 10:
                        target_chapter_indices, chapter_scores_map, _ = select_high_value_chapters_zscore(
                            chapter_facts=cached_facts,
                            min_pct=0.08,
                            max_pct=0.12,
                            vol_batch_size=20,
                        )
                logger.info("Reusing existing factual blueprint for %s (source_id=%s)", novel_title, source_id)

        if blueprint is None:
            # =========================================================================
            # Stage 1: Full-Text Batched Chapter Fact Extraction & Blueprint Synthesis
            # =========================================================================
            self.fact_extractor.lifecycle = self
            checkpoint_file = cache_dir / f"stage1_facts_{source_id}.json" if source_id else None

            chapter_facts: list[ChapterFactExtractContract] = []
            entity_deltas: list[GroupEntityDeltas] = []
            last_anchor_event = ""

            if checkpoint_file and checkpoint_file.exists() and not force_reextract:
                try:
                    cached_data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                    cached_total = cached_data.get("total_chapters")
                    if cached_total is not None and int(cached_total) != total_chapters:
                        logger.info(
                            "Ignoring stale Stage 1 checkpoint for %s (chapter count %s != %s).",
                            novel_title,
                            cached_total,
                            total_chapters,
                        )
                    else:
                        cached_facts = [ChapterFactExtractContract.model_validate(cf) for cf in cached_data.get("chapter_facts", [])]
                        cached_deltas = [GroupEntityDeltas.model_validate(ed) for ed in cached_data.get("entity_deltas", [])]
                        cached_anchor = str(cached_data.get("last_anchor_event", ""))
                        if cached_facts:
                            chapter_facts.extend(cached_facts)
                            entity_deltas.extend(cached_deltas)
                            last_anchor_event = cached_anchor
                            logger.info("Resumed Stage 1 facts extraction for %s: restored %d chapters from checkpoint.", novel_title, len(cached_facts))
                except Exception as e:
                    logger.warning("Failed to load Stage 1 checkpoint for %s: %s", novel_title, e)

            batches = pack_chapters_dynamically(toc, max_chars=7200, max_chapters=3)
            already_extracted_indices = {cf.chapter_index for cf in chapter_facts}

            for b_idx, batch in enumerate(batches, start=1):
                if all(ch.get("index", idx + 1) in already_extracted_indices for idx, ch in enumerate(batch)):
                    continue

                if self._is_cancelled:
                    logger.info("Extraction interrupted by user cancellation during Stage 1.")
                    break
                if not await self._wait_if_paused_or_unloaded(on_progress):
                    break

                res = await self.fact_extractor.extract_batched_chapter_facts(
                    chapters=batch,
                    prior_anchor_event=last_anchor_event,
                )
                for sc in res.chapters:
                    chapter_facts.append(ChapterFactExtractContract.from_compact(sc))
                entity_deltas.append(res.entity_deltas)
                if res.chapters:
                    last_anchor_event = res.chapters[-1].core_event

                if checkpoint_file:
                    try:
                        payload = {
                            "novel_title": novel_title,
                            "total_chapters": total_chapters,
                            "chapter_facts": [cf.model_dump() for cf in chapter_facts],
                            "entity_deltas": [ed.model_dump() for ed in entity_deltas],
                            "last_anchor_event": last_anchor_event,
                            "updated_at": time.time(),
                        }
                        _write_json_atomic(checkpoint_file, payload)
                    except Exception as exc:
                        logger.warning("Failed to save Stage 1 checkpoint for %s: %s", novel_title, exc)

                if self.hardware_guard:
                    await self.hardware_guard.check_and_cooldown_if_hot()
                pct = round((len(chapter_facts) / total_chapters) * 30.0, 1)
                self.status["percent"] = pct
                self.status["phase"] = "STAGE_1_FACTS"
                if on_progress:
                    on_progress(dict(self.status))
                await asyncio.sleep(0)

            if self._is_cancelled or not chapter_facts:
                return 0

            # Consolidate volume facts with entity deltas
            vol_batch_size = 20
            volume_reports = []
            num_vols = max(1, (len(chapter_facts) + vol_batch_size - 1) // vol_batch_size)
            for v_idx in range(num_vols):
                if self._is_cancelled:
                    break
                if not await self._wait_if_paused_or_unloaded(on_progress):
                    break
                v_start = v_idx * vol_batch_size
                v_end = min(len(chapter_facts), (v_idx + 1) * vol_batch_size)
                batch_facts = chapter_facts[v_start:v_end]
                if batch_facts:
                    first_ch_in_vol = batch_facts[0].chapter_index
                    last_ch_in_vol = batch_facts[-1].chapter_index
                    vol_deltas: list[GroupEntityDeltas] = []
                    for b_i, delta in enumerate(entity_deltas):
                        if b_i < len(batches):
                            b_ch_start = batches[b_i][0].get("index", 1)
                            b_ch_end = batches[b_i][-1].get("index", 1)
                            if not (b_ch_end < first_ch_in_vol or b_ch_start > last_ch_in_vol):
                                vol_deltas.append(delta)
                        else:
                            vol_deltas.append(delta)

                    vr = await self.fact_extractor.consolidate_volume_facts(
                        volume_index=v_idx + 1,
                        chapter_facts=batch_facts,
                        volume_entity_deltas=vol_deltas,
                    )
                    volume_reports.append(vr)
                    if self.hardware_guard:
                        await self.hardware_guard.check_and_cooldown_if_hot()
                    pct = round(30.0 + ((v_idx + 1) / num_vols) * 15.0, 1)
                    self.status["percent"] = pct
                    self.status["phase"] = "STAGE_1_CONSOLIDATION"
                    if on_progress:
                        on_progress(dict(self.status))
                    await asyncio.sleep(0)

            if self._is_cancelled:
                return 0

            major_arcs = None
            if num_vols > 4:
                if not await self._wait_if_paused_or_unloaded(on_progress):
                    return 0
                major_arcs = await self.fact_extractor.consolidate_major_arcs(
                    volume_reports=volume_reports,
                    hardware_guard=self.hardware_guard,
                    lifecycle=self,
                )

            # Synthesize macro blueprint with AI-deduced genre
            if not await self._wait_if_paused_or_unloaded(on_progress):
                return 0
            blueprint = await self.fact_extractor.synthesize_macro_blueprint(
                novel_title=novel_title,
                volume_reports=volume_reports,
                chapter_facts=chapter_facts,
                major_arcs=major_arcs,
            )
            curve_data, _ = build_factual_pacing_curve(chapter_facts, total_chapters)
            self.repo.save_book_blueprint(
                source_id=source_id,
                blueprint=blueprint,
                factual_pacing_curve=curve_data,
                total_chapters=total_chapters,
            )

            if checkpoint_file and checkpoint_file.exists():
                try:
                    checkpoint_file.unlink()
                except Exception:
                    pass

            if target_chapter_indices is None and len(chapter_facts) >= 10:
                target_chapter_indices, chapter_scores_map, _ = select_high_value_chapters_zscore(
                    chapter_facts=chapter_facts,
                    min_pct=0.08,
                    max_pct=0.12,
                    vol_batch_size=20,
                )
        else:
            self.status["percent"] = 45.0
            self.status["phase"] = "STAGE_1_SKIPPED_REUSING_BLUEPRINT"
            if on_progress:
                on_progress(dict(self.status))

        # =========================================================================
        # Stage 2: High-Throughput Concurrent Pattern Extraction
        # =========================================================================
        if target_chapter_indices is not None:
            chunk_pairs = split_targeted_chapters_into_chunks(
                toc,
                target_chapter_indices=target_chapter_indices,
                chunk_size=2500,
                overlap=250,
            )
        else:
            raw_chunks = split_novel_into_chunks(raw_text, chunk_size=2500, overlap=250)
            chunk_pairs = [(0, c) for c in raw_chunks]
            del raw_chunks
        # Release the parsed text and chapter map before extraction: together they are
        # the 2nd/3rd full copies of a multi-hundred-MB book in memory.
        del raw_text, toc
        total_chunks = len(chunk_pairs)

        with atomic_transaction(self.repo.db_path) as conn:
            conn.execute(
                "UPDATE ref_sources SET chunk_count = ? WHERE id = ?;",
                (total_chunks, source_id),
            )

        self.status["total_chunks"] = total_chunks
        self.status["phase"] = "STAGE_2_EXTRACTION"
        self.status["percent"] = 50.0
        if on_progress:
            on_progress(dict(self.status))

        # Format complete blueprint prefix for system prompt cache without arbitrary slicing
        gp = convert_simplified_to_traditional(blueprint.genre_primary.strip())
        gs = convert_simplified_to_traditional(blueprint.genre_sub.strip())
        book_genre_tag = f"{gp}:{gs}" if (gp and gs) else (gp or gs)
        archetypes_list = "、".join(
            dict.fromkeys(c.archetype_role for c in blueprint.character_network if c.archetype_role)
        )
        chars_summary = "、".join(
            f"{c.character_name}（基礎定位：{c.archetype_role}）"
            for c in blueprint.character_network
        )
        factions_summary = "、".join(f.faction_name for f in blueprint.faction_geopolitics)
        blueprint_prefix = (
            f"【全域藍圖參考】\n"
            f"書名：《{blueprint.novel_title}》\n"
            f"全書題材：{book_genre_tag}\n"
            f"宏觀大綱：{blueprint.macro_outline}\n"
            f"人物基礎定位索引（僅供識別原文人物，輸出時嚴禁出現任何人物姓名）：{chars_summary}\n"
            f"基礎角色定位集合：[{archetypes_list}]\n"
            f"陣營勢力：{factions_summary}\n"
        )

        concurrency = env_int("AUTONOVEL_LLAMA_NP", 1)
        sem = asyncio.Semaphore(max(1, concurrency))
        processed_count = 0
        start_chunk_idx = 0
        chunk_ckpt = cache_dir / f"stage2_chunks_{source_id}.json"
        if resuming:
            # Prefer the exact per-chunk checkpoint; fall back to entry_count for
            # databases written before the checkpoint file existed.
            resumed_from_checkpoint = False
            if chunk_ckpt.exists():
                try:
                    ck = json.loads(chunk_ckpt.read_text(encoding="utf-8"))
                    if int(ck.get("total_chunks", -1)) == total_chunks:
                        start_chunk_idx = max(
                            0, min(int(ck.get("next_chunk_index", 0)), total_chunks)
                        )
                        resumed_from_checkpoint = True
                except Exception as exc:
                    logger.warning(
                        "Failed to load Stage 2 checkpoint for %s: %s", novel_title, exc
                    )
            if not resumed_from_checkpoint:
                start_chunk_idx = min(int(existing_src["entry_count"]), total_chunks)
            processed_count = start_chunk_idx
            logger.info(
                "Resuming chunk extraction for %s from chunk %d/%d",
                novel_title,
                start_chunk_idx + 1,
                total_chunks,
            )

        async def _process_chunk_worker(c_text: str) -> ReferenceChunkExtractionContract:
            async with sem:
                return await self._extract_single_chunk(
                    c_text, blueprint_prefix=blueprint_prefix
                )

        try:
            # Process in sub-batches of 8 to stream writes and enforce thermal cooldown
            chunk_batch_size = 8
            for batch_start in range(start_chunk_idx, total_chunks, chunk_batch_size):
                if self._is_cancelled:
                    break
                if not await self._wait_if_paused_or_unloaded(on_progress):
                    break

                chunk_slice = chunk_pairs[batch_start : batch_start + chunk_batch_size]
                tasks = [_process_chunk_worker(c_text) for _, c_text in chunk_slice]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                valid_items: list[tuple[int, ReferenceChunkExtractionContract]] = []
                failed_indices: list[int] = []
                for (ch_idx, _), res in zip(chunk_slice, results):
                    if isinstance(res, BaseException):
                        if isinstance(res, asyncio.CancelledError):
                            raise res
                        logger.warning(
                            "Chunk extraction failed at chunk %d (chapter %d), keeping sibling results: %s",
                            batch_start + len(valid_items),
                            ch_idx,
                            res,
                        )
                        failed_indices.append(ch_idx)
                        continue
                    valid_items.append((ch_idx, res))

                if valid_items:
                    # Optimization 1: Batch embed all descriptions and distinct tags in 2 ONNX calls
                    desc_texts = [
                        normalize_quotes_to_traditional(
                            convert_simplified_to_traditional(res.description.strip())
                        )
                        for _, res in valid_items
                    ]
                    desc_vecs = self.embedder.embed_batch(desc_texts)

                    # Gather all unique tags across the batch
                    batch_cleaned_tags: list[list[tuple[str, str]]] = []
                    all_unique_tag_names: dict[str, None] = {}
                    for _, res in valid_items:
                        tagged_items = res.tags_matrix.to_tagged_items() if res.tags_matrix else []
                        clean_tagged = [
                            (convert_simplified_to_traditional(name.strip()), axis)
                            for name, axis in tagged_items
                            if name.strip()
                        ]
                        if book_genre_tag:
                            clean_tagged.insert(0, (book_genre_tag, "CONTEXT"))
                        seen_c = set()
                        deduped_c = []
                        for name, axis in clean_tagged:
                            if name not in seen_c:
                                seen_c.add(name)
                                deduped_c.append((name, axis))
                                all_unique_tag_names[name] = None
                        batch_cleaned_tags.append(deduped_c)

                    unique_tag_names_list = list(all_unique_tag_names.keys())
                    unique_tag_vecs = self.embedder.embed_batch(unique_tag_names_list) if unique_tag_names_list else []
                    tag_vec_map = dict(zip(unique_tag_names_list, unique_tag_vecs))

                    sub_entries: list[dict[str, Any]] = []
                    for idx, ((ch_idx, res), clean_desc, desc_vec, tagged) in enumerate(
                        zip(valid_items, desc_texts, desc_vecs, batch_cleaned_tags)
                    ):
                        clean_snip = normalize_quotes_to_traditional(
                            convert_simplified_to_traditional(res.snippet.strip())
                        )
                        clean_dyn = convert_simplified_to_traditional(res.relational_dynamic.strip())
                        clean_subj = convert_simplified_to_traditional(res.subject_role.strip())
                        clean_obj = convert_simplified_to_traditional(res.object_role.strip())
                        clean_dyn_type = convert_simplified_to_traditional(res.interaction_dynamic.strip())

                        paired = [
                            (name, tag_vec_map[name], axis)
                            for name, axis in tagged
                        ]

                        ch_score = 0.0
                        if chapter_scores_map and ch_idx in chapter_scores_map:
                            ch_score = float(chapter_scores_map[ch_idx])

                        sub_entries.append(
                            {
                                "description": clean_desc,
                                "desc_vec": desc_vec,
                                "snippet": clean_snip,
                                "tag_vectors": paired,
                                "relational_dynamic": clean_dyn,
                                "subject_role": clean_subj,
                                "object_role": clean_obj,
                                "interaction_dynamic": clean_dyn_type,
                                "chapter_index": ch_idx,
                                "exemplar_score": ch_score,
                            }
                        )

                    self.repo.insert_entries_batch(source_id, sub_entries)

                batch_end = batch_start + len(chunk_slice)
                processed_count = min(total_chunks, batch_end)
                self.status["processed_chunks"] = processed_count
                try:
                    ckpt_payload = {
                        "total_chunks": total_chunks,
                        "next_chunk_index": processed_count,
                    }
                    if failed_indices:
                        ckpt_payload["failed_chapters"] = failed_indices
                    _write_json_atomic(chunk_ckpt, ckpt_payload)
                except OSError as exc:
                    logger.warning(
                        "Failed to persist Stage 2 checkpoint for %s: %s", novel_title, exc
                    )
                pct = round(50.0 + (processed_count / max(1, total_chunks)) * 50.0, 1)
                self.status["percent"] = min(100.0, pct)
                if on_progress:
                    on_progress(dict(self.status))

                # Hardware temperature protection loop
                if self.hardware_guard is not None:
                    await self.hardware_guard.check_and_cooldown_if_hot()

                await asyncio.sleep(0)

            self.repo._invalidate_cache()
            if processed_count >= total_chunks:
                chunk_ckpt.unlink(missing_ok=True)
            # Return the chunks actually processed: reporting total_chunks after a
            # cancellation inflates the progress the UI shows.
            return processed_count
        except asyncio.CancelledError:
            self._is_cancelled = True
            logger.info("Extraction task cancelled by asyncio.")
            self.repo._invalidate_cache()
            raise
        finally:
            self.status["active"] = False
            self.status["paused"] = False
            if on_progress:
                on_progress(dict(self.status))

    async def ingest_items(
        self,
        items: list[ReferenceIngestItem],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        fresh_blueprint_paths: set[str] | None = None,
        pre_parsed_novel_map: dict[str, tuple[str, list[dict[str, Any]]]] | None = None,
        completed_indices: set[int] | None = None,
        on_item_finished: Callable[[int, bool, str | None], None] | None = None,
    ) -> int:
        """Process a batch of ReferenceIngestItem entries according to per-file action with failure isolation."""
        self._is_cancelled = False
        batch_total = len(items)
        total = 0
        fresh_set = fresh_blueprint_paths or set()
        completed_set = set(completed_indices) if completed_indices else set()

        try:
            for b_idx, item in enumerate(items, start=1):
                idx = b_idx - 1
                if idx in completed_set:
                    continue
                if self._is_cancelled:
                    break
                act = item.action.upper()
                p = Path(item.file_path)
                if act == "SKIP":
                    completed_set.add(idx)
                    if on_item_finished:
                        on_item_finished(idx, True, None)
                    continue
                if not p.exists():
                    logger.warning("Skills extraction file not found: %s", item.file_path)
                    completed_set.add(idx)
                    self._completed_books.append(
                        {
                            "file_name": p.name,
                            "track": "skills",
                            "chunks": 0,
                            "status": "FAILED",
                            "error": "FILE_NOT_FOUND",
                        }
                    )
                    if on_item_finished:
                        on_item_finished(idx, False, "FILE_NOT_FOUND")
                    self._save_manifest()
                    continue

                is_force = act == "REEXTRACT"
                if is_force and not getattr(item, "extract_pacing", False):
                    existing = self.repo.get_source_by_hash(compute_sha256(p))
                    if existing:
                        self.repo.delete_pacing_templates_by_source(existing["id"])
                pre_parsed = None
                if pre_parsed_novel_map:
                    pre_parsed = pre_parsed_novel_map.get(str(p)) or pre_parsed_novel_map.get(str(p.resolve()))

                try:
                    self._save_manifest()
                    chunks_done = await self.ingest_file(
                        p,
                        force_reextract=is_force,
                        on_progress=on_progress,
                        batch_index=b_idx,
                        batch_total=batch_total,
                        reuse_fresh_blueprint=(str(p.resolve()) in fresh_set or str(p) in fresh_set),
                        pre_parsed_novel=pre_parsed,
                    )
                    total += chunks_done
                    completed_set.add(idx)
                    if not self._is_cancelled:
                        self._completed_books.append(
                            {
                                "file_name": p.name,
                                "track": "skills",
                                "chunks": chunks_done,
                                "status": "SUCCESS",
                                "error": None,
                            }
                        )
                    if on_item_finished:
                        on_item_finished(idx, True, None)
                except asyncio.CancelledError:
                    raise
                except Exception as item_exc:
                    logger.error(
                        "Skills extraction failed for %s: %s; isolating error and continuing batch",
                        p.name,
                        item_exc,
                        exc_info=True,
                    )
                    completed_set.add(idx)
                    self._completed_books.append(
                        {
                            "file_name": p.name,
                            "track": "skills",
                            "chunks": 0,
                            "status": "FAILED",
                            "error": str(item_exc),
                        }
                    )
                    if on_item_finished:
                        on_item_finished(idx, False, str(item_exc))
                finally:
                    if pre_parsed_novel_map:
                        pre_parsed_novel_map.pop(str(p), None)
                        pre_parsed_novel_map.pop(str(p.resolve()), None)
                    self._save_manifest()
        except asyncio.CancelledError:
            self._is_cancelled = True
            logger.info("Batch reference ingestion task cancelled.")
        finally:
            self.status["active"] = False
            self.status["paused"] = False
            if on_progress:
                on_progress(dict(self.status))
        return total

    def enqueue_ingestion_jobs(
        self,
        items: list[ReferenceIngestItem],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> asyncio.Task[int]:
        """Launch non-blocking background asyncio task for reference library extraction."""
        self._is_cancelled = False
        self._bg_task = asyncio.create_task(self.ingest_items(items, on_progress=on_progress))
        return self._bg_task

    def _start_queue_task(
        self,
        on_skills_progress: Callable[[dict[str, Any]], None] | None = None,
        on_pacing_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> asyncio.Task[Any] | None:
        """Start or attach background worker executing sequential queue."""
        if self._bg_task and not self._bg_task.done():
            if on_skills_progress:
                on_skills_progress(self.get_current_status())
            return self._bg_task
        if not self._pending_jobs:
            return None

        self._is_cancelled = False

        async def _run_queue() -> None:
            last_skills_cb = on_skills_progress
            last_pacing_cb = on_pacing_progress
            try:
                while self._pending_jobs and not self._is_cancelled:
                    cur = self._pending_jobs[0]
                    p_items: list[ReferenceIngestItem] = cur["pacing_items"]
                    s_items: list[ReferenceIngestItem] = cur["skill_items"]
                    p_jitter: float = cur["pacing_jitter"]
                    f_reextract: bool = cur["force_reextract"]
                    m_ext: Any = cur["macro_extractor"]
                    s_cb = cur.get("on_skills_progress") or last_skills_cb
                    p_cb = cur.get("on_pacing_progress") or last_pacing_cb
                    last_skills_cb = s_cb
                    last_pacing_cb = p_cb

                    fresh_bp_paths: set[str] = set()
                    cur.setdefault("pacing_completed_indices", [])
                    cur.setdefault("skill_completed_indices", [])
                    cur.setdefault("failed_items", [])

                    if not p_items:
                        self.update_pacing_status(
                            active=False,
                            novel_title="",
                            percent=0.0,
                            phase="IDLE",
                            detail="",
                            batch_index=0,
                            batch_total=0,
                        )
                        if p_cb:
                            p_cb(dict(self.pacing_status))

                    parsed_novel_map: dict[str, tuple[str, list[dict[str, Any]]]] = {}

                    # 1. Track B Pacing Extraction (with item-level fault isolation)
                    if m_ext and p_items:
                        total_p = len(p_items)
                        for p_idx, it in enumerate(p_items, start=1):
                            if (p_idx - 1) in cur["pacing_completed_indices"]:
                                continue
                            if self._is_cancelled:
                                break
                            p = Path(it.file_path)
                            if not p.exists():
                                logger.warning("Pacing file not found: %s; marking as failed", it.file_path)
                                cur["failed_items"].append({"file_name": p.name, "track": "pacing", "error": "FILE_NOT_FOUND"})
                                cur["pacing_completed_indices"].append(p_idx - 1)
                                self._completed_books.append({"file_name": p.name, "track": "pacing", "chunks": 0, "status": "FAILED", "error": "FILE_NOT_FOUND"})
                                self._save_manifest()
                                continue

                            sha = compute_sha256(p)
                            act = it.action.upper()
                            is_force = (act == "REEXTRACT") or f_reextract
                            existing_src = self.repo.get_source_by_hash(sha)
                            if existing_src:
                                source_id = existing_src["id"]
                            else:
                                source_id = self.repo.register_or_reset_source(
                                    sha, p.name, 0, reset_entries=False
                                )

                            self.status = {
                                "active": True,
                                "paused": self._is_paused,
                                "current_file": p.name,
                                "batch_index": p_idx,
                                "batch_total": total_p,
                                "processed_chunks": 0,
                                "total_chunks": 0,
                                "percent": round(((p_idx - 1) / max(1, total_p)) * 100.0, 1),
                                "phase": "TRACK_B_PACING",
                            }
                            self._save_manifest()
                            if s_cb:
                                s_cb(self.get_current_status())

                            def _wrapped_pacing_cb(
                                st: dict[str, Any],
                                _idx: int = p_idx,
                                _tot: int = total_p,
                                _p_cb: Any = p_cb,
                                _s_cb: Any = s_cb,
                            ) -> None:
                                enriched = {
                                    **st,
                                    "batch_index": _idx,
                                    "batch_total": _tot,
                                }
                                self.update_pacing_status(**enriched)
                                if st.get("active"):
                                    self.status["percent"] = st.get("percent", 0.0)
                                    self.status["paused"] = self._is_paused
                                if _p_cb:
                                    _p_cb(dict(self.pacing_status))
                                if _s_cb:
                                    _s_cb(self.get_current_status())

                            try:
                                raw_text, _ = parse_novel_file(p)
                                toc = extract_table_of_contents(raw_text)
                                parsed_novel_map[str(p)] = (raw_text, toc)
                                parsed_novel_map[str(p.resolve())] = (raw_text, toc)
                                title = p.stem
                                await m_ext.extract_from_novel_text(
                                    text=raw_text,
                                    novel_title=title,
                                    source_id=source_id,
                                    jitter_ratio=p_jitter,
                                    hardware_guard=self.hardware_guard,
                                    lifecycle=self,
                                    force_reextract=is_force,
                                    on_progress=_wrapped_pacing_cb,
                                )
                                fresh_bp_paths.add(str(p))
                                fresh_bp_paths.add(str(p.resolve()))
                                cur["pacing_completed_indices"].append(p_idx - 1)
                                if not self._is_cancelled:
                                    self._completed_books.append(
                                        {
                                            "file_name": p.name,
                                            "track": "pacing",
                                            "chunks": 0,
                                            "status": "SUCCESS",
                                            "error": None,
                                        }
                                    )
                            except asyncio.CancelledError:
                                raise
                            except Exception as item_exc:
                                logger.error(
                                    "Pacing extraction failed for %s: %s; isolating error and continuing batch",
                                    p.name,
                                    item_exc,
                                    exc_info=True,
                                )
                                cur["failed_items"].append({"file_name": p.name, "track": "pacing", "error": str(item_exc)})
                                cur["pacing_completed_indices"].append(p_idx - 1)
                                self._completed_books.append(
                                    {
                                        "file_name": p.name,
                                        "track": "pacing",
                                        "chunks": 0,
                                        "status": "FAILED",
                                        "error": str(item_exc),
                                    }
                                )
                            finally:
                                self._save_manifest()

                    # 2. Track A Skills Extraction (with item-level fault isolation)
                    if s_items and not self._is_cancelled:
                        def _on_skill_item_finished(idx: int, success: bool, err_msg: str | None) -> None:
                            cur["skill_completed_indices"].append(idx)
                            if not success and err_msg:
                                cur["failed_items"].append({"index": idx, "track": "skills", "error": err_msg})
                            self._save_manifest()

                        await self.ingest_items(
                            s_items,
                            on_progress=s_cb,
                            fresh_blueprint_paths=fresh_bp_paths,
                            pre_parsed_novel_map=parsed_novel_map,
                            completed_indices=set(cur["skill_completed_indices"]),
                            on_item_finished=_on_skill_item_finished,
                        )

                    # Finished current job batch cleanly
                    if not self._is_cancelled and self._pending_jobs:
                        self._pending_jobs.pop(0)
                        self._save_manifest()

            except asyncio.CancelledError:
                self._is_cancelled = True
                logger.info("Unified reference extraction pipeline cancelled.")
                self.update_pacing_status(active=False, phase="CANCELLED", detail="萃取任務已被使用者手動取消。")
                self._clear_manifest()
            except Exception as exc:
                logger.error("Error in unified reference extraction pipeline: %s", exc, exc_info=True)
                self.update_pacing_status(
                    active=False,
                    phase="ERROR",
                    detail=f"萃取任務異常終止：{exc}",
                )
                self._save_manifest()
            finally:
                self.status["active"] = False
                self.status["paused"] = False
                self.pacing_status["active"] = False
                if not self._pending_jobs:
                    self._clear_manifest()
                if last_skills_cb:
                    last_skills_cb(self.get_current_status())
                if last_pacing_cb:
                    last_pacing_cb(dict(self.pacing_status))

        self._bg_task = asyncio.create_task(_run_queue())
        return self._bg_task

    def enqueue_pipeline(
        self,
        pacing_items: list[ReferenceIngestItem],
        skill_items: list[ReferenceIngestItem],
        pacing_jitter: float = 0.0,
        force_reextract: bool = False,
        macro_extractor: Any = None,
        on_skills_progress: Callable[[dict[str, Any]], None] | None = None,
        on_pacing_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> asyncio.Task[Any]:
        """Enqueue a batch into the sequential background pipeline (Track B pacing -> Track A skills)."""
        self._is_cancelled = False
        job = {
            "pacing_items": list(pacing_items),
            "skill_items": list(skill_items),
            "pacing_jitter": pacing_jitter,
            "force_reextract": force_reextract,
            "macro_extractor": macro_extractor,
            "on_skills_progress": on_skills_progress,
            "on_pacing_progress": on_pacing_progress,
            "pacing_completed_indices": [],
            "skill_completed_indices": [],
            "failed_items": [],
        }
        self._pending_jobs.append(job)
        self._save_manifest()

        if not pacing_items and not self.pacing_status.get("active", False):
            self.update_pacing_status(
                active=False,
                novel_title="",
                percent=0.0,
                phase="IDLE",
                detail="",
                batch_index=0,
                batch_total=0,
            )
            if on_pacing_progress:
                on_pacing_progress(dict(self.pacing_status))

        task = self._start_queue_task(on_skills_progress, on_pacing_progress)
        return task or asyncio.create_task(asyncio.sleep(0))

    async def startup_recovery(
        self,
        macro_extractor: Any = None,
        auto_resume: bool = True,
        on_skills_progress: Callable[[dict[str, Any]], None] | None = None,
        on_pacing_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> bool:
        """Recover pending/in-flight reference extraction queue from manifest after restart."""
        if self._bg_task and not self._bg_task.done():
            return False

        manifest_file = self.manifest_path
        if not manifest_file.exists():
            return False

        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Corrupted extraction manifest ignored: %s", exc)
            return False

        pending_jobs_data = manifest.get("pending_jobs", [])
        if not pending_jobs_data:
            self._clear_manifest()
            return False

        self._completed_books = manifest.get("completed_books", [])

        reconstructed_jobs: list[dict[str, Any]] = []
        for job_dict in pending_jobs_data:
            p_items = [ReferenceIngestItem(**it) for it in job_dict.get("pacing_items", [])]
            s_items = [ReferenceIngestItem(**it) for it in job_dict.get("skill_items", [])]
            p_done = set(job_dict.get("pacing_completed_indices", []))
            s_done = set(job_dict.get("skill_completed_indices", []))

            # Filter out jobs where both tracks are already finished
            if len(p_done) >= len(p_items) and len(s_done) >= len(s_items):
                continue

            reconstructed_jobs.append({
                "pacing_items": p_items,
                "skill_items": s_items,
                "pacing_jitter": float(job_dict.get("pacing_jitter", 0.0)),
                "force_reextract": bool(job_dict.get("force_reextract", False)),
                "macro_extractor": macro_extractor,
                "on_skills_progress": on_skills_progress,
                "on_pacing_progress": on_pacing_progress,
                "pacing_completed_indices": list(p_done),
                "skill_completed_indices": list(s_done),
                "failed_items": list(job_dict.get("failed_items", [])),
            })

        if not reconstructed_jobs:
            self._clear_manifest()
            return False

        self._pending_jobs = reconstructed_jobs
        is_paused = manifest.get("is_paused", False) or manifest.get("state") == "PAUSED"
        self._is_paused = is_paused
        self.status["paused"] = is_paused
        self.pacing_status["paused"] = is_paused

        logger.info(
            "Reference extractor recovered %d pending batch(es) from manifest (paused=%s)",
            len(self._pending_jobs),
            is_paused,
        )

        if auto_resume and not is_paused:
            self._start_queue_task(on_skills_progress, on_pacing_progress)
            return True
        else:
            self.status["active"] = True
            self.status["phase"] = "PAUSED_RECOVERED"
            if on_skills_progress:
                on_skills_progress(self.get_current_status())
            return True

    async def _extract_single_chunk(
        self,
        chunk_text: str,
        blueprint_prefix: str = "",
    ) -> ReferenceChunkExtractionContract:
        """Extract pattern description, snippet, relational_dynamic, and tags from chunk with blueprint context."""
        if not await self._wait_if_paused_or_unloaded():
            raise asyncio.CancelledError()

        if not (self.llm_client and self.llm_client.server_manager.is_healthy()):
            raise AIModelUnavailableError("LLM server is unavailable or unhealthy for chunk pattern extraction.")

        base_sys_prompt = (
            "你是一位專精長篇小說敘事動力學與行文筆法拆解的文學技法分析大師。請深入拆解以下小說切片，提煉可跨題材遷移、高度通用、可直接落地指導正文創作的四維正交寫作技法模組：\n"
            "1. 【全篇繁體中文】：所有文字輸出必須完全使用繁體中文（台灣標準書面語）。\n"
            "2. 【雙層筆法結構（description）】：按雙層架構深入剖析本段運作規律，格式固定為：\n"
            "   「【宏觀戰略】：...；【微觀戰術】：...」\n"
            "   - 【宏觀戰略】：說明本段在全書或分卷情節推進中的結構功能（勢能蓄積、矛盾演進、局勢轉換或波谷承接等結構性作用）。\n"
            "   - 【微觀戰術】：深入剖析具體之文字筆法、感官渲染（光影、氣味、聲響、溫度、觸覺）與語言節奏（語速緩急、句式密度、對白潛台詞）的行文操作。\n"
            "3. 【場景焦點精華區塊（snippet，200～350 字）】：自原文挑選連續 200～350 字最具畫面張力、修辭美感、衝突博弈或對白潛台詞的小說場景焦點精華區塊，具備微觀情境與起承轉合，嚴禁自行改寫或縮寫。\n"
            "4. 【主客體基礎定位（subject_role / object_role / interaction_dynamic）】：\n"
            "   - subject_role 與 object_role：標定行動發起方與承受方在敘事編制中的客觀基礎定位（如主角、配角、長老、敵手、群體、自然環境或自身）。\n"
            "   - interaction_dynamic：簡練界定主體與客體之間的客觀力量作用或情感互動模式。\n"
            "5. 【自然主客因果動力學（relational_dynamic）】：一段精準陳述（40~80字）描述本切片中推動力、承受力或境象的相互作用關係與動態轉移。\n"
            "6. 【正交四象限標籤體系（tags_matrix）】：輸出四維正交技法標籤（結構功能、主客博弈、微觀筆法、情境氛圍），各維度輸出 1 到 4 個客觀可遷移之範式標籤。"
        )
        sys_prompt = f"{base_sys_prompt}\n\n{blueprint_prefix.strip()}" if blueprint_prefix else base_sys_prompt
        user_prompt = f"小說切片正文：\n{chunk_text.strip()}"
        raw: ReferenceChunkExtractionContract | None = None
        max_retries = 3
        attempt = 0
        last_exc: Exception | None = None
        while attempt < max_retries:
            attempt += 1
            if not await self._wait_if_paused_or_unloaded():
                raise asyncio.CancelledError()

            try:
                raw = await self.llm_client.generate_structured(
                    sys_prompt, user_prompt, ReferenceChunkExtractionContract
                )
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exc = e
                is_u = getattr(getattr(self.llm_client, "server_manager", None), "_manually_unloaded", False) is True
                if self._is_paused or is_u:
                    logger.info("Single chunk extraction paused due to user pause or manual model unload; waiting for resume...")
                    attempt -= 1
                    if not await self._wait_if_paused_or_unloaded():
                        raise asyncio.CancelledError()
                    continue
                logger.warning("Single chunk extraction attempt %d failed: %s", attempt, e)
                if attempt < max_retries:
                    await asyncio.sleep(1.0)

        if raw is None:
            raise RuntimeError(f"Failed to extract chunk pattern after {max_retries} attempts: {last_exc}")

        clean_desc = normalize_quotes_to_traditional(
            convert_simplified_to_traditional(raw.description.strip())
        )
        clean_snip = normalize_quotes_to_traditional(
            convert_simplified_to_traditional(raw.snippet.strip())
        )
        clean_dyn = convert_simplified_to_traditional(raw.relational_dynamic.strip())
        clean_subj = convert_simplified_to_traditional(raw.subject_role.strip())
        clean_obj = convert_simplified_to_traditional(raw.object_role.strip())
        clean_dyn_type = convert_simplified_to_traditional(raw.interaction_dynamic.strip())
        clean_matrix = OrthogonalTagsContract(
            structure_tags=[
                convert_simplified_to_traditional(t.strip())
                for t in raw.tags_matrix.structure_tags
                if t.strip()
            ],
            actant_tags=[
                convert_simplified_to_traditional(t.strip())
                for t in raw.tags_matrix.actant_tags
                if t.strip()
            ],
            rhetoric_tags=[
                convert_simplified_to_traditional(t.strip())
                for t in raw.tags_matrix.rhetoric_tags
                if t.strip()
            ],
            context_tags=[
                convert_simplified_to_traditional(t.strip())
                for t in raw.tags_matrix.context_tags
                if t.strip()
            ],
        )
        return ReferenceChunkExtractionContract(
            description=clean_desc,
            snippet=clean_snip,
            relational_dynamic=clean_dyn,
            tags_matrix=clean_matrix,
            subject_role=clean_subj,
            object_role=clean_obj,
            interaction_dynamic=clean_dyn_type,
        )

    def _persist_extracted_entry(
        self,
        source_id: int,
        contract: ReferenceChunkExtractionContract,
        invalidate_cache: bool = True,
    ) -> None:
        """Compute CPU embeddings for description and tags, then insert into SQLite."""
        clean_desc = normalize_quotes_to_traditional(
            convert_simplified_to_traditional(contract.description.strip())
        )
        clean_snip = normalize_quotes_to_traditional(
            convert_simplified_to_traditional(contract.snippet.strip())
        )
        clean_dyn = convert_simplified_to_traditional(contract.relational_dynamic.strip())
        clean_subj = convert_simplified_to_traditional(contract.subject_role.strip())
        clean_obj = convert_simplified_to_traditional(contract.object_role.strip())
        clean_dyn_type = convert_simplified_to_traditional(contract.interaction_dynamic.strip())
        tagged_items = contract.tags_matrix.to_tagged_items() if contract.tags_matrix else []
        clean_tagged_items = [
            (convert_simplified_to_traditional(name.strip()), axis)
            for name, axis in tagged_items
            if name.strip()
        ]
        seen_names = set()
        deduped_items = []
        for name, axis in clean_tagged_items:
            if name not in seen_names:
                seen_names.add(name)
                deduped_items.append((name, axis))

        tag_names = [item[0] for item in deduped_items]
        desc_vec = self.embedder.embed_text(clean_desc)
        tag_vecs = self.embedder.embed_batch(tag_names)
        paired = [
            (item[0], vec, item[1])
            for item, vec in zip(deduped_items, tag_vecs)
        ]
        self.repo.insert_entry_with_tags(
            source_id=source_id,
            description=clean_desc,
            desc_vec=desc_vec,
            snippet=clean_snip,
            tag_vectors=paired,
            subject_role=clean_subj,
            object_role=clean_obj,
            interaction_dynamic=clean_dyn_type,
            relational_dynamic=clean_dyn,
            invalidate_cache=invalidate_cache,
        )

    async def auto_clean_tags(self) -> dict[str, int]:
        """Perform density-guided open taxonomy AI scanning, normalization, and noise filtering."""
        if not (self.llm_client and self.llm_client.server_manager.is_healthy()):
            raise AIModelUnavailableError("本地 AI 模型未就緒或推論服務離線，無法進行標籤自動整頓。")

        tag_freqs = await asyncio.to_thread(self.repo.list_tags_with_frequency)
        if not tag_freqs:
            return {"normalized": 0, "merged": 0, "dropped": 0, "total_affected": 0}

        density_info = await asyncio.to_thread(self.repo.get_density_analysis, 0.45, 2)
        hints = _build_density_hints(
            density_info.get("dense_clusters", []), density_info.get("outliers", [])
        )

        tags_prompt_list = [f"{t['tag_name']} ({t['count']})" for t in tag_freqs[:160]]
        user_prompt = (
            "請分析以下全庫抽取標籤與高維拓撲分佈，制定標籤標準化與清洗規則"
            "（注意：規則中的 old_tag 與 new_tag 必須為純粹標籤名稱，切勿包含括號內的統計數字）：\n\n"
            + "【全庫標籤頻次統計】：\n"
            + "、".join(tags_prompt_list)
            + (f"\n\n{hints}" if hints else "")
        )
        sys_prompt = (
            "你是一位長篇網絡小說文學院士與高維語義拓撲架構大師。\n"
            "請基於聚類密度與頻次拓撲分佈，制定全庫標籤的標準化、歸併與規範化維護規則：\n"
            "1. 【繁體中文鐵律】：所有新標籤（new_tag）必須完全使用繁體中文（台灣標準書面語），嚴禁出現任何簡體字。\n"
            "2. 【精準名稱匹配】：old_tag 必須與輸入清單中的原始標籤字面完全一致，嚴禁夾帶括號頻次或引號。\n"
            "3. 【世界觀題材座標】：將世界觀與流派標籤標準化為「主題材:子題材」之二元層級架構，嚴格封裝核心題材。\n"
            "4. 【主客關係與通用文學技法】：保持主客博弈關係標籤、互動模式標籤，以及純粹的客觀行文機制與敘事技法標籤獨立，不得強制附加題材前綴。\n"
            "5. 【同義異構歸併（MERGE）】：將字面微異、同義近義或拼寫粗糙之變體標籤，全數合併至最具代表性之核心規範標籤。\n"
            "6. 【長尾特色資產保護】：對於獨特、精妙但出現頻次低的罕見文學技法標籤，應優先予以保留（KEEP）或規範化（NORMALIZE），嚴禁盲目清除具有文學價值的長尾標籤；僅對明顯為錯別字或無意義亂碼執行 DROP 操作。"
        )

        last_exc: Exception | None = None
        plan: TagMaintenanceContract | None = None
        for attempt in range(2):
            try:
                plan = await self.llm_client.generate_structured(
                    sys_prompt, user_prompt, TagMaintenanceContract
                )
                break
            except LLMResponseFormatError as exc:
                last_exc = exc
                logger.warning(
                    "Tag maintenance structured output failed (attempt %d/2): %s",
                    attempt + 1,
                    exc,
                )
        if plan is None:
            assert last_exc is not None
            raise last_exc

        for r in plan.rules:
            r.old_tag = _clean_rule_tag_name(r.old_tag)
            if r.new_tag:
                r.new_tag = convert_simplified_to_traditional(
                    _clean_rule_tag_name(r.new_tag)
                )

        return await asyncio.to_thread(
            self.repo.apply_tag_rules, plan.rules, self.embedder
        )
