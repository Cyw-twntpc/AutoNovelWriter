"""Single-chapter execution pipeline: Draft -> Gate A -> Gate B -> Atomic Commit."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

from autonovel.config import RuntimeConfig
from autonovel.db.project_repo import ProjectRepository
from autonovel.engine.context_router import ContextRouter
from autonovel.engine.pacing_monitor import PacingMonitor
from autonovel.engine.prose_sanitizer import (
    convert_simplified_to_traditional,
    sanitize_novel_prose,
)
from autonovel.engine.quality_gate import QualityGateEngine
from autonovel.llm.llm_client import LLMClient
from autonovel.reference.retriever import ReferenceRetriever
from autonovel.schemas.domain import AnomalyType, ChapterStatus
from autonovel.schemas.llm_contracts import ChapterExtractionContract
from autonovel.wizard.qa_engine import AIModelUnavailableError


class GateAFailedError(RuntimeError):
    """Raised when Gate A hard-conflict retries are exhausted; the chapter must not commit."""

    def __init__(self, errors: list[str], gate_a: dict[str, Any] | None = None) -> None:
        self.errors = errors
        self.gate_a = gate_a or {}
        super().__init__(
            "Gate A 硬性衝突重試耗盡，章節拒絕提交：" + "；".join(errors or ["unknown"])
        )


class ChapterPipeline:
    """Executes draft generation, Gate A, Gate B, extraction, and two-phase commit."""

    def __init__(
        self,
        config: RuntimeConfig,
        repo: ProjectRepository,
        router: ContextRouter,
        retriever: ReferenceRetriever,
        pacing: PacingMonitor,
        quality_gate: QualityGateEngine,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Initialize chapter pipeline with all engine sub-components."""
        self.config = config
        self.repo = repo
        self.router = router
        self.retriever = retriever
        self.pacing = pacing
        self.quality_gate = quality_gate
        self.llm_client = llm_client

    async def execute_chapter(
        self,
        chapter_num: int,
        seed: int | None = None,
        on_token: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Run full chapter lifecycle from WRITING to STAGING/REVISING to atomic COMPLETED/REVISED."""
        start_ts = time.perf_counter()
        ch_info = self._load_and_mark_writing(chapter_num)
        if on_status:
            on_status(ChapterStatus.WRITING.value)
        volume_num = int(ch_info["volume_num"])
        self.pacing.check_and_expire_foreshadowing(chapter_num)
        pacing_eval = self.pacing.evaluate_sliding_window(chapter_num)
        ref_digest, _ = await self.retriever.retrieve_and_compress(
            f"{ch_info['title']} {ch_info['beats_json']}"
        )
        ctx = self.router.build_chapter_context(
            chapter_num=chapter_num,
            reference_digest=ref_digest,
            pacing_directive=pacing_eval["pacing_directive"],
        )
        self.repo.update_project_config(forward_retcon_note="")
        gate_a_res, retry_count = await self._run_draft_and_gate_a_loop(
            chapter_num, volume_num, ch_info, ctx, seed, on_token
        )
        final_text, gate_b_res, rev_count, extra_micro = await self._run_gate_b_and_revision(
            chapter_num,
            str(ch_info.get("title") or f"第{chapter_num}章"),
            ctx,
            gate_a_res["sanitized_text"],
            on_token,
            on_status,
        )
        final_text = sanitize_novel_prose(final_text)
        extraction = gate_a_res["extraction"]
        if rev_count > 0:
            # Gate B rewrote the prose, so the Gate A extraction no longer describes the
            # committed text. Re-extract and re-validate hard facts before persisting.
            extraction = await self._extract_chapter_deltas(chapter_num, ch_info, final_text)
            post_errors = self.repo.validate_extraction_hard_facts(volume_num, extraction)
            if post_errors:
                raise GateAFailedError(post_errors, gate_a_res)
        total_micro = int(gate_a_res["micro_rewrite_count"]) + extra_micro
        duration = max(0.01, round(time.perf_counter() - start_ts, 2))
        tps = round(len(final_text) / duration, 2)
        self.repo.commit_chapter_atomically(
            chapter_num=chapter_num,
            final_content=final_text,
            extraction=extraction,
            literary_score=float(gate_b_res["composite_score"]),
            rubric_payload=gate_b_res,
            best_paragraph_text=str(gate_b_res["best_paragraph_text"]),
            best_paragraph_score=float(gate_b_res["best_paragraph_score"]),
            revision_count=rev_count,
            micro_rewrite_count=total_micro,
            contradiction_retry_count=retry_count,
            duration_sec=duration,
        )
        return {
            "chapter_num": chapter_num,
            "volume_num": volume_num,
            "title": gate_a_res["extraction"].chapter_title,
            "word_count": len(final_text),
            "literary_score": gate_b_res["composite_score"],
            "revision_count": rev_count,
            "micro_rewrite_count": total_micro,
            "contradiction_retry_count": retry_count,
            "duration_sec": duration,
            "tokens_per_sec": tps,
        }

    def _load_and_mark_writing(self, chapter_num: int) -> dict[str, Any]:
        """Fetch chapter row and set its status to WRITING."""
        rows = self.repo.fetch_all(
            "SELECT * FROM chapters WHERE chapter_num = ?;", (chapter_num,)
        )
        if not rows:
            raise ValueError(f"Chapter {chapter_num} does not exist in chapters table.")
        self.repo.save_staging_chapter(
            chapter_num, rows[0]["title"], "", status=ChapterStatus.WRITING.value
        )
        return rows[0]

    async def _run_draft_and_gate_a_loop(
        self,
        chapter_num: int,
        volume_num: int,
        ch_info: dict[str, Any],
        ctx: dict[str, Any],
        seed: int | None,
        on_token: Callable[[str], None] | None,
    ) -> tuple[dict[str, Any], int]:
        """Generate draft and enforce Gate A zero-tolerance rules with up to 3 retries."""
        max_retries = self.config.quality_thresholds.max_gate_a_retries
        conflict_warning = ""
        last_gate_a: dict[str, Any] = {}
        for attempt in range(max_retries):
            cur_seed = (seed + attempt) if seed is not None else random.randint(1, 1_000_000_000)
            prompt = ctx["user_prompt"] + (f"\n\n【硬約束修正警告】{conflict_warning}" if conflict_warning else "")
            draft = await self._generate_chapter_draft(
                chapter_num, ch_info, ctx["system_prompt"], prompt, cur_seed, on_token
            )
            self.repo.save_staging_chapter(chapter_num, ch_info["title"], draft)
            extraction = await self._extract_chapter_deltas(chapter_num, ch_info, draft)
            gate_a = await self.quality_gate.run_gate_a(volume_num, draft, extraction)
            last_gate_a = gate_a
            if gate_a["has_minor_drift"] and gate_a["forward_retcon_instruction"]:
                self.repo.update_project_config(
                    forward_retcon_note=gate_a["forward_retcon_instruction"]
                )
                self.repo.log_anomaly(
                    chapter_num=chapter_num,
                    anomaly_type=AnomalyType.SOFT_DRIFT.value,
                    severity="INFO",
                    resolution_method="注入下一章前向圓場指令 (Forward Retcon)",
                    affected_chapters=[chapter_num, chapter_num + 1],
                    detail_diff={"instruction": gate_a["forward_retcon_instruction"]},
                )
            if gate_a["passed"]:
                return gate_a, attempt
            fix_digest, _ = await self.retriever.retrieve_and_compress(
                f"異常修正與邏輯衝突排除：{'；'.join(gate_a['hard_errors'])}",
                fallback_tags=["邏輯自洽", "劇情轉折", "設定修正"],
            )
            conflict_warning = "；".join(gate_a["hard_errors"]) + (
                f"（參考修正手法：{fix_digest}）" if fix_digest else ""
            )
            self.repo.log_anomaly(
                chapter_num=chapter_num,
                anomaly_type=AnomalyType.HARD_CONFLICT.value,
                severity="CRITICAL",
                resolution_method=f"STAGING 攔截並退回重寫當章草稿 (第 {attempt + 1} 次)",
                affected_chapters=[chapter_num],
                detail_diff={"errors": gate_a["hard_errors"]},
            )
        raise GateAFailedError(last_gate_a.get("hard_errors", []), last_gate_a)

    async def _run_gate_b_and_revision(
        self,
        chapter_num: int,
        chapter_title: str,
        ctx: dict[str, Any],
        text: str,
        on_token: Callable[[str], None] | None,
        on_status: Callable[[str], None] | None = None,
    ) -> tuple[str, dict[str, Any], int, int]:
        """Run Gate B Rubric evaluation and trigger at most 1 full revision if score < 80."""
        text = sanitize_novel_prose(text)
        gate_b = await self.quality_gate.run_gate_b(text)
        if gate_b["passed"]:
            return text, gate_b, 0, 0
        self.repo.save_staging_chapter(
            chapter_num, chapter_title, text, status=ChapterStatus.REVISING.value
        )
        if on_status:
            on_status(ChapterStatus.REVISING.value)
        polish_digest, _ = await self.retriever.retrieve_and_compress(
            f"異常修正與文學潤色：{gate_b['revision_guidance']}",
            fallback_tags=["文筆修辭", "對話張力", "場景鋪陳"],
        )
        combined_guidance = str(gate_b["revision_guidance"]) + (
            f"\n【參考書庫潤色技法】{polish_digest}" if polish_digest else ""
        )
        revised_raw = await self._revise_chapter_text(
            ctx["system_prompt"], text, combined_guidance, on_token
        )
        sanitized_rev, extra_micro, _ = await self.quality_gate.scanner.sanitize_text(
            revised_raw
        )
        sanitized_rev = sanitize_novel_prose(sanitized_rev)
        self.repo.save_staging_chapter(
            chapter_num, chapter_title, sanitized_rev, status=ChapterStatus.REVISING.value
        )
        revised_gate_b = await self.quality_gate.run_gate_b(sanitized_rev)
        return sanitized_rev, revised_gate_b, 1, extra_micro

    async def _generate_chapter_draft(
        self,
        chapter_num: int,
        ch_info: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        seed: int,
        on_token: Callable[[str], None] | None,
    ) -> str:
        """Call LLM stream to generate full chapter prose without code truncation or offline fallback."""
        if self.llm_client and self.llm_client.server_manager.is_healthy():
            raw = await self.llm_client.generate_text_stream(
                system_prompt, user_prompt, max_tokens=4200, seed=seed, on_token=on_token
            )
            return sanitize_novel_prose(raw)
        raise AIModelUnavailableError(
            "本地 AI 模型未就緒或推論服務離線，無法撰寫章節正文。請先至環境設置完成模型下載與推論服務設定。"
        )

    async def _revise_chapter_text(
        self,
        system_prompt: str,
        draft_text: str,
        guidance: str,
        on_token: Callable[[str], None] | None,
    ) -> str:
        """Execute single-round Gate B literary polish revision without hard truncation."""
        if self.llm_client and self.llm_client.server_manager.is_healthy():
            revision_system_prompt = (
                f"{system_prompt}\n\n"
                "【文學修訂與深度潤色專用指令】：\n"
                "你目前正在針對文學審核（Gate B）提出的扣分維度與修改指引進行定點精雕細琢。\n"
                "1. 【嚴格修復扣分項】：加強具象五感物性細節，剷除空洞口號；替換高頻俗套慣用比喻，改以新穎冷峻的客觀意象；豐富對話潛台詞與微動作伴隨；加強衝突情境壓迫，並保持章末懸念未決。\n"
                "2. 【保持核心結構】：嚴格保持原章節情節主線與事件因果不變，不得任意刪減核心劇情節點。\n"
                "3. 【純文學輸出】：直接輸出潤色後的完整章節正文散文，嚴禁輸出任何修改說明、標題標籤或元評論。"
            )
            user_prompt = f"【自審修訂指引】\n{guidance}\n\n【待修訂初稿】\n{draft_text}"
            revised = await self.llm_client.generate_text_stream(
                revision_system_prompt, user_prompt, max_tokens=4200, temperature=0.6, on_token=on_token
            )
            return sanitize_novel_prose(revised)
        raise AIModelUnavailableError("本地 AI 模型未就緒或推論服務離線，無法進行文學潤色。")

    async def _extract_chapter_deltas(
        self, chapter_num: int, ch_info: dict[str, Any], text: str
    ) -> ChapterExtractionContract:
        """Extract structured state changes, 3D tension scores, and timeline events from full chapter text."""
        if self.llm_client and self.llm_client.server_manager.is_healthy():
            sys_prompt = (
                "你是一位專精長篇小說世界觀狀態機與伏筆拓撲追蹤的大師。請深入分析本章小說正文，精準抽取章節狀態增量：\n"
                "【輸出規範與鐵律】：\n"
                "1. 【全篇繁體中文】：所有抽取的字串欄位（章節標題、角色名、狀態、伏筆描述、事件摘要、地點）必須完全使用繁體中文（台灣標準書面語），嚴禁出現簡體字。\n"
                "2. 【角色與關係精準】：character_name 必須與正文中出現的名字完全一致，不得擅自簡寫或改字；new_status 僅能填 ALIVE（本章結束時仍存活）或 DEAD（本章確認死亡）；status_note 為 2 至 6 字的精準狀態術語，明確界定其生理傷勢、修為境界、心境轉折或存亡處境；relation_deltas 記錄本章新建立或實質破裂的關係。\n"
                "3. 【伏筆雙向追蹤】：foreshadow_deltas 中，action 為 PLANT（埋設伏線、暗扣或未解因果）或 RESOLVE（閉合伏線、揭曉謎團或兌現前置代價），描述必須客觀具體記錄核心因果或關鍵信物。\n"
                "4. 【時間軸與 3D 評分】：timeline_deltas 總結本章實質推進的大事件；\n"
                "   【3D 節奏評分客觀標準】：長篇小說單章的 goal_progress 通常為 1～2（微觀切片推進），重要轉折為 3，分卷決戰章方可達 4～5。tension_intensity 與 emotional_payoff 請如實反映單章微觀切片，若僅是鋪陳暗湧，切勿虛高標記 4 或 5！"
            )
            raw_ext = await self.llm_client.generate_structured(
                sys_prompt, text, ChapterExtractionContract
            )
            raw_ext.chapter_title = convert_simplified_to_traditional(raw_ext.chapter_title.strip())
            for c in raw_ext.character_deltas:
                c.character_name = convert_simplified_to_traditional(c.character_name.strip())
                c.new_status = convert_simplified_to_traditional(c.new_status.strip())
                c.status_note = convert_simplified_to_traditional(c.status_note.strip())
                c.location_name = convert_simplified_to_traditional(c.location_name.strip())
            for f in raw_ext.foreshadow_deltas:
                f.description = convert_simplified_to_traditional(f.description.strip())
                f.resolution_notes = convert_simplified_to_traditional(f.resolution_notes.strip())
                f.related_characters = [convert_simplified_to_traditional(rc.strip()) for rc in f.related_characters]
            for r in raw_ext.relation_deltas:
                r.source_name = convert_simplified_to_traditional(r.source_name.strip())
                r.relation_type = convert_simplified_to_traditional(r.relation_type.strip())
                r.target_name = convert_simplified_to_traditional(r.target_name.strip())
            for t in raw_ext.timeline_deltas:
                t.track_name = convert_simplified_to_traditional(t.track_name.strip())
                t.in_universe_calendar = convert_simplified_to_traditional(t.in_universe_calendar.strip())
                t.location_name = convert_simplified_to_traditional(t.location_name.strip())
                t.event_summary = convert_simplified_to_traditional(t.event_summary.strip())
            return raw_ext
        raise AIModelUnavailableError("本地 AI 模型未就緒或推論服務離線，無法抽取章節實體變化。")
