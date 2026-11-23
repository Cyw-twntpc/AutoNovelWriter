"""5M-word Milestone Pacing Controller, rolling volume planner, and end-of-volume review."""

from __future__ import annotations

import json
import statistics
from typing import Any

from autonovel.config import QualityThresholdConfig
from autonovel.db.connection import atomic_transaction
from autonovel.db.project_repo import ProjectRepository
from autonovel.engine.prose_sanitizer import convert_simplified_to_traditional
from autonovel.llm.llm_client import LLMClient
from autonovel.reference.retriever import ReferenceRetriever
from autonovel.schemas.domain import AcceptanceReport
from autonovel.schemas.llm_contracts import VolumeOutlineContract
from autonovel.wizard.qa_engine import AIModelUnavailableError


class VolumePlanner:
    """Manages volume-level rolling outlines, end-of-volume acceptance, and 5M-word pacing."""

    def __init__(
        self,
        repo: ProjectRepository,
        retriever: ReferenceRetriever,
        thresholds: QualityThresholdConfig | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Initialize volume planner with repository, reference retriever, and thresholds."""
        self.repo = repo
        self.retriever = retriever
        self.thresholds = thresholds or QualityThresholdConfig()
        self.llm_client = llm_client

    def evaluate_acceptance_report(
        self, volume_num: int | None = None
    ) -> AcceptanceReport:
        """Compute the 5 perfectionist acceptance metrics for a volume or the entire book."""
        if volume_num is not None:
            metrics = self.repo.fetch_all(
                """
                SELECT m.* FROM chapter_metrics m
                JOIN chapters c ON c.chapter_num = m.chapter_num
                WHERE c.volume_num = ? ORDER BY m.chapter_num;
                """,
                (volume_num,),
            )
        else:
            metrics = self.repo.fetch_all(
                "SELECT * FROM chapter_metrics ORDER BY chapter_num;"
            )
        if not metrics:
            return AcceptanceReport(
                hard_conflict_rate=0.0,
                timeline_consistency_rate=0.0,
                foreshadow_recovery_rate=0.0,
                pacing_stability_rate=0.0,
                mean_literary_score=0.0,
                passed=False,
                failed_reasons=["目前尚無已完成章節，待正文生成後自動啟動品質驗收評測"],
                details={"total_chapters": 0, "bottom_20_percent_chapters": []},
            )
        scores = [float(m["literary_score"]) for m in metrics]
        mean_score = round(statistics.mean(scores), 2)
        hard_conflict_rate = self._compute_hard_conflict_rate(volume_num, len(metrics))
        foreshadow_rate = self._compute_foreshadow_recovery_rate()
        pacing_rate = self._compute_pacing_stability_rate(metrics)
        timeline_ok = self._verify_timeline_monotonicity()
        failed: list[str] = []
        if hard_conflict_rate > 0.0:
            failed.append(f"存在境界上限或硬性設定違規 ({hard_conflict_rate:.1%})")
        if not timeline_ok:
            failed.append("時間線因果一致性未達 100%")
        if foreshadow_rate < self.thresholds.foreshadow_recovery_min_rate:
            failed.append(f"伏筆回收率 ({foreshadow_rate:.1%}) 低於 95%")
        if pacing_rate < self.thresholds.pacing_stability_min_rate:
            failed.append(f"節奏平穩度 ({pacing_rate:.1%}) 低於 95%")
        if mean_score < self.thresholds.volume_acceptance_mean_score:
            failed.append(f"文學自審均分 ({mean_score}) 低於 85 分")
        low_chapters = self._find_bottom_20_percent_chapters(
            metrics, force_select=bool(failed)
        )
        report = AcceptanceReport(
            hard_conflict_rate=round(hard_conflict_rate, 4),
            timeline_consistency_rate=1.0 if timeline_ok else 0.0,
            foreshadow_recovery_rate=round(foreshadow_rate, 4),
            pacing_stability_rate=round(pacing_rate, 4),
            mean_literary_score=mean_score,
            passed=len(failed) == 0,
            failed_reasons=failed,
            details={"total_chapters": len(metrics), "bottom_20_percent_chapters": low_chapters},
        )
        if volume_num is not None:
            with atomic_transaction(self.repo.db_path) as conn:
                conn.execute(
                    "UPDATE volumes SET review_report_json = ? WHERE volume_num = ?;",
                    (report.model_dump_json(), volume_num),
                )
        return report

    def _compute_hard_conflict_rate(
        self, volume_num: int | None, total_chapters: int
    ) -> float:
        """Verify 0 active characters exceed the volume's stage-gate rank ceiling."""
        if total_chapters <= 0:
            return 0.0
        if volume_num is not None:
            vols = self.repo.fetch_all(
                "SELECT max_allowed_rank_level FROM volumes WHERE volume_num = ?;",
                (volume_num,),
            )
        else:
            vols = self.repo.fetch_all(
                "SELECT MAX(max_allowed_rank_level) AS max_allowed_rank_level FROM volumes;"
            )
        ceiling = int(vols[0]["max_allowed_rank_level"] or 99) if vols else 99
        violators = self.repo.fetch_all(
            "SELECT COUNT(*) AS c FROM characters WHERE status = 'ALIVE' AND rank_level > ?;",
            (ceiling,),
        )
        v_count = int(violators[0]["c"]) if violators else 0
        return min(1.0, v_count / max(1, total_chapters))

    def _compute_foreshadow_recovery_rate(self) -> float:
        """Compute resolved / due foreshadowing ratio.

        Overdue PENDING items already count as due (they should have paid off by
        now), so the rate cannot stay pinned at 1.0 just because nothing ever writes
        EXPIRED rows to the table.
        """
        rows = self.repo.fetch_all(
            "SELECT status, planted_chapter, expected_span FROM foreshadowing;"
        )
        if not rows:
            return 1.0
        ch_rows = self.repo.fetch_all(
            "SELECT COALESCE(MAX(chapter_num), 0) AS c FROM chapter_metrics;"
        )
        current_ch = int(ch_rows[0]["c"]) if ch_rows else 0
        resolved = 0
        due = 0
        for row in rows:
            status = str(row["status"])
            if status == "RESOLVED":
                resolved += 1
                due += 1
            elif status == "EXPIRED":
                due += 1
            elif status == "PENDING":
                planted = int(row["planted_chapter"] or 0)
                span = max(1, int(row["expected_span"] or 1))
                if current_ch - planted > span:
                    due += 1
        return 1.0 if due == 0 else resolved / due

    @staticmethod
    def _compute_pacing_stability_rate(metrics: list[dict[str, Any]]) -> float:
        """Compute proportion of chapters within normal tension/progress variation."""
        if len(metrics) < 3:
            return 1.0
        abnormal = 0
        for idx in range(2, len(metrics)):
            w3 = metrics[idx - 2 : idx + 1]
            if all(int(r["tension_intensity"]) <= 1 for r in w3) and all(
                int(r["goal_progress"]) <= 1 for r in w3
            ):
                abnormal += 1
        return max(0.0, (len(metrics) - abnormal) / len(metrics))

    def _verify_timeline_monotonicity(self) -> bool:
        """Verify elapsed_days is monotonically non-decreasing per track in chapter order."""
        rows = self.repo.fetch_all(
            "SELECT track_name, chapter_num, elapsed_days FROM timeline "
            "ORDER BY track_name ASC, chapter_num ASC, id ASC;"
        )
        last_seen: dict[str, float] = {}
        for r in rows:
            track = str(r["track_name"])
            days = float(r["elapsed_days"])
            if track in last_seen and days < last_seen[track]:
                return False
            last_seen[track] = days
        return True

    @staticmethod
    def _find_bottom_20_percent_chapters(
        metrics: list[dict[str, Any]], force_select: bool = False
    ) -> list[int]:
        """Identify the lowest-scoring 20% chapters in a volume for targeted re-revision."""
        if not metrics:
            return []
        sorted_m = sorted(metrics, key=lambda m: float(m["literary_score"]))
        k = max(1, len(sorted_m) // 5)
        if force_select:
            return [int(m["chapter_num"]) for m in sorted_m[:k]]
        return [int(m["chapter_num"]) for m in sorted_m[:k] if float(m["literary_score"]) < 85.0]

    def should_conclude_novel(self) -> bool:
        """Check if total words >= min_word_count and all main foreshadowings are resolved."""
        cfg = self.repo.get_project_config()
        min_words = int(cfg.get("min_word_count") or 5_000_000)
        word_rows = self.repo.fetch_all(
            "SELECT COALESCE(SUM(word_count), 0) AS total_w FROM chapter_metrics;"
        )
        total_words = int(word_rows[0]["total_w"]) if word_rows else 0
        if total_words < min_words:
            return False
        unresolved_main = self.repo.fetch_all(
            "SELECT COUNT(*) AS c FROM foreshadowing WHERE scope = 'MAIN' AND status = 'PENDING';"
        )
        return int(unresolved_main[0]["c"]) == 0

    async def plan_next_volume(
        self, next_volume_num: int, chapters_per_volume: int = 10
    ) -> dict[str, Any]:
        """Roll out the next volume outline and chapter Scene Beats under Stage Gate limits."""
        gate = self._resolve_stage_gate(next_volume_num)
        last_ch_rows = self.repo.fetch_all(
            "SELECT COALESCE(MAX(chapter_num), 0) AS max_c FROM chapters;"
        )
        start_ch = int(last_ch_rows[0]["max_c"]) + 1
        ref_digest, _ = await self.retriever.retrieve_and_compress(
            f"第 {next_volume_num} 卷分卷情節規劃：{gate.get('stage_title', '')}，階段里程碑目標：{gate.get('milestone_goal', '')}"
        )
        llm_contract = await self._try_llm_volume_outline(
            next_volume_num, chapters_per_volume, gate, ref_digest
        )
        vol_title = llm_contract.volume_title
        vol_outline = llm_contract.volume_summary
        target_ch_count = len(llm_contract.chapters)
        with atomic_transaction(self.repo.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO volumes (
                    volume_num, title, outline_text, target_chapters,
                    target_word_quota, max_allowed_rank_level, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'PLANNED');
                """,
                (
                    next_volume_num,
                    vol_title,
                    vol_outline,
                    target_ch_count,
                    int(gate.get("target_word_quota", 150000)),
                    int(gate.get("max_allowed_rank_level", next_volume_num * 2)),
                ),
            )
            self._insert_volume_chapters(
                conn, next_volume_num, start_ch, llm_contract
            )
            conn.execute(
                """
                INSERT INTO foreshadowing (
                    description, scope, planted_chapter, expected_span, status, related_character_ids
                ) VALUES (?, 'VOLUME', ?, ?, 'PENDING', '[]');
                """,
                (
                    f"第{next_volume_num}卷核心伏筆：{gate.get('core_mystery_quota') or gate.get('milestone_goal', vol_title)}",
                    start_ch,
                    target_ch_count,
                ),
            )
        return {"volume_num": next_volume_num, "title": vol_title, "start_chapter": start_ch}

    def _resolve_stage_gate(self, next_volume_num: int) -> dict[str, Any]:
        """Fetch the stage gate corresponding to next_volume_num from global_outline_json."""
        cfg = self.repo.get_project_config()
        global_outline = json.loads(cfg.get("global_outline_json") or "{}")
        gates = global_outline.get("stage_gates", [])
        if gates and next_volume_num <= len(gates):
            return gates[next_volume_num - 1]
        if gates:
            # Past the planned outline, extrapolate instead of replaying the final
            # gate forever: a frozen rank ceiling would cap the protagonist forever.
            gate = dict(gates[-1])
            overflow = next_volume_num - len(gates)
            gate["max_allowed_rank_level"] = (
                int(gate.get("max_allowed_rank_level", next_volume_num * 2)) + overflow
            )
            gate["stage_title"] = f"{gate.get('stage_title', '')}（續章 {overflow}）"
            gate.setdefault("target_word_quota", 150000)
            return gate
        return {
            "stage_title": f"第{next_volume_num}階段征途",
            "target_word_quota": 150000,
            "max_allowed_rank_level": next_volume_num * 2,
            "milestone_goal": "推進主線並回收前期伏筆",
        }

    async def _try_llm_volume_outline(
        self,
        next_volume_num: int,
        chapters_per_volume: int,
        gate: dict[str, Any],
        ref_digest: str,
    ) -> VolumeOutlineContract:
        """Generate structured volume outline via LLM when llama-server is healthy."""
        if not (self.llm_client and self.llm_client.server_manager.is_healthy()):
            raise AIModelUnavailableError("本地 AI 模型未就緒或推論服務離線，無法進行分卷大綱規劃。")
        chars = self.repo.fetch_all(
            "SELECT name, rank_level, description FROM characters WHERE status = 'ALIVE' LIMIT 8;"
        )
        pending_fs = self.repo.fetch_all(
            "SELECT id, scope, description FROM foreshadowing WHERE status = 'PENDING' ORDER BY id ASC LIMIT 8;"
        )
        matched_tpl = None
        if hasattr(self, "retriever") and hasattr(self.retriever, "repo"):
            cfg = self.repo.get_project_config()
            gp = cfg.get("genre_primary", "")
            gs = cfg.get("genre_sub", "")
            matched_tpl = self.retriever.repo.match_pacing_template(gp, gs)
        pacing_guidance = ""
        if matched_tpl and matched_tpl.get("stages"):
            pacing_guidance = (
                f"\n參考節奏波浪量化曲線：{json.dumps(matched_tpl['stages'], ensure_ascii=False)}"
            )
        sys_prompt = (
            f"你是百萬字長篇小說首席分卷架構師。請為第 {next_volume_num} 卷規劃分卷大綱與 {chapters_per_volume} 章預排細綱：\n"
            "【分卷規劃鐵律】：\n"
            "1. 【全篇繁體中文】：所有大綱文字、章節標題、Scene Beats 必須完全使用繁體中文（台灣標準書面語），嚴禁出現簡體字。\n"
            f"2. 【境界與階段護欄】：本卷境界上限嚴格受限於 <= {gate.get('max_allowed_rank_level', next_volume_num * 2)} 階，不得過早越級造成戰力崩壞。\n"
            "3. 【角色命名絕對原創】：登場角色必須嚴格遵守原創命名法度，嚴禁借用任何既有經典作品、歷史傳說或流行文化之人名，依循題材世界觀原創音韻與號稱。\n"
            f"4. 【波浪起伏架構】：細綱需完整規劃 {chapters_per_volume} 章，遵循宏觀波浪節奏演繹，由局勢鋪陳、危機遞進、波折膠著推進至局部高潮破局，每章配置 2～3 個功能明確的 Scene Beats，嚴密調控張力流速與因果閉環。"
        )
        user_prompt = (
            f"階段目標：{json.dumps(gate, ensure_ascii=False)}\n"
            f"活躍角色：{json.dumps(chars, ensure_ascii=False)}\n"
            f"待回收伏筆：{json.dumps(pending_fs, ensure_ascii=False)}\n"
            f"參考技法：{ref_digest}{pacing_guidance}"
        )
        raw_contract = await self.llm_client.generate_structured(
            sys_prompt, user_prompt, VolumeOutlineContract, max_tokens=2500
        )
        raw_contract.volume_title = convert_simplified_to_traditional(raw_contract.volume_title.strip())
        raw_contract.volume_summary = convert_simplified_to_traditional(raw_contract.volume_summary.strip())
        for ch in raw_contract.chapters:
            ch.suggested_title = convert_simplified_to_traditional(ch.suggested_title.strip())
            for b in ch.beats:
                b.scene_tag = convert_simplified_to_traditional(b.scene_tag.strip())
                b.plot_objective = convert_simplified_to_traditional(b.plot_objective.strip())
                b.pacing_instruction = convert_simplified_to_traditional(b.pacing_instruction.strip())
                b.involved_characters = [convert_simplified_to_traditional(c.strip()) for c in b.involved_characters]
                b.location_name = convert_simplified_to_traditional(b.location_name.strip())
        return raw_contract

    def _insert_volume_chapters(
        self,
        conn: Any,
        volume_num: int,
        start_ch: int,
        llm_contract: VolumeOutlineContract,
    ) -> None:
        """Insert AI-planned chapter rows with 2-3 Scene Beats each for the new volume."""
        if not llm_contract.chapters:
            raise ValueError("VolumeOutlineContract.chapters must contain AI-planned chapters.")
        for offset, c_item in enumerate(llm_contract.chapters):
            ch_num = start_ch + offset
            raw_title = c_item.suggested_title.strip()
            title = raw_title if raw_title.startswith("第") else f"第{ch_num}章 {raw_title}"
            beats = [b.model_dump() for b in c_item.beats]
            conn.execute(
                """
                INSERT OR REPLACE INTO chapters (chapter_num, volume_num, title, beats_json, status)
                VALUES (?, ?, ?, ?, 'UNWRITTEN');
                """,
                (ch_num, volume_num, title, json.dumps(beats, ensure_ascii=False)),
            )
