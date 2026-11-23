"""3D Tension Sliding Window monitor and foreshadowing expiration tracker."""

from __future__ import annotations

import json
from typing import Any

from autonovel.db.project_repo import ProjectRepository
from autonovel.db.reference_repo import ReferenceRepository
from autonovel.schemas.domain import AnomalyType


class PacingMonitor:
    """Monitors 3D tension metrics (tension, progress, payoff) and foreshadowing spans against reference archetypes."""

    def __init__(
        self,
        repo: ProjectRepository,
        ref_repo: ReferenceRepository | None = None,
    ) -> None:
        """Initialize pacing monitor with project repository and optional reference repository."""
        self.repo = repo
        self.ref_repo = ref_repo

    def evaluate_sliding_window(self, current_chapter_num: int) -> dict[str, Any]:
        """Inspect recent 5 chapters and volume progress for macro pacing and micro tension."""
        # 1. Determine current volume and progress ratio within the volume
        ch_rows = self.repo.fetch_all(
            "SELECT volume_num FROM chapters WHERE chapter_num = ?;",
            (current_chapter_num,),
        )
        vol_num = int(ch_rows[0]["volume_num"]) if ch_rows else 1
        vol_chs = self.repo.fetch_all(
            "SELECT chapter_num FROM chapters WHERE volume_num = ? ORDER BY chapter_num ASC;",
            (vol_num,),
        )
        total_in_vol = max(1, len(vol_chs))
        try:
            ch_idx = [int(r["chapter_num"]) for r in vol_chs].index(current_chapter_num) + 1
        except ValueError:
            ch_idx = 1
        vol_ratio = ch_idx / total_in_vol

        # 2. Check for matching reference pacing template (Track B archetype)
        matched_tpl = None
        target_stage: dict[str, Any] | None = None
        if self.ref_repo:
            cfg = self.repo.get_project_config()
            gp = cfg.get("genre_primary", "")
            gs = cfg.get("genre_sub", "")
            matched_tpl = self.ref_repo.match_pacing_template(gp, gs)
            if matched_tpl and matched_tpl.get("stages"):
                ch_pct = vol_ratio * 100.0
                target_stage = min(
                    matched_tpl["stages"],
                    key=lambda s: abs(float(s["offset_percent"]) - ch_pct),
                )

        # 3. Objective quantitative telemetry from AI-evaluated metrics and reference wave
        rows = self.repo.fetch_all(
            """
            SELECT chapter_num, tension_intensity, goal_progress, emotional_payoff
            FROM chapter_metrics
            ORDER BY chapter_num DESC LIMIT 5;
            """
        )
        ordered = list(reversed(rows))
        alerts: list[dict[str, str]] = []
        directives: list[str] = []

        if target_stage:
            directives.append(
                f"【分卷波浪座標·{matched_tpl['name']}（第{vol_num}卷第{ch_idx}/{total_in_vol}章，進度 {vol_ratio*100:.0f}%）】"
                f"參考階段：{target_stage['pacing_phase']}｜目標張力: Lv.{target_stage['target_tension']}｜"
                f"目標推進: Lv.{target_stage['target_progress']}｜階段任務：{target_stage['dramatic_purpose']}。"
            )
            if ordered:
                last_t = int(ordered[-1]["tension_intensity"])
                target_t = int(target_stage["target_tension"])
                if last_t - target_t >= 2:
                    alerts.append(
                        {
                            "code": "RUSHING_ALERT",
                            "message": f"近期 AI 量化張力 ({last_t}) 高於參考波浪目標張力 ({target_t}) 達 {last_t - target_t} 級。",
                        }
                    )
        else:
            directives.append(
                f"【分卷進度座標】第{vol_num}卷第{ch_idx}/{total_in_vol}章（分卷進度 {vol_ratio*100:.0f}%）。"
            )

        if ordered:
            mean_t = sum(float(r["tension_intensity"]) for r in ordered) / len(ordered)
            mean_p = sum(float(r["goal_progress"]) for r in ordered) / len(ordered)
            mean_e = sum(float(r["emotional_payoff"]) for r in ordered) / len(ordered)
            seq_str = " -> ".join(
                f"Ch.{r['chapter_num']}(T:{r['tension_intensity']},P:{r['goal_progress']},E:{r['emotional_payoff']})"
                for r in ordered
            )
            directives.append(
                f"【近期AI量化評分滑動窗口（近{len(ordered)}章）】序列：{seq_str}｜"
                f"均值：張力={mean_t:.2f}、推進={mean_p:.2f}、情緒共鳴={mean_e:.2f}。"
            )
            if len(ordered) >= 5 and all(
                int(r["tension_intensity"]) <= 1 and int(r["goal_progress"]) <= 1
                for r in ordered[-5:]
            ):
                alerts.append(
                    {
                        "code": "STAGNATION_ALERT",
                        "message": "連續 5 章 AI 量化張力與目標推進度皆 <= 1。",
                    }
                )
            if len(ordered) >= 2 and all(
                int(r["tension_intensity"]) >= 5 for r in ordered[-2:]
            ):
                alerts.append(
                    {
                        "code": "CLIMAX_FATIGUE_ALERT",
                        "message": "連續 2 章 AI 量化張力達極限值 5。",
                    }
                )
            if len(ordered) >= 5 and all(
                int(r["emotional_payoff"]) <= 2 for r in ordered[-5:]
            ):
                alerts.append(
                    {
                        "code": "PAYOFF_DROUGHT_ALERT",
                        "message": "連續 5 章 AI 量化情緒共鳴 <= 2。",
                    }
                )

        return {
            "alerts": alerts,
            "pacing_directive": " ".join(directives),
            "recent_metrics": ordered,
        }

    def check_and_expire_foreshadowing(self, current_chapter_num: int) -> dict[str, Any]:
        """Track pending foreshadowings without destructive status mutation; flag overdue as gentle reminders."""
        pending = self.repo.fetch_all(
            "SELECT * FROM foreshadowing WHERE status = 'PENDING';"
        )
        due_soon: list[dict[str, Any]] = []
        overdue_items: list[dict[str, Any]] = []
        for item in pending:
            planted = int(item["planted_chapter"])
            span = max(1, int(item["expected_span"]))
            elapsed = current_chapter_num - planted
            if elapsed > 2 * span:
                overdue_items.append(dict(item))
            elif elapsed >= span:
                due_soon.append(dict(item))
        # Overdue items stay PENDING forever by design, so without this dedupe the
        # anomaly log grows by one row per overdue item on every single chapter.
        logged_ids: set[int] = set()
        for row in self.repo.fetch_all(
            "SELECT detail_diff_json FROM anomaly_logs WHERE anomaly_type = ?;",
            (AnomalyType.FORESHADOW_EXPIRED.value,),
        ):
            try:
                payload = json.loads(row["detail_diff_json"] or "{}")
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("foreshadow_id") is not None:
                logged_ids.add(int(payload["foreshadow_id"]))
        for item in overdue_items:
            fid = int(item["id"])
            if fid in logged_ids:
                continue
            self.repo.log_anomaly(
                chapter_num=current_chapter_num,
                anomaly_type=AnomalyType.FORESHADOW_EXPIRED.value,
                severity="INFO",
                resolution_method="伏筆超過預期跨度，標記 OVERDUE_PENDING 溫和提醒，保留伏筆生命週期",
                affected_chapters=[current_chapter_num],
                detail_diff={"foreshadow_id": fid, "description": item.get("description", "")},
            )
        return {
            "due_soon": due_soon,
            "overdue_items": overdue_items,
            "expired_ids": [int(x["id"]) for x in overdue_items],
        }
