"""Infinite-depth interactive Q&A engine with real-time contradiction detection."""

from __future__ import annotations

import json
from typing import Any

from autonovel.db.project_repo import ProjectRepository
from autonovel.db.reference_repo import ReferenceRepository
from autonovel.llm.llm_client import LLMClient
from autonovel.schemas.llm_contracts import WizardQuestionContract, WizardStepAnswerContract

QUESTION_PROGRESSION_ASPECTS = [
    ("敘事文風與推進節奏", "確立全書文字語感、鏡頭冷暖色調與章節衝突推進節奏"),
    ("力量／資源階梯體系", "定義角色成長代價、資源爭奪焦點與上限規則"),
    ("主角核心動機與性格缺陷", "塑造主角非臉譜化的行為準則與內在掙扎"),
    ("核心對立勢力與社會結構", "建構多方博弈的派系網脈與秩序矛盾"),
    ("開篇切入鏡頭與危機鉤子", "鎖定第一章的場景氛圍、突發事件與破局方式"),
    ("長線世界觀終極謎團", "埋設貫穿數百萬字的底層真相與宿命伏筆"),
    ("場景美學與標誌性意象", "細化戰鬥／交鋒時的視覺、聽覺與節奏偏好"),
]


def get_blueprint_unlock_threshold(min_word_count: int) -> int:
    """Dynamically determine required Q&A depth threshold based on target word count."""
    if min_word_count < 1_000_000:
        return 7
    elif min_word_count < 3_000_000:
        return 10
    return 14


class AIModelUnavailableError(RuntimeError):
    """Raised when an interactive or generation operation requires live LLM model but server/model is offline."""


class WizardQAEngine:
    """Drives infinite-depth macro-to-micro Q&A and checks answer consistency."""

    def __init__(
        self,
        repo: ProjectRepository,
        llm_client: LLMClient | None = None,
        ref_repo: ReferenceRepository | None = None,
    ) -> None:
        """Initialize Q&A engine with project repository, LLM client, and reference repository."""
        self.repo = repo
        self.llm_client = llm_client
        self.ref_repo = ref_repo
        self._variation_counter = 0

    async def ais_llm_ready(self) -> bool:
        """Probe LLM readiness without blocking the loop or reviving a manual unload."""
        if not self.llm_client:
            return False
        manager = self.llm_client.server_manager
        if getattr(manager, "_manually_unloaded", False):
            return False
        if manager.is_healthy():
            return True
        await manager.ensure_running_async()
        return manager.is_healthy()

    def get_qa_history(self) -> list[dict[str, Any]]:
        """Load current Q&A history list from project_config."""
        cfg = self.repo.get_project_config()
        return json.loads(cfg.get("qa_history_json") or "[]")

    def get_unlock_threshold(self) -> int:
        """Return the current novel's required Q&A threshold based on min_word_count."""
        cfg = self.repo.get_project_config()
        min_words = int(cfg.get("min_word_count") or 5_000_000)
        return get_blueprint_unlock_threshold(min_words)

    async def submit_answer_and_next(
        self,
        question_title: str,
        selected_answers: list[str],
        custom_note: str = "",
        question_aspect: str = "",
    ) -> dict[str, Any]:
        """Validate new answer against history for semantic contradictions via AI and generate next question."""
        history = self.get_qa_history()
        combined_answer = "；".join(selected_answers)
        if custom_note.strip():
            combined_answer = f"{combined_answer}（補充：{custom_note.strip()}）".strip("；")
        new_entry = {
            "step_index": len(history) + 1,
            "question": question_title,
            "aspect": question_aspect,
            "answer": combined_answer,
        }
        history.append(new_entry)
        next_q = await self.generate_next_question(history)
        if next_q.has_contradiction:
            history.pop()
            return {
                "accepted": False,
                "contradiction": {
                    "has_contradiction": True,
                    "reason": next_q.contradiction_reason or "設定存在語義矛盾，請重新選擇或回退。",
                    "conflicting_step_index": next_q.conflicting_step_index or max(1, len(history)),
                },
                "history": history,
                "next_question": None,
            }
        self.repo.update_project_config(
            qa_history_json=json.dumps(history, ensure_ascii=False)
        )
        threshold = self.get_unlock_threshold()
        return {
            "accepted": True,
            "contradiction": None,
            "history": history,
            "next_question": next_q.model_dump(),
            "total_answered": len(history),
            "qa_threshold": threshold,
            "can_build_blueprint": len(history) >= threshold,
        }

    def rollback_to_step(self, step_index: int) -> list[dict[str, Any]]:
        """Truncate Q&A history back to a specific step_index to allow re-answering."""
        history = self.get_qa_history()
        trimmed = [item for item in history if int(item["step_index"]) < step_index]
        self.repo.update_project_config(
            qa_history_json=json.dumps(trimmed, ensure_ascii=False)
        )
        return trimmed

    async def generate_next_question(
        self,
        history: list[dict[str, Any]] | None = None,
        force_variation: bool = False,
    ) -> WizardQuestionContract:
        """Generate the next deeper question strictly via live AI LLM (no heuristic fallback)."""
        if force_variation:
            self._variation_counter += 1
        hist = history if history is not None else self.get_qa_history()
        depth = len(hist)
        cfg = self.repo.get_project_config()
        title = cfg.get("title") or "未命名長篇小說"
        g_pri = cfg.get("genre_primary") or "奇幻"
        g_sub = cfg.get("genre_sub") or "古典修仙"
        samples = self.ref_repo.get_style_and_theme_samples(limit=3) if self.ref_repo else []
        if await self.ais_llm_ready():
            return await self._generate_via_llm(title, g_pri, g_sub, depth, hist, samples)
        raise AIModelUnavailableError(
            "本地 AI 模型未就緒或推論服務離線，無法進行真實 AI 問答衍生。請先於環境設置完成模型下載並啟動推論服務。"
        )

    @classmethod
    def _pick_balanced_aspect_static(cls, history: list[dict[str, Any]]) -> tuple[str, str]:
        """Balance question dimensions across all 7 aspects and deepen progressive layers."""
        if not history:
            return QUESTION_PROGRESSION_ASPECTS[0]

        aspect_counts: dict[str, int] = {asp: 0 for asp, _ in QUESTION_PROGRESSION_ASPECTS}
        for item in history:
            matched = False
            stored_aspect = item.get("aspect")
            if stored_aspect and stored_aspect in aspect_counts:
                aspect_counts[stored_aspect] += 1
                matched = True
            else:
                q_text = item.get("question", "")
                for asp, _ in QUESTION_PROGRESSION_ASPECTS:
                    if asp in q_text:
                        aspect_counts[asp] += 1
                        matched = True
                        break
            if not matched:
                idx = (int(item.get("step_index", 1)) - 1) % len(QUESTION_PROGRESSION_ASPECTS)
                aspect_counts[QUESTION_PROGRESSION_ASPECTS[idx][0]] += 1

        min_count = min(aspect_counts.values())
        candidates = [
            (asp, desc) for asp, desc in QUESTION_PROGRESSION_ASPECTS
            if aspect_counts[asp] == min_count
        ]
        selected_aspect, base_desc = candidates[0]
        if min_count > 0:
            deepen_desc = f"深化「{selected_aspect}」之核心衝突、代價守恆與深層因果（第 {min_count + 1} 階）"
            return selected_aspect, deepen_desc
        return selected_aspect, base_desc

    async def auto_answer_current_question(
        self,
        question_title: str,
        question_aspect: str,
        options: list[str],
        history: list[dict[str, Any]],
    ) -> WizardStepAnswerContract:
        """Invoke Genre-Adaptive Novelist Co-Pilot to select balanced options with macro guidance."""
        cfg = self.repo.get_project_config()
        title = cfg.get("title") or "未命名長篇小說"
        g_pri = cfg.get("genre_primary") or "奇幻"
        g_sub = cfg.get("genre_sub") or "古典修仙"
        if not await self.ais_llm_ready():
            raise AIModelUnavailableError(
                "本地 AI 模型未就緒或推論服務離線，無法執行自動決策。請先於環境設置完成模型下載並啟動推論服務。"
            )
        sys_prompt = (
            "【語言輸出鐵律】你必須全程使用『繁體中文（台灣標準書面語）』輸出！絕對嚴禁輸出任何簡體字！\n"
            f"你是百萬字長篇小說首席特約顧問與小說家。書名：《{title}》，核心題材：{g_pri}（子流派：{g_sub}）。\n"
            "你的職責是站在原作者的宏觀立場，根據小說題材、書名與歷史問答，針對當前問題做出最契合題材風格的決策。\n\n"
            "【必須嚴格遵守的三大核心鐵律】：\n"
            "1. 【題材自適應與溫和理性】：嚴禁採取極端偏鋒、刻意反叛或中二的人格。請全盤權衡情節張力與設定穩定性。\n"
            "2. 【僅在給定選項中挑選】：selected_options 必須從當前題目的 options 清單中挑選 1～2 個最契合題材的具體選項，嚴禁隨意自創無關選項。\n"
            "3. 【嚴禁微觀細節】：thematic_guidance 僅能提供 30～50 字內的宏觀創作原則方針，著重敘事重心與哲理取向。絕對嚴禁捏造具體人名、組織名、功法名稱、地名或微觀分鏡劇本！所有具體實體設定一律留待後續藍圖建構階段處理。"
        )
        user_prompt = json.dumps(
            {
                "current_question": question_title,
                "aspect": question_aspect,
                "options": options,
                "history_context": [h.get("answer", "") for h in history[-6:]],
            },
            ensure_ascii=False,
        )
        assert self.llm_client is not None
        return await self.llm_client.generate_structured(
            sys_prompt, user_prompt, WizardStepAnswerContract
        )

    async def step_auto_turn(
        self,
        question_title: str,
        question_aspect: str,
        options: list[str],
    ) -> dict[str, Any]:
        """Execute a single interactive turn: Role B answers -> saves to DB -> Role A generates next balanced question."""
        history = self.get_qa_history()
        answer_contract = await self.auto_answer_current_question(
            question_title=question_title,
            question_aspect=question_aspect,
            options=options,
            history=history,
        )
        combined = "；".join(answer_contract.selected_options)
        if answer_contract.thematic_guidance.strip():
            combined = f"{combined}（方針：{answer_contract.thematic_guidance.strip()}）"

        new_entry = {
            "step_index": len(history) + 1,
            "question": question_title,
            "aspect": question_aspect,
            "answer": combined,
        }
        history.append(new_entry)
        # Generate before persisting: if the next-question call fails, the user simply
        # retries the same step. Saving first made every retry append a duplicate turn.
        next_q = await self.generate_next_question(history)
        self.repo.update_project_config(
            qa_history_json=json.dumps(history, ensure_ascii=False)
        )
        threshold = self.get_unlock_threshold()
        return {
            "accepted": True,
            "step_index": len(history),
            "answered_title": question_title,
            "answered_selection": answer_contract.selected_options,
            "answered_guidance": answer_contract.thematic_guidance,
            "history": history,
            "next_question": next_q.model_dump(),
            "total_answered": len(history),
            "qa_threshold": threshold,
            "can_build_blueprint": len(history) >= threshold,
        }

    async def _generate_via_llm(
        self,
        title: str,
        g_pri: str,
        g_sub: str,
        depth: int,
        hist: list[dict[str, Any]],
        samples: list[dict[str, str]],
    ) -> WizardQuestionContract:
        """Invoke LLM with GBNF grammar to generate bespoke style/pacing or setting options and check consistency."""
        aspect, desc = self._pick_balanced_aspect_static(hist)
        ref_hint = "；".join(s["description"] for s in samples) if samples else "無外部書庫樣本"
        sys_prompt = (
            "【語言輸出鐵律】你必須全程使用『繁體中文（台灣標準書面語）』輸出！絕對嚴禁輸出任何簡體字！\n"
            f"你是百萬字長篇小說世界觀與文風引導大師。書名：《{title}》，核心題材：{g_pri}（子流派：{g_sub}）。\n"
            f"目前處於第 {depth + 1} 層遞進問答，核心聚焦維度：【{aspect} — {desc}】。\n"
            "請先審查 qa_history 中最新一筆回答是否與先前設定存在不可調和的語義邏輯矛盾；若存在矛盾，將 has_contradiction 設為 true 並說明 contradiction_reason 與 conflicting_step_index。\n"
            "若無矛盾，請根據題材組合、參考書庫養分與已答紀錄，動態生成 1 道深度決策問題與 4～5 個風格迥異、對比強烈的具體選項。\n"
            "【選項設計要求】\n"
            "1. 選項必須具備強烈風格反差，涵蓋正向擴展、代價制衡、深層博弈或異質機制等多元路徑，每項皆需包含明確的世界機制或劇情動機，拒絕平淡刻板！\n"
            "2. question_title 格式為『請決定「…」』，直指核心衝突，不要重複書名或題材名稱。\n"
            "3. question_aspect 僅填維度名稱。\n"
            "4. allow_multi_select 必須設為 true，允許創作者勾選多項進行複合設定融合。"
        )
        user_prompt = json.dumps(
            {"qa_history": hist[-8:], "reference_style_samples": ref_hint, "variation": self._variation_counter},
            ensure_ascii=False,
        )
        assert self.llm_client is not None
        return await self.llm_client.generate_structured(
            sys_prompt, user_prompt, WizardQuestionContract
        )
