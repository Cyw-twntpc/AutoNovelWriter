"""16k Context Router with 1-hop entity graph filtering and 600-800 char tail continuity."""

from __future__ import annotations

import json
from typing import Any

from autonovel.config import ContextBudgetConfig
from autonovel.db.project_repo import ProjectRepository


def estimate_tokens(text: str) -> int:
    """Estimate token count for Chinese/mixed text (~1.15 tokens per character)."""
    if not text:
        return 0
    return int(len(text) * 1.15) + 4


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Fit text within max_tokens by snapping cleanly at paragraph or sentence boundaries."""
    max_chars = max(1, int((max_tokens - 4) / 1.15))
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    last_newline = window.rfind("\n")
    if last_newline >= int(max_chars * 0.5):
        return window[:last_newline].rstrip()
    for sep in ("。", "！", "？", "；"):
        last_punct = window.rfind(sep)
        if last_punct >= int(max_chars * 0.5):
            return window[: last_punct + 1]
    return window


class ContextRouter:
    """Assembles chapter writing prompts within the strict 16k token budget."""

    def __init__(
        self,
        repo: ProjectRepository,
        budget: ContextBudgetConfig | None = None,
    ) -> None:
        """Initialize context router with project repository and token budget config."""
        self.repo = repo
        self.budget = budget or ContextBudgetConfig()

    def slice_previous_chapter_tail(self, current_chapter_num: int) -> str:
        """Extract the trailing 600-800 characters of chapter N-1 for seamless continuity."""
        if current_chapter_num <= 1:
            return ""
        rows = self.repo.fetch_all(
            "SELECT final_content FROM chapters WHERE chapter_num = ?;",
            (current_chapter_num - 1,),
        )
        if not rows or not rows[0].get("final_content"):
            return ""
        prev_text = str(rows[0]["final_content"]).strip()
        if len(prev_text) <= self.budget.previous_tail_max_chars:
            return prev_text
        raw_tail = prev_text[-self.budget.previous_tail_max_chars :]
        newline_pos = raw_tail.find("\n")
        if 0 < newline_pos <= (
            self.budget.previous_tail_max_chars - self.budget.previous_tail_min_chars
        ):
            return raw_tail[newline_pos + 1 :].strip()
        return raw_tail

    def build_chapter_context(
        self,
        chapter_num: int,
        reference_digest: str = "",
        pacing_directive: str = "",
    ) -> dict[str, Any]:
        """Route 1-hop entities, due foreshadowings, anchors, and tail within 16k tokens."""
        cfg = self.repo.get_project_config()
        ch_rows = self.repo.fetch_all(
            "SELECT * FROM chapters WHERE chapter_num = ?;", (chapter_num,)
        )
        ch_row = ch_rows[0] if ch_rows else {"volume_num": 1, "title": f"第{chapter_num}章", "beats_json": "[]"}
        beats = json.loads(ch_row.get("beats_json") or "[]")
        prev_tail = self.slice_previous_chapter_tail(chapter_num)
        system_block = self._build_system_and_anchor_block(cfg)
        settings_block = self._build_routed_settings_block(
            chapter_num, beats, prev_tail, cfg.get("forward_retcon_note", "")
        )
        beats_block = self._build_beats_block(ch_row, beats, pacing_directive)
        ref_block = truncate_to_token_budget(
            reference_digest, self.budget.reference_digest_tokens
        )
        tail_block = truncate_to_token_budget(
            prev_tail, self.budget.previous_tail_tokens
        )
        token_breakdown = {
            "system_and_anchors": estimate_tokens(system_block),
            "routed_settings": estimate_tokens(settings_block),
            "reference_digest": estimate_tokens(ref_block),
            "scene_beats": estimate_tokens(beats_block),
            "previous_tail": estimate_tokens(tail_block),
        }
        token_breakdown["total_prompt_tokens"] = sum(token_breakdown.values())
        return {
            "system_prompt": system_block,
            "user_prompt": self._compose_user_prompt(
                settings_block, ref_block, beats_block, tail_block
            ),
            "previous_tail": tail_block,
            "token_breakdown": token_breakdown,
        }

    def _build_system_and_anchor_block(self, cfg: dict[str, Any]) -> str:
        """Assemble style contract + static golden anchor + dynamic high-score anchor."""
        style = cfg.get("style_contract_text", "")
        static_anchor = cfg.get("static_anchor_text", "")
        dynamic_anchor = cfg.get("dynamic_anchor_text", "")
        parts = [
            "你是一位專精於百萬字長篇小說創作的文學大師。你正在直接撰寫小說的正式連載正文（不是大綱、不是分鏡腳本、不是設定集或結構草案）。\n"
            "【最高核心寫作準則】：\n"
            "1. 【繁體中文鐵律】：全篇正文必須完全且僅能使用繁體中文（台灣標準書面語），嚴禁出現任何簡體字、簡體詞彙或大陸網絡用語。\n"
            "2. 【純文學散文格式】：直接輸出正式的小說內文故事散文。嚴禁輸出任何層級的標題符號或標題語法；嚴禁輸出任何分鏡節拍標籤、場景階段註記、大綱提示詞或結構導引字樣。\n"
            "3. 【嚴禁章末元評論】：嚴禁在章末輸出任何「---」、「【本章結束】」、「【本章完】」、「此章節通過...」、「下章將...」等元評論、總結、後記或劇透。\n"
            "4. 【沈浸式敘事】：將對話、動作、微表情、環境烘托自然交織為連續且起伏有致的散文段落，依據節奏以空行自然分段。\n"
            "5. 【宏觀長篇思維·拒絕短篇化速成】：這是一部長篇小說的單一章節切片，絕對不是獨立微型短篇。單章嚴禁過度堆砌劇情焦點，嚴禁試圖在一章內快速達成宏大目標或徹底消滅強敵！推進必須以整卷為單位波浪式起伏，允許局勢維持膠著、波折與懸念，重在過程的微表情、細節與心理暗流。\n"
            "請嚴格遵守以下風格契約與雙語感錨點：",
            style,
            f"【靜態黃金錨點（底色DNA）】\n{static_anchor}",
        ]
        if dynamic_anchor.strip():
            parts.append(f"【動態高分錨點（近期最佳語感）】\n{dynamic_anchor}")
        combined = "\n\n".join(p for p in parts if p.strip())
        return truncate_to_token_budget(combined, self.budget.system_and_style_tokens)

    def _build_routed_settings_block(
        self,
        chapter_num: int,
        beats: list[dict[str, Any]],
        prev_tail: str,
        forward_retcon: str,
    ) -> str:
        """Select active characters + 1-hop relations + world rules + due foreshadowing."""
        active_chars = self._select_active_characters(beats, prev_tail)
        char_lines = [
            f"- {c['name']}（狀態:{c['status']}, 階梯:{c['rank_level']}）：{c['description']}"
            for c in active_chars
        ]
        char_lines.extend(self._format_active_relations(active_chars))
        loc_lines = self._select_active_locations(beats, active_chars)
        rules = self.repo.fetch_all("SELECT rule_text FROM world_rules LIMIT 8;")
        rule_lines = [f"- {r['rule_text']}" for r in rules]
        foreshadows = self.repo.fetch_all(
            "SELECT id, description, planted_chapter, expected_span FROM foreshadowing WHERE status = 'PENDING' ORDER BY planted_chapter ASC LIMIT 8;"
        )
        fs_lines = []
        for fs in foreshadows:
            overdue = (chapter_num - int(fs["planted_chapter"])) >= int(fs["expected_span"])
            tag = "【優先回收】" if overdue else "【待回收】"
            fs_lines.append(f"- [ID:{fs['id']}]{tag} {fs['description']}")
        timelines = self.repo.fetch_all(
            "SELECT chapter_num, track_name, elapsed_days, event_summary FROM timeline ORDER BY id DESC LIMIT 5;"
        )
        tl_lines = [
            f"- 第{t['chapter_num']}章[{t['track_name']}·第{t['elapsed_days']:.0f}天]：{t['event_summary']}"
            for t in reversed(timelines)
        ]
        sections = [
            "【世界觀核心法則】\n" + ("\n".join(rule_lines) or "無"),
            "【當前場景與地域設定】\n" + ("\n".join(loc_lines) or "無"),
            "【本章關聯角色與一階關係網】\n" + ("\n".join(char_lines) or "無"),
            "【活躍與到期伏筆清單】\n" + ("\n".join(fs_lines) or "無"),
            "【近期多軌時間軸事件】\n" + ("\n".join(tl_lines) or "無"),
        ]
        if forward_retcon.strip():
            sections.append(f"【前向圓場補償指令】\n{forward_retcon.strip()}")
        return truncate_to_token_budget(
            "\n\n".join(sections), self.budget.context_router_settings_tokens
        )

    def _select_active_characters(
        self, beats: list[dict[str, Any]], prev_tail: str
    ) -> list[dict[str, Any]]:
        """Query Beat-named characters, characters in prev_tail, plus 1-hop related characters."""
        mentioned_names: set[str] = set()
        for b in beats:
            for name in b.get("involved_characters", []):
                mentioned_names.add(str(name).strip())
        all_chars = self.repo.fetch_all(
            "SELECT * FROM characters WHERE status != 'ARCHIVED' ORDER BY id;"
        )
        selected_ids: set[int] = set()
        priority: dict[int, tuple[int, int, int, int, int]] = {}
        for ch in all_chars:
            try:
                aliases = json.loads(ch.get("aliases") or "[]")
            except (json.JSONDecodeError, TypeError):
                aliases = []
            if not isinstance(aliases, list):
                aliases = []
            names_to_check = [str(ch["name"]), *[str(a) for a in aliases]]
            is_mentioned = any(n in mentioned_names for n in names_to_check)
            is_in_tail = any(n and n in prev_tail for n in names_to_check)
            if str(ch.get("role_type")) == "PROTAGONIST" or is_mentioned or is_in_tail:
                cid = int(ch["id"])
                selected_ids.add(cid)
                # Rank by relevance instead of id: slicing by id dropped every newly
                # added character once the cap was reached.
                priority[cid] = (
                    0 if str(ch.get("role_type")) == "PROTAGONIST" else 1,
                    0 if is_mentioned else 1,
                    0 if is_in_tail else 1,
                    -int(ch.get("last_appeared_chapter") or 0),
                    cid,
                )
        if selected_ids:
            placeholders = ",".join("?" for _ in selected_ids)
            id_tuple = tuple(selected_ids)
            rels = self.repo.fetch_all(
                f"SELECT source_id, target_id FROM entity_relations "
                f"WHERE status = 'ACTIVE' AND (source_id IN ({placeholders}) OR target_id IN ({placeholders}));",
                id_tuple + id_tuple,
            )
            for r in rels:
                for key in ("source_id", "target_id"):
                    related_id = int(r[key])
                    selected_ids.add(related_id)
                    # 1-hop relations outrank unrelated rows but not direct mentions.
                    priority.setdefault(related_id, (2, 1, 1, 0, related_id))
        return sorted(
            (c for c in all_chars if int(c["id"]) in selected_ids),
            key=lambda c: priority.get(int(c["id"]), (3, 1, 1, 0, int(c["id"]))),
        )[:12]

    def _format_active_relations(self, active_chars: list[dict[str, Any]]) -> list[str]:
        """Format 1-hop interpersonal relationship edges among active characters."""
        if not active_chars:
            return []
        id_to_name = {int(c["id"]): str(c["name"]) for c in active_chars}
        placeholders = ",".join("?" for _ in id_to_name)
        id_tuple = tuple(id_to_name.keys())
        rels = self.repo.fetch_all(
            f"SELECT source_id, relation_type, target_id FROM entity_relations "
            f"WHERE status = 'ACTIVE' AND source_id IN ({placeholders}) AND target_id IN ({placeholders}) LIMIT 10;",
            id_tuple + id_tuple,
        )
        return [
            f"- 關係：{id_to_name[int(r['source_id'])]} —[{r['relation_type']}]→ {id_to_name[int(r['target_id'])]}"
            for r in rels
            if int(r["source_id"]) in id_to_name and int(r["target_id"]) in id_to_name
        ]

    def _select_active_locations(
        self, beats: list[dict[str, Any]], active_chars: list[dict[str, Any]]
    ) -> list[str]:
        """Resolve active scene locations from Beat location_name and character current_location_id."""
        loc_ids = {
            int(c["current_location_id"])
            for c in active_chars
            if c.get("current_location_id") is not None
        }
        beat_loc_names = {
            str(b.get("location_name", "")).strip()
            for b in beats
            if str(b.get("location_name", "")).strip()
        }
        all_locs = self.repo.fetch_all(
            "SELECT id, name, description FROM locations WHERE status = 'ACTIVE' ORDER BY id;"
        )
        matched = [
            loc
            for loc in all_locs
            if int(loc["id"]) in loc_ids or str(loc["name"]) in beat_loc_names
        ]
        if not matched and all_locs:
            matched = all_locs[:2]
        return [
            f"- {loc['name']}" + (f"：{loc['description']}" if loc.get("description") else "")
            for loc in matched[:5]
        ]

    def _build_beats_block(
        self,
        ch_row: dict[str, Any],
        beats: list[dict[str, Any]],
        pacing_directive: str,
    ) -> str:
        """Format chapter title, 2-3 Scene Beats, and sliding-window pacing adjustments."""
        lines = [f"【本章目標標題】{ch_row.get('title', '')}"]
        for b in beats:
            lines.append(
                f"- Beat {b.get('beat_index', 1)}【{b.get('scene_tag', '')}】"
                f"（節奏：{b.get('pacing_instruction', '')}）：{b.get('plot_objective', '')}"
            )
        if pacing_directive.strip():
            lines.append(f"【節奏滑動窗口調控指令】{pacing_directive.strip()}")
        lines.append(
            "【單章情節與情緒密度護欄（防急躁速成鐵律）】\n"
            "- 禁止單章速成：單章絕對禁止解決大目標或打完一場全卷決戰！\n"
            "- 克制推進密度：單章僅限專注演繹局勢演進的單一微觀切片，維持局部衝突的張力延續與深層沉澱，嚴禁在單一章節內強行跨越多個戲劇轉折週期！\n"
            "- 維持局勢懸空：章末應維持衝突懸而未決的張力餘波，保留未解因果與後續危機，嚴禁在單章之內倉促終結全篇核心矛盾。\n"
            "- 重過程輕結果：將筆墨傾注於交鋒過程中的微表情、動作博弈、心理拉扯與五感環境氛圍，嚴禁一筆帶過流水帳。"
        )
        lines.append(
            f"【寫作與字數護欄】請在內心消化上述情節發展與衝突起伏，以純散文直接展開第一段正文（切勿在正文中直接列印 Beat 標題或分鏡標籤），"
            f"全篇使用繁體中文，自然演繹完情節後收尾，字數控制於 "
            f"{self.budget.chapter_soft_min_chars}～{self.budget.chapter_soft_max_chars} 字（硬上限 {self.budget.chapter_hard_max_chars} 字）。"
        )
        return truncate_to_token_budget("\n".join(lines), self.budget.scene_beats_tokens)

    @staticmethod
    def _compose_user_prompt(
        settings_block: str, ref_block: str, beats_block: str, tail_block: str
    ) -> str:
        """Combine routed blocks into the final user prompt for chapter draft generation."""
        parts = [settings_block]
        if ref_block.strip():
            parts.append(f"【參考書庫技法摘要】\n{ref_block}")
        if tail_block.strip():
            parts.append(f"【上一章末尾原文切片（請無縫銜接場景與語氣）】\n{tail_block}")
        parts.append(beats_block)
        parts.append("【輸出指令】：請直接輸出本章小說正文散文，嚴禁輸出任何 Markdown 標題或 Beat 分鏡標籤，嚴禁輸出章末總結評論，全文繁體中文：")
        return "\n\n".join(parts)
