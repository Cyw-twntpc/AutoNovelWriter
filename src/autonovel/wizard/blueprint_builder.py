"""AI blueprint completion (Step 3) and Review Gate workbench manager (Step 4)."""

from __future__ import annotations

import json
from typing import Any

from autonovel.db.connection import atomic_transaction
from autonovel.db.project_repo import ProjectRepository
from autonovel.llm.llm_client import LLMClient
from autonovel.reference.retriever import ReferenceRetriever
from autonovel.schemas.domain import (
    BlacklistConfig,
    PipelineState,
)
from autonovel.schemas.llm_contracts import (
    BlueprintCompletionContract,
    InitialCharacterContract,
)
from autonovel.wizard.qa_engine import AIModelUnavailableError


class BlueprintLockedError(RuntimeError):
    """Raised when a destructive blueprint rebuild is attempted after chapters exist."""


class BlueprintBuilder:
    """Completes world rules, characters, 5M-word stage gates, and style contract."""

    def __init__(
        self,
        repo: ProjectRepository,
        retriever: ReferenceRetriever,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Initialize builder with project repository, reference retriever, and LLM client."""
        self.repo = repo
        self.retriever = retriever
        self.llm_client = llm_client

    async def complete_blueprint(self) -> dict[str, Any]:
        """Generate remaining settings from Q&A history and transition to REVIEW_GATE."""
        cfg = self.repo.get_project_config()
        self._assert_blueprint_mutable(cfg)
        genre_primary = cfg.get("genre_primary", "")
        genre_sub = cfg.get("genre_sub", "")
        min_words = int(cfg.get("min_word_count") or 5_000_000)
        qa_history = json.loads(cfg.get("qa_history_json") or "[]")
        ref_digest, _ = await self.retriever.retrieve_and_compress(
            f"{genre_primary} {genre_sub} 開篇世界觀、文風修辭與大綱設計",
            fallback_tags=[t for t in (genre_primary, genre_sub, "世界觀", "文風修辭") if t],
        )
        blueprint = await self._generate_blueprint_contract(
            genre_primary, genre_sub, min_words, qa_history, ref_digest
        )
        self._persist_blueprint_to_db(blueprint, genre_primary, genre_sub)
        return self.get_review_gate_payload()

    async def _generate_blueprint_contract(
        self,
        genre_primary: str,
        genre_sub: str,
        min_words: int,
        qa_history: list[dict[str, Any]],
        ref_digest: str,
    ) -> BlueprintCompletionContract:
        """Call LLM with GBNF schema to create full blueprint and initial Volume 1 chapters."""
        if self.llm_client:
            if not self.llm_client.server_manager.is_healthy():
                await self.llm_client.server_manager.ensure_running_async()
            if self.llm_client.server_manager.is_healthy():
                matched_tpl = None
                if hasattr(self, "retriever") and hasattr(self.retriever, "repo"):
                    matched_tpl = self.retriever.repo.match_pacing_template(
                        genre_primary, genre_sub
                    )
                pacing_guidance = ""
                if matched_tpl and matched_tpl.get("stages"):
                    pacing_guidance = (
                        f"\n參考節奏波浪量化曲線：{json.dumps(matched_tpl['stages'], ensure_ascii=False)}"
                    )
                sys_prompt = (
                    "【語言輸出鐵律】你必須全程使用『繁體中文（台灣標準書面語）』輸出！絕對嚴禁輸出任何簡體字！\n"
                    f"你是百萬字長篇小說首席世界觀架構師與文學導師。請為『{genre_primary}·{genre_sub}』史詩級長篇小說（目標最低 {min_words} 字）"
                    "深度解析問答歷史，補全完整世界觀、力量階梯、初始角色網、勢力地圖、階段大綱、首卷細綱、六維風格契約書與正文黃金錨點段落。\n\n"
                    "【必須嚴格遵守的六大核心鐵律】：\n"
                    "1. 【角色命名絕對原創】：嚴禁抄襲、套用或借用任何既有文學、影視、動漫、歷史或神話傳說之既有人物姓名與尊號！所有主角、配角與勢力領袖之姓名、稱號、別稱必須 100% 全新原創，依據題材底蘊賦予其獨特音韻美感與性格寓意。\n"
                    "2. 【勢力與地點具體補全】：factions 與 locations 必須各自補全至少 3～5 個具備鮮明矛盾衝突的派系勢力與標誌性地理區域，且每位角色的 faction_name 與 location_name 必須明確歸屬，嚴禁為空字串！\n"
                    "3. 【深層推演拒絕照抄】：切忌字面照搬問答紀錄！必須以使用者的問答答案為因果種子，向上推演深層法則與世界矛盾，向下鋪陳階級壓迫與資源爭奪焦點。\n"
                    "4. 【風格契約書六大維度】：style_contract_text 必須完整規範：①敘事視角之純度與限制邊界；②句長架構與語言流速起伏機制；③物性描寫密度與感官細節標準；④對話容量比例與潛台詞博弈深度；⑤長篇波浪推進節奏與張力釋放曲線；⑥違禁文風紅線與旁白介入禁令。\n"
                    "5. 【靜態黃金錨點段落】：static_anchor_text 必須是一整段 180～250 字具備最高文學張力與感官畫面沉浸感的開篇正文散文，確立全書語言質地與敘事色調，絕對嚴禁填入章節標題、摘要或大綱！\n"
                    "6. 【階梯序列、階段大綱與首卷細綱】：hierarchy_levels 至少 6 階；stage_gates 階段大綱必須對應全書最低目標字數；initial_volume_chapters 必須完整規劃第一卷的 10 章預排細綱（每章包含原創章節標題與 2～3 個具體的 Scene Beats）。"
                )
                user_prompt = (
                    f"創世問答紀錄：{json.dumps(qa_history, ensure_ascii=False)}\n"
                    f"參考書庫風格養分：{ref_digest}{pacing_guidance}"
                )
                return await self.llm_client.generate_structured(
                    sys_prompt, user_prompt, BlueprintCompletionContract, max_tokens=4000
                )
        raise AIModelUnavailableError(
            "本地 AI 模型未就緒或推論服務離線，無法進行 AI 藍圖自動補全。請先至環境設置完成模型下載與推論服務設定。"
        )

    def _persist_blueprint_to_db(
        self, bp: BlueprintCompletionContract, genre_primary: str, genre_sub: str = ""
    ) -> None:
        """Write completed blueprint into novel_project.db tables and Volume 1 outline."""
        blacklist = BlacklistConfig(
            tier1_global_cliches=[],
            tier2_genre_anachronisms=list(bp.genre_blacklist_terms),
            tier3_user_custom=[],
        )
        global_outline = {
            "opening_plan": bp.opening_plan,
            "ending_draft": bp.ending_draft,
            "stage_gates": [g.model_dump() for g in bp.stage_gates],
        }
        with atomic_transaction(self.repo.db_path) as conn:
            # A full rebuild replaces the whole narrative graph; child tables must be
            # cleared first or the new rows will collide with stale history/timeline.
            self._reset_narrative_state(conn)
            conn.execute("DELETE FROM world_rules;")
            for rule in bp.world_rules:
                conn.execute("INSERT OR IGNORE INTO world_rules (rule_text) VALUES (?);", (rule,))
            conn.execute("DELETE FROM hierarchy_levels;")
            for hl in bp.hierarchy_levels:
                conn.execute(
                    "INSERT INTO hierarchy_levels (category, level_num, level_name, description) VALUES (?, ?, ?, ?);",
                    (hl.category, hl.level_num, hl.level_name, hl.description),
                )
            for fac in bp.factions:
                conn.execute("INSERT OR IGNORE INTO factions (name) VALUES (?);", (fac,))
            for loc in bp.locations:
                conn.execute("INSERT OR IGNORE INTO locations (name) VALUES (?);", (loc,))
            self._seed_initial_characters(conn, bp.characters)
            self._seed_first_volume(conn, bp, genre_primary, genre_sub)
            # The wizard lets the user name the book first; the blueprint's own title
            # must not silently overwrite a user-chosen name.
            title_row = conn.execute(
                "SELECT title FROM project_config WHERE id = 1;"
            ).fetchone()
            existing_title = str(title_row["title"] or "").strip() if title_row else ""
            final_title = (
                existing_title
                if existing_title and existing_title != "未命名長篇小說"
                else bp.novel_title
            )
            conn.execute(
                """
                UPDATE project_config
                SET title = ?, global_outline_json = ?, style_contract_text = ?,
                    static_anchor_text = ?, blacklist_json = ?, pipeline_state = ?
                WHERE id = 1;
                """,
                (
                    final_title,
                    json.dumps(global_outline, ensure_ascii=False),
                    bp.style_contract_text,
                    bp.static_anchor_text,
                    blacklist.model_dump_json(),
                    PipelineState.REVIEW_GATE.value,
                ),
            )

    @staticmethod
    def _reset_character_graph(conn: Any) -> None:
        """Delete history/relations before characters to satisfy foreign keys."""
        conn.execute("DELETE FROM character_history;")
        conn.execute("DELETE FROM entity_relations;")
        conn.execute("DELETE FROM characters;")

    @staticmethod
    def _reset_narrative_state(conn: Any) -> None:
        """Clear every chapter-derived table in foreign-key safe order.

        Chapter metrics, timeline and foreshadowing carry no ON DELETE CASCADE, so
        deleting chapters alone leaves orphan rows that poison later statistics.
        """
        conn.execute("DELETE FROM chapter_metrics;")
        conn.execute("DELETE FROM timeline;")
        conn.execute("DELETE FROM foreshadowing;")
        conn.execute("DELETE FROM chapters;")
        conn.execute("DELETE FROM volumes;")
        conn.execute("DELETE FROM character_history;")
        conn.execute("DELETE FROM entity_relations;")
        conn.execute("DELETE FROM characters;")
        conn.execute("DELETE FROM locations;")
        conn.execute("DELETE FROM factions;")

    def _seed_initial_characters(
        self, conn: Any, characters: list[InitialCharacterContract]
    ) -> None:
        """Insert initial characters into characters table (caller clears old rows)."""
        for ch in characters:
            fac_id = self.repo._ensure_named_entity(conn, "factions", ch.faction_name)
            loc_id = self.repo._ensure_named_entity(conn, "locations", ch.location_name)
            conn.execute(
                """
                INSERT INTO characters (
                    name, aliases, role_type, status, rank_level,
                    faction_id, current_location_id, description
                ) VALUES (?, ?, ?, 'ALIVE', ?, ?, ?, ?);
                """,
                (
                    ch.name,
                    json.dumps(ch.aliases, ensure_ascii=False),
                    ch.role_type,
                    ch.rank_level,
                    fac_id,
                    loc_id,
                    ch.description,
                ),
            )

    def _seed_first_volume(
        self,
        conn: Any,
        bp: BlueprintCompletionContract,
        genre_primary: str = "",
        genre_sub: str = "",
    ) -> None:
        """Initialize Volume 1 and its AI-planned chapters without hardcoded templates."""
        if not bp.initial_volume_chapters:
            raise ValueError(
                "BlueprintCompletionContract.initial_volume_chapters is empty; Volume 1 chapters must be generated by AI."
            )
        conn.execute("DELETE FROM chapters;")
        conn.execute("DELETE FROM volumes;")
        first_gate = bp.stage_gates[0]
        target_chapters = len(bp.initial_volume_chapters)
        conn.execute(
            """
            INSERT INTO volumes (
                volume_num, title, outline_text, target_chapters,
                target_word_quota, max_allowed_rank_level, status
            ) VALUES (1, ?, ?, ?, ?, ?, 'PLANNED');
            """,
            (
                f"第一卷：{first_gate.stage_title}",
                bp.opening_plan,
                target_chapters,
                first_gate.target_word_quota,
                first_gate.max_allowed_rank_level,
            ),
        )
        protagonist = bp.characters[0].name if bp.characters else ""
        conn.execute("DELETE FROM foreshadowing;")
        conn.execute(
            """
            INSERT INTO foreshadowing (
                description, scope, planted_chapter, expected_span, status, related_character_ids
            ) VALUES (?, 'VOLUME', 1, ?, 'PENDING', ?);
            """,
            (
                f"首卷大綱核心謎團：{first_gate.core_mystery_quota}",
                target_chapters,
                json.dumps([protagonist] if protagonist else [], ensure_ascii=False),
            ),
        )
        for idx, ch_contract in enumerate(bp.initial_volume_chapters, start=1):
            raw_title = ch_contract.suggested_title.strip()
            formatted_title = (
                raw_title if raw_title.startswith("第") else f"第{idx}章 {raw_title}"
            )
            beats = [b.model_dump() for b in ch_contract.beats]
            conn.execute(
                """
                INSERT INTO chapters (chapter_num, volume_num, title, beats_json, status)
                VALUES (?, 1, ?, ?, 'UNWRITTEN');
                """,
                (idx, formatted_title, json.dumps(beats, ensure_ascii=False)),
            )

    def get_review_gate_payload(self) -> dict[str, Any]:
        """Return all 5 structured tabs for the Step 4 Review Gate Workbench."""
        cfg = self.repo.get_project_config()
        return {
            "project_config": cfg,
            "world_rules": self.repo.fetch_all("SELECT * FROM world_rules ORDER BY id;"),
            "hierarchy_levels": self.repo.fetch_all(
                "SELECT * FROM hierarchy_levels ORDER BY level_num;"
            ),
            "characters": self.repo.fetch_all(
                """
                SELECT c.*, COALESCE(f.name, '') AS faction_name, COALESCE(l.name, '') AS location_name
                FROM characters c
                LEFT JOIN factions f ON f.id = c.faction_id
                LEFT JOIN locations l ON l.id = c.current_location_id
                ORDER BY c.id;
                """
            ),
            "factions": self.repo.fetch_all(
                "SELECT * FROM factions WHERE status = 'ACTIVE' ORDER BY id;"
            ),
            "locations": self.repo.fetch_all(
                "SELECT * FROM locations WHERE status = 'ACTIVE' ORDER BY id;"
            ),
            "volumes": self.repo.fetch_all("SELECT * FROM volumes ORDER BY volume_num;"),
            "chapters": self.repo.fetch_all(
                "SELECT chapter_num, volume_num, title, beats_json, status FROM chapters ORDER BY chapter_num;"
            ),
        }

    def update_review_gate_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply user modifications from all 5 tabs of the Review Gate Workbench into SQLite."""
        self._assert_blueprint_mutable()
        with atomic_transaction(self.repo.db_path) as conn:
            self._update_review_config_fields(conn, payload)
            self._update_review_world_and_hierarchy(conn, payload)
            self._update_review_entities(conn, payload)
            self._update_review_outline_and_chapters(conn, payload)
        return self.get_review_gate_payload()

    @staticmethod
    def _update_review_config_fields(conn: Any, payload: dict[str, Any]) -> None:
        """Update title, style contract, static anchor, blacklist, and global outline."""
        if "title" in payload:
            conn.execute(
                "UPDATE project_config SET title = ? WHERE id = 1;", (payload["title"],)
            )
        if "style_contract_text" in payload:
            conn.execute(
                "UPDATE project_config SET style_contract_text = ? WHERE id = 1;",
                (payload["style_contract_text"],),
            )
        if "static_anchor_text" in payload:
            conn.execute(
                "UPDATE project_config SET static_anchor_text = ? WHERE id = 1;",
                (payload["static_anchor_text"],),
            )
        if "blacklist" in payload:
            conn.execute(
                "UPDATE project_config SET blacklist_json = ? WHERE id = 1;",
                (json.dumps(payload["blacklist"], ensure_ascii=False),),
            )
        if "global_outline" in payload and isinstance(payload["global_outline"], dict):
            conn.execute(
                "UPDATE project_config SET global_outline_json = ? WHERE id = 1;",
                (json.dumps(payload["global_outline"], ensure_ascii=False),),
            )

    @staticmethod
    def _update_review_world_and_hierarchy(conn: Any, payload: dict[str, Any]) -> None:
        """Update world_rules and hierarchy_levels tables from user edits."""
        if "world_rules" in payload:
            conn.execute("DELETE FROM world_rules;")
            for r in payload["world_rules"]:
                text = r["rule_text"] if isinstance(r, dict) else str(r)
                if text.strip():
                    conn.execute(
                        "INSERT OR IGNORE INTO world_rules (rule_text) VALUES (?);",
                        (text.strip(),),
                    )
        if "hierarchy_levels" in payload:
            conn.execute("DELETE FROM hierarchy_levels;")
            for idx, item in enumerate(payload["hierarchy_levels"], start=1):
                if isinstance(item, dict):
                    lvl = int(item.get("level_num") or idx)
                    name = str(item.get("level_name") or f"第{lvl}階").strip()
                    desc = str(item.get("description") or "").strip()
                    cat = str(item.get("category") or "CULTIVATION").strip()
                    conn.execute(
                        "INSERT INTO hierarchy_levels (category, level_num, level_name, description) VALUES (?, ?, ?, ?);",
                        (cat, lvl, name, desc),
                    )

    def _update_review_entities(self, conn: Any, payload: dict[str, Any]) -> None:
        """Update factions, locations, and initial characters from user edits."""
        if "factions" in payload:
            names = [
                (f.get("name", "") if isinstance(f, dict) else str(f)).strip()
                for f in payload["factions"]
            ]
            names = [n for n in names if n]
            for name in names:
                conn.execute("INSERT OR IGNORE INTO factions (name) VALUES (?);", (name,))
                conn.execute(
                    "UPDATE factions SET status = 'ACTIVE' WHERE name = ?;", (name,)
                )
            # Soft-delete instead of DELETE: characters.current_location_id /
            # faction_id reference these rows, and a hard delete breaks the FK.
            self._archive_removed_entities(conn, "factions", names)
        if "locations" in payload:
            names = [
                (loc.get("name", "") if isinstance(loc, dict) else str(loc)).strip()
                for loc in payload["locations"]
            ]
            names = [n for n in names if n]
            for name in names:
                conn.execute("INSERT OR IGNORE INTO locations (name) VALUES (?);", (name,))
                conn.execute(
                    "UPDATE locations SET status = 'ACTIVE' WHERE name = ?;", (name,)
                )
            self._archive_removed_entities(conn, "locations", names)
        if "characters" in payload:
            parsed: list[InitialCharacterContract] = []
            for ch in payload["characters"]:
                if isinstance(ch, dict) and str(ch.get("name", "")).strip():
                    aliases = ch.get("aliases", [])
                    if isinstance(aliases, str):
                        try:
                            aliases = json.loads(aliases)
                        except ValueError:
                            aliases = [a.strip() for a in aliases.split(",") if a.strip()]
                    parsed.append(
                        InitialCharacterContract(
                            name=str(ch["name"]).strip(),
                            aliases=aliases if isinstance(aliases, list) else [],
                            role_type=str(ch.get("role_type") or "MAJOR"),
                            rank_level=int(ch.get("rank_level") or 1),
                            faction_name=str(ch.get("faction_name") or ""),
                            location_name=str(ch.get("location_name") or ""),
                            description=str(ch.get("description") or ""),
                        )
                    )
            if parsed:
                self._reset_character_graph(conn)
                self._seed_initial_characters(conn, parsed)

    @staticmethod
    def _archive_removed_entities(conn: Any, table: str, keep_names: list[str]) -> None:
        """Mark rows missing from the edited list as ARCHIVED (soft delete)."""
        if table not in ("factions", "locations"):
            raise ValueError(f"Unsupported entity table: {table}")
        if not keep_names:
            conn.execute(
                f"UPDATE {table} SET status = 'ARCHIVED' WHERE status = 'ACTIVE';"
            )
            return
        placeholders = ", ".join("?" for _ in keep_names)
        conn.execute(
            f"UPDATE {table} SET status = 'ARCHIVED' "
            f"WHERE status = 'ACTIVE' AND name NOT IN ({placeholders});",
            tuple(keep_names),
        )

    @staticmethod
    def _update_review_outline_and_chapters(conn: Any, payload: dict[str, Any]) -> None:
        """Update pre-planned Volume 1 chapter titles and Scene Beats."""
        if "chapters" not in payload:
            return
        for ch in payload["chapters"]:
            if not isinstance(ch, dict) or "chapter_num" not in ch:
                continue
            ch_num = int(ch["chapter_num"])
            title = str(ch.get("title") or f"第{ch_num}章").strip()
            beats_raw = ch.get("beats_json", "[]")
            beats_str = (
                beats_raw
                if isinstance(beats_raw, str)
                else json.dumps(beats_raw, ensure_ascii=False)
            )
            conn.execute(
                "UPDATE chapters SET title = ?, beats_json = ? WHERE chapter_num = ?;",
                (title, beats_str, ch_num),
            )

    def _assert_blueprint_mutable(self, cfg: dict[str, Any] | None = None) -> None:
        """Refuse destructive blueprint rebuilds once serialization has started.

        Deleting chapters/volumes/characters after prose exists is irreversible, so
        only WIZARD and REVIEW_GATE are allowed to rewrite the blueprint.
        """
        state = (cfg or self.repo.get_project_config()).get("pipeline_state")
        mutable = {PipelineState.WIZARD.value, PipelineState.REVIEW_GATE.value}
        if state not in mutable:
            raise BlueprintLockedError(
                f"目前專案狀態為 {state}，已有正文或連載紀錄，禁止重建藍圖"
                "（會刪除全部章節）。請先匯出備份，或另開新專案。"
            )

    async def regenerate_review_section(self, section: str) -> dict[str, Any]:
        """Regenerate a single Review Gate tab section without overwriting other tabs."""
        cfg = self.repo.get_project_config()
        self._assert_blueprint_mutable(cfg)
        genre_primary = cfg.get("genre_primary", "")
        genre_sub = cfg.get("genre_sub", "")
        min_words = int(cfg.get("min_word_count") or 5_000_000)
        qa_history = json.loads(cfg.get("qa_history_json") or "[]")
        ref_digest, _ = await self.retriever.retrieve_and_compress(
            f"{genre_primary} {genre_sub} 局部重抽：{section}"
        )
        bp = await self._generate_blueprint_contract(
            genre_primary, genre_sub, min_words, qa_history, ref_digest
        )
        self._apply_regenerated_section(section, bp, genre_primary, genre_sub)
        return self.get_review_gate_payload()

    def _apply_regenerated_section(
        self,
        section: str,
        bp: BlueprintCompletionContract,
        genre_primary: str,
        genre_sub: str = "",
    ) -> None:
        """Persist only the requested regenerated section into SQLite."""
        with atomic_transaction(self.repo.db_path) as conn:
            if section == "world":
                conn.execute("DELETE FROM world_rules;")
                for rule in bp.world_rules:
                    conn.execute("INSERT OR IGNORE INTO world_rules (rule_text) VALUES (?);", (rule,))
                conn.execute("DELETE FROM hierarchy_levels;")
                for hl in bp.hierarchy_levels:
                    conn.execute(
                        "INSERT INTO hierarchy_levels (category, level_num, level_name, description) VALUES (?, ?, ?, ?);",
                        (hl.category, hl.level_num, hl.level_name, hl.description),
                    )
            elif section == "characters":
                self._reset_character_graph(conn)
                self._seed_initial_characters(conn, bp.characters)
            elif section == "outline":
                global_outline = {
                    "opening_plan": bp.opening_plan,
                    "ending_draft": bp.ending_draft,
                    "stage_gates": [g.model_dump() for g in bp.stage_gates],
                }
                conn.execute(
                    "UPDATE project_config SET global_outline_json = ? WHERE id = 1;",
                    (json.dumps(global_outline, ensure_ascii=False),),
                )
                self._seed_first_volume(conn, bp, genre_primary, genre_sub)
            elif section == "style":
                conn.execute(
                    "UPDATE project_config SET style_contract_text = ?, static_anchor_text = ? WHERE id = 1;",
                    (bp.style_contract_text, bp.static_anchor_text),
                )
            elif section == "blacklist":
                bl = BlacklistConfig(
                    tier1_global_cliches=[],
                    tier2_genre_anachronisms=list(bp.genre_blacklist_terms),
                    tier3_user_custom=[],
                )
                conn.execute(
                    "UPDATE project_config SET blacklist_json = ? WHERE id = 1;",
                    (bl.model_dump_json(),),
                )
