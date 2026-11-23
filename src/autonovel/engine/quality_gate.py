"""Two-Gate Quality Control: Gate A (Hard Facts + Micro-Rewrite) and Gate B (100-pt LLM Rubric)."""

from __future__ import annotations

from typing import Any

from autonovel.config import QualityThresholdConfig
from autonovel.db.project_repo import ProjectRepository
from autonovel.engine.blacklist_scanner import BlacklistScanner
from autonovel.engine.entity_resolver import EntityResolver
from autonovel.engine.prose_sanitizer import (
    convert_simplified_to_traditional,
    sanitize_novel_prose,
)
from autonovel.llm.llm_client import LLMClient
from autonovel.schemas.llm_contracts import (
    ChapterExtractionContract,
    GateASemanticCheckContract,
    GateBRubricContract,
    RubricItemResult,
)
from autonovel.wizard.qa_engine import AIModelUnavailableError


def compute_weighted_rubric_score(
    rubric_items: list[RubricItemResult],
) -> dict[str, Any]:
    """Calculate 100-pt literary quality score directly from 10-item LLM Rubric without mathematical predictions.

    Weights: RHETORIC 40%, DIALOGUE 35%, SCENE 25%.
    Item point values: PASS = 10.0, PARTIAL = 5.0, FAIL = 0.0.
    """
    pts_map = {"PASS": 10.0, "PARTIAL": 5.0, "FAIL": 0.0}
    dim_earned: dict[str, float] = {"RHETORIC": 0.0, "DIALOGUE": 0.0, "SCENE": 0.0}
    dim_count: dict[str, int] = {"RHETORIC": 0, "DIALOGUE": 0, "SCENE": 0}
    for item in rubric_items:
        dim = item.dimension
        dim_earned[dim] = dim_earned.get(dim, 0.0) + pts_map.get(item.verdict, 5.0)
        dim_count[dim] = dim_count.get(dim, 0) + 1

    rhetoric_pct = (dim_earned["RHETORIC"] / max(1.0, dim_count["RHETORIC"] * 10.0)) * 100.0 if dim_count["RHETORIC"] else 0.0
    dialogue_pct = (dim_earned["DIALOGUE"] / max(1.0, dim_count["DIALOGUE"] * 10.0)) * 100.0 if dim_count["DIALOGUE"] else 0.0
    scene_pct = (dim_earned["SCENE"] / max(1.0, dim_count["SCENE"] * 10.0)) * 100.0 if dim_count["SCENE"] else 0.0

    composite = round(
        0.40 * rhetoric_pct + 0.35 * dialogue_pct + 0.25 * scene_pct,
        2,
    )
    return {
        "composite_score": composite,
        "python_score_30": 0.0,
        "llm_score_70": composite,
        "dimension_scores": {
            "RHETORIC": round(rhetoric_pct, 2),
            "DIALOGUE": round(dialogue_pct, 2),
            "SCENE": round(scene_pct, 2),
        },
    }


class QualityGateEngine:
    """Executes Gate A (blacklist + hard/core conflict check) and Gate B (Rubric scoring)."""

    def __init__(
        self,
        repo: ProjectRepository,
        scanner: BlacklistScanner,
        resolver: EntityResolver,
        thresholds: QualityThresholdConfig | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Initialize quality gate engine with repo, scanner, resolver, and thresholds."""
        self.repo = repo
        self.scanner = scanner
        self.resolver = resolver
        self.thresholds = thresholds or QualityThresholdConfig()
        self.llm_client = llm_client

    async def run_gate_a(
        self,
        volume_num: int,
        draft_text: str,
        extraction: ChapterExtractionContract,
    ) -> dict[str, Any]:
        """Execute Gate A: sanitize blacklisted terms and check SQLite hard/semantic rules."""
        sanitized_text, micro_count, hit_terms = await self.scanner.sanitize_text(draft_text)
        sanitized_text = sanitize_novel_prose(sanitized_text)
        normalized_ext = self.resolver.normalize_extraction(extraction)
        hard_errors = self.repo.validate_extraction_hard_facts(volume_num, normalized_ext)
        if hit_terms:
            hard_errors.append(f"Residual blacklisted terms detected: {', '.join(hit_terms)}")
        semantic = await self._check_semantic_consistency(sanitized_text)
        if semantic.has_core_contradiction:
            hard_errors.append(f"Core semantic conflict: {semantic.core_contradiction_detail}")
        return {
            "passed": len(hard_errors) == 0,
            "sanitized_text": sanitized_text,
            "micro_rewrite_count": micro_count,
            "hit_blacklist_terms": hit_terms,
            "hard_errors": hard_errors,
            "has_minor_drift": semantic.has_minor_drift,
            "forward_retcon_instruction": semantic.forward_retcon_instruction,
            "extraction": normalized_ext,
        }

    async def _check_semantic_consistency(
        self, text: str
    ) -> GateASemanticCheckContract:
        """Run LLM semantic consistency and format purity check against world rules on full chapter text."""
        if self.llm_client and self.llm_client.server_manager.is_healthy():
            rules = self.repo.fetch_all("SELECT rule_text FROM world_rules;")
            rules_str = "\n".join(r["rule_text"] for r in rules)
            sys_prompt = (
                "你是一位極致嚴謹的小說世界觀邏輯與正文格式純度裁判。請比對以下世界觀核心法則與章節完整正文，審查是否存在設定矛盾或格式污染：\n"
                "【審查原則與鐵律】：\n"
                "1. 全程使用繁體中文（台灣標準書面語）輸出。\n"
                "2. 【核心矛盾與格式違規 (core contradiction)】：指無法調和的硬傷，包含：(a) 因果秩序逆亂、戰力境界逾越上限、時空定位斷裂或違背世界觀底層硬性鐵律；(b) 正文中夾雜 Markdown 標題符號、大綱分鏡節拍標籤（Beat 標記）或章末元評論（如本章完、下章預告、作者總結等）。若發現任一情況，has_core_contradiction 設為 true，並在 core_contradiction_detail 具體指明問題點。\n"
                "3. 【次要漂移與前向圓場 (forward retcon)】：指微小的數值偏差或細節模糊，可透過後續劇情自圓其說。若發現，has_minor_drift 設為 true，並在 forward_retcon_instruction 給出具體、可直接執行的下一章前向圓場指令（繁體中文），切勿空泛。"
            )
            res = await self.llm_client.generate_structured(
                sys_prompt, f"法則：\n{rules_str}\n\n正文：\n{text}", GateASemanticCheckContract
            )
            if res.core_contradiction_detail:
                res.core_contradiction_detail = convert_simplified_to_traditional(res.core_contradiction_detail.strip())
            if res.forward_retcon_instruction:
                res.forward_retcon_instruction = convert_simplified_to_traditional(res.forward_retcon_instruction.strip())
            return res
        raise AIModelUnavailableError("本地 AI 模型未就緒或推論服務離線，無法進行世界觀語義一致性審查。")

    async def run_gate_b(self, text: str) -> dict[str, Any]:
        """Execute Gate B: 100-pt 10-item LLM Rubric evaluation without mathematical predictions."""
        clean_text = sanitize_novel_prose(text)
        rubric_contract = await self._evaluate_rubric(clean_text)
        scored = compute_weighted_rubric_score(rubric_contract.rubric_items)
        dims = scored["dimension_scores"]
        min_dim = min(dims.values()) if dims else 0.0
        passed = (
            scored["composite_score"] >= self.thresholds.single_chapter_pass_score
            and min_dim >= self.thresholds.single_dimension_min_score
        )
        return {
            "passed": passed,
            "composite_score": scored["composite_score"],
            "dimension_scores": dims,
            "python_details": {},
            "rubric_items": [i.model_dump() for i in rubric_contract.rubric_items],
            "revision_guidance": rubric_contract.revision_guidance,
            "best_paragraph_text": rubric_contract.best_paragraph_text,
            "best_paragraph_score": rubric_contract.best_paragraph_score,
        }

    async def _evaluate_rubric(self, text: str) -> GateBRubricContract:
        """Evaluate 10 concrete rubric items and extract the best 120-char paragraph."""
        if self.llm_client and self.llm_client.server_manager.is_healthy():
            sys_prompt = (
                "你是一位極致嚴苛、審稿標準極高的長篇網絡小說金牌主編與資深文審。你的職責是「精準挑刺、剔除注水、拒絕平庸」，"
                "逼出作者的最高文學潛力。你絕不充當討好作者的好好先生，嚴格杜絕通脹高分！\n\n"
                "【10 項具體文學 Rubric 檢核標準（VERDICT 僅能為 PASS、PARTIAL、FAIL）】：\n"
                "=== 【修辭與文風 (RHETORIC - 40% 權重)】===\n"
                "- R1 [具象細節 vs 抽象空詞]：必須具備光影、冷暖、氣味或微動作等具象五感描寫。若充斥缺乏物性支撐的抽象概括大詞 -> FAIL；具備基本感官捕捉但立體感不足 -> PARTIAL；物性、光影與肌理細節飽滿生動 -> PASS。\n"
                "- R2 [語言節奏與句長起伏]：長短句交錯起伏，日常沉穩舒展，衝突短句疾進。若全篇句式均勻死板、節奏拖沓 -> FAIL；句式略有變化但張力不足 -> PARTIAL；長短錯落節奏凌厲 -> PASS。\n"
                "- R3 [修辭原創與去套路化]：嚴禁高頻流通的套路化比喻與定型化慣用語。若出現高頻陳腐慣用修辭 -> FAIL；比喻平實但缺乏獨創意象 -> PARTIAL；具備深刻冷峻的原創文學通感與意象構建 -> PASS。\n"
                "- R4 [視角純度 (POV)]：嚴格第三人稱限制視角，嚴禁作者突兀跳出全知說教或出現現代流行語出戲評論。若視角跳脫或作者主觀說教 -> FAIL；微有旁白主觀介入 -> PARTIAL；視角沉浸純粹 -> PASS。\n\n"
                "=== 【人物對話 (DIALOGUE - 35% 權重)】===\n"
                "- D1 [話中藏鋒與潛台詞]：對話必須具備心理試探、利益博弈或言外之意，嚴禁直球宣讀設定或問答機器人。若台詞直白宣讀劇情 -> FAIL；平實但推進情節 -> PARTIAL；句句藏鋒言不由衷 -> PASS。\n"
                "- D2 [人物語氣辨識度]：遮住說話者名字，讀者仍能依據用語口吻辨識角色性格。若千人一面、語氣雷同 -> FAIL；略有身份區別但個性平平 -> PARTIAL；口吻鮮明如見其人 -> PASS。\n"
                "- D3 [神態伴隨與微表情]：對話必須伴隨呼吸、視線轉移、肢體微動作或物件互動，嚴禁連續純台詞對撞。若連續對話缺乏任何伴隨動作與神態烘托 -> FAIL；動作描寫單一機械 -> PARTIAL；言行互證，微表情與潛台詞高度呼應 -> PASS。\n\n"
                "=== 【場景與張力 (SCENE - 25% 權重)】===\n"
                "- S1 [動態衝突與情境壓迫]：場景必須具備清晰的情勢變化或威脅逼近，不可流水帳。若平鋪直敘無危機感，或違反長篇小說推進規律、在單章內急躁速成解決宏大目標 -> FAIL；衝突化解過於倉促或推進速度失控 -> PARTIAL；局部危機步步緊逼、代價真實且克制單章推進速度 -> PASS。\n"
                "- S2 [環境烘托與空間立體]：場景不可為抽象白房間，環境光影必須烘托角色心境與局勢。若無環境或環境脫節 -> FAIL；僅有靜態背景交待 -> PARTIAL；環境動態烘托極佳 -> PASS。\n"
                "- S3 [章末扣子與懸念推動]：章末必須具備未解懸念、局勢反轉或關鍵行動啟程，激發翻頁慾望。若平淡收場無期待感，或急於將所有危機圓滿化解、毫無懸空餘味 -> FAIL；收束平穩但鉤子不強 -> PARTIAL；懸念尖銳扣人心弦、危機懸而未決 -> PASS。\n\n"
                "【極嚴評分鐵律】：\n"
                "1. 【全篇繁體中文】：所有評語（evidence_quote, revision_guidance）必須使用繁體中文（台灣標準書面語）。\n"
                "2. 【打分分佈防通脹鐵律】：絕大多數初稿章節的合理通過得分應在 75～85 分之間！正常未潤色草稿通常應有 3～5 項為 PARTIAL 或 FAIL。唯有經精雕細琢、毫無瑕疵的段落方能評出 90 分以上。絕對禁止不負責任地全給 PASS！\n"
                "3. 【證據引用 (evidence_quote)】：判定 PASS 者必須在 evidence_quote 引述 10～30 字無可挑剔的原創精湛文字證據；判定 FAIL 或 PARTIAL 者必須引述有瑕疵的原文句子。\n"
                "4. 【動態黃金錨點段落 (best_paragraph_text)】：必須直接從正文中摘錄 80～120 字最具文學底蘊與畫面感的純散文段落（嚴禁標題或大綱）。\n"
                "5. 【修訂指引 (revision_guidance)】：若有任何一項為 PARTIAL 或 FAIL，必須具體指明修改方向與筆法改善建議。"
            )
            res = await self.llm_client.generate_structured(
                sys_prompt, text, GateBRubricContract
            )
            if res.best_paragraph_text:
                res.best_paragraph_text = sanitize_novel_prose(res.best_paragraph_text)
            if res.revision_guidance:
                res.revision_guidance = convert_simplified_to_traditional(res.revision_guidance.strip())
            for item in res.rubric_items:
                if item.evidence_quote:
                    item.evidence_quote = convert_simplified_to_traditional(item.evidence_quote.strip())
            return res
        raise AIModelUnavailableError("本地 AI 模型未就緒或推論服務離線，無法進行 10 項文學 Rubric 評分。")
