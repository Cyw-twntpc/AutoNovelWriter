"""Pytest fixtures and mock LLM test doubles for AutoNovelWriter test suite."""

from __future__ import annotations

import re
from typing import Any, Callable

import pytest

from autonovel.config import RuntimeConfig
from autonovel.schemas.domain import SceneBeat, StageGateQuota
from autonovel.schemas.llm_contracts import (
    BatchedFactExtractionContract,
    BlueprintCompletionContract,
    BookAnchorDiscoveryContract,
    BookBlueprintSynthesisContract,
    ChapterExtractionContract,
    ChapterFactExtractContract,
    ChapterOutlineContract,
    CharacterArchetypeContract,
    EntityArchetypeMappingContract,
    FactionGeopoliticsContract,
    GateASemanticCheckContract,
    GateBRubricContract,
    GroupEntityDeltas,
    HierarchyItemContract,
    InitialCharacterContract,
    MacroPacingTemplateContract,
    MajorArcConsolidationContract,
    OrthogonalTagsContract,
    PacingStageTemplateContract,
    PowerHierarchyContract,
    ReferenceChunkExtractionContract,
    ReferenceQueryTagsContract,
    RubricItemResult,
    SingleChapterCompactFact,
    TagMaintenanceContract,
    TagRuleContract,
    VolumeFactConsolidationContract,
    VolumeOutlineContract,
    WizardQuestionContract,
    WizardStepAnswerContract,
)


class MockLlamaServerManager:
    """Mock process manager simulating healthy llama-server instance."""

    def __init__(self, healthy: bool = True) -> None:
        self._healthy = healthy
        self._manually_unloaded: bool = False

    def is_healthy(self) -> bool:
        if self._manually_unloaded:
            return False
        return self._healthy

    def ensure_running(self) -> bool:
        self._manually_unloaded = False
        return self._healthy

    async def ensure_running_async(self) -> bool:
        """Async variant used by the production readiness probe."""
        return self.ensure_running()

    async def restart_server_async(self) -> bool:
        """Async variant used by the LLM client watchdog."""
        return self.ensure_running()

    def stop_server(self, manual: bool = False) -> bool:
        if manual:
            self._manually_unloaded = True
        return True


def _build_mock_blueprint_contract() -> BlueprintCompletionContract:
    """Construct a compliant BlueprintCompletionContract for testing."""
    gates = [
        StageGateQuota(
            stage_index=i + 1,
            stage_title=f"第{i + 1}階段·破局征途",
            target_word_quota=1_000_000,
            max_allowed_rank_level=(i + 1) * 2,
            core_mystery_quota=f"揭曉第 {i + 1} 層世界法則真相與核心伏筆線索",
            milestone_goal=f"完成第 {i + 1} 階段主線破局並晉升至階梯 {(i + 1) * 2}",
        )
        for i in range(5)
    ]
    levels = [
        HierarchyItemContract(level_num=i, level_name=name, description=desc)
        for i, (name, desc) in enumerate(
            [
                ("第一階·初窺門徑", "感應天地氣機，淬鍊體魄根基"),
                ("第二階·凝元築基", "氣海成形，初步掌握術法外放"),
                ("第三階·金丹大道", "精氣神凝結如一，壽元與戰力質變"),
                ("第四階·元嬰化神", "神識通玄，初步觸及空間與天地法則"),
                ("第五階·返虛合道", "洞悉萬物本源，言出法隨"),
                ("第六階·渡劫登仙", "超脫凡俗桎梏，直面紀元大劫"),
            ],
            start=1,
        )
    ]
    initial_chapters = [
        ChapterOutlineContract(
            chapter_offset=i,
            suggested_title=f"第{i}章 寒淵風起",
            beats=[
                SceneBeat(
                    beat_index=1,
                    scene_tag="局勢鋪陳",
                    pacing_instruction="沉穩鋪墊環境氛圍",
                    plot_objective=f"第 {i} 章 Beat 1：推演局勢並察覺異動",
                    conflict_type="INTERNAL",
                    tension_target=2,
                    involved_characters=["林墨"],
                    location_name="北境寒淵",
                ),
                SceneBeat(
                    beat_index=2,
                    scene_tag="暗流交鋒",
                    pacing_instruction="短句疾進凸顯張力",
                    plot_objective=f"第 {i} 章 Beat 2：局部試探交鋒並留下懸念",
                    conflict_type="INTERPERSONAL",
                    tension_target=3,
                    involved_characters=["林墨", "蘇清寒"],
                    location_name="北境寒淵",
                ),
            ],
        )
        for i in range(1, 11)
    ]
    return BlueprintCompletionContract(
        novel_title="《古典修仙紀元：蒼穹古卷》",
        world_rules=[
            "天地法則守恆：一切修為晉升與禁術施展皆須支付對等代價，強行跨階必遭反噬。",
            "因果氣機牽引：高階修士與古寶出世皆會在時間軸與地域留下可追查的痕跡。",
            "勢力制衡鐵律：任何單一宗門或家族無法無代價壟斷全部高階資源。",
        ],
        hierarchy_levels=levels,
        factions=["青雲劍宗", "玄陰閣", "大衍皇朝"],
        locations=["北境寒淵", "青雲山脈", "天樞帝都"],
        characters=[
            InitialCharacterContract(
                name="林墨",
                aliases=["林師兄", "墨淵客"],
                role_type="PROTAGONIST",
                rank_level=1,
                faction_name="青雲劍宗",
                location_name="北境寒淵",
                description="出身北境寒門，心性堅韌沉穩，行事果決，擅長於絕境中推演破局。",
            ),
            InitialCharacterContract(
                name="蘇清寒",
                aliases=["蘇仙子", "清寒師姐"],
                role_type="MAJOR",
                rank_level=2,
                faction_name="青雲劍宗",
                location_name="青雲山脈",
                description="劍宗內門執劍弟子，外冷內剛，背負古族血脈隱秘。",
            ),
            InitialCharacterContract(
                name="沈蒼",
                aliases=["沈長老", "玄陰大長老"],
                role_type="ANTAGONIST",
                rank_level=3,
                faction_name="玄陰閣",
                location_name="北境寒淵",
                description="城府極深的幕後佈局者，企圖奪取北境古墓本源。",
            ),
        ],
        opening_plan="以北境寒淵古墓異動為切入點，主角林墨於生死試煉中取得殘缺古卷，捲入青雲劍宗與玄陰閣的暗中博弈。",
        ending_draft="歷經九州動盪與古界法則重塑，林墨解開紀元輪迴之謎，斬斷舊神枷鎖，開創全新修行秩序。",
        stage_gates=gates,
        initial_volume_chapters=initial_chapters,
        style_contract_text=(
            "【全書風格契約書】\n"
            "1. 敘事視角與語感：採用第三人稱限制視角，文字講究冷峻凝練、具象五感描寫，以動作與環境折射心理。\n"
            "2. 節奏控制：遵循『基調穩定＋局部變調』原則，日常密謀段語速沉穩綿密，戰鬥交鋒段採短句疾進。\n"
            "3. 修辭要求：嚴禁空洞形容詞堆砌與說明文式總結，所有威壓與交鋒須透過器物、光影與氣機細節具象呈現。"
        ),
        static_anchor_text=(
            "殘陽如血，將斷崖上的青石染得一片暗紅。林墨指尖輕撫過劍脊上的裂紋，寒風捲起他微濕的衣袂，"
            "遠處山谷間傳來低沉的鐘鳴，每一聲都像是敲在緊繃的弓弦之上，沉重而肅殺。"
        ),
        genre_blacklist_terms=[],
    )


def _build_mock_chapter_stream_text(on_token: Callable[[str], None] | None = None) -> str:
    """Return multi-paragraph literary Chinese chapter text for mock LLM streaming."""
    paragraphs = [
        "夜色如墨，寒風掠過斷崖古松，發出低沉而綿長的嗚咽。林墨立於石階盡頭，指尖輕叩腰間古劍的吞口，目光穿透層層霧靄，落在遠處燈火明滅的谷地。自上一場風波平息以來，此間的氣機便愈發晦暗不明，彷彿有一張無形的大網正在夜幕下緩緩收緊。",
        "「此處的陣紋已被人動過手腳。」蘇清寒緩步走來，衣袂間帶著幾分山巔寒霜的清冽。她垂眸審視著青石縫隙中殘留的暗金色粉末，聲音清冷而篤定，「來人手法極其老辣，不僅抹去了原有的靈力烙印，更在暗處布下了三重反噬禁制，顯然是衝著古卷本源而來。」",
        "林墨神色微凝，蹲下身以指腹撚起一絲微涼的石屑。空氣中瀰漫著淡淡的鐵鏽與焦木氣息，遠處山澗水聲潺潺，卻掩不住暗處潛伏者的呼吸律動。他並未急於拔劍，而是在腦海中將各方勢力的動向逐一推演，將每一處破綻與退路皆算入分毫之間。",
        "「既然對方設局請君入甕，我們便順勢敲山震虎。」林墨緩緩起身，眸底掠過一抹深邃的寒芒。兩人對視一眼，默契地分左右掠入密林深處，腳尖點過枯葉卻不發出一絲聲響，唯有衣角在夜風中獵獵作響。",
        "就在他們踏入古壇外圍的剎那，四周石柱驟然亮起刺目的血芒！數道凌厲無匹的破空聲自頭頂裂雲而下，刀氣縱橫交錯，瞬間將丈許方圓的青石劈出數道深達數寸的溝壑。",
        "「來得好！」林墨低喝一聲，手中古劍鏘然出鞘。劍光如寒潭驚電，在狹窄的石陣間拉出一道森然弧線，精準無比地格開迎面斬落的重刃。金鐵交鳴之聲震得四周松針簌簌墜落，火星四濺映亮了偷襲者驚愕的臉龐。",
        "蘇清寒同時掐訣引動劍陣，數十道澄澈劍氣如游魚般封死敵方側翼退路。短短數息之間，兩人一攻一控，步法交錯如行雲流水。林墨瞅準敵首舊力已盡、新力未生的瞬息破綻，欺身進逼，劍脊重重拍在對方護身玉佩之上。",
        "伴隨著清脆的碎裂聲，黑衣統領悶哼倒退，嘴角溢出一絲鮮血，眼中滿是難以置信。他未料到這名年輕修士對氣機流轉的把握竟精妙至此，不敢再有半點戀戰，捏碎一枚遁符化作黑煙遁入山嵐深處。",
        "風聲漸歇，古壇中央卻緩緩浮現出一道從未見過的古老星圖印記，微弱的螢光在夜色中規律搏動，指向北方更深處的未知秘境。林墨收劍入鞘，拾起地上遺落的半枚玄鐵令牌，深知真正的風暴才剛剛揭開序幕。",
    ]
    body = "\n\n".join(paragraphs * 3)
    if on_token:
        for idx in range(0, len(body), 120):
            on_token(body[idx : idx + 120])
    return body


class MockLLMClient:
    """Mock LLM client returning compliant structured Pydantic contracts and stream chunks."""

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.server_manager = MockLlamaServerManager()

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[Any],
        **kwargs: Any,
    ) -> Any:
        if schema_model == WizardQuestionContract:
            if "無魔寫實" in user_prompt and "御劍飛行" in user_prompt:
                return WizardQuestionContract(
                    has_contradiction=True,
                    contradiction_reason="前序設定為無魔寫實基調，與御劍飛行修仙法術存在核心設定衝突。",
                    contradiction_suggestion="請改選符合寫實基調的能力體系，或調整世界觀基調。",
                    question_title="請重新決定「劍道心法與修為階梯」",
                    question_aspect="力量／資源階梯體系",
                    options=["凝氣築基飛升", "內家氣血武道", "神識御劍", "萬法歸宗"],
                    allow_multi_select=True,
                )
            return WizardQuestionContract(
                has_contradiction=False,
                question_title="請決定「劍道心法與修為階梯」",
                question_aspect="力量／資源階梯體系",
                options=["無魔寫實純歷史凡人無超凡", "凝氣築基飛升", "神識御劍", "萬法歸宗"],
                allow_multi_select=True,
            )
        if schema_model == WizardStepAnswerContract:
            return WizardStepAnswerContract(
                selected_options=["凝氣築基飛升"],
                thematic_guidance="偏向循序漸進與道法自然，注重心性歷練與代價平衡。",
            )
        if schema_model == BlueprintCompletionContract:
            return _build_mock_blueprint_contract()
        if schema_model == ChapterExtractionContract:
            return ChapterExtractionContract(
                chapter_title="第1章 劍出北境",
                tension_intensity=3,
                goal_progress=3,
                emotional_payoff=3,
                character_deltas=[],
                foreshadow_deltas=[],
                relation_deltas=[],
                timeline_deltas=[],
            )
        if schema_model == GateASemanticCheckContract:
            return GateASemanticCheckContract(
                has_core_contradiction=False,
                has_minor_drift=False,
                core_contradiction_reasons=[],
                minor_drift_reasons=[],
                forward_retcon_instruction="",
            )
        if schema_model == GateBRubricContract:
            return GateBRubricContract(
                rubric_items=[
                    RubricItemResult(
                        item_id=f"R{i}",
                        dimension="RHETORIC",
                        verdict="PASS",
                        evidence_quote="文筆流暢自然",
                    )
                    for i in range(1, 5)
                ]
                + [
                    RubricItemResult(
                        item_id=f"D{i}",
                        dimension="DIALOGUE",
                        verdict="PASS",
                        evidence_quote="對白簡潔有張力",
                    )
                    for i in range(1, 4)
                ]
                + [
                    RubricItemResult(
                        item_id=f"S{i}",
                        dimension="SCENE",
                        verdict="PASS",
                        evidence_quote="畫面感與氣機描摹生動",
                    )
                    for i in range(1, 4)
                ],
                revision_guidance="",
                best_paragraph_text="殘陽如血，將斷崖上的青石染得一片暗紅。林墨指尖輕撫過劍脊上的裂紋，寒風捲起他微濕的衣袂。",
                best_paragraph_score=92.0,
            )
        if schema_model == VolumeOutlineContract:
            return VolumeOutlineContract(
                volume_title="第一卷·初出茅廬",
                volume_summary="第一卷主線綱要",
                chapters=[
                    ChapterOutlineContract(
                        chapter_offset=i,
                        suggested_title=f"第{i}章 劍影初現",
                        beats=[
                            SceneBeat(
                                beat_index=1,
                                plot_objective="推演局勢",
                                conflict_type="INTERNAL",
                                tension_target=3,
                            ),
                            SceneBeat(
                                beat_index=2,
                                plot_objective="迎擊敵鋒",
                                conflict_type="INTERPERSONAL",
                                tension_target=4,
                            ),
                        ],
                    )
                    for i in range(1, 11)
                ],
            )
        if schema_model == BookAnchorDiscoveryContract:
            return BookAnchorDiscoveryContract(
                genre_anchor="仙俠:修真",
                entity_ledger=[
                    EntityArchetypeMappingContract(entity_name="林墨", archetype_role="核心行動者"),
                    EntityArchetypeMappingContract(entity_name="純陽子", archetype_role="宗門上位者"),
                    EntityArchetypeMappingContract(entity_name="噬金蟲", archetype_role="伴生異獸"),
                ],
            )
        if schema_model == ReferenceChunkExtractionContract:
            return ReferenceChunkExtractionContract(
                description="青石斷崖之上劍氣縱橫激盪，林墨凝神靜氣引動天地殺機。",
                snippet="青石斷崖之上，劍氣縱橫激盪，林墨凝神靜氣，以靈氣引動天地殺機。",
                tags_matrix=OrthogonalTagsContract(
                    structure_tags=["劍道殺伐", "推進"],
                    actant_tags=["交鋒", "戰鬥張力"],
                    rhetoric_tags=["劍氣", "寒潭"],
                    context_tags=["仙俠:劍道殺伐", "仙俠"],
                ),
                subject_role="主角",
                object_role="環境",
                interaction_dynamic="威權壓迫",
                relational_dynamic="行動方以凌厲劍意壓迫敵手，引發對方強烈戒備與抵禦",
            )
        if schema_model == ReferenceQueryTagsContract:
            return ReferenceQueryTagsContract(
                query_tags_matrix=OrthogonalTagsContract(
                    structure_tags=["推進"],
                    actant_tags=["戰鬥張力"],
                    rhetoric_tags=["劍氣"],
                    context_tags=["仙俠"],
                ),
                subject_role="主角",
                object_role="環境",
                interaction_dynamic="威權壓迫",
                relational_dynamic="行動方以凌厲劍意壓迫敵手，引發對方強烈戒備與抵禦",
            )
        if schema_model == BatchedFactExtractionContract:
            matches = re.findall(r"===\s*(?:第\s*(\d+)\s*章|原始章節編號\s*(\d+))：(.*?)\s*===", user_prompt)
            if not matches:
                parsed_pairs = [("1", "第一章 劍起青萍")]
            else:
                parsed_pairs = [(m[0] or m[1], m[2]) for m in matches]
            compacts = [
                SingleChapterCompactFact(
                    chapter_index=int(idx_str),
                    chapter_title=title_str.strip(),
                    core_event=f"第{idx_str}章核心事件：主角於試煉中突破困境並推進局勢。",
                    pacing_phase="CLIMAX" if int(idx_str) == 4 else "COMPLICATION",
                    actual_tension=4 if int(idx_str) >= 3 else 2,
                    actual_progress=4 if int(idx_str) >= 3 else 2,
                    score_causal=4.0 if int(idx_str) >= 3 else 3.0,
                    score_craft=4.0 if int(idx_str) >= 3 else 3.0,
                    score_depth=3.5 if int(idx_str) >= 3 else 3.0,
                )
                for idx_str, title_str in parsed_pairs
            ]
            return BatchedFactExtractionContract(
                chapters=compacts,
                entity_deltas=GroupEntityDeltas(
                    character_milestones=["主角修為穩固，道心愈發堅韌"],
                    faction_dynamics=["宗門外圍探子有所察覺，暗流漸生"],
                    world_rules_and_hierarchy=["天地靈氣運轉法則與境界屏障"],
                ),
            )
        if schema_model == ChapterFactExtractContract:
            return ChapterFactExtractContract(
                chapter_index=1,
                chapter_title="第一章 劍起青萍",
                core_event="少年在斷崖參悟劍氣，引發異象並擊退潛在窺視者。",
                pacing_phase="BUILDUP",
                character_dynamics=["主角修為穩固，道心愈發堅韌"],
                faction_shifts=["宗門外圍探子有所察覺，暗流漸生"],
                actual_tension=3,
                actual_progress=3,
            )
        if schema_model == VolumeFactConsolidationContract:
            return VolumeFactConsolidationContract(
                volume_index=1,
                volume_title="第一卷（第1～20章）",
                volume_arc_summary="本卷主角歷經斷崖苦修與宗門初試，破除外部阻礙並奠定道基。",
                core_conflict_resolution="階段性外敵威脅暫告平息，獲取晉升資格。",
                character_milestones=["主角突破瓶頸，晉升核心位階"],
                faction_geopolitics_summary="三大宗門暗中博弈，暫維表面和平。",
                world_rules_and_hierarchy_summary="確立基礎修煉位階與天地靈氣守恆規則。",
            )
        if schema_model == MajorArcConsolidationContract:
            return MajorArcConsolidationContract(
                arc_index=1,
                covered_volumes="第 1 卷至第 4 卷",
                arc_title="第一幕：潛龍出淵",
                arc_summary="主角自微末崛起，逐步揭開上古秘境真相。",
                conflict_and_resolution="擊潰外圍強敵，晉入更高層次天地。",
                character_milestones_summary=["主角晉升金丹境界"],
                faction_geopolitics_summary="正魔兩道重新洗牌。",
                world_rules_and_hierarchy_summary="高階法則與界域壁壘顯現。",
            )
        if schema_model == BookBlueprintSynthesisContract:
            return BookBlueprintSynthesisContract(
                novel_title="凡人經典",
                genre_primary="仙俠",
                genre_sub="凡人修真",
                macro_outline="開端破局：少年求道；中點反轉：宗門遭劫；終極決戰：逆轉天命，重塑秩序。",
                character_network=[
                    CharacterArchetypeContract(
                        character_name="林墨",
                        archetype_role="主角",
                        faction_name="散修",
                        background_and_motive="歷經生死考驗，追尋天地大道",
                    ),
                    CharacterArchetypeContract(
                        character_name="長老",
                        archetype_role="反派",
                        faction_name="青雲門",
                        background_and_motive="維護門閥威望，打壓異數",
                    ),
                    CharacterArchetypeContract(
                        character_name="道友",
                        archetype_role="男配角",
                        faction_name="天機閣",
                        background_and_motive="情報交換，守望相助",
                    ),
                ],
                faction_geopolitics=[
                    FactionGeopoliticsContract(
                        faction_name="青雲門",
                        doctrine_and_nature="正道魁首，等級森嚴",
                        power_tier="頂級勢力",
                        rival_factions=["幽冥宗"],
                    )
                ],
                hierarchy_rules=[
                    PowerHierarchyContract(
                        level_num=1,
                        level_name="煉氣期",
                        core_attributes="引氣入體，神識初顯",
                        breakthrough_criteria="聚氣成海，築就道基",
                    )
                ],
                world_laws=["天道無親，常與善人", "大道爭鋒，退一步萬劫不復"],
            )
        if schema_model == TagMaintenanceContract:
            return TagMaintenanceContract(
                rules=[
                    TagRuleContract(
                        old_tag="仙俠:劍道殺伐", new_tag="仙俠:純陽劍道", action="NORMALIZE"
                    )
                ]
            )
        if schema_model == MacroPacingTemplateContract:
            return MacroPacingTemplateContract(
                template_name="凡人經典·標準宏觀節奏波浪",
                genre_primary="仙俠",
                genre_sub="凡人修真",
                source_novel_title="凡人經典",
                total_chapters_analyzed=5,
                narrative_archetype="經典起承轉合逆襲流",
                pacing_tags=["#慢熱蓄勢", "#雙峰高潮", "#階梯式推進"],
                summary="標準起承轉合波浪曲線",
                stages=[
                    PacingStageTemplateContract(
                        stage_index=1,
                        offset_percent=0.0,
                        pacing_phase="BUILDUP",
                        target_tension=2,
                        target_progress=2,
                        target_density=2,
                        dramatic_purpose="鋪陳環境常規",
                    ),
                    PacingStageTemplateContract(
                        stage_index=2,
                        offset_percent=25.0,
                        pacing_phase="COMPLICATION",
                        target_tension=3,
                        target_progress=3,
                        target_density=3,
                        dramatic_purpose="核心矛盾初露",
                    ),
                    PacingStageTemplateContract(
                        stage_index=3,
                        offset_percent=55.0,
                        pacing_phase="COMPLICATION",
                        target_tension=4,
                        target_progress=3,
                        target_density=4,
                        dramatic_purpose="危機升級",
                    ),
                    PacingStageTemplateContract(
                        stage_index=4,
                        offset_percent=78.0,
                        pacing_phase="CLIMAX",
                        target_tension=5,
                        target_progress=5,
                        target_density=5,
                        dramatic_purpose="全卷決戰大爆發",
                    ),
                    PacingStageTemplateContract(
                        stage_index=5,
                        offset_percent=100.0,
                        pacing_phase="RESOLUTION",
                        target_tension=2,
                        target_progress=2,
                        target_density=2,
                        dramatic_purpose="餘波平復",
                    ),
                ],
            )
        raise ValueError(f"MockLLMClient unsupported schema model: {schema_model}")

    async def generate_text_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4200,
        temperature: float = 0.72,
        seed: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        if "改寫目標句子" in system_prompt or "修辭微調" in system_prompt:
            return "林墨凝視著眼前的陣法，神色微斂，暗自推演著破陣的得失利弊與諸般變化。"
        if "寫作操作清單" in system_prompt or "技法與邏輯內化" in system_prompt or "理解並內化" in system_prompt:
            return "【參考技法內化】透過冷冽環境描寫與短句動作交錯烘托拔劍對決的壓迫感，主客交鋒展現凌厲的主動壓制態勢。"
        return _build_mock_chapter_stream_text(on_token)


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Provide a mock LLMClient instance."""
    return MockLLMClient()
