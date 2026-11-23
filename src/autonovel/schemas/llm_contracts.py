"""Strict JSON Schema Pydantic contracts for llama-server GBNF constrained decoding."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from autonovel.schemas.domain import (
    ForeshadowScope,
    HierarchyChangeType,
    SceneBeat,
    StageGateQuota,
)


class WizardQuestionContract(BaseModel):
    """Structured output for generating the next interactive Q&A step."""

    has_contradiction: bool = False
    contradiction_reason: str = ""
    conflicting_step_index: int | None = None
    question_title: str
    question_aspect: str
    options: list[str] = Field(..., min_length=4, max_length=6)
    allow_multi_select: bool = True


class WizardStepAnswerContract(BaseModel):
    """Structured decision produced by Genre-Adaptive Novelist Co-Pilot."""

    selected_options: list[str] = Field(..., min_length=1, max_length=3)
    thematic_guidance: str = Field(
        default="",
        description="宏觀方向方針，嚴禁具體人名道具微觀情節",
    )


class HierarchyItemContract(BaseModel):
    """Structured definition of a cultivation or power rank level."""

    category: str = "CULTIVATION"
    level_num: int = Field(..., ge=1)
    level_name: str
    description: str


class InitialCharacterContract(BaseModel):
    """Structured initial character card produced during AI blueprint completion."""

    name: str
    aliases: list[str] = Field(default_factory=list)
    role_type: str = "PROTAGONIST"
    rank_level: int = 1
    faction_name: str = ""
    location_name: str = ""
    description: str


class ChapterOutlineContract(BaseModel):
    """Pre-planned chapter outline within a volume."""

    chapter_offset: int = Field(..., ge=1)
    suggested_title: str
    beats: list[SceneBeat] = Field(..., min_length=2, max_length=3)


class BlueprintCompletionContract(BaseModel):
    """Structured output for Step 3 AI blueprint completion."""

    novel_title: str
    world_rules: list[str] = Field(..., min_length=3)
    hierarchy_levels: list[HierarchyItemContract] = Field(..., min_length=3)
    factions: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    characters: list[InitialCharacterContract] = Field(..., min_length=3)
    opening_plan: str
    ending_draft: str
    stage_gates: list[StageGateQuota] = Field(..., min_length=3)
    style_contract_text: str
    static_anchor_text: str = Field(
        ...,
        min_length=50,
        description="開篇定場文風示範段落（純粹小說正文，至少50字，嚴禁包含章節標題或大綱說明）",
    )
    genre_blacklist_terms: list[str] = Field(default_factory=list)
    initial_volume_chapters: list[ChapterOutlineContract] = Field(
        default_factory=list,
        description="首卷預排章節細綱清單（由AI根據世界觀與開篇計畫直接規劃每章標題與場景節拍）",
    )

    @field_validator("static_anchor_text")
    @classmethod
    def _validate_static_anchor_text(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 50:
            raise ValueError("static_anchor_text must contain at least 50 characters of genuine prose.")
        return cleaned


class VolumeOutlineContract(BaseModel):
    """Structured output for rolling volume planning."""

    volume_title: str
    volume_summary: str
    chapters: list[ChapterOutlineContract] = Field(..., min_length=1)


class RubricItemResult(BaseModel):
    """Single item evaluation in the 10-item LLM literary rubric."""

    item_id: str
    dimension: Literal["RHETORIC", "DIALOGUE", "SCENE"]
    verdict: Literal["PASS", "PARTIAL", "FAIL"]
    evidence_quote: str = ""


class GateASemanticCheckContract(BaseModel):
    """Structured output for Gate A semantic consistency and drift check."""

    has_core_contradiction: bool = False
    core_contradiction_detail: str = ""
    has_minor_drift: bool = False
    minor_drift_note: str = ""
    forward_retcon_instruction: str = ""


class GateBRubricContract(BaseModel):
    """Structured output for Gate B 10-item literary rubric and best paragraph anchor."""

    rubric_items: list[RubricItemResult] = Field(..., min_length=10, max_length=10)
    revision_guidance: str = ""
    best_paragraph_text: str
    best_paragraph_score: float = Field(..., ge=0.0, le=100.0)


class CharacterDeltaContract(BaseModel):
    """Extracted character state update at the end of a chapter."""

    character_name: str
    aliases: list[str] = Field(default_factory=list)
    new_status: Literal["ALIVE", "DEAD"] = "ALIVE"
    # Free-form 2-6 character flavour term (injury/realm/mood) kept separate from the
    # ALIVE/DEAD lifecycle enum that every query filters on.
    status_note: str = ""
    new_rank_level: int | None = None
    change_type: HierarchyChangeType = HierarchyChangeType.NORMAL_BREAKTHROUGH
    change_reason: str = ""
    location_name: str = ""
    faction_name: str = ""
    description_update: str = ""


class ForeshadowDeltaContract(BaseModel):
    """Extracted new or resolved foreshadowing item at the end of a chapter.

    Resolution targets are matched by ``description`` because the model never sees
    database primary keys; an id field would only invite hallucinated matches.
    """

    action: Literal["PLANT", "RESOLVE"]
    description: str
    scope: ForeshadowScope = ForeshadowScope.SHORT
    related_characters: list[str] = Field(default_factory=list)
    resolution_notes: str = ""


class RelationDeltaContract(BaseModel):
    """Extracted interpersonal or factional relationship update."""

    source_name: str
    relation_type: str
    target_name: str
    status: str = "ACTIVE"


class TimelineDeltaContract(BaseModel):
    """Extracted timeline event for a narrative track."""

    track_name: str = "主角線"
    duration_days: float = Field(default=1.0, ge=0.0)
    in_universe_calendar: str = ""
    location_name: str = ""
    event_summary: str
    is_sync_point: bool = False
    sync_point_label: str = ""


class ChapterExtractionContract(BaseModel):
    """Complete structured post-chapter extraction payload."""

    chapter_title: str
    tension_intensity: int = Field(..., ge=1, le=5)
    goal_progress: int = Field(..., ge=1, le=5)
    emotional_payoff: int = Field(..., ge=1, le=5)
    character_deltas: list[CharacterDeltaContract] = Field(default_factory=list)
    foreshadow_deltas: list[ForeshadowDeltaContract] = Field(default_factory=list)
    relation_deltas: list[RelationDeltaContract] = Field(default_factory=list)
    timeline_deltas: list[TimelineDeltaContract] = Field(default_factory=list)


class OrthogonalTagsContract(BaseModel):
    """Orthogonal 4-axis ontology tags for reference novel extraction."""

    structure_tags: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="宏觀結構功能：段落在劇情推演與篇章力學中的結構性作用（如：勢能蓄積、暗線埋伏、轉折頓悟、承前啟後）",
    )
    actant_tags: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="主客博弈動力：人物、勢力或主客體間之力量交鋒與動態張力（如：位階壓制、試探博弈、理念背離、利益結盟）",
    )
    rhetoric_tags: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="微觀行文筆法：文字修辭、感官渲染與節奏調控（如：五感通感、蒙太奇組接、對白潛台詞、句式緩急）",
    )
    context_tags: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="情境氛圍座標：客觀世界觀定位與特定場景氛圍（如：暗黑懸疑、朝堂肅殺、宗門世道、市井煙火）",
    )

    def to_tagged_items(self) -> list[tuple[str, str]]:
        """Return list of (tag_name, axis_type) pairs."""
        items: list[tuple[str, str]] = []
        for t in self.structure_tags:
            if t and t.strip():
                items.append((t.strip(), "STRUCTURE"))
        for t in self.actant_tags:
            if t and t.strip():
                items.append((t.strip(), "ACTANT"))
        for t in self.rhetoric_tags:
            if t and t.strip():
                items.append((t.strip(), "RHETORIC"))
        for t in self.context_tags:
            if t and t.strip():
                items.append((t.strip(), "CONTEXT"))
        return items

    def all_tags(self) -> list[str]:
        """Return flat list of all tags across axes."""
        return [item[0] for item in self.to_tagged_items()]


class ReferenceChunkExtractionContract(BaseModel):
    """Structured knowledge entry extracted from a reference novel chunk."""

    description: str = Field(..., max_length=1000)
    snippet: str = Field(..., max_length=1500)
    tags_matrix: OrthogonalTagsContract = Field(
        ...,
        description="正交四象限文學本體標籤體系（結構功能、主客博弈、微觀筆法、情境氛圍）",
    )
    relational_dynamic: str = Field(
        default="",
        max_length=400,
        description="自然主客因果動力學描述：結合全局藍圖與當前切片情境，界定本段敘事力量的作用方向與結構張力。",
    )
    subject_role: str = Field(
        default="",
        max_length=100,
        description="主體基礎定位：標定本切片中主動發起行動方在敘事編制中的最基礎通俗定位或客觀實體類別（如主配角性別編制、敵我職能、自然環境等）。",
    )
    object_role: str = Field(
        default="",
        max_length=100,
        description="客體基礎定位：標定本切片中承受作用力或博弈目標之最基礎通俗定位或客觀實體類別。",
    )
    interaction_dynamic: str = Field(
        default="",
        max_length=100,
        description="主客互動模式：以簡練詞彙界定主體與客體之間的客觀力量作用或情感互動模式。",
    )


class ReferenceQueryTagsContract(BaseModel):
    """Query tags generated by AI for Track-2 reference library retrieval."""

    query_tags_matrix: OrthogonalTagsContract = Field(
        ...,
        description="四維正交檢索標籤體系",
    )
    relational_dynamic: str = Field(
        default="",
        description="當前章節任務所屬之自然主客因果動力學方向",
    )
    subject_role: str = Field(
        default="",
        description="當前章節任務發起方之敘事編制基礎定位",
    )
    object_role: str = Field(
        default="",
        description="當前章節任務承受或博弈方之敘事編制基礎定位",
    )
    interaction_dynamic: str = Field(
        default="",
        description="主客互動模式（界定主體與客體之間的客觀力量作用或情感互動模式）",
    )


class TagRuleContract(BaseModel):
    """Rule to normalize, merge, or drop an extracted reference library tag."""

    old_tag: str
    new_tag: str = ""
    action: Literal["KEEP", "NORMALIZE", "MERGE", "DROP"] = "KEEP"


class TagMaintenanceContract(BaseModel):
    """Structured global tag maintenance plan produced by AI."""

    rules: list[TagRuleContract] = Field(default_factory=list)


class PacingStageTemplateContract(BaseModel):
    """Single stage/milestone in a macro pacing curve template."""

    stage_index: int
    offset_percent: float = Field(..., ge=0.0, le=100.0)
    pacing_phase: Literal["BUILDUP", "COMPLICATION", "CLIMAX", "RESOLUTION"]
    target_tension: int = Field(..., ge=1, le=5)
    target_progress: int = Field(..., ge=1, le=5)
    target_density: int = Field(..., ge=1, le=5)
    dramatic_purpose: str = Field(
        ...,
        description="抽象戲劇任務：說明該進度節點在全書情節動力學中所擔負之結構性推進行動，嚴禁包含小說具體角色名或特定事件流水帳",
    )


class MacroPacingTemplateContract(BaseModel):
    """Full macro pacing wave archetype extracted from a reference novel."""

    template_name: str
    genre_primary: str
    genre_sub: str
    source_novel_title: str
    total_chapters_analyzed: int
    narrative_archetype: str
    pacing_tags: list[str] = Field(
        default_factory=list,
        description="波浪特徵標籤：反映起承轉合形態、衝突密度與推進速率之巨觀屬性標籤（3～6個）",
    )
    summary: str
    stages: list[PacingStageTemplateContract] = Field(..., min_length=4)
    chapter_curve_data: list[dict[str, Any]] = Field(
        default_factory=list,
        description="逐章100%歸一化高精度平滑波浪數據序列（含每章百分比、平滑張力、推進度、密度）",
    )


class EntityArchetypeMappingContract(BaseModel):
    """Mapping between a concrete novel entity name and an abstract dramatic archetype."""

    entity_name: str
    archetype_role: str


class BookAnchorDiscoveryContract(BaseModel):
    """Structured genre anchor and entity archetype ledger discovered from the opening text of a novel."""

    genre_anchor: str = Field(..., description="母題材座標：以「主題材:子題材」之層級架構界定世界觀底層設定，嚴禁多餘符號")
    entity_ledger: list[EntityArchetypeMappingContract] = Field(
        default_factory=list,
        description="開篇核心實體至可跨文本遷移抽象原型角色之映射清單，archetype_role 嚴禁包含具體專有名詞",
    )


class SingleChapterCompactFact(BaseModel):
    """High-density chapter factual increment and 3D literary exemplar score."""

    chapter_index: int
    chapter_title: str
    # Allow up to 800 chars to match ChapterFactExtractContract.core_event so single-chapter
    # fallback conversion never fails ValidationError on valid detailed summaries.
    core_event: str = Field(
        ...,
        max_length=800,
        description="客觀總結本章實質發生的核心事件因果鏈（40~80字，起因-衝突-結果，嚴禁流水帳）",
    )
    pacing_phase: Literal["BUILDUP", "COMPLICATION", "CLIMAX", "RESOLUTION"] = Field(
        default="COMPLICATION",
        description="本章於敘事結構中之戲劇階段定性（BUILDUP: 鋪墊蓄勢 / COMPLICATION: 衝突推進與博弈拉扯 / CLIMAX: 矛盾爆發與決戰高潮 / RESOLUTION: 餘波收束與戰果結算）",
    )
    actual_tension: int = Field(..., ge=1, le=5, description="客觀評定本章文字實質呈現的衝突激烈程度（1=平緩日常，3=矛盾對峙，5=生死決戰）")
    actual_progress: int = Field(..., ge=1, le=5, description="客觀評定本章實質推動全書核心主線的進展幅度（1~5）")
    score_causal: float = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="局勢相變度：關鍵因果約束狀態之不可逆位移量（1.0=平衡維持，5.0=關鍵約束徹底失效並引發全局軌道轉移）。",
    )
    score_craft: float = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="修辭潛流度：表層文字與深層實質結構的偏離度（潛台詞張力），及語言符號組織秩序的自洽形式感（1.0=表深重合事實交代，5.0=表深高度對立且形式自洽）。",
    )
    score_depth: float = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="共鳴認知度：文本跨越文字符號對心智引發的情感共振強度、知覺沉浸度與哲思啟發深度（1.0=心智無波瀾，5.0=深刻知覺共時化與認知昇華）。",
    )


class GroupEntityDeltas(BaseModel):
    """Substantive entity, faction, and world rule increments for a batched group of chapters."""

    character_milestones: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="本組各章內登場核心角色的實質位階突破、重大身分轉變、核心動機揭露或生死立場逆轉（僅記錄實質重大增量，無則留空）",
    )
    faction_dynamics: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="本組各章內涉及的新組織登場、陣營關係演變或利益結構變動（無則留空）",
    )
    world_rules_and_hierarchy: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="本組各章內新揭曉或實質變更的位階階梯、組織制度、底層運轉鐵律或不可違背之代價禁忌（無則留空）",
    )


class BatchedFactExtractionContract(BaseModel):
    """Structured output for batched chapter extraction in a single LLM invocation."""

    chapters: list[SingleChapterCompactFact] = Field(..., min_length=1, max_length=4)
    entity_deltas: GroupEntityDeltas = Field(default_factory=GroupEntityDeltas)


class ChapterFactExtractContract(BaseModel):
    """Structured chapter factual increment extracted during Stage 1 full-text deep read."""

    chapter_index: int = 1
    chapter_title: str = ""
    core_event: str = Field(
        ...,
        max_length=800,
        description="客觀總結本章實質發生的核心事件因果（60~150字，起因-衝突-結果，嚴禁長篇流水帳）",
    )
    pacing_phase: Literal["BUILDUP", "COMPLICATION", "CLIMAX", "RESOLUTION"] = Field(
        default="COMPLICATION",
        description="本章於敘事結構中之戲劇階段定性（BUILDUP: 鋪墊蓄勢 / COMPLICATION: 衝突推進與博弈拉扯 / CLIMAX: 矛盾爆發與決戰高潮 / RESOLUTION: 餘波收束與戰果結算）",
    )
    character_dynamics: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="本章登場或涉及的核心角色立場動態、態度變化或位階突破（最多8條）",
    )
    faction_shifts: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="本章涉及的勢力交鋒或陣營利益變動（最多6條）",
    )
    actual_tension: int = Field(..., ge=1, le=5, description="客觀評定本章文字實質呈現的衝突激烈程度（1=平緩日常，3=矛盾對峙，5=生死決戰）")
    actual_progress: int = Field(..., ge=1, le=5, description="客觀評定本章實質推動全書核心主線的進展幅度（1~5）")
    # 三維原始文學價值評分（1.0 ～ 5.0，精確至一位小數）
    score_causal: float = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="局勢相變度：關鍵因果約束狀態之不可逆位移量（1.0=平衡維持，5.0=關鍵約束徹底失效並引發全局軌道轉移）。",
    )
    score_craft: float = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="修辭潛流度：表層文字與深層實質結構的偏離度（潛台詞張力），及語言符號組織秩序的自洽形式感（1.0=表深重合事實交代，5.0=表深高度對立且形式自洽）。",
    )
    score_depth: float = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="共鳴認知度：文本跨越文字符號對心智引發的情感共振強度、知覺沉浸度與哲思啟發深度（1.0=心智無波瀾，5.0=深刻知覺共時化與認知昇華）。",
    )

    @classmethod
    def from_compact(
        cls,
        compact: SingleChapterCompactFact,
        character_dynamics: list[str] | None = None,
        faction_shifts: list[str] | None = None,
    ) -> ChapterFactExtractContract:
        """Convert a SingleChapterCompactFact to ChapterFactExtractContract."""
        return cls(
            chapter_index=compact.chapter_index,
            chapter_title=compact.chapter_title,
            core_event=compact.core_event,
            pacing_phase=compact.pacing_phase,
            actual_tension=compact.actual_tension,
            actual_progress=compact.actual_progress,
            score_causal=compact.score_causal,
            score_craft=compact.score_craft,
            score_depth=compact.score_depth,
            character_dynamics=character_dynamics or [],
            faction_shifts=faction_shifts or [],
        )


def calculate_chapter_exemplar_score(causal: float, craft: float, depth: float) -> float:
    """Calculate weighted exemplar score using 7:2:1 peak-led formula (range 2.0 to 10.0).

    Formula: Highest * 1.4 + Middle * 0.4 + Lowest * 0.2
    """
    scores = sorted([float(causal), float(craft), float(depth)], reverse=True)
    final_score = (scores[0] * 1.4) + (scores[1] * 0.4) + (scores[2] * 0.2)
    return round(final_score, 2)


class VolumeFactConsolidationContract(BaseModel):
    """Consolidated facts and dramatic arc for a volume/arc of chapters."""

    volume_index: int = 1
    volume_title: str = ""
    volume_arc_summary: str = Field(
        ...,
        max_length=1200,
        description="本卷核心情節弧起承轉合與主線推進走勢（200~400字）",
    )
    core_conflict_resolution: str = Field(
        ...,
        max_length=800,
        description="本卷核心矛盾與主要勢力交鋒的解決或激化情況（100~200字）",
    )
    character_milestones: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="核心角色在本卷達成的關鍵成長、陣營轉換或生死狀態（最多10條）",
    )
    faction_geopolitics_summary: str = Field(
        ...,
        max_length=800,
        description="本卷結束時各大勢力實力格局天平演變現況（100~200字）",
    )
    world_rules_and_hierarchy_summary: str = Field(
        default="",
        max_length=800,
        description="本卷揭曉或驗證的核心位階階梯特徵、晉升門檻、組織制度與不可違背之底層運轉法則或代價禁忌（100~200字；若題材無特殊位階，請總結行業/社會/法律規則）",
    )


class MajorArcConsolidationContract(BaseModel):
    """Consolidated report for a major narrative arc (merging 7~8 volumes) to eliminate truncation."""

    arc_index: int = 1
    arc_title: str = ""
    covered_volumes: str = ""
    arc_summary: str = Field(
        ...,
        max_length=1500,
        description="本大幕情節起承轉合主線走勢與重大戰略轉折（300~500字）",
    )
    conflict_and_resolution: str = Field(
        ...,
        max_length=800,
        description="本大幕核心矛盾爆發與收束結果（150~250字）",
    )
    character_milestones_summary: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="本大幕核心角色之命運轉折、陣營歸屬與實力突破",
    )
    faction_geopolitics_summary: str = Field(
        ...,
        max_length=800,
        description="本大幕結束時各大勢力門派版圖與地緣利益天平現況（150~250字）",
    )
    world_rules_and_hierarchy_summary: str = Field(
        default="",
        max_length=800,
        description="本大幕揭示或深化之位階階梯特徵、核心體制與底層運轉法則或代價禁忌（150~250字；若無特殊位階體系，請總結大局規則）",
    )


class CharacterArchetypeContract(BaseModel):
    """Mapping between a novel character entity and its dramatic archetype role in the blueprint."""

    character_name: str = Field(
        ...,
        max_length=50,
        description="角色於原著文本中之正式稱謂（嚴禁填入抽象功能代號）",
    )
    archetype_role: str = Field(
        ...,
        max_length=50,
        description="敘事編制基礎定位（2～3字，僅標定該角色於全書編制中之通俗主配角性別定位、敵我陣營職能或非人生物類別；嚴禁堆砌心境狀態、命運修飾語或形容詞，嚴禁包含角色本名或專有名詞）",
    )
    faction_name: str = Field(default="", max_length=50, description="角色所屬的核心勢力或組織名稱")
    background_and_motive: str = Field(..., max_length=400, description="人物背景、核心動機與命運軌跡簡述")


class FactionGeopoliticsContract(BaseModel):
    """Faction/organization profile and geopolitical alignment in the blueprint."""

    faction_name: str = Field(..., max_length=50)
    doctrine_and_nature: str = Field(..., max_length=400, description="組織或陣營宗旨、處事風格與核心利益訴求")
    power_tier: str = Field(..., max_length=50, description="該組織於全局勢力結構中所處之支配力與資源層級定位")
    rival_factions: list[str] = Field(default_factory=list, max_length=8, description="主要敵對或競爭勢力")


class PowerHierarchyContract(BaseModel):
    """Power/rank level definition in the blueprint."""

    level_num: int = Field(..., ge=1)
    level_name: str = Field(..., max_length=50)
    core_attributes: str = Field(..., max_length=400, description="該位階核心能力特徵、權限範圍、生命質變或專業表現")
    breakthrough_criteria: str = Field(..., max_length=400, description="晉階突破之條件、門檻、考驗、瓶頸或代價")


class BookBlueprintSynthesisContract(BaseModel):
    """Complete macro factual blueprint synthesized from all volume fact reports."""

    novel_title: str
    genre_primary: str = Field(..., description="依據全書實質世界觀設定與核心衝突性質判定之主題材分類")
    genre_sub: str = Field(..., description="依據全書敘事結構與核心母題判定之子題材分類")
    macro_outline: str = Field(
        ...,
        max_length=3000,
        description="全書宏觀大綱骨幹（500~800字，結構化四幕架構：【第一幕·開端破局】、【第二幕·深入博弈】、【第三幕·高潮相變】、【第四幕·終局收束】）",
    )
    character_network: list[CharacterArchetypeContract] = Field(..., min_length=2, max_length=12, description="全書登場主配角人物原型與背景網絡（最多12人）")
    faction_geopolitics: list[FactionGeopoliticsContract] = Field(..., max_length=10, description="世界觀勢力陣營格局與利益天平（最多10個組織）")
    hierarchy_rules: list[PowerHierarchyContract] = Field(default_factory=list, max_length=15, description="實力、修煉或職級權力晉階等級階梯與位階特徵（最多15階；若題材無等級階梯晉升體系，可回傳空陣列 []）")
    world_laws: list[str] = Field(default_factory=list, max_length=10, description="不可違背之底層世界觀運轉鐵律、制度邊界、超自然代價或社會核心禁忌（最多10條）")

