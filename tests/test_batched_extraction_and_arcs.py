"""Tests for Track B Batched Fact Extraction, Lossless Splitting, Dynamic Packing, and Major Arc Consolidation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from autonovel.reference.macro_extractor import (
    ChapterFactExtractor,
    pack_chapters_dynamically,
    reconcile_batched_compact_facts,
    split_long_chapter_lossless,
)
from autonovel.schemas.llm_contracts import (
    BatchedFactExtractionContract,
    BookBlueprintSynthesisContract,
    ChapterFactExtractContract,
    CharacterArchetypeContract,
    FactionGeopoliticsContract,
    GroupEntityDeltas,
    MajorArcConsolidationContract,
    SingleChapterCompactFact,
    VolumeFactConsolidationContract,
)


def test_split_long_chapter_lossless_short_and_long():
    """Verify split_long_chapter_lossless keeps short text in one segment and losslessly splits long text."""
    short_text = "這是一段不足一萬字的標準章節內文。\n" * 100
    segments_short = split_long_chapter_lossless(short_text, max_segment_chars=10000)
    assert len(segments_short) == 1
    assert segments_short[0] == short_text.strip()

    # Long text (e.g., ~16,000 chars) is split at paragraph/sentence boundaries without losing characters
    para = "夜色深沉，山谷間迴盪著激越的劍鳴聲，雙方修士各自催動法寶激烈交鋒。\n"
    long_text = (para * 500).strip()
    assert len(long_text) > 15000

    segments = split_long_chapter_lossless(long_text, max_segment_chars=7200)
    assert len(segments) >= 3
    for seg in segments:
        assert len(seg) <= 7200
    # Verify lossless preservation of all content lines
    rejoined_lines = [line for seg in segments for line in seg.splitlines() if line.strip()]
    original_lines = [line for line in long_text.splitlines() if line.strip()]
    assert rejoined_lines == original_lines


def test_pack_chapters_dynamically_scenarios():
    """Verify dynamic token-budget packing respects character limits and chapter count."""
    toc_uniform = [
        {"index": i, "title": f"第{i}章 標題", "text": "內文" * 750}
        for i in range(1, 6)
    ]
    batches_a = pack_chapters_dynamically(toc_uniform, max_chars=7200, max_chapters=3)
    assert len(batches_a) == 2
    assert len(batches_a[0]) == 3
    assert len(batches_a[1]) == 2
    assert [c["index"] for c in batches_a[0]] == [1, 2, 3]
    assert [c["index"] for c in batches_a[1]] == [4, 5]

    toc_boundary = [
        {"index": 1, "title": "第1章", "text": "甲" * 3500},
        {"index": 2, "title": "第2章", "text": "乙" * 3500},
        {"index": 3, "title": "第3章", "text": "丙" * 1000},
    ]
    batches_b = pack_chapters_dynamically(toc_boundary, max_chars=7200, max_chapters=3)
    assert len(batches_b) == 2
    assert len(batches_b[0]) == 2
    assert len(batches_b[1]) == 1

    toc_oversized = [
        {"index": 1, "title": "第1章 超長章節", "text": "超" * 8500},
        {"index": 2, "title": "第2章 正常章節", "text": "常" * 1500},
    ]
    batches_c = pack_chapters_dynamically(toc_oversized, max_chars=7200, max_chapters=3)
    assert len(batches_c) == 2
    assert len(batches_c[0]) == 1
    assert len(batches_c[1]) == 1


def test_compact_fact_to_full_contract_conversion():
    """Verify SingleChapterCompactFact converts accurately to ChapterFactExtractContract."""
    compact = SingleChapterCompactFact(
        chapter_index=42,
        chapter_title="第四十二章 京察風雲",
        core_event="主角於調查機構查明假銀案關鍵線索",
        pacing_phase="CLIMAX",
        actual_tension=4,
        actual_progress=4,
        score_causal=4.5,
        score_craft=4.0,
        score_depth=3.5,
    )

    full = ChapterFactExtractContract.from_compact(compact)
    assert full.chapter_index == 42
    assert full.chapter_title == "第四十二章 京察風雲"
    assert full.core_event == "主角於調查機構查明假銀案關鍵線索"
    assert full.pacing_phase == "CLIMAX"
    assert full.actual_tension == 4
    assert full.actual_progress == 4
    assert full.score_causal == 4.5
    assert full.score_craft == 4.0
    assert full.score_depth == 3.5
    assert full.character_dynamics == []
    assert full.faction_shifts == []


def test_schema_entity_deltas_and_volume_consolidation():
    """Verify GroupEntityDeltas and VolumeFactConsolidationContract schema completeness."""
    deltas = GroupEntityDeltas(
        character_milestones=["主角晉升煉氣境", "統領賜佩刀與腰牌"],
        faction_dynamics=["執法衛所與文官集團暗流湧動"],
        world_rules_and_hierarchy=["武道體系：練精、練氣、練神階梯"],
    )
    assert len(deltas.character_milestones) == 2
    assert len(deltas.faction_dynamics) == 1
    assert len(deltas.world_rules_and_hierarchy) == 1

    vol = VolumeFactConsolidationContract(
        volume_index=1,
        volume_title="第一卷（第1～20章）",
        volume_arc_summary="主角破案立足執法衙門。",
        core_conflict_resolution="稅銀案告破，幕後黑手暫退。",
        character_milestones=["主角確立武道根基"],
        faction_geopolitics_summary="朝堂帝黨與相黨對立加深。",
        world_rules_and_hierarchy_summary="確立武道與儒道基本品級規則，品級森嚴不得僭越。",
    )
    assert vol.volume_index == 1
    assert "武道與儒道" in vol.world_rules_and_hierarchy_summary


def test_major_arc_consolidation_contract_schema():
    """Verify MajorArcConsolidationContract validation and fields."""
    arc = MajorArcConsolidationContract(
        arc_index=1,
        covered_volumes="第 1 卷至第 8 卷",
        arc_title="【第一幕：京察暗流·邊州破局】",
        arc_summary="主角由底層捕快步入核心體系，歷經京察風波與邊州平叛。",
        conflict_and_resolution="邊州叛亂平定，但幕後推手潛逃海外，朝廷矛盾進一步激化。",
        character_milestones_summary=["主角晉升中階品級", "統領戰略佈局初現"],
        faction_geopolitics_summary="執法機構與文官權衡，叛軍勢力伏誅。",
        world_rules_and_hierarchy_summary="展現中低階武者與道門金丹境界之威能，氣運因果律初步浮現。",
    )
    assert arc.arc_index == 1
    assert arc.covered_volumes == "第 1 卷至第 8 卷"
    assert "邊州破局" in arc.arc_title
    assert len(arc.character_milestones_summary) == 2


@pytest.mark.asyncio
async def test_extract_batched_chapter_facts_mocked():
    """Verify extract_batched_chapter_facts formats prompt, injects anchor, and converts Traditional Chinese."""
    mock_llm = MagicMock()
    mock_llm.server_manager.is_healthy.return_value = True

    mock_response = BatchedFactExtractionContract(
        chapters=[
            SingleChapterCompactFact(
                chapter_index=1,
                chapter_title="第一章 狱中苏醒",
                core_event="主角在牢房中苏醒并思索脱困之策",
                pacing_phase="BUILDUP",
                actual_tension=3,
                actual_progress=3,
                score_causal=3.5,
                score_craft=3.5,
                score_depth=3.0,
            ),
            SingleChapterCompactFact(
                chapter_index=2,
                chapter_title="第二章 炼金破案",
                core_event="主角运用化学知识侦破假银之谜",
                pacing_phase="CLIMAX",
                actual_tension=4,
                actual_progress=4,
                score_causal=4.5,
                score_craft=4.0,
                score_depth=3.5,
            ),
        ],
        entity_deltas=GroupEntityDeltas(
            character_milestones=["主角化解死刑危机"],
            faction_dynamics=["术士机构关注炼金术"],
            world_rules_and_hierarchy=["术士体系炼金规则"],
        ),
    )
    mock_llm.generate_structured = AsyncMock(return_value=mock_response)

    extractor = ChapterFactExtractor(mock_llm)
    batch_input = [
        {"index": 1, "title": "第一章 獄中蘇醒", "text": "內文一"},
        {"index": 2, "title": "第二章 煉金破案", "text": "內文二"},
    ]
    anchor = "前一組結尾：家族遭誣陷全族收押。"

    res = await extractor.extract_batched_chapter_facts(
        chapters=batch_input,
        prior_anchor_event=anchor,
    )

    assert mock_llm.generate_structured.call_count == 1
    call_args = mock_llm.generate_structured.call_args
    user_prompt = call_args[0][1]
    assert "前組終局錨點" in user_prompt
    assert anchor in user_prompt

    assert "獄中蘇醒" in res.chapters[0].chapter_title
    assert "脫困之策" in res.chapters[0].core_event
    assert "主角化解死刑危機" in res.entity_deltas.character_milestones[0]
    assert "術士機構關注煉金術" in res.entity_deltas.faction_dynamics[0]


@pytest.mark.asyncio
async def test_consolidate_major_arcs_mocked():
    """Verify consolidate_major_arcs partitions volumes into 6 arcs and calls LLM per arc."""
    mock_llm = MagicMock()
    mock_llm.server_manager.is_healthy.return_value = True

    volume_reports = [
        VolumeFactConsolidationContract(
            volume_index=i,
            volume_title=f"第{i}卷",
            volume_arc_summary=f"第{i}卷劇情走勢說明",
            core_conflict_resolution=f"第{i}卷矛盾化解",
            character_milestones=[f"角色第{i}階段突破"],
            faction_geopolitics_summary=f"勢力第{i}階段格局",
            world_rules_and_hierarchy_summary=f"法則第{i}階段規則",
        )
        for i in range(1, 13)
    ]

    def make_mock_arc(sys_prompt, user_prompt, contract_type, **kwargs):
        return MajorArcConsolidationContract(
            arc_index=1,
            covered_volumes="",
            arc_title="情節大幕標題",
            arc_summary="大幕情節走勢總結",
            conflict_and_resolution="大幕核心衝突收束",
            character_milestones_summary=["核心突破"],
            faction_geopolitics_summary="勢力平衡",
            world_rules_and_hierarchy_summary="境界法則",
        )

    mock_llm.generate_structured = AsyncMock(side_effect=make_mock_arc)

    extractor = ChapterFactExtractor(mock_llm)
    arcs = await extractor.consolidate_major_arcs(volume_reports, num_arcs=6)

    assert len(arcs) == 6
    assert mock_llm.generate_structured.call_count == 6
    assert arcs[0].arc_index == 1
    assert arcs[0].covered_volumes == "第 1 卷至第 2 卷"
    assert arcs[5].arc_index == 6
    assert arcs[5].covered_volumes == "第 11 卷至第 12 卷"


@pytest.mark.asyncio
async def test_consolidate_major_arcs_adaptive_short_and_auto():
    """Verify adaptive major arc consolidation skips short novels (<=4 vols) and auto-computes arcs for longer novels."""
    mock_llm = MagicMock()
    mock_llm.server_manager.is_healthy.return_value = True

    extractor = ChapterFactExtractor(mock_llm)

    short_reports = [
        VolumeFactConsolidationContract(
            volume_index=i,
            volume_title=f"第{i}卷",
            volume_arc_summary="短篇走勢",
            core_conflict_resolution="矛盾解決",
            character_milestones=[],
            faction_geopolitics_summary="勢力",
            world_rules_and_hierarchy_summary="規則",
        )
        for i in range(1, 4)
    ]
    short_arcs = await extractor.consolidate_major_arcs(short_reports, num_arcs=None)
    assert short_arcs == []
    assert mock_llm.generate_structured.call_count == 0

    def make_mock_arc(sys_prompt, user_prompt, contract_type, **kwargs):
        return MajorArcConsolidationContract(
            arc_index=1,
            covered_volumes="",
            arc_title="自動大幕",
            arc_summary="大幕走勢",
            conflict_and_resolution="矛盾收束",
            character_milestones_summary=[],
            faction_geopolitics_summary="勢力平衡",
            world_rules_and_hierarchy_summary="體系規則",
        )

    mock_llm.generate_structured = AsyncMock(side_effect=make_mock_arc)
    med_reports = [
        VolumeFactConsolidationContract(
            volume_index=i,
            volume_title=f"第{i}卷",
            volume_arc_summary="中篇走勢",
            core_conflict_resolution="矛盾解決",
            character_milestones=[],
            faction_geopolitics_summary="勢力",
            world_rules_and_hierarchy_summary="規則",
        )
        for i in range(1, 11)
    ]
    med_arcs = await extractor.consolidate_major_arcs(med_reports, num_arcs=None)
    assert len(med_arcs) == 2
    assert mock_llm.generate_structured.call_count == 2


def test_vol_deltas_precise_intersection_variable_batches():
    """Verify slicing entity deltas by chapter intersection works with variable-sized dynamic batches."""
    batches = [
        [{"index": 1}, {"index": 2}, {"index": 3}],
        [{"index": 4}],
        [{"index": 5}, {"index": 6}, {"index": 7}],
    ]
    deltas = [
        GroupEntityDeltas(character_milestones=["Delta_Batch_0"]),
        GroupEntityDeltas(character_milestones=["Delta_Batch_1"]),
        GroupEntityDeltas(character_milestones=["Delta_Batch_2"]),
    ]

    first_ch_a, last_ch_a = 1, 4
    vol_a_deltas = []
    for b_i, delta in enumerate(deltas):
        b_ch_start = batches[b_i][0]["index"]
        b_ch_end = batches[b_i][-1]["index"]
        if not (b_ch_end < first_ch_a or b_ch_start > last_ch_a):
            vol_a_deltas.append(delta)

    assert len(vol_a_deltas) == 2
    assert vol_a_deltas[0].character_milestones == ["Delta_Batch_0"]
    assert vol_a_deltas[1].character_milestones == ["Delta_Batch_1"]

    first_ch_b, last_ch_b = 5, 7
    vol_b_deltas = []
    for b_i, delta in enumerate(deltas):
        b_ch_start = batches[b_i][0]["index"]
        b_ch_end = batches[b_i][-1]["index"]
        if not (b_ch_end < first_ch_b or b_ch_start > last_ch_b):
            vol_b_deltas.append(delta)

    assert len(vol_b_deltas) == 1
    assert vol_b_deltas[0].character_milestones == ["Delta_Batch_2"]


@pytest.mark.asyncio
async def test_synthesize_macro_blueprint_neutral_genre():
    """Verify synthesize_macro_blueprint supports non-fantasy genres and empty hierarchy rules."""
    mock_llm = MagicMock()
    mock_llm.server_manager.is_healthy.return_value = True

    mock_blueprint = BookBlueprintSynthesisContract(
        novel_title="大江東去",
        genre_primary="歷史",
        genre_sub="改革開放",
        macro_outline="時代浪潮下三位主角在國企、鄉鎮企業與外企中的命運起伏與時代搏擊。",
        character_network=[
            CharacterArchetypeContract(
                character_name="宋運輝",
                archetype_role="技術理想開拓者",
                faction_name="金州化工國企",
                background_and_motive="憑藉高考改變命運的技術官僚。",
            ),
            CharacterArchetypeContract(
                character_name="雷東寶",
                archetype_role="鄉土改革行動者",
                faction_name="小雷家村集體",
                background_and_motive="雷厲風行的農村改革領頭人。",
            ),
        ],
        faction_geopolitics=[
            FactionGeopoliticsContract(
                faction_name="金州化工廠",
                doctrine_and_nature="傳統重工業國企體制，派系林立。",
                power_tier="大型國有骨幹企業",
            ),
        ],
        hierarchy_rules=[],
        world_laws=["時代政策變遷決定行業生死", "集體制度與市場經濟的法律與倫理碰撞"],
    )
    mock_llm.generate_structured = AsyncMock(return_value=mock_blueprint)

    extractor = ChapterFactExtractor(mock_llm)
    res = await extractor.synthesize_macro_blueprint(
        novel_title="大江東去",
        volume_reports=[
            VolumeFactConsolidationContract(
                volume_index=1,
                volume_title="第一卷",
                volume_arc_summary="宋運輝考入大學，雷東寶退伍回村。",
                core_conflict_resolution="貧困枷鎖打破",
                character_milestones=["高考破局"],
                faction_geopolitics_summary="國企與農村體制分化",
                world_rules_and_hierarchy_summary="改革開放初期政策規則",
            )
        ],
        chapter_facts=[],
        genre_primary="歷史",
        genre_sub="改革開放",
        major_arcs=None,
    )

    assert res.novel_title == "大江東去"
    assert res.genre_primary == "歷史"
    assert res.hierarchy_rules == []
    assert len(res.world_laws) == 2

    call_args = mock_llm.generate_structured.call_args
    user_prompt = call_args[0][1]
    assert "全書各卷事實收斂匯報" in user_prompt
    assert "位階與法則" in user_prompt
    assert "修煉與法則" not in user_prompt


def test_reconcile_batched_compact_facts_strict_validation_no_fabrication():
    """Verify reconcile_batched_compact_facts enforces strict 1:1 count without code merging or fabrication."""
    input_chapters = [
        {"index": 105, "title": "第一百零五章 京城風雲", "text": "內文A"},
        {"index": 106, "title": "第一百零六章 暗流湧動", "text": "內文B"},
    ]

    # Scenario 1: Mismatched count (3 items vs 2 chapters) MUST raise ValueError instead of code merging/fabricating
    mismatched = [
        SingleChapterCompactFact(
            chapter_index=105,
            chapter_title="第一百零五章（上）",
            core_event="事件一",
            actual_tension=3,
            actual_progress=3,
        ),
        SingleChapterCompactFact(
            chapter_index=105,
            chapter_title="第一百零五章（下）",
            core_event="事件二",
            actual_tension=5,
            actual_progress=4,
        ),
        SingleChapterCompactFact(
            chapter_index=106,
            chapter_title="第一百零六章 暗流湧動",
            core_event="事件三",
            actual_tension=4,
            actual_progress=3,
        ),
    ]
    with pytest.raises(ValueError, match="Batched chapter count mismatch"):
        reconcile_batched_compact_facts(mismatched, input_chapters)

    # Scenario 2: Exact 1:1 count preserves full core_event text without truncation
    long_event = "主角深入敵營查明真相，揭開多方勢力暗中勾結的完整因果鏈條。" * 5
    valid_extracted = [
        SingleChapterCompactFact(
            chapter_index=1,
            chapter_title="第一章",
            core_event=long_event,
            pacing_phase="CLIMAX",
            actual_tension=5,
            actual_progress=4,
            score_causal=4.5,
            score_craft=4.0,
            score_depth=4.0,
        ),
        SingleChapterCompactFact(
            chapter_index=2,
            chapter_title="第二章",
            core_event="事件乙",
            pacing_phase="RESOLUTION",
            actual_tension=2,
            actual_progress=2,
            score_causal=3.5,
            score_craft=3.5,
            score_depth=3.5,
        ),
    ]
    reconciled = reconcile_batched_compact_facts(valid_extracted, input_chapters)
    assert len(reconciled) == 2
    assert reconciled[0].chapter_index == 105
    assert reconciled[1].chapter_index == 106
    assert reconciled[0].core_event == long_event
    assert reconciled[0].pacing_phase == "CLIMAX"
    assert reconciled[1].pacing_phase == "RESOLUTION"


@pytest.mark.asyncio
async def test_extract_batched_chapter_facts_fallback_on_count_mismatch():
    """Verify extract_batched_chapter_facts falls back to individual AI chapter extraction when batch count mismatches."""
    mock_llm = MagicMock()
    mock_llm.server_manager.is_healthy.return_value = True

    bad_batch_response = BatchedFactExtractionContract(
        chapters=[
            SingleChapterCompactFact(
                chapter_index=201,
                chapter_title="第二百零一章",
                core_event="僅回傳一章",
                actual_tension=3,
                actual_progress=3,
            )
        ],
        entity_deltas=GroupEntityDeltas(),
    )

    def side_effect(sys_prompt, user_prompt, contract_type, **kwargs):
        if contract_type == BatchedFactExtractionContract:
            return bad_batch_response
        if contract_type == ChapterFactExtractContract:
            idx = 201 if "201" in sys_prompt else 202
            return ChapterFactExtractContract(
                chapter_index=idx,
                chapter_title=f"第{idx}章 獨立重抽",
                core_event=f"第{idx}章由AI單章完整重抽之核心事件",
                pacing_phase="COMPLICATION",
                character_dynamics=[f"角色狀態變化{idx}"],
                faction_shifts=[f"勢力動向{idx}"],
                actual_tension=4,
                actual_progress=4,
            )
        raise ValueError(f"Unexpected contract: {contract_type}")

    mock_llm.generate_structured = AsyncMock(side_effect=side_effect)

    extractor = ChapterFactExtractor(mock_llm)
    batch_input = [
        {"index": 201, "title": "第二百零一章 查案", "text": "內文201"},
        {"index": 202, "title": "第二百零二章 決斷", "text": "內文202"},
    ]

    res = await extractor.extract_batched_chapter_facts(batch_input)
    assert len(res.chapters) == 2
    assert res.chapters[0].chapter_index == 201
    assert res.chapters[1].chapter_index == 202
    assert "由AI單章完整重抽" in res.chapters[0].core_event
    assert "由AI單章完整重抽" in res.chapters[1].core_event


@pytest.mark.asyncio
async def test_extract_batched_chapter_facts_single_chapter_fallback_and_long_core_event():
    """Verify 1-chapter batches also fall back cleanly on count mismatch and preserve >200-char core_event."""
    mock_llm = MagicMock()
    mock_llm.server_manager.is_healthy.return_value = True

    over_split_response = BatchedFactExtractionContract(
        chapters=[
            SingleChapterCompactFact(
                chapter_index=770,
                chapter_title="第1079章（前半）",
                core_event="前半場事件",
                actual_tension=3,
                actual_progress=3,
            ),
            SingleChapterCompactFact(
                chapter_index=771,
                chapter_title="第1079章（後半）",
                core_event="後半場事件",
                actual_tension=4,
                actual_progress=3,
            ),
        ],
        entity_deltas=GroupEntityDeltas(),
    )
    long_summary = "主角於滄州暗中佈局扶危濟困並整合流民與邊軍舊部，" * 10  # 240 chars (> 200, <= 800)
    assert 200 < len(long_summary) <= 800

    def side_effect(sys_prompt, user_prompt, contract_type, **kwargs):
        if contract_type == BatchedFactExtractionContract:
            return over_split_response
        if contract_type == ChapterFactExtractContract:
            return ChapterFactExtractContract(
                chapter_index=770,
                chapter_title="第1079章 天命之人",
                core_event=long_summary,
                pacing_phase="COMPLICATION",
                character_dynamics=["主角確立根據地"],
                faction_shifts=["滄州地方勢力重組"],
                actual_tension=4,
                actual_progress=4,
            )
        raise ValueError(f"Unexpected contract: {contract_type}")

    mock_llm.generate_structured = AsyncMock(side_effect=side_effect)

    extractor = ChapterFactExtractor(mock_llm)
    single_batch = [
        {"index": 770, "title": "第1079章 天命之人", "text": "單章正文內容"},
    ]

    res = await extractor.extract_batched_chapter_facts(single_batch)
    assert len(res.chapters) == 1
    assert res.chapters[0].chapter_index == 770
    assert res.chapters[0].core_event == long_summary


@pytest.mark.asyncio
async def test_synthesize_macro_blueprint_character_decoupling_and_acts_prompt():
    """Verify blueprint synthesis prompt mandates 4-act structural outline and general character decoupling without specific book exemplars."""
    mock_llm = MagicMock()
    mock_llm.server_manager.is_healthy.return_value = True

    mock_blueprint = BookBlueprintSynthesisContract(
        novel_title="測試長篇",
        genre_primary="仙俠",
        genre_sub="懸疑",
        macro_outline="【第一幕·開端破局】開局破案。\n【第二幕·深入博弈】中段風雲。\n【第三幕·高潮相變】終極決戰。\n【第四幕·終局收束】大劫平定。",
        character_network=[
            CharacterArchetypeContract(
                character_name="主角甲",
                archetype_role="核心破局行動者",
                faction_name="執法機構",
                background_and_motive="身負特殊使命的調查者。",
            ),
            CharacterArchetypeContract(
                character_name="統領乙",
                archetype_role="戰略佈局引路者",
                faction_name="中樞機構",
                background_and_motive="統籌全局的上位佈局者。",
            ),
        ],
        faction_geopolitics=[],
        hierarchy_rules=[],
        world_laws=["力量施展伴隨對等代價"],
    )
    mock_llm.generate_structured = AsyncMock(return_value=mock_blueprint)

    extractor = ChapterFactExtractor(mock_llm)
    res = await extractor.synthesize_macro_blueprint(
        novel_title="測試長篇",
        volume_reports=[],
        chapter_facts=[],
        genre_primary="仙俠",
        genre_sub="懸疑",
    )

    call_args = mock_llm.generate_structured.call_args
    sys_prompt = call_args[0][0]

    assert "500～800 字" in sys_prompt
    assert "【第一幕·開端破局】" in sys_prompt
    assert "【第二幕·深入博弈】" in sys_prompt
    assert "【第三幕·高潮相變】" in sys_prompt
    assert "【第四幕·終局收束】" in sys_prompt
    assert "嚴格區分角色稱謂與基礎定位" in sys_prompt
    # Verify Iron Rule 4: No specific novel character names in system prompt
    assert "許七安" not in sys_prompt
    assert "魏淵" not in sys_prompt
    assert call_args[1]["max_tokens"] == 4096
    assert res.character_network[0].character_name == "主角甲"


@pytest.mark.asyncio
async def test_consolidate_major_arcs_two_tier_for_epic_novels():
    """Verify 78 volumes triggers two-tier tree reduction without context overflow."""
    mock_llm = MagicMock()
    mock_llm.server_manager.is_healthy.return_value = True

    volume_reports = [
        VolumeFactConsolidationContract(
            volume_index=i,
            volume_title=f"第{i}卷",
            volume_arc_summary=f"第{i}卷走勢",
            core_conflict_resolution=f"第{i}卷矛盾",
            character_milestones=[f"突破{i}"],
            faction_geopolitics_summary=f"勢力{i}",
            world_rules_and_hierarchy_summary=f"規則{i}",
        )
        for i in range(1, 79)
    ]

    call_prompts = []

    def make_mock_arc(sys_prompt, user_prompt, contract_type, **kwargs):
        call_prompts.append(user_prompt)
        return MajorArcConsolidationContract(
            arc_index=1,
            covered_volumes="",
            arc_title="情節大幕標題",
            arc_summary="大幕情節走勢總結",
            conflict_and_resolution="大幕核心衝突收束",
            character_milestones_summary=["核心突破"],
            faction_geopolitics_summary="勢力平衡",
            world_rules_and_hierarchy_summary="境界法則",
        )

    mock_llm.generate_structured = AsyncMock(side_effect=make_mock_arc)
    extractor = ChapterFactExtractor(mock_llm)

    arcs = await extractor.consolidate_major_arcs(volume_reports)

    assert len(arcs) == 5
    assert mock_llm.generate_structured.call_count == 18
    for p in call_prompts:
        assert p.count("=== 第") <= 6


@pytest.mark.asyncio
async def test_synthesize_macro_blueprint_auto_consolidates_when_missing_arcs():
    """Verify synthesize_macro_blueprint automatically triggers hierarchical reduction when major_arcs is omitted."""
    mock_llm = MagicMock()
    mock_llm.server_manager.is_healthy.return_value = True

    volume_reports = [
        VolumeFactConsolidationContract(
            volume_index=i,
            volume_title=f"第{i}卷",
            volume_arc_summary=f"第{i}卷走勢",
            core_conflict_resolution=f"第{i}卷矛盾",
            character_milestones=[f"突破{i}"],
            faction_geopolitics_summary=f"勢力{i}",
            world_rules_and_hierarchy_summary=f"規則{i}",
        )
        for i in range(1, 16)
    ]

    mock_blueprint = BookBlueprintSynthesisContract(
        novel_title="長篇測試",
        genre_primary="仙俠",
        genre_sub="通用",
        macro_outline="【第一幕·開端破局】測試\n【第二幕·深入博弈】測試\n【第三幕·高潮相變】測試\n【第四幕·終局收束】測試",
        character_network=[
            CharacterArchetypeContract(
                character_name="主角",
                archetype_role="命運破局者",
                faction_name="宗門",
                background_and_motive="追求大道",
            ),
            CharacterArchetypeContract(
                character_name="導師",
                archetype_role="領路護道人",
                faction_name="宗門",
                background_and_motive="指引主角成長",
            ),
        ],
        faction_geopolitics=[],
        hierarchy_rules=[],
        world_laws=[],
    )

    def side_effect(sys_prompt, user_prompt, contract_type, **kwargs):
        if contract_type == MajorArcConsolidationContract:
            return MajorArcConsolidationContract(
                arc_index=1,
                covered_volumes="",
                arc_title="自動整合大幕",
                arc_summary="大幕走勢",
                conflict_and_resolution="衝突解決",
                character_milestones_summary=[],
                faction_geopolitics_summary="勢力",
                world_rules_and_hierarchy_summary="法則",
            )
        return mock_blueprint

    mock_llm.generate_structured = AsyncMock(side_effect=side_effect)
    extractor = ChapterFactExtractor(mock_llm)

    bp = await extractor.synthesize_macro_blueprint(
        novel_title="長篇測試",
        volume_reports=volume_reports,
        chapter_facts=[],
        major_arcs=None,
    )

    assert bp.novel_title == "長篇測試"
    last_call = mock_llm.generate_structured.call_args_list[-1]
    blueprint_user_prompt = last_call[0][1]
    assert "大情節幕宏觀收斂匯報" in blueprint_user_prompt
    assert "自動整合大幕" in blueprint_user_prompt
    assert blueprint_user_prompt.count("=== 自動整合大幕") <= 6
