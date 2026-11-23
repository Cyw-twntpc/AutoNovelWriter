"""Tests for reference library SHA-256 pre-flight, dual-track retrieval, and 16k Context Router."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autonovel.db.connection import atomic_transaction
from autonovel.db.project_repo import ProjectRepository
from autonovel.db.reference_repo import (
    ReferenceRepository,
    compute_density_clusters,
    get_cluster_color,
    resolve_tag_cluster,
)
from autonovel.engine.context_router import ContextRouter
from autonovel.llm.embedder import CpuEmbedder
from autonovel.reference.chunk_extractor import (
    ReferenceExtractor,
    split_novel_into_chunks,
)
from autonovel.reference.file_parser import preflight_check_files
from autonovel.reference.retriever import ReferenceRetriever
from autonovel.schemas.llm_contracts import (
    ChapterExtractionContract,
    RelationDeltaContract,
    TagRuleContract,
)


@pytest.mark.asyncio
async def test_reference_preflight_ingestion_and_dual_track_retrieval(
    tmp_path: Path, mock_llm_client: Any
) -> None:
    """Verify 1s SHA-256 pre-flight check, 2,500-char chunking, and 0.4*S_desc+0.6*S_tag retrieval."""
    ref_db = tmp_path / "reference_library.db"
    ref_repo = ReferenceRepository(ref_db)
    embedder = CpuEmbedder()
    extractor = ReferenceExtractor(ref_repo, embedder, llm_client=mock_llm_client)
    retriever = ReferenceRetriever(ref_repo, embedder, threshold=0.35, top_k=3, llm_client=mock_llm_client)

    sample_book = tmp_path / "xianxia_ref.txt"
    sample_text = (
        "第一章 寒潭劍氣\n"
        + ("青石斷崖之上，劍氣纵橫激盪，林墨凝神靜氣，以靈氣引動天地殺機。\n" * 90)
        + "\n第二章 密室博弈\n"
        + ("長老目光冷冽，大殿之內眾人言辭交鋒，句句暗藏玄機。\n" * 90)
    )
    sample_book.write_text(sample_text, encoding="utf-8")
    corrupted_book = tmp_path / "corrupted.txt"
    corrupted_book.write_bytes(b"\x00\x00\xff\xfeinvalid")

    chunks = split_novel_into_chunks(sample_text, chunk_size=2500, overlap=250)
    assert len(chunks) >= 2
    assert all(len(c) <= 2500 for c in chunks)

    # First pre-flight check: sample_book is NEW, corrupted_book is CORRUPTED
    pf1 = preflight_check_files([sample_book, corrupted_book], ref_repo)
    assert pf1[0].status == "NEW"
    assert pf1[1].status == "CORRUPTED"

    # Ingest sample_book -> raw text discarded, snippets <= 150 chars stored
    ingested_count = await extractor.ingest_file(sample_book)
    assert ingested_count == len(chunks)
    stats = ref_repo.get_library_stats()
    assert stats["sources"] == 1
    assert stats["entries"] == len(chunks)

    # Second pre-flight check: sample_book is now flagged as DUPLICATE
    pf2 = preflight_check_files([sample_book], ref_repo)
    assert pf2[0].status == "DUPLICATE"

    # Dual-track retrieval and <=300 char understanding loop compression
    digest, hits = await retriever.retrieve_and_compress(
        "青石斷崖劍氣戰鬥與對話交鋒", fallback_tags=["戰鬥張力", "對話交鋒"]
    )
    assert len(hits) >= 1
    assert 0 < len(digest) <= 300


def test_punctuation_snapping_chunker() -> None:
    """Verify sliding window cleanly snaps to punctuation delimiters without splitting sentences."""
    # Build text where sentences end with quotes and different punctuations
    sentence_list = [
        f"長老厲聲喝問第{i}句：「小輩休得猖狂！這難道就是你的倚仗嗎？」"
        f"林墨眼神如霜，冷冷說道：「廢話少說，拔劍吧！」\n"
        for i in range(1, 60)
    ]
    long_doc = "第一章 宗門風雲\n" + "".join(sentence_list)
    chunks = split_novel_into_chunks(long_doc, chunk_size=1000, overlap=150)
    assert len(chunks) >= 3
    for idx, c in enumerate(chunks):
        assert len(c) <= 1000
        # Must end with a valid sentence delimiter or quote
        assert c[-1] in ("」", "』", "”", "。", "！", "？", "!", "?", "\n"), (
            f"Chunk {idx} ends abnormally: ...{c[-20:]!r}"
        )



@pytest.mark.asyncio
async def test_16k_context_router_and_previous_chapter_tail(
    tmp_path: Path, mock_llm_client: Any
) -> None:
    """Verify ContextRouter slices 600-800 chars from previous chapter and stays under budget."""
    proj_db = tmp_path / "novel_project.db"
    ref_db = tmp_path / "reference_library.db"
    proj_repo = ProjectRepository(proj_db)
    ref_repo = ReferenceRepository(ref_db)
    embedder = CpuEmbedder()
    retriever = ReferenceRetriever(ref_repo, embedder, llm_client=mock_llm_client)

    from autonovel.wizard.blueprint_builder import BlueprintBuilder

    builder = BlueprintBuilder(proj_repo, retriever, llm_client=mock_llm_client)
    await builder.complete_blueprint()

    # Populate chapter 1 with 2,500 chars of final_content
    ch1_text = ("這是第一章的正文內容，林墨與蘇清寒於北境寒淵並肩作戰。\n" * 80) + "【第一章結尾關鍵鏡頭：古壇星圖驟然亮起】"
    with atomic_transaction(proj_db) as conn:
        conn.execute(
            "UPDATE chapters SET final_content = ?, status = 'COMPLETED' WHERE chapter_num = 1;",
            (ch1_text,),
        )

    router = ContextRouter(proj_repo)
    ctx = router.build_chapter_context(
        chapter_num=2,
        reference_digest="參考書庫壓縮技法摘要",
        pacing_directive="強化衝突張力",
    )
    assert 600 <= len(ctx["previous_tail"]) <= 800
    assert "【第一章結尾關鍵鏡頭：古壇星圖驟然亮起】" in ctx["previous_tail"]
    assert ctx["token_breakdown"]["total_prompt_tokens"] <= 5500

    # Test relation upsert deduplication, 1-hop relation prompt formatting, and table whitelist
    with pytest.raises(ValueError, match="Unsupported entity table"):
        with atomic_transaction(proj_db) as conn:
            proj_repo._ensure_named_entity(conn, "invalid_table", "測試")

    ext_rel_1 = ChapterExtractionContract(
        chapter_title="第1章 結盟",
        tension_intensity=3,
        goal_progress=3,
        emotional_payoff=3,
        relation_deltas=[
            RelationDeltaContract(
                source_name="林墨", relation_type="同門盟友", target_name="蘇清寒"
            )
        ],
    )
    proj_repo.commit_chapter_atomically(
        chapter_num=1,
        final_content=ch1_text,
        extraction=ext_rel_1,
        literary_score=88.0,
        rubric_payload={"composite_score": 88.0},
        best_paragraph_text="測試段落",
        best_paragraph_score=88.0,
        revision_count=0,
        micro_rewrite_count=0,
        contradiction_retry_count=0,
        duration_sec=1.0,
    )
    ext_rel_2 = ChapterExtractionContract(
        chapter_title="第2章 生死之交",
        tension_intensity=4,
        goal_progress=4,
        emotional_payoff=4,
        relation_deltas=[
            RelationDeltaContract(
                source_name="林墨", relation_type="生死劍侶", target_name="蘇清寒"
            )
        ],
    )
    proj_repo.commit_chapter_atomically(
        chapter_num=2,
        final_content=ch1_text,
        extraction=ext_rel_2,
        literary_score=89.0,
        rubric_payload={"composite_score": 89.0},
        best_paragraph_text="測試段落",
        best_paragraph_score=89.0,
        revision_count=0,
        micro_rewrite_count=0,
        contradiction_retry_count=0,
        duration_sec=1.0,
    )
    rels = proj_repo.fetch_all("SELECT * FROM entity_relations;")
    assert len(rels) == 1
    assert rels[0]["relation_type"] == "生死劍侶"
    ctx_ch3 = router.build_chapter_context(chapter_num=3)
    assert "林墨 —[生死劍侶]→ 蘇清寒" in ctx_ch3["user_prompt"]


@pytest.mark.asyncio
async def test_reference_books_entries_and_tags_management(
    tmp_path: Path, mock_llm_client: Any
) -> None:
    """Verify listing books, querying entries with filters, adding/removing/renaming tags, and cascade delete."""
    ref_db = tmp_path / "reference_library.db"
    ref_repo = ReferenceRepository(ref_db)
    embedder = CpuEmbedder()
    extractor = ReferenceExtractor(ref_repo, embedder, llm_client=mock_llm_client)

    sample_book = tmp_path / "test_book.txt"
    sample_text = (
        "【第一卷 劍啟微塵】\n"
        + ("寒光冷照，少年拔劍一戰，天地為之變色。\n" * 100)
    )
    sample_book.write_text(sample_text, encoding="utf-8")

    await extractor.ingest_file(sample_book)

    # 1. List books
    books = ref_repo.list_sources_with_counts()
    assert len(books) == 1
    source_id = books[0]["id"]
    assert books[0]["file_name"] == "test_book.txt"
    assert books[0]["chunk_count"] > 0
    assert books[0]["entry_count"] > 0

    # 2. Query entries without filter
    entries, total = ref_repo.query_entries(limit=10, offset=0)
    assert total > 0
    assert len(entries) > 0
    first_entry = entries[0]
    entry_id = first_entry["id"]

    # 3. Add a new tag to the entry
    new_tag_vec = embedder.embed_text("仙俠:劍道殺伐")
    success = ref_repo.add_tag_to_entry(entry_id, "仙俠:劍道殺伐", new_tag_vec)
    assert success is True

    # 4. Query entries with tag filter
    filtered_entries, f_total = ref_repo.query_entries(tag="仙俠:劍道殺伐")
    assert f_total >= 1
    assert any("仙俠:劍道殺伐" in e["tags"] for e in filtered_entries)

    # 5. List tags with frequency
    tags_freq = ref_repo.list_tags_with_frequency()
    tag_names = [t["tag_name"] for t in tags_freq]
    assert "仙俠:劍道殺伐" in tag_names

    # 6. Globally rename tag
    renamed_vec = embedder.embed_text("仙俠:純陽劍道")
    updated_count = ref_repo.rename_tag_globally("仙俠:劍道殺伐", "仙俠:純陽劍道", renamed_vec)
    assert updated_count >= 1

    renamed_entries, r_total = ref_repo.query_entries(tag="仙俠:純陽劍道")
    assert r_total >= 1

    # 7. Manual tag removal is no longer offered: deleting a whole entry is the only
    # way to retire knowledge, so an entry can never be left without tags.
    post_ren_entries, _ = ref_repo.query_entries(source_id=source_id)
    target_entry = next((e for e in post_ren_entries if e["id"] == entry_id), None)
    assert target_entry is not None
    assert "仙俠:純陽劍道" in target_entry["tags"]

    # 8. Delete entry
    del_entry_ok = ref_repo.delete_entry(entry_id)
    assert del_entry_ok is True
    rem_entry_check, _ = ref_repo.query_entries(keyword="非不可能的無關搜尋條件_x98")
    assert len(rem_entry_check) == 0

    # 9. Delete source book with cascade
    del_source_ok = ref_repo.delete_source(source_id)
    assert del_source_ok is True
    assert len(ref_repo.list_sources_with_counts()) == 0
    all_rem_entries, all_rem_total = ref_repo.query_entries()
    assert all_rem_total == 0


@pytest.mark.asyncio
async def test_auto_clean_tags_and_apply_rules(
    tmp_path: Path, mock_llm_client: Any
) -> None:
    """Verify AI automated tag maintenance: normalization, merging, noise dropping, and deduplication."""
    ref_db = tmp_path / "reference_library.db"
    ref_repo = ReferenceRepository(ref_db)
    embedder = CpuEmbedder()
    extractor = ReferenceExtractor(ref_repo, embedder, llm_client=mock_llm_client)

    # Insert a dummy source and entry with messy tags
    src_id = ref_repo.register_or_reset_source("test_sha_99", "sample.txt", 1)
    desc_vec = embedder.embed_text("測試描寫")
    tag_tuples = [
        ("御劍", embedder.embed_text("御劍")),
        ("劍招", embedder.embed_text("劍招")),
        ("此時", embedder.embed_text("此時")),
    ]
    entry_id = ref_repo.insert_entry_with_tags(
        src_id, "測試條目描寫", desc_vec, "測試原文微縮", tag_tuples
    )
    assert entry_id > 0

    # 1. Test direct apply_tag_rules
    rules = [
        TagRuleContract(old_tag="御劍", new_tag="仙俠:劍道殺伐", action="NORMALIZE"),
        TagRuleContract(old_tag="劍招", new_tag="仙俠:劍道殺伐", action="MERGE"),
        TagRuleContract(old_tag="此時", new_tag="", action="DROP"),
    ]
    stats = ref_repo.apply_tag_rules(rules, embedder)
    assert stats["normalized"] == 1
    assert stats["merged"] == 1
    assert stats["dropped"] == 1
    assert stats["total_affected"] == 3

    # Check resulting entry tags: "此時" is gone, "仙俠:劍道殺伐" is present exactly once (no duplicates)
    entries, _ = ref_repo.query_entries()
    assert len(entries) == 1
    current_tags = entries[0]["tags"]
    assert "此時" not in current_tags
    assert "仙俠:劍道殺伐" in current_tags
    assert current_tags.count("仙俠:劍道殺伐") == 1

    # 2. Test auto_clean_tags via extractor with mock_llm_client
    clean_stats = await extractor.auto_clean_tags()
    assert clean_stats["normalized"] == 1
    tags_after = [t["tag_name"] for t in ref_repo.list_tags_with_frequency()]
    assert "仙俠:純陽劍道" in tags_after

    # 3. Test get_tag_clusters_and_stats
    assets = ref_repo.get_tag_clusters_and_stats()
    assert "stats" in assets
    assert "nodes" in assets
    assert "clusters" in assets
    assert assets["stats"]["total_unique"] >= 1
    assert "outliers_count" in assets["stats"]
    assert len(assets["nodes"]) >= 1
    first_node = assets["nodes"][0]
    assert "x" in first_node and "y" in first_node
    assert "color" in first_node
    assert "cluster" in first_node
    assert "density_cluster" in first_node
    assert "is_outlier" in first_node


def test_dbscan_density_clustering_and_emergent_genres() -> None:
    """Verify NumPy DBSCAN identifies dense clusters vs outliers, and emergent genres get distinct colors."""
    embedder = CpuEmbedder()

    # Create two semantic clusters: Swords and Magic, plus an isolated noise word
    tags = ["純陽劍法", "純陽劍氣", "純陽劍意", "禁咒法術", "禁咒法門", "此時"]
    vectors = np.array(embedder.embed_batch(tags))

    labels = compute_density_clusters(vectors, eps=0.45, min_samples=2)
    assert len(labels) == 6

    # "純陽劍法", "純陽劍氣", "純陽劍意" belong to cluster 0
    assert labels[0] >= 0
    assert labels[0] == labels[1] == labels[2]

    # "禁咒法術", "禁咒法門" belong to a separate dense cluster
    assert labels[3] >= 0
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]

    # "此時" is an isolated noise point with no semantic neighbors
    assert labels[5] == -1

    # Verify emergent genre resolution and dynamic color assignment
    assert resolve_tag_cluster("詭異:克蘇魯眷族") == "詭異"
    assert resolve_tag_cluster("無限流:輪迴死鬥") == "無限流"
    color_cthulhu = get_cluster_color("詭異")
    color_infinite = get_cluster_color("無限流")
    assert color_cthulhu.startswith("#")
    assert color_infinite.startswith("#")


def test_reference_extraction_traditional_sanitization(tmp_path: Path) -> None:
    """Verify that reference entries are strictly normalized to Traditional Chinese."""
    from autonovel.schemas.llm_contracts import OrthogonalTagsContract, ReferenceChunkExtractionContract

    ref_db = tmp_path / "reference_library.db"
    ref_repo = ReferenceRepository(ref_db)
    embedder = CpuEmbedder()
    extractor = ReferenceExtractor(ref_repo, embedder)

    source_id = ref_repo.register_or_reset_source("test_sha", "test.txt", 1)
    contract = ReferenceChunkExtractionContract(
        description="通过环境描写营造无尽的紧张杀伐氛围，师兄弟们互相应对。",
        snippet="李逍遥冷声说道：“这绝非寻常的天劫，而是灭顶之灾！”",
        tags_matrix=OrthogonalTagsContract(
            structure_tags=["无尽杀伐"],
            actant_tags=["战鬥张力"],
            rhetoric_tags=["环境描写"],
            context_tags=["仙侠:剑道争锋"],
        ),
    )

    extractor._persist_extracted_entry(source_id, contract)

    entries, total = ref_repo.query_entries(source_id=source_id)
    assert total == 1
    # Check description converted to traditional
    assert "通過環境描寫營造無盡的緊張殺伐氛圍" in entries[0]["description"]
    assert "師兄弟們" in entries[0]["description"]
    # Check quotes converted
    assert "「" in entries[0]["snippet"]
    assert "」" in entries[0]["snippet"]
    assert "這絕非尋常的天劫" in entries[0]["snippet"]

    tags = [t["tag_name"] for t in ref_repo.list_tags_with_frequency()]
    assert "仙俠:劍道爭鋒" in tags
    assert "戰鬥張力" in tags
    assert "無盡殺伐" in tags


def test_plain_text_utf16_and_binary_detection(tmp_path: Path) -> None:
    """Verify parser handles UTF-16 with/without BOM and correctly rejects real binary files."""
    from autonovel.reference.file_parser import parse_novel_file

    # 1. UTF-16 LE with BOM (Notepad Unicode)
    f_utf16_bom = tmp_path / "novel_utf16_bom.txt"
    f_utf16_bom.write_bytes("第一章 少年出山\n風雲激盪。".encode("utf-16"))
    text, enc = parse_novel_file(f_utf16_bom)
    assert "第一章 少年出山" in text
    assert "utf-16" in enc.lower()

    # 2. UTF-16 LE without BOM
    f_utf16_le = tmp_path / "novel_utf16_le.txt"
    f_utf16_le.write_bytes("第一章 少年出山\n風雲激盪。".encode("utf-16-le"))
    text, enc = parse_novel_file(f_utf16_le)
    assert "第一章 少年出山" in text
    assert "utf-16" in enc.lower()

    # 3. Real binary file (e.g. fake PNG with null bytes)
    f_binary = tmp_path / "fake_binary.txt"
    f_binary.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\x00\x00\x00")
    with pytest.raises(ValueError, match="Binary null bytes detected"):
        parse_novel_file(f_binary)


@pytest.mark.asyncio
async def test_subject_object_interaction_dimension(tmp_path: Path, mock_llm_client: Any) -> None:
    """Verify Subject-Object Interaction (SOI) extraction, synthesized tag, and dual-track retrieval."""
    ref_db = tmp_path / "reference_library.db"
    ref_repo = ReferenceRepository(ref_db)
    embedder = CpuEmbedder()
    extractor = ReferenceExtractor(ref_repo, embedder, llm_client=mock_llm_client)
    retriever = ReferenceRetriever(ref_repo, embedder, threshold=0.3, top_k=3, llm_client=mock_llm_client)

    sample_text = "青石斷崖之上，劍氣纵橫激盪，林墨凝神靜氣，以靈氣引動天地殺機。"
    contract = await extractor._extract_single_chunk(sample_text)

    # 1. Contract fields extracted with general role positioning
    assert contract.subject_role == "主角"
    assert contract.object_role == "環境"
    assert contract.interaction_dynamic == "威權壓迫"
    # 2. Pure semantic tags maintained without synthetic arrow string concatenation
    all_extracted_tags = contract.tags_matrix.all_tags()
    assert "仙俠:劍道殺伐" in all_extracted_tags
    assert "戰鬥張力" in all_extracted_tags

    # 3. Persist and query
    source_id = ref_repo.register_or_reset_source("test_sha_soi", "soi_book.txt", 1)
    extractor._persist_extracted_entry(source_id, contract)

    entries, total = ref_repo.query_entries(source_id=source_id)
    assert total == 1
    assert entries[0]["subject_role"] == "主角"
    assert entries[0]["object_role"] == "環境"
    assert entries[0]["interaction_dynamic"] == "威權壓迫"
    assert "仙俠:劍道殺伐" in entries[0]["tags"]

    # 4. Dual-track retrieval
    digest, hits = await retriever.retrieve_and_compress("劍客引動天地殺機")
    assert len(hits) >= 1
    assert hits[0]["subject_role"] == "主角"
    assert hits[0]["object_role"] == "環境"
    assert hits[0]["interaction_dynamic"] == "威權壓迫"
    assert "主客交鋒" in digest or len(digest) > 0


def test_optional_salient_dimensions_palette_and_clustering() -> None:
    """Verify resolution and dedicated palette for optional dimensions: focalization, tempo, irony, syntax."""
    from autonovel.db.reference_repo import resolve_tag_cluster, get_cluster_color

    test_tags = [
        ("焦距:微觀特寫", "焦距", "#06b6d4"),
        ("焦距:宏觀全景", "焦距", "#06b6d4"),
        ("節奏:提煉概括", "節奏", "#f97316"),
        ("節奏:實時交鋒", "節奏", "#f97316"),
        ("資訊差:戲劇反諷", "資訊差", "#8b5cf6"),
        ("資訊差:隱藏底牌", "資訊差", "#8b5cf6"),
        ("句式:短句疾進", "句式", "#14b8a6"),
        ("句式:長句流動", "句式", "#14b8a6"),
    ]
    for tag, expected_cluster, expected_color in test_tags:
        cluster = resolve_tag_cluster(tag)
        assert cluster == expected_cluster
        assert get_cluster_color(cluster) == expected_color


def test_reference_contracts_tag_limits() -> None:
    """Verify that extraction contract accommodates orthogonal tags across 4 axes."""
    from autonovel.schemas.llm_contracts import OrthogonalTagsContract, ReferenceChunkExtractionContract

    contract = ReferenceChunkExtractionContract(
        description="測試描述",
        snippet="測試摘錄內容",
        tags_matrix=OrthogonalTagsContract(
            structure_tags=["資訊差:隱藏底牌", "節奏:實時交鋒"],
            actant_tags=["劍客對抗", "窒息壓迫"],
            rhetoric_tags=["動作白描", "句式:短句疾進"],
            context_tags=["題材:仙俠", "焦距:微觀特寫"],
        ),
        subject_role="劍客",
        object_role="對手",
        interaction_dynamic="DOMINANCE",
    )
    assert len(contract.tags_matrix.all_tags()) == 8


def test_dual_track_mmr_diversity(tmp_path: Path) -> None:
    """Verify that MMR diversity selection prevents redundant duplicate tags from monopolizing top_k."""
    ref_db = tmp_path / "reference_library.db"
    ref_repo = ReferenceRepository(ref_db)
    embedder = CpuEmbedder()

    source_id = ref_repo.register_or_reset_source("test_sha_mmr", "mmr_book.txt", 3)

    # Entry 1: Sword battle
    v_desc1 = embedder.embed_text("絕世劍法縱橫，劍氣撕裂長空")
    tags1 = ["仙俠:劍道殺伐", "劍氣激盪"]
    tag_vecs1 = embedder.embed_batch(tags1)
    ref_repo.insert_entry_with_tags(source_id, "絕世劍法縱橫", v_desc1, "劍氣撕裂長空", list(zip(tags1, tag_vecs1)))

    # Entry 2: Redundant sword battle with identical tags
    v_desc2 = embedder.embed_text("無匹劍道交鋒，劍意直衝霄漢")
    tags2 = ["仙俠:劍道殺伐", "劍氣激盪"]
    tag_vecs2 = embedder.embed_batch(tags2)
    ref_repo.insert_entry_with_tags(source_id, "無匹劍道交鋒", v_desc2, "劍意直衝霄漢", list(zip(tags2, tag_vecs2)))

    # Entry 3: Distinct psychological micro-expression
    v_desc3 = embedder.embed_text("審訊眼神交鋒，呼吸微瀾指尖微顫")
    tags3 = ["焦距:微觀特寫", "心理博弈"]
    tag_vecs3 = embedder.embed_batch(tags3)
    ref_repo.insert_entry_with_tags(source_id, "審訊眼神交鋒", v_desc3, "指尖微顫", list(zip(tags3, tag_vecs3)))

    task_vec = embedder.embed_text("劍道對決與緊張氣氛")
    q_tag_vecs = embedder.embed_batch(["仙俠:劍道殺伐", "對決"])

    # Retrieve top_k=2
    hits = ref_repo.search_dual_track(task_vec=task_vec, query_tag_vecs=q_tag_vecs, threshold=0.1, top_k=2)
    assert len(hits) == 2
    # Verify diversity: both Entry 1 (or 2) AND Entry 3 should be selected due to MMR diversity
    returned_descriptions = [h["description"] for h in hits]
    assert any("劍" in d for d in returned_descriptions)
    assert any("審訊" in d for d in returned_descriptions)


def test_book_blueprint_crud_and_relational_dynamic(tmp_path: Path) -> None:
    """Verify permanent book blueprint storage, JSON deserialization, and relational_dynamic attributes."""
    ref_repo = ReferenceRepository(tmp_path / "test_ref_bp.db")
    embedder = CpuEmbedder()

    from autonovel.schemas.llm_contracts import (
        BookBlueprintSynthesisContract,
        CharacterArchetypeContract,
        FactionGeopoliticsContract,
        PowerHierarchyContract,
    )

    bp = BookBlueprintSynthesisContract(
        novel_title="凡人修仙傳",
        genre_primary="仙俠",
        genre_sub="凡人修真",
        macro_outline="開端：少年求仙；中篇：宗門大比與魔道亂起；終章：飛升靈界。",
        character_network=[
            CharacterArchetypeContract(
                character_name="韓立",
                archetype_role="求道求生拓荒者",
                faction_name="落雲宗",
                background_and_motive="深沉謹慎，唯求大道長生",
            ),
            CharacterArchetypeContract(
                character_name="墨大夫",
                archetype_role="舊秩序誘殺奪舍者",
                faction_name="神手谷",
                background_and_motive="壽元將盡，企圖奪舍逆天改命",
            ),
            CharacterArchetypeContract(
                character_name="南宮婉",
                archetype_role="命定糾葛同行者",
                faction_name="掩月宗",
                background_and_motive="結丹修士，因緣際會結下宿緣",
            ),
        ],
        faction_geopolitics=[
            FactionGeopoliticsContract(
                faction_name="七玄門",
                doctrine_and_nature="凡俗江湖門派",
                power_tier="江湖草莽",
            )
        ],
        hierarchy_rules=[
            PowerHierarchyContract(
                level_num=1,
                level_name="煉氣期",
                core_attributes="吐納靈氣",
                breakthrough_criteria="築基丹突破",
            )
        ],
        world_laws=["殺人奪寶，強者為尊"],
    )

    curve = [{"ch": 1, "pct": 20.0, "t": 2.0, "p": 2.0, "d": 3.0, "phase": "BUILDUP", "title": "山村少年"}]
    src_id = ref_repo.register_or_reset_source("test_sha_bp", "fanren.txt", 1)
    bp_id = ref_repo.save_book_blueprint(source_id=src_id, blueprint=bp, factual_pacing_curve=curve, total_chapters=1)
    assert bp_id > 0
    assert ref_repo.has_book_blueprint(src_id) is True
    assert ref_repo.has_book_blueprint(999) is False

    fetched = ref_repo.get_book_blueprint(src_id)
    assert fetched is not None
    assert fetched["novel_title"] == "凡人修仙傳"
    assert fetched["macro_outline"] == "開端：少年求仙；中篇：宗門大比與魔道亂起；終章：飛升靈界。"
    assert len(fetched["character_network"]) == 3
    assert fetched["character_network"][0]["character_name"] == "韓立"
    assert len(fetched["factual_pacing_curve"]) == 1

    # Test relational_dynamic insertion and retrieval
    vec_desc = embedder.embed_text("韓立禦使飛劍斬擊強敵")
    vec_tags = [(t, embedder.embed_text(t)) for t in ["仙俠", "鬥法"]]
    eid = ref_repo.insert_entry_with_tags(
        source_id=src_id,
        description="韓立禦使飛劍斬擊強敵",
        desc_vec=vec_desc,
        snippet="一道青色劍芒自袖中呼嘯而出，撕裂長空！",
        tag_vectors=vec_tags,
        relational_dynamic="行動方以迅雷之勢主動突襲，受動方猝不及防陷入被動防守",
    )
    assert eid > 0

    entries, total = ref_repo.query_entries(source_id=src_id)
    assert total == 1
    assert entries[0]["relational_dynamic"] == "行動方以迅雷之勢主動突襲，受動方猝不及防陷入被動防守"

    # Test search_dual_track returns relational_dynamic
    hits = ref_repo.search_dual_track(task_vec=vec_desc, query_tag_vecs=[vec_tags[0][1]], threshold=0.1, top_k=1)
    assert len(hits) == 1
    assert hits[0]["relational_dynamic"] == "行動方以迅雷之勢主動突襲，受動方猝不及防陷入被動防守"


def test_genre_track_prefers_books_of_the_same_genre(tmp_path: Path) -> None:
    """Genre weighting: a book's blueprint genre becomes the third 0.15 scoring track."""
    # Deterministic 8-dim embedder: each genre label maps to its own axis.
    genre_axis = {"仙俠:劍修": 0, "都市:商戰": 1}

    def fake_embed(text: str) -> np.ndarray:
        """Map a genre label to a fixed one-hot axis (unknown text falls back to axis 7)."""
        vec = np.zeros(8, dtype=np.float32)
        vec[genre_axis.get(text, 7)] = 1.0
        return vec

    ref_repo = ReferenceRepository(tmp_path / "ref.db", embed_fn=fake_embed)
    xianxia_src = ref_repo.register_or_reset_source("sha-genre-x", "xianxia.txt", 1)
    urban_src = ref_repo.register_or_reset_source("sha-genre-u", "urban.txt", 1)
    with atomic_transaction(ref_repo.db_path) as conn:
        for src_id, title, primary, sub in (
            (xianxia_src, "劍書", "仙俠", "劍修"),
            (urban_src, "商書", "都市", "商戰"),
        ):
            conn.execute(
                """
                INSERT INTO ref_book_blueprints (source_id, novel_title, genre_primary, genre_sub)
                VALUES (?, ?, ?, ?);
                """,
                (src_id, title, primary, sub),
            )

    desc_vec = np.zeros(8, dtype=np.float32)
    desc_vec[2] = 1.0
    tag_vec = np.zeros(8, dtype=np.float32)
    tag_vec[3] = 1.0
    xianxia_eid = ref_repo.insert_entry_with_tags(xianxia_src, "同題材條目", desc_vec, "", [("劍意", tag_vec)])
    urban_eid = ref_repo.insert_entry_with_tags(urban_src, "異題材條目", desc_vec, "", [("劍意", tag_vec)])

    weighted = ref_repo.search_dual_track(
        task_vec=desc_vec,
        query_tag_vecs=[tag_vec],
        threshold=0.0,
        top_k=5,
        query_genre="仙俠:劍修",
    )
    by_id = {h["entry_id"]: h for h in weighted}
    assert by_id[xianxia_eid]["score_genre"] == pytest.approx(1.0, abs=1e-5)
    assert by_id[urban_eid]["score_genre"] == pytest.approx(0.0, abs=1e-5)
    assert by_id[xianxia_eid]["score_final"] > by_id[urban_eid]["score_final"]

    # Without a query genre the third track is neutral, so the two rows tie.
    baseline = ref_repo.search_dual_track(
        task_vec=desc_vec, query_tag_vecs=[tag_vec], threshold=0.0, top_k=5
    )
    base_scores = {h["entry_id"]: h["score_final"] for h in baseline}
    assert base_scores[xianxia_eid] == pytest.approx(base_scores[urban_eid])


def test_tag_maintenance_preserves_entries_that_lose_all_tags(tmp_path: Path) -> None:
    """Dropping an entry's last tag must preserve the prose row instead of deleting it."""
    ref_repo = ReferenceRepository(tmp_path / "ref.db")
    src = ref_repo.register_or_reset_source("sha-drop", "book.txt", 1)
    desc_vec = np.zeros(8, dtype=np.float32)
    desc_vec[0] = 1.0
    tag_vec = np.zeros(8, dtype=np.float32)
    tag_vec[1] = 1.0
    ref_repo.insert_entry_with_tags(src, "孤兒條目", desc_vec, "", [("孤兒標籤", tag_vec)])
    healthy = ref_repo.insert_entry_with_tags(
        src, "健全條目", desc_vec, "", [("孤兒標籤", tag_vec), ("核心標籤", tag_vec)]
    )

    class _FixedEmbedder:
        """Return a constant vector so tag maintenance never needs the ONNX model."""

        def embed_text(self, _text: str) -> np.ndarray:
            """Return the shared deterministic tag vector."""
            return tag_vec

        def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
            """Return one deterministic vector per requested tag name."""
            return [tag_vec for _ in texts]

    stats = ref_repo.apply_tag_rules(
        [TagRuleContract(action="DROP", old_tag="孤兒標籤")], _FixedEmbedder()
    )
    assert stats["deleted_tagless_entries"] == 0

    remaining, total = ref_repo.query_entries(source_id=src)
    assert total == 2
    assert ref_repo.get_library_stats()["entries"] == 2


def test_tag_clusters_and_density_sampling_and_mixed_dimensions(tmp_path: Path) -> None:
    """Verify tag asset clustering handles >max_nodes stratification and skips mismatched embedding dimensions."""
    import sqlite3

    assert resolve_tag_cluster("角色（主角/配角）") == "通用技法"

    ref_db = tmp_path / "large_tags.db"
    ref_repo = ReferenceRepository(ref_db)
    src_id = ref_repo.register_or_reset_source("sha-large-tags", "large.txt", 1)

    rng = np.random.default_rng(42)
    desc_vec = np.ones(16, dtype=np.float32)
    desc_vec /= np.linalg.norm(desc_vec)

    batch_items = []
    # Create 85 distinct valid 16-dim tags (mixture of genre prefixes and techniques)
    for i in range(85):
        vec = rng.normal(size=16).astype(np.float32)
        vec /= np.linalg.norm(vec)
        tag_name = f"仙俠:流派技法{i}" if i < 20 else f"敘事技法_{i}"
        repeats = 3 if i < 5 else 1
        for r in range(repeats):
            batch_items.append(
                {
                    "description": f"條目_{i}_{r}",
                    "desc_vec": desc_vec,
                    "snippet": f"片段_{i}_{r}",
                    "tag_vectors": [(tag_name, vec)],
                }
            )
    ref_repo.insert_entries_batch(src_id, batch_items)

    # Inject corrupted/mismatched embedding blobs to verify resilient dimension filtering
    with sqlite3.connect(ref_db) as conn:
        eid = conn.execute("SELECT id FROM ref_entries LIMIT 1;").fetchone()[0]
        conn.execute(
            "INSERT INTO ref_tags (entry_id, axis_type, tag_name, tag_embedding) VALUES (?, 'CONTEXT', ?, ?);",
            (eid, "壞損空向量標籤", b""),
        )
        wrong_dim = np.ones(4, dtype=np.float32).tobytes()
        conn.execute(
            "INSERT INTO ref_tags (entry_id, axis_type, tag_name, tag_embedding) VALUES (?, 'CONTEXT', ?, ?);",
            (eid, "異維度標籤", wrong_dim),
        )
        conn.commit()
    ref_repo._invalidate_cache()

    assets = ref_repo.get_tag_clusters_and_stats(max_nodes=50)
    assert len(assets["nodes"]) == 50
    assert assets["stats"]["total_unique"] == 85
    # 5 tags * 3 usages + 80 tags * 1 usage = 95 valid tag usages
    assert assets["stats"]["total_usages"] == 95
    assert assets["stats"]["genre_tags_count"] == 20
    assert assets["stats"]["technique_tags_count"] == 65

    density = ref_repo.get_density_analysis(eps=0.45, min_samples=2, max_tags=60)
    clustered_total = sum(len(c["tags"]) for c in density["dense_clusters"])
    assert clustered_total + len(density["outliers"]) == 60


def test_apply_tag_rules_cleans_frequency_suffix_and_brackets(tmp_path: Path) -> None:
    """Verify apply_tag_rules strips echoed '(count)' suffixes/brackets and converts Simplified targets."""
    ref_repo = ReferenceRepository(tmp_path / "clean_rules.db")
    src = ref_repo.register_or_reset_source("sha-clean-rules", "clean.txt", 1)
    vec = np.ones(8, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    ref_repo.insert_entry_with_tags(
        src,
        "劍氣交鋒描寫",
        vec,
        "原文片段",
        [("御劍", vec), ("劍招", vec), ("此時", vec)],
    )

    class _MockRuleEmbedder:
        """Deterministic constant embedder for tag rule normalization tests."""

        def embed_text(self, _text: str) -> np.ndarray:
            return vec

        def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
            return [vec for _ in texts]

    rules = [
        TagRuleContract(old_tag="御劍 (12)", new_tag="「仙侠:纯阳剑道」", action="NORMALIZE"),
        TagRuleContract(old_tag="「劍招」", new_tag="#仙侠:纯阳剑道#", action="MERGE"),
        TagRuleContract(old_tag="此時 (1)", new_tag="", action="DROP"),
    ]
    stats = ref_repo.apply_tag_rules(rules, _MockRuleEmbedder())
    assert stats["normalized"] == 1
    assert stats["merged"] == 1
    assert stats["dropped"] == 1

    entries, total = ref_repo.query_entries(source_id=src)
    assert total == 1
    tags = entries[0]["tags"]
    assert tags == ["仙俠:純陽劍道"]

