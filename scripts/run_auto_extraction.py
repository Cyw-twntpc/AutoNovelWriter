"""Autonomous Takeover Script for Full-Novel Extraction Pipeline.

Executes Track B (Chapter Facts, Volume Consolidation, Macro Blueprint, Pacing Curve)
and Track A (Chunking, Embeddings, Tag Extraction) for reference novel '大奉打更人.txt'.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Set environment variables for optimal extraction execution
os.environ["AUTONOVEL_LLAMA_NP"] = "1"  # Dedicate entire 16k context window to single extraction slot
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np

from autonovel.config import RuntimeConfig, get_logger
from autonovel.db.project_schema import init_project_database
from autonovel.db.reference_repo import ReferenceRepository
from autonovel.db.reference_schema import init_reference_database
from autonovel.engine.hardware_guard import HardwareGuard
from autonovel.llm.embedder import CpuEmbedder
from autonovel.llm.llama_manager import LlamaServerManager
from autonovel.llm.llm_client import LLMClient
from autonovel.reference.chunk_extractor import ReferenceExtractor
from autonovel.reference.file_parser import compute_sha256, parse_novel_file
from autonovel.reference.macro_extractor import (
    ChapterFactExtractor,
    MacroPacingExtractor,
    build_factual_pacing_curve,
    extract_table_of_contents,
    pack_chapters_dynamically,
    select_high_value_chapters_zscore,
)
from autonovel.schemas.llm_contracts import (
    ChapterFactExtractContract,
    GroupEntityDeltas,
    MacroPacingTemplateContract,
    MajorArcConsolidationContract,
    VolumeFactConsolidationContract,
    calculate_chapter_exemplar_score,
)

logger = get_logger("run_auto_extraction")


class AutonomousExtractionRunner:
    """Manages end-to-end extraction, persistent monitoring, and post-flight quality checks."""

    def __init__(
        self,
        novel_filename: str = "大奉打更人.txt",
        genre_primary: str | None = None,
        genre_sub: str | None = None,
    ) -> None:
        self.novel_path = Path(novel_filename).resolve()
        if not self.novel_path.is_file():
            raise FileNotFoundError(f"Target novel file not found: {self.novel_path}")

        self.genre_primary = genre_primary
        self.genre_sub = genre_sub

        self.cfg = RuntimeConfig()
        # Initialize databases
        init_reference_database(self.cfg.reference_db_path)
        init_project_database(self.cfg.project_db_path)

        self.ref_repo = ReferenceRepository(self.cfg.reference_db_path)
        self.hardware_guard = HardwareGuard(self.cfg)

        self.llama_manager = LlamaServerManager(self.cfg)
        self.llm_client = LLMClient(self.cfg, server_manager=self.llama_manager)
        self.embedder = CpuEmbedder()

        self.macro_extractor = MacroPacingExtractor(self.ref_repo, llm_client=self.llm_client)
        self.fact_extractor = ChapterFactExtractor(llm_client=self.llm_client)
        self.ref_extractor = ReferenceExtractor(
            repo=self.ref_repo,
            embedder=self.embedder,
            llm_client=self.llm_client,
            hardware_guard=self.hardware_guard,
        )

    def ensure_server(self) -> None:
        """Verify or launch local llama-server process."""
        print(f"[INIT] Checking llama-server at {self.cfg.llama_server_url}...")
        if not self.llama_manager.is_healthy():
            print("[INIT] Starting llama-server process (np=1, 16k context)...")
            started = self.llama_manager.ensure_running()
            if not started:
                raise RuntimeError("Failed to start llama-server process. Check binary and model paths.")
        print("[INIT] llama-server is healthy and ready.")

    async def run(self) -> dict[str, object]:
        """Execute complete Track B and Track A pipeline with live telemetry."""
        self.ensure_server()

        start_time = time.time()
        print("\n==================================================")
        print(f"  AUTONOMOUS EXTRACTION RUNNER: {self.novel_path.name}")
        print("==================================================\n")

        # 1. Parse novel text and detect encoding
        print(f"[STAGE 0] Decoding {self.novel_path.name}...")
        raw_text, detected_enc = parse_novel_file(self.novel_path)
        sha = compute_sha256(self.novel_path)
        novel_title = self.novel_path.stem
        print(f"[STAGE 0] Encoding: {detected_enc} | Length: {len(raw_text):,} chars | SHA: {sha[:16]}...")

        # 2. Extract TOC and dynamically bin-pack chapters
        toc = extract_table_of_contents(raw_text)
        total_chapters = len(toc)
        batches = pack_chapters_dynamically(toc, max_chars=7200, max_chapters=3)
        total_batches = len(batches)
        print(f"[STAGE 0] Parsed {total_chapters} chapters from TOC.")
        print(f"[STAGE 0] Packed into {total_batches} dynamic groups (max 7,200 chars / up to 3 chapters per group).\n")

        source_id = self.ref_repo.register_or_reset_source(sha, self.novel_path.name, 0)

        # 3. Track B: Batched Chapter Fact & Entity Delta Extraction with Checkpointing
        print(f"[TRACK B] Commencing Stage 1: Batched Chapter Fact & Entity Delta Extraction ({total_batches} groups, {total_chapters} chapters)...")
        checkpoint_path = self.cfg.data_dir / f"extraction_checkpoint_{sha[:12]}.json"
        chapter_facts: list[ChapterFactExtractContract] = []
        entity_deltas: list[GroupEntityDeltas] = []
        volume_reports: list[VolumeFactConsolidationContract] = []
        major_arcs: list[MajorArcConsolidationContract] = []
        last_anchor_event = ""
        start_batch_idx = 1

        if checkpoint_path.is_file():
            try:
                ckpt_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if ckpt_data.get("sha") == sha and ckpt_data.get("total_chapters") == total_chapters:
                    saved_facts = ckpt_data.get("chapter_facts", [])
                    if saved_facts and "score_causal" in saved_facts[0]:
                        chapter_facts = [ChapterFactExtractContract(**item) for item in saved_facts]
                        saved_deltas = ckpt_data.get("entity_deltas", [])
                        entity_deltas = [GroupEntityDeltas(**item) for item in saved_deltas]
                        saved_vols = ckpt_data.get("volume_reports", [])
                        volume_reports = [VolumeFactConsolidationContract(**item) for item in saved_vols]
                        saved_arcs = ckpt_data.get("major_arcs", [])
                        major_arcs = [MajorArcConsolidationContract(**item) for item in saved_arcs]
                        last_anchor_event = ckpt_data.get("last_anchor_event", "")

                        # Calculate start_batch_idx from extracted chapters count
                        extracted_count = len(chapter_facts)
                        cum_count = 0
                        for b_i, b in enumerate(batches, start=1):
                            cum_count += len(b)
                            if cum_count > extracted_count:
                                start_batch_idx = b_i
                                break
                        else:
                            start_batch_idx = total_batches + 1

                        print(f"[CHECKPOINT] Resuming from checkpoint: {len(chapter_facts)}/{total_chapters} chapters ({start_batch_idx - 1}/{total_batches} groups) already extracted.")
                    else:
                        print("[CHECKPOINT] Stored checkpoint lacks 3D evaluation scores; restarting fresh to guarantee high-value scoring.")
                        checkpoint_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[CHECKPOINT] Could not load checkpoint ({e}); starting from beginning.")

        fact_start = time.time()
        for b_idx in range(start_batch_idx, total_batches + 1):
            batch = batches[b_idx - 1]
            first_ch_idx = batch[0].get("index", 1)
            last_ch_idx = batch[-1].get("index", 1)
            batch_chars = sum(len(ch.get("text", "")) for ch in batch)

            t0 = time.time()
            res = await self.fact_extractor.extract_batched_chapter_facts(
                chapters=batch,
                prior_anchor_event=last_anchor_event,
            )
            dt = time.time() - t0

            for sc in res.chapters:
                cf = ChapterFactExtractContract.from_compact(sc)
                chapter_facts.append(cf)
            entity_deltas.append(res.entity_deltas)

            if res.chapters:
                last_anchor_event = res.chapters[-1].core_event

            # Save checkpoint after every batch to guarantee zero data loss
            checkpoint_path.write_text(
                json.dumps({
                    "sha": sha,
                    "total_chapters": total_chapters,
                    "total_batches": total_batches,
                    "current_phase": "STAGE_1_BATCHED_FACTS",
                    "chapter_facts": [c.model_dump() for c in chapter_facts],
                    "entity_deltas": [d.model_dump() for d in entity_deltas],
                    "volume_reports": [vr.model_dump() for vr in volume_reports],
                    "major_arcs": [ma.model_dump() for ma in major_arcs],
                    "last_anchor_event": last_anchor_event,
                }, ensure_ascii=False),
                encoding="utf-8",
            )

            # Thermal guard check
            await self.hardware_guard.check_and_cooldown_if_hot()

            if b_idx % 2 == 0 or b_idx == 1 or b_idx == total_batches:
                elapsed = time.time() - fact_start
                rate = (b_idx - start_batch_idx + 1) / max(1.0, elapsed)
                remaining_sec = (total_batches - b_idx) / max(0.001, rate)
                last_sc = res.chapters[-1] if res.chapters else None
                cur_exemplar = calculate_chapter_exemplar_score(last_sc.score_causal, last_sc.score_craft, last_sc.score_depth) if last_sc else 0.0
                print(
                    f"[Group {b_idx:3d}/{total_batches:3d}] ({len(chapter_facts)/total_chapters*100:5.1f}%) "
                    f"Ch.{first_ch_idx:03d}~{last_ch_idx:03d} ({batch_chars}c) -> "
                    f"Latest=({last_sc.score_causal:.1f}, {last_sc.score_craft:.1f}, {last_sc.score_depth:.1f}) -> Exemplar={cur_exemplar:.2f} "
                    f"[{dt:.2f}s | Speed: {rate*60:.1f} grp/min | ETA: {remaining_sec/60:.1f}m]"
                )

        # 4. Volume Consolidation & Major Arc Reduction & Blueprint Synthesis
        if not self.ref_repo.has_book_blueprint(source_id):
            vol_batch_size = 20
            num_vols = max(1, (total_chapters + vol_batch_size - 1) // vol_batch_size)
            print(f"\n[TRACK B] Stage 2: Consolidating {num_vols} volumes (batches of {vol_batch_size} chapters with entity deltas)...")

            start_vol_idx = len(volume_reports) + 1
            if start_vol_idx > 1:
                print(f"[CHECKPOINT] Resuming volume consolidation: {len(volume_reports)}/{num_vols} volumes already consolidated.")

            for v_idx in range(start_vol_idx, num_vols + 1):
                v_start = (v_idx - 1) * vol_batch_size
                v_end = min(total_chapters, v_idx * vol_batch_size)
                batch = chapter_facts[v_start:v_end]
                # Accurately slice entity_deltas matching this volume's actual chapter range [first_ch, last_ch]
                first_ch_in_vol = batch[0].chapter_index if batch else (v_start + 1)
                last_ch_in_vol = batch[-1].chapter_index if batch else v_end
                vol_deltas = []
                for b_i, delta in enumerate(entity_deltas):
                    if b_i < len(batches):
                        b_ch_start = batches[b_i][0].get("index", 1)
                        b_ch_end = batches[b_i][-1].get("index", 1)
                        if not (b_ch_end < first_ch_in_vol or b_ch_start > last_ch_in_vol):
                            vol_deltas.append(delta)
                    else:
                        vol_deltas.append(delta)

                if batch:
                    t0 = time.time()
                    vr = await self.fact_extractor.consolidate_volume_facts(
                        volume_index=v_idx,
                        chapter_facts=batch,
                        volume_entity_deltas=vol_deltas,
                    )
                    dt = time.time() - t0
                    volume_reports.append(vr)
                    print(f"  Vol.{v_idx:02d}/{num_vols:02d} ({vr.volume_title}) consolidated in {dt:.2f}s.")

                    # Save checkpoint with volume_reports
                    checkpoint_path.write_text(
                        json.dumps({
                            "sha": sha,
                            "total_chapters": total_chapters,
                            "total_batches": total_batches,
                            "current_phase": "STAGE_2_VOLUMES",
                            "chapter_facts": [c.model_dump() for c in chapter_facts],
                            "entity_deltas": [d.model_dump() for d in entity_deltas],
                            "volume_reports": [v.model_dump() for v in volume_reports],
                            "major_arcs": [ma.model_dump() for ma in major_arcs],
                            "last_anchor_event": last_anchor_event,
                        }, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    await self.hardware_guard.check_and_cooldown_if_hot()

            # 4.5 Hierarchical Reduction: Consolidate Volumes into Major Narrative Arcs (Adaptive)
            if not major_arcs and num_vols > 4:
                num_arcs = min(8, max(2, (num_vols + 4) // 5))
                print(f"\n[TRACK B] Stage 2.5: Reducing {len(volume_reports)} volumes into {num_arcs} Major Narrative Arcs (Zero-Truncation Tree)...")
                t0 = time.time()
                major_arcs = await self.fact_extractor.consolidate_major_arcs(
                    volume_reports=volume_reports,
                    num_arcs=num_arcs,
                    hardware_guard=self.hardware_guard,
                )
                dt = time.time() - t0
                print(f"[TRACK B] {len(major_arcs)} Major Arcs synthesized in {dt:.2f}s:")
                for ma in major_arcs:
                    print(f"  - {ma.arc_title} ({ma.covered_volumes})")

                # Save checkpoint with major_arcs
                checkpoint_path.write_text(
                    json.dumps({
                        "sha": sha,
                        "total_chapters": total_chapters,
                        "total_batches": total_batches,
                        "current_phase": "STAGE_2_5_MAJOR_ARCS",
                        "chapter_facts": [c.model_dump() for c in chapter_facts],
                        "entity_deltas": [d.model_dump() for d in entity_deltas],
                        "volume_reports": [v.model_dump() for v in volume_reports],
                        "major_arcs": [ma.model_dump() for ma in major_arcs],
                        "last_anchor_event": last_anchor_event,
                    }, ensure_ascii=False),
                    encoding="utf-8",
                )
            elif num_vols <= 4:
                print(f"\n[TRACK B] Compact length ({num_vols} volumes <= 4); skipping Major Arcs and synthesizing blueprint directly from volume reports.")

            # 5. Global Macro Blueprint Synthesis
            arc_label = f"{len(major_arcs)} Major Arcs" if major_arcs else "Volume Reports"
            print(f"\n[TRACK B] Stage 3: Synthesizing Global Macro Blueprint from {arc_label} for 《{novel_title}》...")
            t0 = time.time()
            blueprint = await self.fact_extractor.synthesize_macro_blueprint(
                novel_title=novel_title,
                volume_reports=volume_reports,
                chapter_facts=chapter_facts,
                genre_primary=self.genre_primary,
                genre_sub=self.genre_sub,
                major_arcs=major_arcs,
            )
            self.genre_primary = blueprint.genre_primary
            self.genre_sub = blueprint.genre_sub
            dt = time.time() - t0
            print(f"[TRACK B] Blueprint synthesized in {dt:.2f}s:")
            print(f"  - Macro Outline: {len(blueprint.macro_outline)} chars")
            print(f"  - Character Network: {len(blueprint.character_network)} archetypes")
            print(f"  - Factions: {len(blueprint.faction_geopolitics)} organizations")
            print(f"  - Hierarchy Rules: {len(blueprint.hierarchy_rules)} tiers")
            print(f"  - World Laws: {len(blueprint.world_laws)} iron rules")

            # 6. Build factual pacing curve and save blueprint
            curve_data, stages = build_factual_pacing_curve(chapter_facts, total_chapters)
            self.ref_repo.save_book_blueprint(
                source_id=source_id,
                blueprint=blueprint,
                factual_pacing_curve=curve_data,
                total_chapters=total_chapters,
            )

            climax_offset = 75.0
            for s in stages:
                if s.pacing_phase == "CLIMAX":
                    climax_offset = s.offset_percent
                    break

            template = MacroPacingTemplateContract(
                template_name=f"{novel_title}·標準宏觀節奏波浪",
                genre_primary=self.genre_primary,
                genre_sub=self.genre_sub,
                source_novel_title=novel_title,
                total_chapters_analyzed=total_chapters,
                narrative_archetype=f"{self.genre_primary}·實體情節波浪流",
                pacing_tags=["#章節級高解析度", "#客觀事實收斂", "#起承轉合"],
                summary=f"提取自《{novel_title}》全書 {total_chapters} 章事實解構之高精度真實波浪曲線，決戰於 {climax_offset:.0f}% 處爆發。",
                stages=stages,
                chapter_curve_data=curve_data,
            )
            self.ref_repo.save_pacing_template(template, source_id=source_id)
            print("[TRACK B] Factual pacing template and book blueprint saved to reference database.\n")
        else:
            print(f"[TRACK B] Reusing existing synthesized blueprint and factual pacing curve for 《{novel_title}》.\n")

        # 6.5 Select High-Value Exemplar Chapters for Targeted Track A Ingestion
        print("\n[SAMPLING] Selecting high-value exemplar chapters via Dual-Tier Z-score proportion sampling...")
        target_chapters, chapter_scores_map, sampling_stats = select_high_value_chapters_zscore(
            chapter_facts=chapter_facts,
            min_pct=0.08,
            max_pct=0.12,
            vol_batch_size=20,
        )
        print("[SAMPLING] Evaluation Complete:")
        print(f"  - Total Chapters: {sampling_stats['total_chapters']}")
        print(f"  - Score Mean: {sampling_stats['global_mean_score']:.2f} (Std: {sampling_stats['global_std_score']:.2f}, Range: [{sampling_stats['min_score']}, {sampling_stats['max_score']}])")
        print(f"  - Global Outliers (Z >= 1.5): {sampling_stats['global_outliers_count']}")
        print(f"  - Volume Peaks (Z_vol >= 1.0): {sampling_stats['volume_peaks_count']}")
        print(f"  - Selected Chapters: {sampling_stats['selected_count']} ({sampling_stats['selected_proportion']*100:.1f}% of novel, spanning Ch.{sampling_stats['min_selected_chapter']} to Ch.{sampling_stats['max_selected_chapter']})")

        # 7. Track A: Skills and Chunk Pattern Ingestion on Selected Chapters
        print(f"\n[TRACK A] Commencing Targeted Chunk Ingestion & Semantic Tagging for 《{novel_title}》...")
        track_a_start = time.time()
        force_reextract_chunks = os.environ.get("FORCE_REEXTRACT_CHUNKS", "1") == "1"
        total_chunks = await self.ref_extractor.ingest_file(
            file_path=self.novel_path,
            force_reextract=force_reextract_chunks,
            on_progress=lambda p: None,
            target_chapter_indices=target_chapters,
            chapter_scores_map=chapter_scores_map,
        )
        track_a_duration = time.time() - track_a_start
        print(f"[TRACK A] Ingested {total_chunks} chunks from {len(target_chapters)} high-value chapters in {track_a_duration:.2f}s.\n")

        total_duration = time.time() - start_time
        checkpoint_path.unlink(missing_ok=True)
        print("==================================================")
        print(f"  EXTRACTION FINISHED in {total_duration/60:.2f} minutes")
        print("==================================================\n")

        # 8. Post-Flight Quality Audit
        audit = self.perform_quality_audit(
            source_id=source_id,
            total_chapters=total_chapters,
            sampling_stats=sampling_stats,
        )
        return audit

    def perform_quality_audit(
        self,
        source_id: int,
        total_chapters: int,
        sampling_stats: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        """Verify the quality of extracted blueprint, pacing curve, chunks, and semantic tags."""
        print(f"[QUALITY AUDIT] Running post-flight inspection on Source #{source_id}...")
        bp = self.ref_repo.get_book_blueprint(source_id)
        entries, total_entries = self.ref_repo.query_entries(source_id=source_id, limit=2000)
        tags_dist = self.ref_repo.list_tags_with_frequency()
        pacing_tpls = self.ref_repo.list_pacing_templates()
        clusters_info = self.ref_repo.get_tag_clusters_and_stats()

        total_tags = sum(t["count"] for t in tags_dist)
        unique_tags = len(tags_dist)
        high_freq_tags = [t for t in tags_dist if t["count"] >= 3]
        subj_count = len([e for e in entries if e.get("subject_role")])
        obj_count = len([e for e in entries if e.get("object_role")])

        entry_chapter_indices = [e["chapter_index"] for e in entries if e.get("chapter_index")]
        entry_scores = [e["exemplar_score"] for e in entries if e.get("exemplar_score")]

        audit_results = {
            "source_id": source_id,
            "novel_title": self.novel_path.stem,
            "total_chapters": total_chapters,
            "blueprint_present": bp is not None,
            "blueprint_macro_outline_chars": len(bp.get("macro_outline", "")) if bp else 0,
            "character_network_count": len(bp.get("character_network", [])) if bp else 0,
            "faction_geopolitics_count": len(bp.get("faction_geopolitics", [])) if bp else 0,
            "hierarchy_rules_count": len(bp.get("hierarchy_rules", [])) if bp else 0,
            "world_laws_count": len(bp.get("world_laws", [])) if bp else 0,
            "factual_pacing_points": len(bp.get("factual_pacing_curve", [])) if bp else 0,
            "pacing_templates_saved": len(pacing_tpls),
            "chunks_ingested": total_entries,
            "subject_roles_assigned": subj_count,
            "object_roles_assigned": obj_count,
            "unique_tags_count": unique_tags,
            "total_tag_assignments": total_tags,
            "high_freq_tags_count": len(high_freq_tags),
            "clusters_count": clusters_info.get("stats", {}).get("clusters_count", 0),
            "chapter_span_min": min(entry_chapter_indices) if entry_chapter_indices else 0,
            "chapter_span_max": max(entry_chapter_indices) if entry_chapter_indices else 0,
            "mean_exemplar_score": round(float(np.mean(entry_scores)), 2) if entry_scores else 0.0,
            "sampling_stats": sampling_stats or {},
            "top_15_tags": tags_dist[:15],
            "character_archetypes": [
                {"name": c.get("character_name"), "role": c.get("archetype_role")}
                for c in (bp.get("character_network", []) if bp else [])
            ],
            "world_laws": bp.get("world_laws", []) if bp else [],
        }

        audit_file = self.cfg.data_dir / "extraction_audit_report.json"
        audit_file.write_text(json.dumps(audit_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[QUALITY AUDIT] Audit results written to {audit_file}:")
        print(f"  - Blueprint Outline: {audit_results['blueprint_macro_outline_chars']} chars")
        print(f"  - Character Archetypes: {audit_results['character_network_count']}")
        print(f"  - Chunks Ingested: {audit_results['chunks_ingested']}")
        print(f"  - Subject/Object Roles Assigned: {audit_results['subject_roles_assigned']}/{audit_results['chunks_ingested']}")
        print(f"  - Chapter Span: Ch.{audit_results['chapter_span_min']} to Ch.{audit_results['chapter_span_max']}")
        print(f"  - Mean Exemplar Score: {audit_results['mean_exemplar_score']}")
        print(f"  - Unique Tags: {audit_results['unique_tags_count']}")
        print(f"  - Top Tags: {', '.join(t['tag_name'] for t in tags_dist[:10])}")
        return audit_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Novel Extraction Runner")
    parser.add_argument("--novel", default="大奉打更人.txt", help="Path to novel text file")
    parser.add_argument("--genre-primary", default=None, help="Primary genre (e.g. 仙俠, 科幻, 都市, 懸疑, 歷史)")
    parser.add_argument("--genre-sub", default=None, help="Sub genre (e.g. 懸疑探案修仙, 賽博朋克, 職場商戰)")
    args = parser.parse_args()

    runner = AutonomousExtractionRunner(
        novel_filename=args.novel,
        genre_primary=args.genre_primary,
        genre_sub=args.genre_sub,
    )
    asyncio.run(runner.run())
