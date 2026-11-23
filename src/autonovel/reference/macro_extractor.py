"""Macro plot pacing, chapter fact extraction, and factual book blueprint synthesis."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from autonovel.db.reference_repo import ReferenceRepository
from autonovel.engine.prose_sanitizer import (
    convert_simplified_to_traditional,
    strip_web_novel_noise,
)
from autonovel.llm.llm_client import LLMClient
from autonovel.schemas.llm_contracts import (
    BatchedFactExtractionContract,
    BookBlueprintSynthesisContract,
    ChapterFactExtractContract,
    GroupEntityDeltas,
    MacroPacingTemplateContract,
    MajorArcConsolidationContract,
    PacingStageTemplateContract,
    SingleChapterCompactFact,
    VolumeFactConsolidationContract,
    calculate_chapter_exemplar_score,
)
from autonovel.wizard.qa_engine import AIModelUnavailableError

logger = logging.getLogger(__name__)

_NUM_CHARS = r"零一二三四五六七八九十百千萬〇兩两0-9０-９"
_WS = r"[ \t\u3000]"
_TITLE_SEP = r"[ \t\u3000：:、.\-—·（(【\[「『]"
# Match chapter/volume headers cleanly while avoiding dialogue quotes and prose phrases
# such as "第一節課，..." or "第一回合，...". Allows !/? inside titles and period-ending
# titles after a separator for 章 headers; uses lookahead newline so consecutive headers match.
CHAPTER_HEADER_RE = re.compile(
    r"^" + _WS + r"*("
    r"(?:[=＊*—\-~～【\[ \t\u3000]*"
    r"(?:[^\n\r，。！？；：「」『』“”]{0,16}?" + _WS + r"+)?"
    r"(?:"
    r"(?:第[" + _NUM_CHARS + r"]+[卷集部篇節]" + _WS + r"*第?[" + _NUM_CHARS + r"]+章|第[" + _NUM_CHARS + r"]+章)"
    r"(?:" + _TITLE_SEP + r"[^\n\r]{0,45}|[^\n\r，。；：][^。\n\r]{0,40})?"
    r"|"
    r"(?:第[" + _NUM_CHARS + r"]+[卷集部篇]" + _WS + r"*第?[" + _NUM_CHARS + r"]+[節回幕話]|第[" + _NUM_CHARS + r"]+[節回卷集部篇幕話])"
    r"(?:" + _TITLE_SEP + r"[^\n\r。]{0,40}|[0-9０-９][^\n\r。]{0,40})?"
    r")"
    r"|(?:序章|序言|楔子|引子|尾聲|尾声|番外(?:篇)?)[^\n\r。]{0,30}"
    r"|序" + _WS + r"*"
    r"|(?:Chapter|Part|Episode|Day|Section)" + _WS + r"*[0-9０-９]+[^\n\r。]{0,30}"
    r"|(?:[（(【\[][" + _NUM_CHARS + r"]+[）)】\]])" + _WS + r"*[^\n\r。]{0,30}"
    r"))\r?(?=\n|$)",
    re.IGNORECASE | re.MULTILINE,
)


def _is_context_size_exceeded(exc: Exception) -> bool:
    """Detect whether an exception is an HTTP 400 context size overflow from llama-server."""
    msg = str(exc).lower()
    if "exceed_context_size" in msg or "exceeds the available context size" in msg:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = (resp.text or "").lower()
            if "exceed_context_size" in body or "exceeds the available context size" in body:
                return True
        except Exception as body_exc:
            # Some transports raise while materializing the body; treat as non-matching.
            logger.debug("Could not read error response body: %s", body_exc)
    return False


def extract_table_of_contents(text: str) -> list[dict[str, Any]]:
    """Extract chapter sequence, titles, word count, and text bodies from novel text."""
    cleaned = strip_web_novel_noise(text.strip()).strip()
    if not cleaned:
        return []

    matches = list(CHAPTER_HEADER_RE.finditer(cleaned))
    if not matches:
        if len(cleaned) > 10000:
            sections = split_long_chapter_lossless(cleaned, max_segment_chars=6000)
            return [
                {
                    "index": idx,
                    "title": f"全篇第{idx}節",
                    "char_count": len(seg),
                    "text": seg,
                }
                for idx, seg in enumerate(sections, start=1)
                if seg.strip()
            ]
        return [
            {
                "index": 1,
                "title": "全篇第一部",
                "char_count": len(cleaned),
                "text": cleaned,
            }
        ]

    toc: list[dict[str, Any]] = []
    # Keep any meaningful front matter, not only prefaces longer than 600 chars:
    # short forewords were silently dropped from the chapter map.
    if matches[0].start() > 0:
        pre_body = cleaned[: matches[0].start()].strip()
        if pre_body:
            toc.append(
                {
                    "index": 1,
                    "title": "引子",
                    "char_count": len(pre_body),
                    "text": pre_body,
                }
            )
    current_prefix = ""
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        chapter_body = cleaned[start_pos:end_pos].strip()
        if len(chapter_body) == 0:
            # Volume or section banner without body; carry over as contextual prefix for the following chapter
            current_prefix = title
            continue

        full_title = f"{current_prefix} · {title}" if current_prefix else title
        current_prefix = ""
        toc.append(
            {
                "index": len(toc) + 1,
                "title": full_title,
                "char_count": len(chapter_body),
                "text": chapter_body,
            }
        )
    return toc


def split_long_chapter_lossless(text: str, max_segment_chars: int = 10000) -> list[str]:
    """Split an exceptionally long chapter into contiguous lossless segments at paragraph/sentence boundaries."""
    cleaned = text.strip()
    if not cleaned:
        return [""]
    if len(cleaned) <= max_segment_chars:
        return [cleaned]

    segments: list[str] = []
    start = 0
    n = len(cleaned)
    while start < n:
        if n - start <= max_segment_chars:
            segments.append(cleaned[start:].strip())
            break
        end = start + max_segment_chars
        # Find nearest paragraph or sentence boundary before end
        cut = cleaned.rfind("\n", start + max_segment_chars // 2, end)
        if cut == -1:
            cut = cleaned.rfind("。", start + max_segment_chars // 2, end)
        if cut == -1:
            cut = end
        else:
            cut += 1
        seg = cleaned[start:cut].strip()
        if seg:
            segments.append(seg)
        start = cut
    return segments or [cleaned]


def select_high_value_chapters_zscore(
    chapter_facts: list[ChapterFactExtractContract],
    min_pct: float = 0.08,
    max_pct: float = 0.12,
    vol_batch_size: int = 20,
) -> tuple[set[int], dict[int, float], dict[str, Any]]:
    """Select high-value exemplar chapters across the novel using dual-tier Z-score proportion sampling on AI scores."""
    n = len(chapter_facts)
    if n == 0:
        return set(), {}, {}

    chapter_scores_map: dict[int, float] = {}
    for cf in chapter_facts:
        score = calculate_chapter_exemplar_score(
            cf.score_causal, cf.score_craft, cf.score_depth
        )
        chapter_scores_map[cf.chapter_index] = score

    scores = [chapter_scores_map[cf.chapter_index] for cf in chapter_facts]
    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    if sigma < 1e-6:
        sigma = 1.0

    global_outliers: set[int] = {
        cf.chapter_index
        for cf in chapter_facts
        if (chapter_scores_map[cf.chapter_index] - mu) / sigma >= 1.5
    }

    volume_peaks: set[int] = set()
    num_vols = max(1, (n + vol_batch_size - 1) // vol_batch_size)
    for v_idx in range(num_vols):
        v_start = v_idx * vol_batch_size
        v_end = min(n, (v_idx + 1) * vol_batch_size)
        vol_cf = chapter_facts[v_start:v_end]
        if not vol_cf:
            continue
        vol_scores = [chapter_scores_map[c.chapter_index] for c in vol_cf]
        v_mu = float(np.mean(vol_scores))
        v_sigma = float(np.std(vol_scores))
        if v_sigma < 1e-6:
            v_sigma = 1.0
        for c in vol_cf:
            s = chapter_scores_map[c.chapter_index]
            if (s - v_mu) / v_sigma >= 1.0 and s >= mu:
                volume_peaks.add(c.chapter_index)

    candidates = global_outliers | volume_peaks

    min_count = max(1, int(round(n * min_pct)))
    max_count = max(min_count, int(round(n * max_pct)))

    if len(candidates) < min_count:
        sorted_all = sorted(
            chapter_facts,
            key=lambda c: chapter_scores_map[c.chapter_index],
            reverse=True,
        )
        for c in sorted_all:
            candidates.add(c.chapter_index)
            if len(candidates) >= min_count:
                break
    elif len(candidates) > max_count:
        sorted_candidates = sorted(
            candidates,
            key=lambda idx: chapter_scores_map[idx],
            reverse=True,
        )
        candidates = set(sorted_candidates[:max_count])

    stats = {
        "total_chapters": n,
        "global_mean_score": round(mu, 3),
        "global_std_score": round(sigma, 3),
        "min_score": round(float(np.min(scores)), 2),
        "max_score": round(float(np.max(scores)), 2),
        "global_outliers_count": len(global_outliers),
        "volume_peaks_count": len(volume_peaks),
        "selected_count": len(candidates),
        "selected_proportion": round(len(candidates) / max(1, n), 4),
        "min_selected_chapter": min(candidates) if candidates else 0,
        "max_selected_chapter": max(candidates) if candidates else 0,
    }
    return candidates, chapter_scores_map, stats


def pack_chapters_dynamically(
    toc: list[dict[str, Any]],
    max_chars: int = 7200,
    max_chapters: int = 3,
) -> list[list[dict[str, Any]]]:
    """Greedily bin-pack chapters into dynamic groups bounded by prose length and chapter count.

    1. Chapters are added consecutively while total characters <= max_chars and count <= max_chapters.
    2. If a single chapter exceeds max_chars, it forms its own group of size 1.
    """
    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_chars = 0

    for ch in toc:
        ch_text = ch.get("text", "")
        ch_len = len(ch_text)

        # If current batch is empty, add chapter unconditionally
        if not current_batch:
            current_batch.append(ch)
            current_chars = ch_len
            if ch_len >= max_chars or len(current_batch) >= max_chapters:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            continue

        # Check if adding this chapter exceeds capacity
        if current_chars + ch_len <= max_chars and len(current_batch) < max_chapters:
            current_batch.append(ch)
            current_chars += ch_len
            if len(current_batch) >= max_chapters or current_chars >= max_chars:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
        else:
            # Seal current batch and start new batch with this chapter
            batches.append(current_batch)
            current_batch = [ch]
            current_chars = ch_len
            if ch_len >= max_chars or len(current_batch) >= max_chapters:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0

    if current_batch:
        batches.append(current_batch)

    return batches


def reconcile_batched_compact_facts(
    extracted_compacts: list[SingleChapterCompactFact],
    input_chapters: list[dict[str, Any]],
) -> list[SingleChapterCompactFact]:
    """Verify strict 1:1 count alignment with input chapters and bind chapter metadata without mutating prose."""
    if not input_chapters:
        return []

    if len(extracted_compacts) != len(input_chapters):
        raise ValueError(
            f"Batched chapter count mismatch: expected {len(input_chapters)} chapters, "
            f"received {len(extracted_compacts)} from LLM."
        )

    reconciled: list[SingleChapterCompactFact] = []
    for i, ch in enumerate(input_chapters):
        exp_idx = int(ch.get("index", i + 1))
        target_title = convert_simplified_to_traditional(str(ch.get("title", f"第{exp_idx}章")))
        chosen = extracted_compacts[i]
        chosen.chapter_index = exp_idx
        chosen.chapter_title = target_title
        chosen.core_event = convert_simplified_to_traditional(chosen.core_event.strip())
        chosen.score_causal = round(float(chosen.score_causal), 1)
        chosen.score_craft = round(float(chosen.score_craft), 1)
        chosen.score_depth = round(float(chosen.score_depth), 1)
        reconciled.append(chosen)

    return reconciled


def build_factual_pacing_curve(
    chapter_facts: list[ChapterFactExtractContract],
    total_chapters: int,
) -> tuple[list[dict[str, Any]], list[PacingStageTemplateContract]]:
    """Build factual pacing wave curve and milestone stages directly from AI-evaluated chapter facts."""
    n = len(chapter_facts)
    if n == 0:
        raise ValueError("Cannot build factual pacing curve from empty chapter facts.")

    curve_data: list[dict[str, Any]] = []
    for idx, cf in enumerate(chapter_facts):
        # Use the real chapter number: enumerating positions shifts every label when
        # the sampled chapter set has gaps.
        ch_idx = int(cf.chapter_index or idx + 1)
        pct = round((ch_idx / max(1, total_chapters)) * 100.0, 2)
        curve_data.append(
            {
                "ch": ch_idx,
                "pct": pct,
                "t": float(cf.actual_tension),
                "p": float(cf.actual_progress),
                "d": 3.0,
                "phase": cf.pacing_phase,
                "title": cf.chapter_title or f"第{ch_idx}章",
                "purpose": cf.core_event,
                "score_causal": float(cf.score_causal),
                "score_craft": float(cf.score_craft),
                "score_depth": float(cf.score_depth),
            }
        )

    if n < 4:
        ratios = [0.0, 33.3, 66.7, 100.0]
        stages = []
        for i, pct in enumerate(ratios):
            src = curve_data[min(i, n - 1)]
            stages.append(
                PacingStageTemplateContract(
                    stage_index=i + 1,
                    offset_percent=pct,
                    pacing_phase=src["phase"],
                    target_tension=int(src["t"]),
                    target_progress=int(src["p"]),
                    target_density=3,
                    dramatic_purpose=f"第{src['ch']}章【{src['title']}】：{src['purpose']}",
                )
            )
        return curve_data, stages

    if n <= 20:
        stages = []
        for idx, pt in enumerate(curve_data):
            stages.append(
                PacingStageTemplateContract(
                    stage_index=idx + 1,
                    offset_percent=pt["pct"],
                    pacing_phase=pt["phase"],
                    target_tension=int(pt["t"]),
                    target_progress=int(pt["p"]),
                    target_density=3,
                    dramatic_purpose=f"第{pt['ch']}章【{pt['title']}】：{pt['purpose']}",
                )
            )
    else:
        num_bins = min(16, max(6, n // 10))
        stages = []
        for b in range(num_bins):
            s_idx = round(b * n / num_bins)
            e_idx = round((b + 1) * n / num_bins)
            bin_pts = curve_data[s_idx:e_idx]
            if not bin_pts:
                bin_pts = [curve_data[min(s_idx, n - 1)]]
            offset_pct = 0.0 if b == 0 else (100.0 if b == num_bins - 1 else round(((s_idx + e_idx - 1) / (2.0 * max(1, n - 1))) * 100.0, 1))
            peak_pt = max(bin_pts, key=lambda p: (p["t"], p["p"]))
            max_t = int(peak_pt["t"])
            max_p = int(peak_pt["p"])
            phase = peak_pt["phase"]
            ch_range = f"第 {s_idx + 1}～{e_idx} 章" if e_idx > s_idx + 1 else f"第 {s_idx + 1} 章"
            stages.append(
                PacingStageTemplateContract(
                    stage_index=b + 1,
                    offset_percent=offset_pct,
                    pacing_phase=phase,
                    target_tension=max_t,
                    target_progress=max_p,
                    target_density=3,
                    dramatic_purpose=f"{ch_range}：{peak_pt['purpose']}",
                )
            )

    return curve_data, stages


def reconstruct_chapter_facts_from_curve(
    curve_data: list[dict[str, Any]],
) -> list[ChapterFactExtractContract]:
    """Reconstruct ChapterFactExtractContract objects from cached factual_pacing_curve entries."""
    reconstructed: list[ChapterFactExtractContract] = []
    for idx, pt in enumerate(curve_data, start=1):
        ch_idx = int(pt.get("ch", idx))
        phase_raw = str(pt.get("phase", "BUILDUP")).upper()
        if phase_raw not in ("BUILDUP", "COMPLICATION", "CLIMAX", "RESOLUTION"):
            phase_raw = "BUILDUP"
        reconstructed.append(
            ChapterFactExtractContract(
                chapter_index=ch_idx,
                chapter_title=str(pt.get("title") or f"第{ch_idx}章"),
                core_event=str(pt.get("purpose") or pt.get("title") or f"第{ch_idx}章事件"),
                pacing_phase=phase_raw,  # type: ignore[arg-type]
                actual_tension=max(1, min(5, int(round(float(pt.get("t", 3.0)))))),
                actual_progress=max(1, min(5, int(round(float(pt.get("p", 3.0)))))),
                score_causal=max(1.0, min(5.0, float(pt.get("score_causal", 3.0)))),
                score_craft=max(1.0, min(5.0, float(pt.get("score_craft", 3.0)))),
                score_depth=max(1.0, min(5.0, float(pt.get("score_depth", 3.0)))),
            )
        )
    return reconstructed


def apply_pacing_jitter(
    stages: list[PacingStageTemplateContract],
    jitter_ratio: float = 0.1,
    seed: int | None = None,
) -> list[PacingStageTemplateContract]:
    """Inject stochastic phase jitter into middle stages to prevent mechanical repetition."""
    if len(stages) <= 2 or jitter_ratio <= 0.0:
        return stages

    rng = random.Random(seed)
    jittered: list[PacingStageTemplateContract] = []
    n = len(stages)

    for i, st in enumerate(stages):
        if i == 0:
            jittered.append(st.model_copy(update={"offset_percent": 0.0}))
        elif i == n - 1:
            jittered.append(st.model_copy(update={"offset_percent": 100.0}))
        else:
            prev_offset = jittered[i - 1].offset_percent
            next_offset = stages[i + 1].offset_percent
            max_delta = (next_offset - prev_offset) * jitter_ratio * 0.5
            shift = rng.uniform(-max_delta, max_delta)
            new_offset = round(max(prev_offset + 2.0, min(next_offset - 2.0, st.offset_percent + shift)), 1)
            jittered.append(st.model_copy(update={"offset_percent": new_offset}))

    return jittered


class ChapterFactExtractor:
    """Extracts chapter-level facts, consolidates volume arcs, and synthesizes permanent factual blueprints."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        lifecycle: Any | None = None,
    ) -> None:
        """Initialize chapter fact extractor with optional LLM client and lifecycle manager."""
        self.llm_client = llm_client
        self.lifecycle = lifecycle

    async def _wait_if_paused_or_unloaded(self) -> bool:
        """Pause loop: wait while extraction is paused or model is manually unloaded. Returns False if cancelled."""
        paused = False
        while True:
            if self.lifecycle and getattr(self.lifecycle, "is_cancelled", lambda: False)() is True:
                return False
            is_p = bool(self.lifecycle and getattr(self.lifecycle, "is_paused", lambda: False)() is True)
            sm = getattr(self.llm_client, "server_manager", None)
            is_u = getattr(sm, "_manually_unloaded", False) is True
            if not is_p and not is_u:
                break
            paused = True
            await asyncio.sleep(0.5)
        if paused:
            # Resumed: wait until server is healthy before continuing
            while True:
                if self.lifecycle and getattr(self.lifecycle, "is_cancelled", lambda: False)() is True:
                    return False
                sm = getattr(self.llm_client, "server_manager", None)
                if sm and bool(sm.is_healthy()):
                    break
                await asyncio.sleep(0.5)
        return True

    async def _extract_single_segment_facts(
        self,
        chapter_index: int,
        chapter_title: str,
        segment_text: str,
        prior_anchor_event: str = "",
        segment_label: str = "",
    ) -> ChapterFactExtractContract:
        """Extract objective factual increment from a single contiguous chapter or segment."""
        sys_prompt = (
            "你是一位極致客觀的小說事實解構與文體學評估專家。請從給予的小說正文中，忠實於原文事實，精準解構客觀事實增量、判定戲劇階段並評定寫作範式價值。\n\n"
            "【解構與評估規範】：\n"
            "1. 【全篇繁體中文】：所有文字輸出必須完全使用繁體中文（台灣標準書面語）。\n"
            "2. 【core_event】：客觀總結本章實質發生的核心事件因果鏈條（100字以內，起因-衝突-結果，嚴禁主觀抒情或編造）。\n"
            "3. 【pacing_phase】：判定本章於敘事結構中之戲劇階段定性（BUILDUP: 鋪墊蓄勢 / COMPLICATION: 衝突推進與博弈拉扯 / CLIMAX: 矛盾爆發與決戰高潮 / RESOLUTION: 餘波收束與戰果結算）。\n"
            "4. 【character_dynamics】：提取登場核心角色的實質行動、立場變化、動機展露或位階突破（每條一項，無則留空）。\n"
            "5. 【faction_shifts】：提取涉及的勢力交鋒、陣營利益變動或衝突升級（每條一項，無則留空）。\n"
            "6. 【actual_tension (1~5)】：基於真實衝突劇烈程度打分（1=平緩日常，3=矛盾對峙，5=生死決戰）。\n"
            "7. 【actual_progress (1~5)】：基於對全書主線目標推動的實質幅度打分（1=微幅鋪墊，3=階段進展，5=重大跨越）。\n"
            "8. 【章節範式價值三維客觀評估（各評定 1.0～5.0 分，允許一位小數）】：\n"
            "   (a) 局勢相變度（score_causal）：度量章節終點相對於起點在情節因果約束狀態上的不可逆位移量。以故事世界的穩態平衡框架（各方力場均衡、既定規則約定、角色認知邊界）是否受到打破為判定依據。約束狀態維持原狀者為最低（1.0），關鍵約束根本性失效並引發全局軌道不可逆轉移者為最高（5.0）。\n"
            "   (b) 修辭潛流度（score_craft）：度量表層符號與深層實質結構的張力差（潛台詞），及語言符號組織秩序的主動自洽形式感。表層言行與深層意圖完全重疊、語言僅作透明事實交代者為最低（1.0）；表深層高度偏離、潛台詞張力強烈，且語言架構展現高度自洽形式感者為最高（5.0）。\n"
            "   (c) 共鳴認知度（score_depth）：度量文本跨越文字符號對人類心智引發的情感共振強度、知覺沉浸度與哲思啟發深度。心智無波瀾、僅接收冷靜事實告知者為最低（1.0）；知覺強烈共時化沉浸、引發深層情感共振或觸發人性本質認知昇華者為最高（5.0）。\n"
            "   【錨點防通脹標準】：1.0=常態基線零位移；2.0=局部弱度擾動；3.0=基準合格態；4.0=顯著突破態；5.0=極限典範態。"
        )
        anchor_str = f"【前序終局錨點】：\n{prior_anchor_event.strip()}\n\n" if prior_anchor_event.strip() else ""
        header = f"第 {chapter_index} 章：{chapter_title}" + (f"（{segment_label}）" if segment_label else "")
        user_prompt = f"{anchor_str}{header}\n\n正文內容：\n{segment_text.strip()}"

        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                if not await self._wait_if_paused_or_unloaded():
                    raise asyncio.CancelledError()
                temp = 0.20 + (attempt - 1) * 0.08
                res = await self.llm_client.generate_structured(
                    sys_prompt, user_prompt, ChapterFactExtractContract, max_tokens=3072, temperature=temp
                )
                res.chapter_index = chapter_index
                res.chapter_title = convert_simplified_to_traditional(chapter_title)
                res.core_event = convert_simplified_to_traditional(res.core_event.strip())
                res.character_dynamics = [convert_simplified_to_traditional(c.strip()) for c in res.character_dynamics if c.strip()]
                res.faction_shifts = [convert_simplified_to_traditional(f.strip()) for f in res.faction_shifts if f.strip()]
                res.score_causal = round(float(res.score_causal), 1)
                res.score_craft = round(float(res.score_craft), 1)
                res.score_depth = round(float(res.score_depth), 1)
                return res
            except Exception as exc:
                last_exc = exc
                if self.lifecycle and getattr(self.lifecycle, "is_cancelled", lambda: False)() is True:
                    raise asyncio.CancelledError()
                is_p = bool(self.lifecycle and getattr(self.lifecycle, "is_paused", lambda: False)() is True)
                sm = getattr(self.llm_client, "server_manager", None)
                is_u = getattr(sm, "_manually_unloaded", False) is True
                if is_p or is_u:
                    logger.info("Single segment extraction paused due to user pause or manual model unload; waiting for resume...")
                    if not await self._wait_if_paused_or_unloaded():
                        raise asyncio.CancelledError()
                    continue
                if _is_context_size_exceeded(exc) and len(segment_text) > 800:
                    logger.warning(
                        "Segment for chapter %d ('%s' %s) exceeded context window (%d chars); splitting losslessly in half.",
                        chapter_index,
                        chapter_title,
                        segment_label,
                        len(segment_text),
                    )
                    sub_segs = split_long_chapter_lossless(
                        segment_text, max_segment_chars=max(600, len(segment_text) // 2)
                    )
                    if len(sub_segs) > 1:
                        sub_facts: list[ChapterFactExtractContract] = []
                        sub_anchor = prior_anchor_event
                        for sub_i, sub_s in enumerate(sub_segs, start=1):
                            sf = await self._extract_single_segment_facts(
                                chapter_index=chapter_index,
                                chapter_title=chapter_title,
                                segment_text=sub_s,
                                prior_anchor_event=sub_anchor,
                                segment_label=f"{segment_label or '分段'}·子段{sub_i}/{len(sub_segs)}",
                            )
                            sub_facts.append(sf)
                            sub_anchor = sf.core_event
                        return await self._consolidate_segment_facts_tree(
                            chapter_index=chapter_index,
                            chapter_title=chapter_title,
                            segment_facts=sub_facts,
                            prior_anchor_event=prior_anchor_event,
                        )
                logger.warning(
                    "Extraction attempt %d failed for chapter %d ('%s' %s): %s",
                    attempt,
                    chapter_index,
                    chapter_title,
                    segment_label,
                    exc,
                )
                if attempt < 3:
                    await asyncio.sleep(1.0)

        raise RuntimeError(
            f"Failed to extract chapter facts for chapter {chapter_index} ('{chapter_title}') after 3 attempts: {last_exc}"
        )

    @staticmethod
    def _format_segment_facts_summary(facts: list[ChapterFactExtractContract]) -> str:
        """Format a group of segment facts into a structured text block for hierarchical consolidation."""
        return "\n\n".join(
            f"【第 {i} 分段事實】\n"
            f"- 核心事件：{sf.core_event}\n"
            f"- 戲劇階段：{sf.pacing_phase} | 張力：{sf.actual_tension} | 推進：{sf.actual_progress}\n"
            f"- 三維評分：相變={sf.score_causal}, 修辭={sf.score_craft}, 共鳴={sf.score_depth}\n"
            f"- 角色動態：{', '.join(sf.character_dynamics[:6])}\n"
            f"- 勢力交鋒：{', '.join(sf.faction_shifts[:6])}"
            for i, sf in enumerate(facts, start=1)
        )

    async def _consolidate_segment_facts_tree(
        self,
        chapter_index: int,
        chapter_title: str,
        segment_facts: list[ChapterFactExtractContract],
        prior_anchor_event: str = "",
        max_group_size: int = 6,
    ) -> ChapterFactExtractContract:
        """Hierarchically consolidate multi-segment chapter facts using B-Tree reduction (k <= 6).

        Guarantees that no consolidation prompt ever exceeds ~2,400 input tokens regardless of
        how many segments a chapter contains (e.g. 70+ segments in an unchaptered or mega-chapter text).
        """
        current_facts = list(segment_facts)
        total_initial = len(current_facts)
        tier = 1

        while len(current_facts) > max_group_size:
            next_tier_facts: list[ChapterFactExtractContract] = []
            num_groups = (len(current_facts) + max_group_size - 1) // max_group_size
            for g_idx in range(num_groups):
                chunk = current_facts[g_idx * max_group_size : (g_idx + 1) * max_group_size]
                seg_summary = self._format_segment_facts_summary(chunk)
                group_anchor = (
                    prior_anchor_event
                    if g_idx == 0
                    else (next_tier_facts[-1].core_event if next_tier_facts else "")
                )
                cf = await self._extract_single_segment_facts(
                    chapter_index=chapter_index,
                    chapter_title=chapter_title,
                    segment_text=(
                        f"本章為超長章節（共 {total_initial} 段），以下為第 {tier} 層第 {g_idx + 1}/{num_groups} 組"
                        f"分段事實，請收斂為本組階段整體事實與評分：\n\n{seg_summary}"
                    ),
                    prior_anchor_event=group_anchor,
                    segment_label=f"第{tier}層收斂 {g_idx + 1}/{num_groups}",
                )
                next_tier_facts.append(cf)
            current_facts = next_tier_facts
            tier += 1

        if len(current_facts) == 1:
            return current_facts[0]

        seg_summary = self._format_segment_facts_summary(current_facts)
        return await self._extract_single_segment_facts(
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            segment_text=f"本章為超長章節，共分為 {total_initial} 段完整無損閱讀，請將以下各分段事實收斂為本章整體事實與評分：\n\n{seg_summary}",
            prior_anchor_event=prior_anchor_event,
            segment_label="全章無損匯總",
        )

    async def extract_chapter_facts(
        self,
        chapter_index: int,
        chapter_title: str,
        chapter_text: str,
        prior_anchor_event: str = "",
    ) -> ChapterFactExtractContract:
        """Extract objective factual increment for a single chapter with lossless segmentation and B-Tree consolidation."""
        if not await self._wait_if_paused_or_unloaded():
            raise asyncio.CancelledError()
        if not self.llm_client or not self.llm_client.server_manager.is_healthy():
            raise AIModelUnavailableError("LLM server is unavailable or unhealthy for literary fact extraction.")

        segments = split_long_chapter_lossless(chapter_text, max_segment_chars=7200)
        if len(segments) == 1:
            return await self._extract_single_segment_facts(
                chapter_index=chapter_index,
                chapter_title=chapter_title,
                segment_text=segments[0],
                prior_anchor_event=prior_anchor_event,
            )

        segment_facts: list[ChapterFactExtractContract] = []
        cur_anchor = prior_anchor_event
        for s_idx, seg in enumerate(segments, start=1):
            sf = await self._extract_single_segment_facts(
                chapter_index=chapter_index,
                chapter_title=chapter_title,
                segment_text=seg,
                prior_anchor_event=cur_anchor,
                segment_label=f"第 {s_idx}/{len(segments)} 分段",
            )
            segment_facts.append(sf)
            cur_anchor = sf.core_event

        return await self._consolidate_segment_facts_tree(
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            segment_facts=segment_facts,
            prior_anchor_event=prior_anchor_event,
        )

    async def extract_batched_chapter_facts(
        self,
        chapters: list[dict[str, Any]],
        prior_anchor_event: str = "",
    ) -> BatchedFactExtractionContract:
        """Extract high-density factual increments and 3D ratings for a dynamic group of 1~3 chapters."""
        if not await self._wait_if_paused_or_unloaded():
            raise asyncio.CancelledError()
        if not self.llm_client or not self.llm_client.server_manager.is_healthy():
            raise AIModelUnavailableError("LLM server is unavailable or unhealthy for literary fact extraction.")

        # If any chapter in the group exceeds single-pass segment threshold, extract chapters individually losslessly
        if any(len(str(ch.get("text", "")).strip()) > 7200 for ch in chapters):
            return await self._extract_batch_via_single_chapters(chapters, prior_anchor_event)

        ch_count = len(chapters)
        ch_indices_str = ", ".join(str(c.get("index", idx + 1)) for idx, c in enumerate(chapters))
        sys_prompt = (
            "你是一位極致客觀的小說事實解構與文體學評估專家。\n"
            f"請閱讀以下連續小說章節正文（本組共包含 {ch_count} 個章節，章節序號分別為：{ch_indices_str}）。\n"
            "請嚴格依序為傳入的每個章節解構客觀事實、判定戲劇階段與三維範式評分，並在末尾總結本組之實質角色、勢力與法則增量。\n\n"
            "【解構與評估鐵律】：\n"
            "1. 【全篇繁體中文】：所有文字輸出必須完全使用繁體中文（台灣標準書面語）。\n"
            f"2. 【章節獨立事實 (chapters)】：本組輸入恰好包含 {ch_count} 章，輸出陣列 chapters 的長度【必須且僅能恰好等於 {ch_count}】！\n"
            "   - 嚴格一對一對齊：輸出陣列的每一項必須精準對應上述輸入章節序號與標題，嚴禁將單一章節拆分成多項輸出，嚴禁遺漏任何章節，亦嚴禁額外憑空生成！\n"
            "   - chapter_index, chapter_title：精準對應輸入章節的序號與標題。\n"
            "   - core_event：客觀總結本章實質發生的核心事件因果鏈（40~80字，起因-衝突-結果，嚴禁流水帳或抒情）。\n"
            "   - pacing_phase：判定本章於敘事結構中之戲劇階段定性（BUILDUP: 鋪墊蓄勢 / COMPLICATION: 衝突推進與博弈拉扯 / CLIMAX: 矛盾爆發與決戰高潮 / RESOLUTION: 餘波收束與戰果結算）。\n"
            "   - actual_tension (1~5)：客觀評定本章文字實質呈現的衝突激烈程度（1=平緩日常，3=矛盾對峙，5=生死決戰）。\n"
            "   - actual_progress (1~5)：客觀評定本章實質推動全書核心主線的進展幅度（1~5）。\n"
            "   - 【章節範式價值三維評估（各評定 1.0～5.0 分，允許一位小數）】：\n"
            "     (a) 局勢相變度（score_causal）：度量章節終點相對於起點在情節因果約束狀態上的不可逆位移量。第一章必須對照【前組終局錨點】評定是否打破穩態平衡；後續章節對照組內前章評定。約束狀態維持原狀者為最低（1.0），關鍵約束根本性失效並引發全局軌道不可逆轉移者為最高（5.0）。\n"
            "     (b) 修辭潛流度（score_craft）：度量表層符號與深層實質結構的張力差（潛台詞），及語言符號組織秩序的主動自洽形式感（1.0=表深重合透明交代，5.0=表深高度對立且形式自洽）。\n"
            "     (c) 共鳴認知度（score_depth）：度量文本跨越文字符號對人類心智引發的情感共振強度、知覺沉浸度與哲思啟發深度（1.0=心智無波瀾，5.0=深刻知覺共時化與認知昇華）。\n"
            "     【錨點防通脹標準】：1.0=常態基線零位移；2.0=局部弱度擾動；3.0=基準合格態；4.0=顯著突破態；5.0=極限典範態。\n"
            "3. 【組級實質增量 (entity_deltas)】：集中總結本組章節內發生的實質增量（僅記錄實質重大變化，無則留空，嚴禁佔位廢話）：\n"
            "   - character_milestones：核心角色之重大位階突破、身分轉換、動機揭露或生死立場逆轉。\n"
            "   - faction_dynamics：新勢力登場、陣營關係演變或利益結構衝突。\n"
            "   - world_rules_and_hierarchy：新揭曉或實質變更的位階階梯、組織制度、運轉鐵律或不可違背之代價禁忌（無則留空）。"
        )

        anchor_section = ""
        if prior_anchor_event.strip():
            anchor_section = f"【前組終局錨點（上一組末章狀態）】：\n{prior_anchor_event.strip()}\n\n"

        prose_blocks = []
        for ch in chapters:
            idx = ch.get("index", 1)
            title = ch.get("title", f"第{idx}章")
            text = str(ch.get("text", "")).strip()
            prose_blocks.append(f"=== 第 {idx} 章：{title} ===\n{text}")

        user_prompt = f"{anchor_section}" + "\n\n".join(prose_blocks)

        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                if not await self._wait_if_paused_or_unloaded():
                    raise asyncio.CancelledError()
                temp = 0.20 + (attempt - 1) * 0.08
                res = await self.llm_client.generate_structured(
                    sys_prompt, user_prompt, BatchedFactExtractionContract, max_tokens=3500, temperature=temp
                )
                res.chapters = reconcile_batched_compact_facts(res.chapters, chapters)

                res.entity_deltas.character_milestones = [
                    convert_simplified_to_traditional(m.strip())
                    for m in res.entity_deltas.character_milestones
                    if m.strip()
                ]
                res.entity_deltas.faction_dynamics = [
                    convert_simplified_to_traditional(f.strip())
                    for f in res.entity_deltas.faction_dynamics
                    if f.strip()
                ]
                res.entity_deltas.world_rules_and_hierarchy = [
                    convert_simplified_to_traditional(w.strip())
                    for w in res.entity_deltas.world_rules_and_hierarchy
                    if w.strip()
                ]
                return res
            except Exception as exc:
                last_exc = exc
                if self.lifecycle and getattr(self.lifecycle, "is_cancelled", lambda: False)() is True:
                    raise asyncio.CancelledError()
                is_p = bool(self.lifecycle and getattr(self.lifecycle, "is_paused", lambda: False)() is True)
                sm = getattr(self.llm_client, "server_manager", None)
                is_u = getattr(sm, "_manually_unloaded", False) is True
                if is_p or is_u:
                    logger.info("Batched extraction paused due to user pause or manual model unload; waiting for resume...")
                    if not await self._wait_if_paused_or_unloaded():
                        raise asyncio.CancelledError()
                    continue
                if _is_context_size_exceeded(exc):
                    logger.warning(
                        "Batch %s exceeded context window; immediately falling back to individual lossless chapter extraction.",
                        [c.get("index") for c in chapters],
                    )
                    return await self._extract_batch_via_single_chapters(chapters, prior_anchor_event)
                logger.warning(
                    "Batched extraction attempt %d failed for chapters %s: %s",
                    attempt,
                    [c.get("index") for c in chapters],
                    exc,
                )
                if attempt < 3:
                    await asyncio.sleep(1.0)

        # Fall back to single-chapter object extraction (ChapterFactExtractContract) for any
        # batch size (including 1-chapter batches) so an LLM array-length mismatch cannot abort a book.
        if last_exc is not None:
            logger.info(
                "Falling back to individual lossless chapter extraction for batch %s after batch failure: %s",
                [c.get("index") for c in chapters],
                last_exc,
            )
            return await self._extract_batch_via_single_chapters(chapters, prior_anchor_event)

        raise RuntimeError(
            f"Failed to extract batched facts for chapters {[c.get('index') for c in chapters]} after 3 attempts: {last_exc}"
        )

    async def _extract_batch_via_single_chapters(
        self,
        chapters: list[dict[str, Any]],
        prior_anchor_event: str = "",
    ) -> BatchedFactExtractionContract:
        """Extract each chapter in a batch individually via AI without any truncation or synthetic placeholders."""
        compact_list: list[SingleChapterCompactFact] = []
        char_milestones: list[str] = []
        faction_dyn: list[str] = []
        cur_anchor = prior_anchor_event

        for idx, ch in enumerate(chapters):
            ch_idx = int(ch.get("index", idx + 1))
            ch_title = str(ch.get("title", f"第{ch_idx}章"))
            ch_text = str(ch.get("text", ""))
            cf = await self.extract_chapter_facts(
                chapter_index=ch_idx,
                chapter_title=ch_title,
                chapter_text=ch_text,
                prior_anchor_event=cur_anchor,
            )
            compact_list.append(
                SingleChapterCompactFact(
                    chapter_index=cf.chapter_index,
                    chapter_title=cf.chapter_title,
                    core_event=cf.core_event,
                    pacing_phase=cf.pacing_phase,
                    actual_tension=cf.actual_tension,
                    actual_progress=cf.actual_progress,
                    score_causal=cf.score_causal,
                    score_craft=cf.score_craft,
                    score_depth=cf.score_depth,
                )
            )
            char_milestones.extend(cf.character_dynamics)
            faction_dyn.extend(cf.faction_shifts)
            cur_anchor = cf.core_event

        # The compact batched path intentionally skips world-rule extraction; log it so
        # a thin blueprint is traceable instead of looking like a silent model failure.
        logger.info(
            "Batched chapter facts produced without world_rules_and_hierarchy (chapters=%d).",
            len(compact_list),
        )
        return BatchedFactExtractionContract(
            chapters=compact_list,
            entity_deltas=GroupEntityDeltas(
                character_milestones=list(dict.fromkeys(char_milestones))[:5],
                faction_dynamics=list(dict.fromkeys(faction_dyn))[:5],
                world_rules_and_hierarchy=[],
            ),
        )

    async def consolidate_volume_facts(
        self,
        volume_index: int,
        chapter_facts: list[ChapterFactExtractContract],
        volume_entity_deltas: list[GroupEntityDeltas] | None = None,
    ) -> VolumeFactConsolidationContract:
        """Consolidate a cluster/volume of chapter facts into a coherent volume-level dramatic arc."""
        ch_count = len(chapter_facts)
        first_ch = chapter_facts[0].chapter_index if chapter_facts else 1
        last_ch = chapter_facts[-1].chapter_index if chapter_facts else 1
        vol_title = f"第{volume_index}卷（第{first_ch}～{last_ch}章）"

        if not await self._wait_if_paused_or_unloaded():
            raise asyncio.CancelledError()
        if not self.llm_client or not self.llm_client.server_manager.is_healthy():
            raise AIModelUnavailableError("LLM server is unavailable or unhealthy for volume consolidation.")

        facts_lines = [
            f"- 第{cf.chapter_index}章【{cf.chapter_title}】(張力:{cf.actual_tension}, 推進:{cf.actual_progress}): {cf.core_event}"
            for cf in chapter_facts
        ]
        deltas_lines = []
        if volume_entity_deltas:
            all_chars = list(dict.fromkeys(m for d in volume_entity_deltas for m in d.character_milestones if m.strip()))
            all_factions = list(dict.fromkeys(f for d in volume_entity_deltas for f in d.faction_dynamics if f.strip()))
            all_rules = list(dict.fromkeys(r for d in volume_entity_deltas for r in d.world_rules_and_hierarchy if r.strip()))
            if all_chars:
                deltas_lines.append("【本卷角色關鍵突破與動態】：\n- " + "\n- ".join(all_chars))
            if all_factions:
                deltas_lines.append("【本卷勢力版圖與利益衝突】：\n- " + "\n- ".join(all_factions))
            if all_rules:
                deltas_lines.append("【本卷位階體系與底層法則記錄】：\n- " + "\n- ".join(all_rules))

        facts_summary = "\n".join(facts_lines)
        if deltas_lines:
            facts_summary += "\n\n" + "\n\n".join(deltas_lines)

        sys_prompt = (
            "你是一位長篇小說分卷大綱整合專家。請根據給予的章節事實清單與實質增量記錄，收斂整合出本卷的宏觀劇情弧、核心矛盾結果、角色關鍵成長、勢力格局天平與世界觀規則體系。\n\n"
            "【整合規範與鐵律】：\n"
            "1. 全程使用繁體中文（台灣標準書面語）輸出。\n"
            "2. 100%完全忠於輸入的章節事實與實質增量，嚴禁憑空捏造未發生的情節。\n"
            "3. volume_arc_summary：梳理本卷情節的起承轉合主線走勢（200~350字）。\n"
            "4. core_conflict_resolution：說明本卷核心矛盾的激化或解決結果（100~200字）。\n"
            "5. character_milestones：提煉核心角色在本卷取得的關鍵突破、身份轉換或生死變故（最多8條）。\n"
            "6. faction_geopolitics_summary：總結本卷結束時各大勢力實力格局天平演變現況（100~200字）。\n"
            "7. world_rules_and_hierarchy_summary：總結本卷揭曉或驗證的核心位階階梯特徵、晉升門檻、組織制度與不可違背之底層運轉法則或代價禁忌（若題材無特殊位階，請總結行業/社會/法律規則，100~200字）。"
        )
        user_prompt = f"分卷：{vol_title}（共{ch_count}章）\n\n事實與實體增量清單：\n{facts_summary}"
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                if not await self._wait_if_paused_or_unloaded():
                    raise asyncio.CancelledError()
                temp = 0.20 + (attempt - 1) * 0.08
                res = await self.llm_client.generate_structured(
                    sys_prompt, user_prompt, VolumeFactConsolidationContract, max_tokens=3072, temperature=temp
                )
                res.volume_index = volume_index
                res.volume_title = vol_title
                res.volume_arc_summary = convert_simplified_to_traditional(res.volume_arc_summary.strip())
                res.core_conflict_resolution = convert_simplified_to_traditional(res.core_conflict_resolution.strip())
                res.character_milestones = [convert_simplified_to_traditional(m.strip()) for m in res.character_milestones if m.strip()]
                res.faction_geopolitics_summary = convert_simplified_to_traditional(res.faction_geopolitics_summary.strip())
                res.world_rules_and_hierarchy_summary = convert_simplified_to_traditional(res.world_rules_and_hierarchy_summary.strip())
                return res
            except Exception as exc:
                last_exc = exc
                if self.lifecycle and getattr(self.lifecycle, "is_cancelled", lambda: False)() is True:
                    raise asyncio.CancelledError()
                is_p = bool(self.lifecycle and getattr(self.lifecycle, "is_paused", lambda: False)() is True)
                sm = getattr(self.llm_client, "server_manager", None)
                is_u = getattr(sm, "_manually_unloaded", False) is True
                if is_p or is_u:
                    logger.info("Volume consolidation paused due to user pause or manual model unload; waiting for resume...")
                    if not await self._wait_if_paused_or_unloaded():
                        raise asyncio.CancelledError()
                    continue
                if _is_context_size_exceeded(exc) and len(chapter_facts) > 1:
                    mid = len(chapter_facts) // 2
                    logger.warning(
                        "Volume %d consolidation exceeded context window (%d chapters); splitting into two sub-volumes.",
                        volume_index,
                        len(chapter_facts),
                    )
                    sub1 = await self.consolidate_volume_facts(
                        volume_index=volume_index,
                        chapter_facts=chapter_facts[:mid],
                        volume_entity_deltas=None,
                    )
                    sub2 = await self.consolidate_volume_facts(
                        volume_index=volume_index,
                        chapter_facts=chapter_facts[mid:],
                        volume_entity_deltas=None,
                    )
                    merged_facts = [
                        ChapterFactExtractContract(
                            chapter_index=first_ch,
                            chapter_title=f"{sub1.volume_title}上半卷",
                            core_event=f"{sub1.volume_arc_summary} {sub1.core_conflict_resolution}",
                            character_dynamics=sub1.character_milestones,
                            faction_shifts=[sub1.faction_geopolitics_summary],
                        ),
                        ChapterFactExtractContract(
                            chapter_index=last_ch,
                            chapter_title=f"{sub2.volume_title}下半卷",
                            core_event=f"{sub2.volume_arc_summary} {sub2.core_conflict_resolution}",
                            character_dynamics=sub2.character_milestones,
                            faction_shifts=[sub2.faction_geopolitics_summary],
                        ),
                    ]
                    return await self.consolidate_volume_facts(
                        volume_index=volume_index,
                        chapter_facts=merged_facts,
                        volume_entity_deltas=None,
                    )
                logger.warning("Volume consolidation attempt %d failed for volume %d: %s", attempt, volume_index, exc)
                if attempt < 3:
                    await asyncio.sleep(1.0)

        raise RuntimeError(f"Failed to consolidate volume {volume_index} after 3 attempts: {last_exc}")

    async def _consolidate_arc_slice(
        self,
        volume_reports: list[VolumeFactConsolidationContract],
        target_count: int,
        max_vols_per_chunk: int = 6,
        hardware_guard: Any | None = None,
        lifecycle: Any | None = None,
        is_sub_tier: bool = False,
    ) -> list[MajorArcConsolidationContract]:
        """Consolidate a contiguous slice of volume reports into MajorArcConsolidationContract items."""
        total_vols = len(volume_reports)
        vols_per_arc = min(max_vols_per_chunk, max(1, (total_vols + target_count - 1) // target_count))
        actual_count = (total_vols + vols_per_arc - 1) // vols_per_arc
        major_arcs: list[MajorArcConsolidationContract] = []

        sys_prompt = (
            "你是一位通曉長篇敘事宏觀結構的殿堂級文學總策劃。\n"
            "請將以下若干卷的分卷報告，依據情節主線發展脈絡，收斂整合成一個宏觀情節大幕（Major Narrative Arc）。\n\n"
            "【整合規範】：\n"
            "1. 全程使用繁體中文（台灣標準書面語）輸出。\n"
            "2. 100%忠於分卷報告事實，客觀整合。\n"
            "3. arc_title：提煉概括本大幕的結構性主題標題（以幕次與核心因果轉折命題命名）。\n"
            "4. arc_summary：梳理本大幕情節起承轉合主線走勢與重大戰略轉折（300~500字）。\n"
            "5. conflict_and_resolution：本大幕核心矛盾爆發與收束結果（150~250字）。\n"
            "6. character_milestones_summary：本大幕核心角色之命運轉折、陣營歸屬與實力/位階突破（最多8條）。\n"
            "7. faction_geopolitics_summary：本大幕結束時各大勢力門派版圖與地緣/利益天平現況（150~250字）。\n"
            "8. world_rules_and_hierarchy_summary：本大幕揭示或深化之位階階梯特徵、核心體制與底層運轉法則或代價禁忌（150~250字；若無特殊位階體系，請總結大局規則）。"
        )

        for a_idx in range(actual_count):
            if lifecycle and lifecycle.is_cancelled():
                logger.info("Arc consolidation cancelled by user.")
                break
            while lifecycle and lifecycle.is_paused():
                if lifecycle.is_cancelled():
                    break
                await asyncio.sleep(0.5)
            if lifecycle and lifecycle.is_cancelled():
                break

            v_start = a_idx * vols_per_arc
            v_end = min(total_vols, (a_idx + 1) * vols_per_arc)
            arc_vols = volume_reports[v_start:v_end]
            if not arc_vols:
                break

            cov_str = f"第 {arc_vols[0].volume_index} 卷至第 {arc_vols[-1].volume_index} 卷"
            vols_text = "\n\n".join(
                f"=== {vr.volume_title} ===\n"
                f"劇情走勢：{vr.volume_arc_summary}\n"
                f"核心矛盾結果：{vr.core_conflict_resolution}\n"
                f"角色里程碑：{', '.join(vr.character_milestones)}\n"
                f"勢力現況：{vr.faction_geopolitics_summary}\n"
                f"位階與法則：{vr.world_rules_and_hierarchy_summary}"
                for vr in arc_vols
            )

            tier_label = "階段子大幕" if is_sub_tier else "情節大幕"
            user_prompt = f"{tier_label}編號：第 {a_idx + 1} 幕（涵蓋：{cov_str}）\n\n所屬分卷報告清單：\n{vols_text}"
            last_exc: Exception | None = None
            for attempt in range(1, 4):
                try:
                    if not await self._wait_if_paused_or_unloaded():
                        raise asyncio.CancelledError()
                    temp = 0.20 + (attempt - 1) * 0.08
                    res = await self.llm_client.generate_structured(
                        sys_prompt, user_prompt, MajorArcConsolidationContract, max_tokens=2500, temperature=temp
                    )
                    res.arc_index = a_idx + 1
                    res.covered_volumes = cov_str
                    res.arc_title = convert_simplified_to_traditional(res.arc_title.strip())
                    res.arc_summary = convert_simplified_to_traditional(res.arc_summary.strip())
                    res.conflict_and_resolution = convert_simplified_to_traditional(res.conflict_and_resolution.strip())
                    res.character_milestones_summary = [convert_simplified_to_traditional(m.strip()) for m in res.character_milestones_summary if m.strip()]
                    res.faction_geopolitics_summary = convert_simplified_to_traditional(res.faction_geopolitics_summary.strip())
                    res.world_rules_and_hierarchy_summary = convert_simplified_to_traditional(res.world_rules_and_hierarchy_summary.strip())
                    major_arcs.append(res)
                    if hardware_guard:
                        await hardware_guard.check_and_cooldown_if_hot()
                    break
                except Exception as exc:
                    last_exc = exc
                    if self.lifecycle and getattr(self.lifecycle, "is_cancelled", lambda: False)() is True:
                        raise asyncio.CancelledError()
                    is_p = bool(self.lifecycle and getattr(self.lifecycle, "is_paused", lambda: False)() is True)
                    sm = getattr(self.llm_client, "server_manager", None)
                    is_u = getattr(sm, "_manually_unloaded", False) is True
                    if is_p or is_u:
                        logger.info("Arc consolidation paused due to user pause or manual model unload; waiting for resume...")
                        if not await self._wait_if_paused_or_unloaded():
                            raise asyncio.CancelledError()
                        continue
                    logger.warning("Arc consolidation attempt %d failed for arc %d: %s", attempt, a_idx + 1, exc)
                    if attempt < 3:
                        await asyncio.sleep(1.0)
            else:
                raise RuntimeError(f"Failed to consolidate {tier_label} {a_idx + 1} after 3 attempts: {last_exc}")

        return major_arcs

    async def _consolidate_intermediate_arcs(
        self,
        sub_arcs: list[MajorArcConsolidationContract],
        target_count: int = 5,
        hardware_guard: Any | None = None,
        lifecycle: Any | None = None,
    ) -> list[MajorArcConsolidationContract]:
        """Consolidate Tier-1 sub-arcs into final 4~6 Major Narrative Arcs."""
        total_subs = len(sub_arcs)
        if total_subs <= target_count:
            return sub_arcs

        subs_per_arc = max(1, (total_subs + target_count - 1) // target_count)
        final_arcs: list[MajorArcConsolidationContract] = []

        sys_prompt = (
            "你是一位通曉長篇敘事宏觀結構的殿堂級文學總策劃。\n"
            "請將以下若干個連續情節子階段（子大幕），收斂整合成全書核心宏觀情節大幕（Major Narrative Arc）。\n\n"
            "【整合規範】：\n"
            "1. 全程使用繁體中文（台灣標準書面語）輸出。\n"
            "2. 100%忠於輸入的各子階段情節事實，精準收斂全局演變。\n"
            "3. arc_title：提煉概括本大幕的結構性主題標題（以幕次與核心因果轉折命題命名）。\n"
            "4. arc_summary：梳理本大幕情節起承轉合主線走勢與重大戰略轉折（300~500字）。\n"
            "5. conflict_and_resolution：本大幕核心矛盾爆發與收束結果（150~250字）。\n"
            "6. character_milestones_summary：本大幕核心角色之命運轉折、陣營歸屬與實力/位階突破（最多8條）。\n"
            "7. faction_geopolitics_summary：本大幕結束時各大勢力門派版圖與地緣/利益天平現況（150~250字）。\n"
            "8. world_rules_and_hierarchy_summary：本大幕揭示或深化之位階階梯特徵、核心體制與底層運轉法則或代價禁忌（150~250字；若無特殊位階體系，請總結大局規則）。"
        )

        for a_idx in range(target_count):
            if lifecycle and lifecycle.is_cancelled():
                logger.info("Tier-2 arc consolidation cancelled by user.")
                break
            while lifecycle and lifecycle.is_paused():
                if lifecycle.is_cancelled():
                    break
                await asyncio.sleep(0.5)
            if lifecycle and lifecycle.is_cancelled():
                break

            s_start = a_idx * subs_per_arc
            s_end = min(total_subs, (a_idx + 1) * subs_per_arc)
            chunk = sub_arcs[s_start:s_end]
            if not chunk:
                break

            cov_first = chunk[0].covered_volumes.split("至")[0].strip()
            cov_last = chunk[-1].covered_volumes.split("至")[-1].strip()
            cov_str = f"{cov_first}至{cov_last}"

            subs_text = "\n\n".join(
                f"=== {sa.arc_title} ({sa.covered_volumes}) ===\n"
                f"劇情走勢：{sa.arc_summary}\n"
                f"核心矛盾結果：{sa.conflict_and_resolution}\n"
                f"角色里程碑：{', '.join(sa.character_milestones_summary)}\n"
                f"勢力現況：{sa.faction_geopolitics_summary}\n"
                f"位階與法則：{sa.world_rules_and_hierarchy_summary}"
                for sa in chunk
            )

            user_prompt = f"宏觀情節大幕編號：第 {a_idx + 1} 幕（涵蓋：{cov_str}）\n\n所屬情節子階段清單：\n{subs_text}"
            last_exc: Exception | None = None
            for attempt in range(1, 4):
                try:
                    if not await self._wait_if_paused_or_unloaded():
                        raise asyncio.CancelledError()
                    temp = 0.20 + (attempt - 1) * 0.08
                    res = await self.llm_client.generate_structured(
                        sys_prompt, user_prompt, MajorArcConsolidationContract, max_tokens=2500, temperature=temp
                    )
                    res.arc_index = a_idx + 1
                    res.covered_volumes = cov_str
                    res.arc_title = convert_simplified_to_traditional(res.arc_title.strip())
                    res.arc_summary = convert_simplified_to_traditional(res.arc_summary.strip())
                    res.conflict_and_resolution = convert_simplified_to_traditional(res.conflict_and_resolution.strip())
                    res.character_milestones_summary = [convert_simplified_to_traditional(m.strip()) for m in res.character_milestones_summary if m.strip()]
                    res.faction_geopolitics_summary = convert_simplified_to_traditional(res.faction_geopolitics_summary.strip())
                    res.world_rules_and_hierarchy_summary = convert_simplified_to_traditional(res.world_rules_and_hierarchy_summary.strip())
                    final_arcs.append(res)
                    if hardware_guard:
                        await hardware_guard.check_and_cooldown_if_hot()
                    break
                except Exception as exc:
                    last_exc = exc
                    if self.lifecycle and getattr(self.lifecycle, "is_cancelled", lambda: False)() is True:
                        raise asyncio.CancelledError()
                    is_p = bool(self.lifecycle and getattr(self.lifecycle, "is_paused", lambda: False)() is True)
                    sm = getattr(self.llm_client, "server_manager", None)
                    is_u = getattr(sm, "_manually_unloaded", False) is True
                    if is_p or is_u:
                        logger.info("Intermediate arc consolidation paused due to user pause or manual model unload; waiting for resume...")
                        if not await self._wait_if_paused_or_unloaded():
                            raise asyncio.CancelledError()
                        continue
                    logger.warning("Tier-2 arc consolidation attempt %d failed for arc %d: %s", attempt, a_idx + 1, exc)
                    if attempt < 3:
                        await asyncio.sleep(1.0)
            else:
                raise RuntimeError(f"Failed to consolidate final major arc {a_idx + 1} after 3 attempts: {last_exc}")

        return final_arcs

    async def consolidate_major_arcs(
        self,
        volume_reports: list[VolumeFactConsolidationContract],
        num_arcs: int | None = None,
        hardware_guard: Any | None = None,
        lifecycle: Any | None = None,
    ) -> list[MajorArcConsolidationContract]:
        """Hierarchically consolidate volume reports into major narrative arcs to eliminate blueprint prompt truncation.

        For novels with up to ~36 volumes, performs single-tier reduction into 2~6 major arcs.
        For epic-length novels (>36 volumes, e.g. 1500+ chapters), performs a two-tier tree reduction
        guaranteeing prompt token counts never exceed context limits.
        """
        if not volume_reports:
            return []
        total_vols = len(volume_reports)
        if total_vols <= 4 and num_arcs is None:
            return []

        if not self.llm_client or not self.llm_client.server_manager.is_healthy():
            raise AIModelUnavailableError("LLM server is unavailable or unhealthy for major arc consolidation.")

        # Mathematically derived B-Tree reduction:
        # Branching factor k <= 6 guarantees prompt input tokens <= 4,800 tokens << 16,128 context window.
        # Target final count: M_final in [2, 6], matching standard dramatic act structures.
        # Convergence depth: D = max(0, ceil(log_k(N / M_final))).
        if num_arcs is None:
            target_final_arcs = min(6, max(2, (total_vols + 4) // 5))
        else:
            target_final_arcs = min(6, max(2, num_arcs))

        # Initial reduction from volume_reports
        initial_target = min(target_final_arcs, total_vols) if total_vols <= 36 else (total_vols + 5) // 6
        current_arcs = await self._consolidate_arc_slice(
            volume_reports=volume_reports,
            target_count=initial_target,
            max_vols_per_chunk=6,
            hardware_guard=hardware_guard,
            lifecycle=lifecycle,
            is_sub_tier=(total_vols > 36),
        )

        tier = 2
        while len(current_arcs) > target_final_arcs:
            if lifecycle and lifecycle.is_cancelled():
                break
            next_target = target_final_arcs if len(current_arcs) <= 36 else (len(current_arcs) + 5) // 6
            logger.info(
                "Running Tier-%d reduction: reducing %d sub-arcs to %d arcs (<= 6 per chunk).",
                tier,
                len(current_arcs),
                next_target,
            )
            current_arcs = await self._consolidate_intermediate_arcs(
                sub_arcs=current_arcs,
                target_count=next_target,
                hardware_guard=hardware_guard,
                lifecycle=lifecycle,
            )
            tier += 1

        return current_arcs

    async def synthesize_macro_blueprint(
        self,
        novel_title: str,
        volume_reports: list[VolumeFactConsolidationContract],
        chapter_facts: list[ChapterFactExtractContract],
        genre_primary: str | None = None,
        genre_sub: str | None = None,
        major_arcs: list[MajorArcConsolidationContract] | None = None,
    ) -> BookBlueprintSynthesisContract:
        """Synthesize permanent macro blueprint from consolidated major arcs and volume reports."""
        if not self.llm_client or not self.llm_client.server_manager.is_healthy():
            raise AIModelUnavailableError("LLM server is unavailable or unhealthy for blueprint synthesis.")

        # Autonomous defense: If major_arcs was not provided but volume_reports > 4, auto-consolidate
        if not major_arcs and len(volume_reports) > 4:
            logger.info(
                "Blueprint synthesis received %d volumes without major arcs; triggering automatic hierarchical reduction.",
                len(volume_reports),
            )
            major_arcs = await self.consolidate_major_arcs(volume_reports)

        if major_arcs:
            summary_content = "\n\n".join(
                f"=== {arc.arc_title} ({arc.covered_volumes}) ===\n"
                f"劇情走勢：{arc.arc_summary}\n"
                f"核心矛盾結果：{arc.conflict_and_resolution}\n"
                f"角色里程碑：{', '.join(arc.character_milestones_summary)}\n"
                f"勢力現況：{arc.faction_geopolitics_summary}\n"
                f"位階與法則：{arc.world_rules_and_hierarchy_summary}"
                for arc in major_arcs
            )
            header_label = f"全書 {len(major_arcs)} 大情節幕宏觀收斂匯報"
        else:
            summary_content = "\n\n".join(
                f"=== {vr.volume_title} ===\n"
                f"劇情弧：{vr.volume_arc_summary}\n"
                f"核心矛盾：{vr.core_conflict_resolution}\n"
                f"角色里程碑：{', '.join(vr.character_milestones)}\n"
                f"勢力現況：{vr.faction_geopolitics_summary}\n"
                f"位階與法則：{vr.world_rules_and_hierarchy_summary}"
                for vr in volume_reports
            )
            header_label = "全書各卷事實收斂匯報"

        sys_prompt = (
            "你是一位精通長篇小說底層架構設計的殿堂級總策劃。\n"
            "請根據整本書的宏觀情節收斂報告，站在俯瞰全域的高度，提煉收斂出全書的永久事實藍圖。\n\n"
            "【提煉標準與鐵律】：\n"
            "1. 【全篇繁體中文】：所有文字輸出必須完全使用繁體中文（台灣標準書面語）。\n"
            "2. 【genre_primary / genre_sub】：依據全書實質世界觀設定、力量或社會體系與核心敘事母題，客觀判定其主題材分類與子題材分類。\n"
            "3. 【macro_outline】：全書宏觀大綱骨幹深度解析（長度要求 500～800 字）。站在全局高維俯瞰視角，必須結構化劃分為四幕展開：\n"
            "   - 【第一幕·開端破局】：初始平衡狀態打破、核心懸念與主角核心動機確立。\n"
            "   - 【第二幕·深入博弈】：阻力層層加碼、多方勢力交錯牽制、中點重大危機或認知反轉。\n"
            "   - 【第三幕·高潮相變】：全局矛盾徹底白熱化、關鍵約束崩潰與不可逆終極決戰。\n"
            "   - 【第四幕·終局收束】：秩序重塑、伏筆收攏、代價結算與深層主題昇華。\n"
            "4. 【character_network】：提煉全書核心角色與非人實體網絡（最多12項）。請注意【嚴格區分角色稱謂與基礎定位】：\n"
            "   - character_name：必須填寫原著文本中的真實角色稱謂（嚴禁在此欄填寫定位類別或代號）。\n"
            "   - archetype_role：填寫 2～3 字最基礎、最通俗的敘事編制定位或生物類別（僅標定其在全書編制中的主配角性別定位、敵我陣營職能或非人生物類別；嚴禁堆砌心理狀態、性格形容詞或命運修飾語，此欄位嚴禁出現角色本名或特定世界觀專有名詞）。\n"
            "   - faction_name：填寫角色所屬核心勢力或組織。\n"
            "   - background_and_motive：人物背景、核心動機與命運軌跡簡述。\n"
            "5. 【faction_geopolitics】：勢力陣營格局天平（組織或陣營宗旨、利益訴求、支配力層級定位與敵對關係）。\n"
            "6. 【hierarchy_rules】：實力、修煉或職級權力晉階等級階梯與位階特徵（位階序號、位階名、核心特徵、突破門檻代價；若該題材無等級階梯晉升體系，可回傳空陣列 []）。\n"
            "7. 【world_laws】：不可違背之底層世界觀運轉鐵律、制度邊界、超自然代價或社會核心禁忌。"
        )
        genre_hint = (
            f"指定題材：{genre_primary} / {genre_sub}\n\n"
            if genre_primary and genre_primary not in ("通用", "")
            else "題材判定：請由下方全書事實匯報客觀歸納主題材（genre_primary）與子題材（genre_sub）\n\n"
        )
        user_prompt = (
            f"書名：《{novel_title}》\n"
            f"{genre_hint}"
            f"{header_label}：\n{summary_content}"
        )
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                if not await self._wait_if_paused_or_unloaded():
                    raise asyncio.CancelledError()
                temp = 0.20 + (attempt - 1) * 0.08
                res = await self.llm_client.generate_structured(
                    sys_prompt, user_prompt, BookBlueprintSynthesisContract, max_tokens=4096, temperature=temp
                )
                res.novel_title = convert_simplified_to_traditional(novel_title)
                if genre_primary and genre_primary not in ("通用", ""):
                    res.genre_primary = convert_simplified_to_traditional(genre_primary.strip())
                else:
                    res.genre_primary = convert_simplified_to_traditional(res.genre_primary.strip()) or "文學"
                if genre_sub and genre_sub not in ("通用", "綜合", ""):
                    res.genre_sub = convert_simplified_to_traditional(genre_sub.strip())
                else:
                    res.genre_sub = convert_simplified_to_traditional(res.genre_sub.strip()) or "長篇敘事"
                res.macro_outline = convert_simplified_to_traditional(res.macro_outline.strip())
                for c in res.character_network:
                    c.character_name = convert_simplified_to_traditional(c.character_name.strip())
                    c.archetype_role = convert_simplified_to_traditional(c.archetype_role.strip())
                    c.faction_name = convert_simplified_to_traditional(c.faction_name.strip())
                    c.background_and_motive = convert_simplified_to_traditional(c.background_and_motive.strip())
                for f in res.faction_geopolitics:
                    f.faction_name = convert_simplified_to_traditional(f.faction_name.strip())
                    f.doctrine_and_nature = convert_simplified_to_traditional(f.doctrine_and_nature.strip())
                    f.power_tier = convert_simplified_to_traditional(f.power_tier.strip())
                    f.rival_factions = [convert_simplified_to_traditional(rf.strip()) for rf in f.rival_factions if rf.strip()]
                for h in res.hierarchy_rules:
                    h.level_name = convert_simplified_to_traditional(h.level_name.strip())
                    h.core_attributes = convert_simplified_to_traditional(h.core_attributes.strip())
                    h.breakthrough_criteria = convert_simplified_to_traditional(h.breakthrough_criteria.strip())
                res.world_laws = [convert_simplified_to_traditional(w.strip()) for w in res.world_laws if w.strip()]
                return res
            except Exception as exc:
                last_exc = exc
                if self.lifecycle and getattr(self.lifecycle, "is_cancelled", lambda: False)() is True:
                    raise asyncio.CancelledError()
                is_p = bool(self.lifecycle and getattr(self.lifecycle, "is_paused", lambda: False)() is True)
                sm = getattr(self.llm_client, "server_manager", None)
                is_u = getattr(sm, "_manually_unloaded", False) is True
                if is_p or is_u:
                    logger.info("Blueprint synthesis paused due to user pause or manual model unload; waiting for resume...")
                    if not await self._wait_if_paused_or_unloaded():
                        raise asyncio.CancelledError()
                    continue
                logger.warning("Blueprint synthesis attempt %d failed: %s", attempt, exc)
                if attempt < 3:
                    await asyncio.sleep(1.0)

        raise RuntimeError(f"Failed to synthesize macro blueprint for 《{novel_title}》 after 3 attempts: {last_exc}")


class MacroPacingExtractor:
    """Extracts macro narrative wave archetypes and synthesizes factual book blueprints."""

    def __init__(
        self,
        repo: ReferenceRepository,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Initialize macro extractor with reference repository and optional LLM client."""
        self.repo = repo
        self._llm_client = llm_client
        self.fact_extractor = ChapterFactExtractor(llm_client=llm_client)

    @property
    def llm_client(self) -> LLMClient | None:
        return self._llm_client

    @llm_client.setter
    def llm_client(self, client: LLMClient | None) -> None:
        self._llm_client = client
        if hasattr(self, "fact_extractor") and self.fact_extractor:
            self.fact_extractor.llm_client = client

    async def extract_from_novel_text(
        self,
        text: str,
        novel_title: str,
        genre_primary: str | None = None,
        genre_sub: str | None = None,
        source_id: int | None = None,
        jitter_ratio: float = 0.0,
        hardware_guard: Any | None = None,
        lifecycle: Any | None = None,
        force_reextract: bool = False,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> MacroPacingTemplateContract:
        """Process novel text into factual chapter evaluations, synthesize blueprint and pacing template."""
        if on_progress:
            on_progress({
                "active": True,
                "novel_title": novel_title,
                "percent": 5.0,
                "phase": "TOC_PARSING",
                "detail": f"正在解析《{novel_title}》全書目錄骨幹與章節全文...",
            })

        toc = extract_table_of_contents(text)
        total_chapters = len(toc)

        if source_id is not None and not force_reextract and self.repo.has_book_blueprint(source_id):
            existing_bp = self.repo.get_book_blueprint(source_id)
            if existing_bp:
                curve_data = existing_bp.get("factual_pacing_curve", [])
                total_chapters = existing_bp.get("total_chapters_analyzed", len(toc))
                templates = [t for t in self.repo.list_pacing_templates() if t.get("source_id") == source_id]
                if templates:
                    existing_tpl = self.repo.get_pacing_template(templates[0]["id"])
                    if existing_tpl:
                        if on_progress:
                            on_progress({
                                "active": False,
                                "novel_title": novel_title,
                                "percent": 100.0,
                                "phase": "PACING_COMPLETE",
                                "detail": f"《{novel_title}》重用既有事實藍圖與波浪範本！",
                            })
                        stages = [PacingStageTemplateContract(**s) for s in existing_tpl["stages"]]
                        resolved_gp = existing_tpl.get("genre_primary") or existing_bp.get("genre_primary") or genre_primary or "文學"
                        resolved_gs = existing_tpl.get("genre_sub") or existing_bp.get("genre_sub") or genre_sub or "長篇敘事"
                        return MacroPacingTemplateContract(
                            template_name=existing_tpl["name"],
                            genre_primary=resolved_gp,
                            genre_sub=resolved_gs,
                            source_novel_title=novel_title,
                            total_chapters_analyzed=total_chapters,
                            narrative_archetype=existing_tpl.get("archetype", f"{resolved_gp}·實體情節波浪流"),
                            pacing_tags=existing_tpl.get("pacing_tags", ["#客觀事實收斂"]),
                            summary=existing_tpl.get("summary", ""),
                            stages=stages,
                            chapter_curve_data=curve_data,
                        )
                elif curve_data:
                    cached_facts = reconstruct_chapter_facts_from_curve(curve_data)
                    if cached_facts:
                        resolved_gp = existing_bp.get("genre_primary") or genre_primary or "文學"
                        resolved_gs = existing_bp.get("genre_sub") or genre_sub or "長篇敘事"
                        _, stages = build_factual_pacing_curve(cached_facts, total_chapters)
                        if jitter_ratio > 0.0:
                            stages = apply_pacing_jitter(stages, jitter_ratio=jitter_ratio)
                        climax_offset = 75.0
                        for s in stages:
                            if s.pacing_phase == "CLIMAX":
                                climax_offset = s.offset_percent
                                break
                        template = MacroPacingTemplateContract(
                            template_name=f"{novel_title}·標準宏觀節奏波浪",
                            genre_primary=resolved_gp,
                            genre_sub=resolved_gs,
                            source_novel_title=novel_title,
                            total_chapters_analyzed=total_chapters,
                            narrative_archetype=f"{resolved_gp}·實體情節波浪流",
                            pacing_tags=["#章節級高解析度", "#客觀事實收斂", "#起承轉合"],
                            summary=f"提取自《{novel_title}》全書 {total_chapters} 章事實解構之高精度真實波浪曲線，決戰於 {climax_offset:.0f}% 處爆發。",
                            stages=stages,
                            chapter_curve_data=curve_data,
                        )
                        self.repo.save_pacing_template(template, source_id=source_id)
                        if on_progress:
                            on_progress({
                                "active": False,
                                "novel_title": novel_title,
                                "percent": 100.0,
                                "phase": "PACING_COMPLETE",
                                "detail": f"《{novel_title}》自既有事實藍圖快速合成波浪曲線完成！",
                            })
                        return template

        if lifecycle:
            self.fact_extractor.lifecycle = lifecycle
        if not await self.fact_extractor._wait_if_paused_or_unloaded():
            return None

        if not self.llm_client or not self.llm_client.server_manager.is_healthy():
            raise AIModelUnavailableError(
                "LLM server is unavailable or unhealthy; macro pacing and blueprint extraction require AI analysis."
            )

        # Stage 1: Dynamic batching with incremental checkpoint recovery for fault-tolerant long-text analysis
        cache_dir = self.repo.db_path.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = cache_dir / f"stage1_facts_{source_id}.json" if source_id else None

        chapter_facts: list[ChapterFactExtractContract] = []
        entity_deltas: list[GroupEntityDeltas] = []
        last_anchor_event = ""

        if checkpoint_file and checkpoint_file.exists() and not force_reextract:
            try:
                cached_data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                cached_total = cached_data.get("total_chapters")
                if cached_total is not None and int(cached_total) != total_chapters:
                    logger.info(
                        "Ignoring stale Stage 1 checkpoint for %s (chapter count %s != %s).",
                        novel_title,
                        cached_total,
                        total_chapters,
                    )
                else:
                    cached_facts = [ChapterFactExtractContract.model_validate(cf) for cf in cached_data.get("chapter_facts", [])]
                    cached_deltas = [GroupEntityDeltas.model_validate(ed) for ed in cached_data.get("entity_deltas", [])]
                    cached_anchor = str(cached_data.get("last_anchor_event", ""))
                    if cached_facts:
                        chapter_facts.extend(cached_facts)
                        entity_deltas.extend(cached_deltas)
                        last_anchor_event = cached_anchor
                        logger.info("Resumed Stage 1 facts extraction for %s: restored %d chapters from checkpoint.", novel_title, len(cached_facts))
            except Exception as e:
                logger.warning("Failed to load Stage 1 checkpoint for %s: %s", novel_title, e)

        batches = pack_chapters_dynamically(toc, max_chars=7200, max_chapters=3)
        total_batches = len(batches)
        already_extracted_indices = {cf.chapter_index for cf in chapter_facts}

        for b_idx, batch in enumerate(batches, start=1):
            if all(ch.get("index", idx + 1) in already_extracted_indices for idx, ch in enumerate(batch)):
                continue

            if lifecycle and lifecycle.is_cancelled():
                logger.info("Macro pacing extraction cancelled by user during Stage 1.")
                break
            if not await self.fact_extractor._wait_if_paused_or_unloaded():
                break

            res = await self.fact_extractor.extract_batched_chapter_facts(
                chapters=batch,
                prior_anchor_event=last_anchor_event,
            )
            for sc in res.chapters:
                chapter_facts.append(ChapterFactExtractContract.from_compact(sc))
            entity_deltas.append(res.entity_deltas)
            if res.chapters:
                last_anchor_event = res.chapters[-1].core_event

            if checkpoint_file:
                try:
                    payload = {
                        "novel_title": novel_title,
                        "total_chapters": total_chapters,
                        "chapter_facts": [cf.model_dump() for cf in chapter_facts],
                        "entity_deltas": [ed.model_dump() for ed in entity_deltas],
                        "last_anchor_event": last_anchor_event,
                        "updated_at": time.time(),
                    }
                    tmp_ckpt = checkpoint_file.with_name(checkpoint_file.name + ".tmp")
                    tmp_ckpt.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    os.replace(tmp_ckpt, checkpoint_file)
                except Exception as e:
                    logger.warning("Failed to save Stage 1 checkpoint: %s", e)

            if hardware_guard:
                await hardware_guard.check_and_cooldown_if_hot()
            if on_progress:
                pct = 10.0 + (len(chapter_facts) / max(1, total_chapters)) * 50.0
                last_title = res.chapters[-1].chapter_title if res.chapters else ""
                on_progress({
                    "active": True,
                    "novel_title": novel_title,
                    "percent": round(pct, 1),
                    "phase": "FACT_EXTRACTION",
                    "detail": f"正在深讀解構第 {len(chapter_facts)}/{total_chapters} 章客觀事實（批次 {b_idx}/{total_batches}）：{last_title}...",
                })

        if (lifecycle and lifecycle.is_cancelled()) or not chapter_facts:
            return None

        # Step 2: Consolidate volumes (chunks of 20 chapters) with entity_deltas
        vol_batch_size = 20
        volume_reports: list[VolumeFactConsolidationContract] = []
        num_vols = max(1, (len(chapter_facts) + vol_batch_size - 1) // vol_batch_size)
        for v_idx in range(num_vols):
            if lifecycle and lifecycle.is_cancelled():
                logger.info("Macro pacing extraction cancelled during volume consolidation.")
                break
            if not await self.fact_extractor._wait_if_paused_or_unloaded():
                break

            v_start = v_idx * vol_batch_size
            v_end = min(len(chapter_facts), (v_idx + 1) * vol_batch_size)
            batch_facts = chapter_facts[v_start:v_end]
            if batch_facts:
                first_ch_in_vol = batch_facts[0].chapter_index
                last_ch_in_vol = batch_facts[-1].chapter_index
                vol_deltas: list[GroupEntityDeltas] = []
                for b_i, delta in enumerate(entity_deltas):
                    if b_i < len(batches):
                        b_ch_start = batches[b_i][0].get("index", 1)
                        b_ch_end = batches[b_i][-1].get("index", 1)
                        if not (b_ch_end < first_ch_in_vol or b_ch_start > last_ch_in_vol):
                            vol_deltas.append(delta)
                    else:
                        vol_deltas.append(delta)

                vr = await self.fact_extractor.consolidate_volume_facts(
                    volume_index=v_idx + 1,
                    chapter_facts=batch_facts,
                    volume_entity_deltas=vol_deltas,
                )
                volume_reports.append(vr)
                if hardware_guard:
                    await hardware_guard.check_and_cooldown_if_hot()
                if on_progress:
                    pct = 60.0 + (v_idx + 1) / num_vols * 20.0
                    on_progress({
                        "active": True,
                        "novel_title": novel_title,
                        "percent": round(pct, 1),
                        "phase": "VOLUME_CONSOLIDATION",
                        "detail": f"正在整合第 {v_idx + 1}/{num_vols} 卷宏觀劇情弧與勢力天平...",
                    })

        if lifecycle and lifecycle.is_cancelled():
            return None

        # Step 2.5: Hierarchical Reduction to Major Narrative Arcs (Tree Reduction)
        major_arcs: list[MajorArcConsolidationContract] | None = None
        if num_vols > 4:
            if not await self.fact_extractor._wait_if_paused_or_unloaded():
                return None
            if on_progress:
                on_progress({
                    "active": True,
                    "novel_title": novel_title,
                    "percent": 80.0,
                    "phase": "MAJOR_ARCS_CONSOLIDATION",
                    "detail": f"正在將《{novel_title}》全書 {num_vols} 卷樹狀收斂為宏觀情節大幕...",
                })
            major_arcs = await self.fact_extractor.consolidate_major_arcs(
                volume_reports=volume_reports,
                hardware_guard=hardware_guard,
                lifecycle=lifecycle,
            )

        # Step 3: Synthesize macro blueprint
        if not await self.fact_extractor._wait_if_paused_or_unloaded():
            return None
        if on_progress:
            on_progress({
                "active": True,
                "novel_title": novel_title,
                "percent": 85.0,
                "phase": "BLUEPRINT_SYNTHESIS",
                "detail": f"正在收斂《{novel_title}》全書永久事實藍圖（大綱骨幹、角色原型網絡、等級法則）...",
            })

        blueprint = await self.fact_extractor.synthesize_macro_blueprint(
            novel_title=novel_title,
            volume_reports=volume_reports,
            chapter_facts=chapter_facts,
            genre_primary=genre_primary,
            genre_sub=genre_sub,
            major_arcs=major_arcs,
        )

        resolved_gp = blueprint.genre_primary
        resolved_gs = blueprint.genre_sub

        curve_data, stages = build_factual_pacing_curve(chapter_facts, total_chapters)

        if source_id is not None:
            self.repo.save_book_blueprint(
                source_id=source_id,
                blueprint=blueprint,
                factual_pacing_curve=curve_data,
                total_chapters=total_chapters,
            )

        if jitter_ratio > 0.0:
            stages = apply_pacing_jitter(stages, jitter_ratio=jitter_ratio)

        climax_offset = 75.0
        for s in stages:
            if s.pacing_phase == "CLIMAX":
                climax_offset = s.offset_percent
                break

        template = MacroPacingTemplateContract(
            template_name=f"{novel_title}·標準宏觀節奏波浪",
            genre_primary=resolved_gp,
            genre_sub=resolved_gs,
            source_novel_title=novel_title,
            total_chapters_analyzed=total_chapters,
            narrative_archetype=f"{resolved_gp}·實體情節波浪流",
            pacing_tags=["#章節級高解析度", "#客觀事實收斂", "#起承轉合"],
            summary=f"提取自《{novel_title}》全書 {total_chapters} 章事實解構之高精度真實波浪曲線，決戰於 {climax_offset:.0f}% 處爆發。",
            stages=stages,
            chapter_curve_data=curve_data,
        )

        self.repo.save_pacing_template(template, source_id=source_id)

        if checkpoint_file and checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
            except Exception:
                pass

        if on_progress:
            on_progress({
                "active": False,
                "novel_title": novel_title,
                "percent": 100.0,
                "phase": "PACING_COMPLETE",
                "detail": f"《{novel_title}》全書事實藍圖與波浪曲線建構完成！",
            })

        return template
