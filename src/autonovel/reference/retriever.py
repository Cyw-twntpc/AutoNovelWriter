"""Dual-track reference retrieval (0.4*desc + 0.45*tag + 0.15*genre >= 0.72) and 300-char digest loop."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from autonovel.db.reference_repo import ReferenceRepository
from autonovel.engine.prose_sanitizer import (
    convert_simplified_to_traditional,
    sanitize_novel_prose,
)
from autonovel.llm.embedder import CpuEmbedder
from autonovel.llm.llm_client import LLMClient
from autonovel.schemas.llm_contracts import ReferenceQueryTagsContract
from autonovel.wizard.qa_engine import AIModelUnavailableError

logger = logging.getLogger(__name__)


class ReferenceRetriever:
    """Task-oriented dual-track retriever and understanding-loop compressor."""

    def __init__(
        self,
        repo: ReferenceRepository,
        embedder: CpuEmbedder,
        llm_client: LLMClient | None = None,
        threshold: float = 0.72,
        top_k: int = 3,
        genre_provider: Callable[[], str] | None = None,
    ) -> None:
        """Initialize retriever with repository, CPU embedder, and score threshold."""
        self.repo = repo
        self.embedder = embedder
        self.llm_client = llm_client
        self.threshold = threshold
        self.top_k = top_k
        # Supplies the active project's "主:子" genre so the third scoring track can
        # reward books of the same genre. Late-bound so project switches are picked up.
        self.genre_provider = genre_provider

    async def retrieve_and_compress(
        self, task_description: str, fallback_tags: list[str] | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        """Run dual-track query and compress top-K matches into a writing guidance digest."""
        query_tags = await self._generate_query_tags(task_description, fallback_tags)
        task_vec = self.embedder.embed_text(task_description)
        tag_vecs = self.embedder.embed_batch(query_tags)
        hits = self.repo.search_dual_track(
            task_vec=task_vec,
            query_tag_vecs=tag_vecs,
            threshold=self.threshold,
            top_k=self.top_k,
            query_genre=self._current_genre(),
        )
        if not hits:
            return "", []
        digest = await self._compress_understanding_loop(task_description, hits)
        return digest, hits

    def _current_genre(self) -> str:
        """Return the active project's "主:子" genre label, or "" when unavailable."""
        if self.genre_provider is None:
            return ""
        try:
            return str(self.genre_provider() or "").strip()
        except Exception as exc:  # noqa: BLE001 - optional boost must never break retrieval
            # Genre weighting is an optional boost; a stale project DB or a bad
            # provider must not take down reference retrieval.
            logger.warning("Genre provider failed: %s", exc)
            return ""

    async def _generate_query_tags(
        self, task_description: str, fallback_tags: list[str] | None
    ) -> list[str]:
        """Generate 3-6 semantic query tags with subject-object dynamic via LLM."""
        if not (self.llm_client and self.llm_client.server_manager.is_healthy()):
            raise AIModelUnavailableError("LLM server is unavailable or unhealthy for query tag generation.")

        sys_prompt = (
            "你是一位專精長篇小說敘事動力學的檢索架構專家。請分析當前章節寫作任務，按第一性原理演繹四維正交檢索標籤與主客體基礎定位：\n"
            "1. 【全篇繁體中文】：所有文字輸出必須完全使用繁體中文（台灣標準書面語）。\n"
            "2. 【四維正交檢索座標（query_tags_matrix）】：\n"
            "   - 結構功能（structure_tags）：當前任務推進之宏觀情節力學功能（如：蓄勢、轉折、頓悟、破局、餘波承接）；\n"
            "   - 主客博弈（actant_tags）：主客體在敘事場域中的力量交鋒本質（如：位階壓制、試探博弈、理念背離、利益結盟）；\n"
            "   - 微觀筆法（rhetoric_tags）：核心行文機制、修辭手法、感官渲染與語言節奏（如：五感通感、蒙太奇組接、對白潛台詞、句式緩急）；\n"
            "   - 情境氛圍（context_tags）：題材座標與場景氛圍（如：暗黑懸疑、朝堂肅殺、宗門世道、市井煙火）。\n"
            "3. 【主客體基礎定位】：\n"
            "   - subject_role 與 object_role：標定行動發起方與承受方之敘事編制客觀定位（如主角、配角、長老、敵手、自然環境等）；\n"
            "   - interaction_dynamic：簡練界定主體與客體之間的客觀力量作用或情感互動模式。"
        )
        res = await self.llm_client.generate_structured(
            sys_prompt, task_description, ReferenceQueryTagsContract
        )
        matrix_tags = res.query_tags_matrix.all_tags() if res.query_tags_matrix else []
        clean_matrix_tags = [
            convert_simplified_to_traditional(t.strip())
            for t in matrix_tags
            if t.strip()
        ]
        deduped = [t for t in dict.fromkeys(clean_matrix_tags) if t]
        if fallback_tags:
            for ft in fallback_tags:
                clean_ft = convert_simplified_to_traditional(ft.strip())
                if clean_ft and clean_ft not in deduped:
                    deduped.append(clean_ft)
        return deduped

    async def _compress_understanding_loop(
        self, task_description: str, hits: list[dict[str, Any]]
    ) -> str:
        """Synthesize retrieved patterns into an original writing guidance digest via LLM."""
        if not (self.llm_client and self.llm_client.server_manager.is_healthy()):
            raise AIModelUnavailableError("LLM server is unavailable or unhealthy for reference digest synthesis.")

        formatted_items: list[str] = []
        for h in hits:
            dyn_info = h.get("relational_dynamic", "")
            if not dyn_info:
                s_r = h.get("subject_role", "")
                o_r = h.get("object_role", "")
                dyn = h.get("interaction_dynamic", "")
                dyn_info = f"{s_r}->{o_r} [{dyn}]" if (s_r or o_r) else ""
            so_info = f"（動力學關係：{dyn_info}）" if dyn_info else ""
            desc = convert_simplified_to_traditional(h.get("description", ""))
            snip = convert_simplified_to_traditional(h.get("snippet", ""))
            formatted_items.append(f"- 手法：{desc}{so_info}（示範：「{snip}」）")
        formatted = "\n".join(formatted_items)

        sys_prompt = (
            "你是一位專精百萬字長篇小說的首席文字總監。請深入消化參考材料中的寫作手法精髓，針對當前章節寫作任務，提煉出一份不超過 220 字、純幹貨、可直接落地的【寫作操作清單】。\n"
            "【輸出規範】：\n"
            "1. 全程繁體中文，嚴禁任何簡體字。\n"
            "2. 嚴禁任何客套開場白或泛泛而談的理論說教，直接輸出指令。\n"
            "3. 採條列式輸出三大落地指令：\n"
            "   ①【鏡頭與五感】：指定具體的光影色彩、環境溫度、聲音動態或物質材質細節；\n"
            "   ②【主客交鋒與心理】：指定主動方與被動方的力量因果交互與微表情折射方式；\n"
            "   ③【節奏與句式】：指定日常鋪墊句式延展或交鋒轉折句式緊湊的切換節點。\n"
            "4. 嚴禁出現參考書中的具體人名、組織名或特定劇情梗概。"
        )
        user_prompt = f"當前任務：{task_description}\n參考材料：\n{formatted}"
        text = await self.llm_client.generate_text_stream(
            sys_prompt, user_prompt, max_tokens=320, temperature=0.4
        )
        return sanitize_novel_prose(text)
