"""Repository for reference_library.db supporting SHA-256 deduplication and vector search."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from autonovel.config import get_logger
from autonovel.db.connection import atomic_transaction, read_connection
from autonovel.db.reference_schema import init_reference_database
from autonovel.engine.prose_sanitizer import convert_simplified_to_traditional
from autonovel.schemas.llm_contracts import (
    BookBlueprintSynthesisContract,
    MacroPacingTemplateContract,
    PacingStageTemplateContract,
    TagRuleContract,
)

logger = get_logger(__name__)

# Hard ceiling on rows held in the in-memory similarity matrix. A 512-dim float32
# vector is ~2 KB, so 20k entries cost ~40 MB for descriptions plus tags; without a
# cap a large library would be fully resident on every process.
MAX_MATRIX_ENTRIES = 20_000


def vector_to_blob(vec: np.ndarray) -> bytes:
    """Serialize a float32 numpy vector into raw bytes for SQLite BLOB storage."""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr = arr / norm
    return arr.tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    """Deserialize a SQLite BLOB back into a normalized float32 numpy vector."""
    return np.frombuffer(blob, dtype=np.float32)


STANDARD_CLUSTER_COLORS: dict[str, str] = {
    "境象": "#10b981",       # Emerald Green for Ambient / Scenery
    "主態": "#38bdf8",       # Sky Blue for Solo Agentive
    "主客": "#a855f7",       # Violet for Subject-Object Dynamics
    "焦距": "#06b6d4",       # Electric Cyan for Focalization (Micro/Macro)
    "視角": "#06b6d4",       # Electric Cyan for Perspective
    "節奏": "#f97316",       # Bright Orange for Narrative Tempo & Pacing
    "時序": "#f97316",       # Bright Orange for Temporality
    "資訊差": "#8b5cf6",     # Purple for Information Asymmetry / Irony
    "句式": "#14b8a6",       # Teal for Syntactic Density & Rhythm
    "風格": "#14b8a6",       # Teal for Prose Stylistics
    "題材": "#f59e0b",       # Amber Orange for Themes / Genres
    "文學技法": "#ec4899",   # Rose Pink for Literary Techniques
    "技法": "#ec4899",       # Rose Pink
    "感官情緒": "#ef4444",   # Crimson Red for Sensory & Emotions
    "心理層次": "#6366f1",   # Indigo for Deep Psychology
    "仙俠": "#10b981",
    "武俠": "#f59e0b",
    "奇幻": "#8b5cf6",
    "科幻": "#06b6d4",
    "懸疑": "#ef4444",
    "歷史": "#d97706",
    "都市": "#3b82f6",
    "末日": "#64748b",
    "通用技法": "#ec4899",
    "其他": "#94a3b8",
}

EXTENDED_EMERGENT_PALETTE: list[str] = [
    "#14b8a6",  # Teal
    "#a855f7",  # Violet
    "#f97316",  # Bright Orange
    "#84cc16",  # Lime
    "#e11d48",  # Rose
    "#6366f1",  # Indigo
    "#0284c7",  # Sky Blue
    "#d946ef",  # Fuchsia
    "#059669",  # Deep Emerald
    "#eab308",  # Yellow
]


def get_cluster_color(cluster_name: str) -> str:
    """Return color for standard genre or assign deterministic color from extended palette."""
    if cluster_name in STANDARD_CLUSTER_COLORS:
        return STANDARD_CLUSTER_COLORS[cluster_name]
    # Python's built-in hash() is salted per process (PYTHONHASHSEED), which made the
    # same tag change color across restarts. Derive the index from a stable digest.
    digest = hashlib.sha256(cluster_name.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % len(EXTENDED_EMERGENT_PALETTE)
    return EXTENDED_EMERGENT_PALETTE[idx]


def compute_density_clusters(
    vectors: np.ndarray,
    eps: float = 0.45,
    min_samples: int = 2,
) -> np.ndarray:
    """Pure NumPy DBSCAN clustering on normalized embeddings using cosine distance.

    Returns an array of cluster labels, where -1 denotes noise / isolated outlier.
    """
    n = len(vectors)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([-1], dtype=int)

    # Normalize as float32 to keep pairwise similarity compact and avoid a duplicate distance matrix
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = (vectors / norms).astype(np.float32, copy=False)
    sim_threshold = 1.0 - eps - 1e-6

    # Materialize adjacency only up to 3,000 items (~9 MB bool); larger inputs query row-by-row
    adj = (np.dot(normalized, normalized.T) >= sim_threshold) if n <= 3000 else None

    def _neighbors_of(idx: int) -> np.ndarray:
        if adj is not None:
            return np.flatnonzero(adj[idx])
        return np.flatnonzero((normalized @ normalized[idx]) >= sim_threshold)

    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    queued = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = _neighbors_of(i)
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        queue: deque[int] = deque()
        for idx in neighbors:
            idx_int = int(idx)
            if idx_int != i and not queued[idx_int]:
                queued[idx_int] = True
                queue.append(idx_int)
        while queue:
            q = queue.popleft()
            if not visited[q]:
                visited[q] = True
                q_neighbors = _neighbors_of(q)
                if len(q_neighbors) >= min_samples:
                    for qn in q_neighbors:
                        qn_int = int(qn)
                        if not visited[qn_int] and not queued[qn_int]:
                            queued[qn_int] = True
                            queue.append(qn_int)
            if labels[q] == -1:
                labels[q] = cluster_id
        cluster_id += 1

    return labels


def resolve_tag_cluster(name: str) -> str:
    """Resolve high-level theme or literary technique cluster name from structured tag string."""
    for sep in (":", "：", "·", "/"):
        if sep in name:
            p = name.split(sep, 1)[0].strip()
            # Skip separators inside parenthetical notes such as "角色名（主角/配角）"
            if ("（" in p and "）" not in p) or ("(" in p and ")" not in p):
                continue
            return p if p else "其他"
    return "通用技法"


def _clean_rule_tag_name(raw: str) -> str:
    """Strip surrounding quotes/brackets, hash prefix, and trailing '(count)' frequency notes."""
    cleaned = raw.strip()
    cleaned = re.sub(r"\s*[（(]\d+\s*次?[）)]\s*$", "", cleaned).strip()
    return cleaned.strip("「」\"'【】[]# ").strip()


def project_2d_pca(vectors: np.ndarray) -> np.ndarray:
    """Project normalized high-dimensional vectors to 2D coordinates via SVD."""
    if len(vectors) < 2:
        return np.zeros((len(vectors), 2))
    x_c = vectors - np.mean(vectors, axis=0)
    _, _, vt = np.linalg.svd(x_c, full_matrices=False)
    coords_raw = np.dot(x_c, vt[:2].T)
    max_v = float(np.max(np.abs(coords_raw)))
    if max_v > 1e-6:
        return coords_raw / max_v
    return coords_raw


class ReferenceRepository:
    """Data access layer for decoupled cross-project reference_library.db."""

    def __init__(
        self,
        db_path: Path,
        embed_fn: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        """Initialize reference library database schema and in-memory matrix cache."""
        self.db_path = db_path
        init_reference_database(self.db_path)
        self._matrix_cache: dict[str, Any] | None = None
        self._tag_assets_cache: dict[str, Any] | None = None
        # Genre weighting needs text->vector conversion for the "主:子" genre label.
        # The embedder is injected lazily because the daemon builds the repository
        # before its CpuEmbedder instance exists.
        self._embed_fn = embed_fn
        self._genre_vec_cache: dict[str, np.ndarray | None] = {}

    def set_embed_fn(self, embed_fn: Callable[[str], np.ndarray] | None) -> None:
        """Attach (or replace) the text embedder used for genre weighting."""
        self._embed_fn = embed_fn
        self._genre_vec_cache = {}
        self._matrix_cache = None
        self._tag_assets_cache = None

    def _invalidate_cache(self) -> None:
        """Invalidate in-memory matrix and tag cache on repository mutations."""
        self._matrix_cache = None
        self._tag_assets_cache = None
        self._genre_vec_cache = {}

    def _genre_vector(self, genre_text: str) -> np.ndarray | None:
        """Return a memoized normalised vector for a genre label, or None when unavailable."""
        key = genre_text.strip()
        if not key or self._embed_fn is None:
            return None
        if key in self._genre_vec_cache:
            return self._genre_vec_cache[key]
        try:
            vec = blob_to_vector(vector_to_blob(self._embed_fn(key)))
        except Exception as exc:  # noqa: BLE001 - genre weighting is optional
            # A missing/broken embedder must degrade the genre track to 0, not
            # abort the whole retrieval (and the cache stores the failure).
            logger.warning("Genre embedding failed for %r: %s", key, exc)
            vec = None
        self._genre_vec_cache[key] = vec
        return vec

    def _get_or_build_matrix_cache(self) -> dict[str, Any] | None:
        """Fetch all entries and tags, returning stacked 2D numpy matrices or None if empty."""
        if self._matrix_cache is not None:
            return self._matrix_cache

        with read_connection(self.db_path) as conn:
            total_entries = int(
                conn.execute("SELECT COUNT(*) AS n FROM ref_entries;").fetchone()["n"]
            )
            if total_entries == 0:
                return None
            # Keep the newest rows when the library exceeds the in-memory ceiling;
            # re-extracted/imported material is the most useful for retrieval.
            entries = conn.execute(
                """
                SELECT * FROM (
                    SELECT id, source_id, description, desc_embedding, snippet,
                           COALESCE(subject_role, '') AS subject_role,
                           COALESCE(object_role, '') AS object_role,
                           COALESCE(interaction_dynamic, '') AS interaction_dynamic,
                           COALESCE(relational_dynamic, '') AS relational_dynamic
                    FROM ref_entries
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC;
                """,
                (MAX_MATRIX_ENTRIES,),
            ).fetchall()
            if len(entries) < total_entries:
                logger.warning(
                    "Reference matrix cache capped at %d of %d entries; older rows are not searchable.",
                    MAX_MATRIX_ENTRIES,
                    total_entries,
                )
            tags_rows = conn.execute(
                """
                SELECT t.entry_id, t.tag_name, t.tag_embedding
                FROM ref_tags t
                WHERE t.entry_id >= ?
                ORDER BY t.entry_id ASC;
                """,
                (int(entries[0]["id"]),),
            ).fetchall()
            # An entry inherits its book genre through source_id -> ref_book_blueprints;
            # no per-tag provenance column is needed.
            genre_rows = conn.execute(
                """
                SELECT source_id, genre_primary, genre_sub
                FROM ref_book_blueprints;
                """
            ).fetchall()

        genre_by_source: dict[int, str] = {}
        for gr in genre_rows:
            gp = str(gr["genre_primary"] or "").strip()
            gs = str(gr["genre_sub"] or "").strip()
            label = f"{gp}:{gs}" if (gp and gs) else (gp or gs)
            if label:
                genre_by_source[int(gr["source_id"])] = label
        genre_vec_by_source: dict[int, np.ndarray] = {}
        for src_id, label in genre_by_source.items():
            vec = self._genre_vector(label)
            if vec is not None:
                genre_vec_by_source[src_id] = vec

        tag_map: dict[int, list[tuple[str, np.ndarray]]] = {}
        tag_freq: dict[str, int] = {}
        for tr in tags_rows:
            eid = int(tr["entry_id"])
            t_name = str(tr["tag_name"])
            tag_freq[t_name] = tag_freq.get(t_name, 0) + 1
            tag_map.setdefault(eid, []).append(
                (t_name, blob_to_vector(tr["tag_embedding"]))
            )

        # A library may mix embedding models (or hold a truncated blob). np.vstack would
        # raise ValueError and break every search, so keep the dominant dimension and
        # drop the outliers from both the matrix and the row index.
        desc_vecs = [blob_to_vector(r["desc_embedding"]) for r in entries]
        dim_counts = Counter(v.size for v in desc_vecs if v.size > 0)
        if dim_counts:
            # Tie-break on the newest row: the most recently ingested entry reflects
            # the embedding model currently in use.
            max_count = max(dim_counts.values())
            expected_dim = next(
                vec.size
                for vec in reversed(desc_vecs)
                if vec.size > 0 and dim_counts[vec.size] == max_count
            )
        else:
            expected_dim = 0
        kept_rows: list[Any] = []
        kept_vecs: list[np.ndarray] = []
        for row, vec in zip(entries, desc_vecs, strict=False):
            if vec.size == expected_dim:
                kept_rows.append(row)
                kept_vecs.append(vec)
        if len(kept_rows) < len(entries):
            logger.warning(
                "Dropped %d reference entries whose embedding dimension differs from %d.",
                len(entries) - len(kept_rows),
                expected_dim,
            )
        entries = kept_rows
        if not entries:
            return None
        # Tag embeddings must share the same dimension as descriptions; one stale row
        # would otherwise make the tag matrix non-square against the query matrix.
        tag_map = {
            eid: [(name, vec) for name, vec in tlist if vec.size == expected_dim]
            for eid, tlist in tag_map.items()
        }
        tag_map = {eid: tlist for eid, tlist in tag_map.items() if tlist}
        desc_matrix = np.vstack(kept_vecs)
        kept_ids = {int(r["id"]) for r in entries}
        tag_map = {eid: tl for eid, tl in tag_map.items() if eid in kept_ids}
        entry_tag_matrices = {
            eid: np.vstack([t[1] for t in tlist]) for eid, tlist in tag_map.items()
        }
        # Map each surviving entry to its book's genre vector (0.0 weight when absent).
        entry_genre_matrix = np.zeros((len(entries), expected_dim), dtype=np.float32)
        genre_hits = 0
        for pos, row in enumerate(entries):
            gvec = genre_vec_by_source.get(int(row["source_id"]))
            if gvec is not None and gvec.size == expected_dim:
                entry_genre_matrix[pos] = gvec
                genre_hits += 1

        self._matrix_cache = {
            "entries": entries,
            "desc_matrix": desc_matrix,
            "tag_map": tag_map,
            "entry_tag_matrices": entry_tag_matrices,
            "tag_freq": tag_freq,
            "genre_matrix": entry_genre_matrix,
            "genre_coverage": genre_hits,
        }
        return self._matrix_cache

    def has_source_hash(self, file_sha256: str) -> bool:
        """Check whether a file SHA-256 hash has already been imported."""
        with read_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM ref_sources WHERE file_sha256 = ?;", (file_sha256,)
            ).fetchone()
            return row is not None

    def get_source_by_hash(self, file_sha256: str) -> dict[str, Any] | None:
        """Fetch existing source record by SHA-256 hash including track completion flags."""
        with read_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT s.id, s.file_sha256, s.file_name, s.imported_at, s.chunk_count, s.entry_count,
                       EXISTS(SELECT 1 FROM ref_pacing_templates p WHERE p.source_id = s.id) AS has_pacing,
                       EXISTS(SELECT 1 FROM ref_book_blueprints b WHERE b.source_id = s.id) AS has_blueprint,
                       COALESCE((SELECT b.total_chapters_analyzed FROM ref_book_blueprints b WHERE b.source_id = s.id LIMIT 1), 0) AS total_chapters
                FROM ref_sources s
                WHERE s.file_sha256 = ?;
                """,
                (file_sha256,),
            ).fetchone()
            if not row:
                return None
            return {
                "id": int(row["id"]),
                "file_sha256": str(row["file_sha256"]),
                "file_name": str(row["file_name"]),
                "imported_at": str(row["imported_at"]),
                "chunk_count": int(row["chunk_count"]),
                "entry_count": int(row["entry_count"]),
                "has_skills": int(row["entry_count"]) > 0,
                "has_pacing": bool(row["has_pacing"]),
                "has_blueprint": bool(row["has_blueprint"]),
                "total_chapters": int(row["total_chapters"]),
            }

    def get_source(self, source_id: int) -> dict[str, Any] | None:
        """Fetch a source record (id, name, hash) by primary key."""
        with read_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, file_sha256, file_name FROM ref_sources WHERE id = ?;",
                (source_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "id": int(row["id"]),
                "file_sha256": str(row["file_sha256"]),
                "file_name": str(row["file_name"]),
            }

    def count_sources_by_file_name(self, file_name: str, exclude_id: int) -> int:
        """Count other sources that still reference the same uploaded file name."""
        with read_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM ref_sources WHERE file_name = ? AND id != ?;",
                (file_name, exclude_id),
            ).fetchone()
            return int(row["n"] or 0) if row else 0

    def register_or_reset_source(
        self,
        file_sha256: str,
        file_name: str,
        chunk_count: int,
        reset_entries: bool = True,
    ) -> int:
        """Create a source record or delete its old entries if re-extracting skills."""
        with atomic_transaction(self.db_path) as conn:
            existing = conn.execute(
                "SELECT id FROM ref_sources WHERE file_sha256 = ?;", (file_sha256,)
            ).fetchone()
            if existing:
                src_id = int(existing["id"])
                if reset_entries:
                    conn.execute("DELETE FROM ref_entries WHERE source_id = ?;", (src_id,))
                    conn.execute(
                        """
                        UPDATE ref_sources
                        SET file_name = ?, chunk_count = ?, entry_count = 0, imported_at = datetime('now')
                        WHERE id = ?;
                        """,
                        (file_name, chunk_count, src_id),
                    )
                    self._invalidate_cache()
                else:
                    conn.execute(
                        """
                        UPDATE ref_sources
                        SET file_name = ?, imported_at = datetime('now')
                        WHERE id = ?;
                        """,
                        (file_name, src_id),
                    )
                return src_id
            cursor = conn.execute(
                """
                INSERT INTO ref_sources (file_sha256, file_name, chunk_count, entry_count)
                VALUES (?, ?, ?, 0);
                """,
                (file_sha256, file_name, chunk_count),
            )
            return int(cursor.lastrowid)

    def insert_entry_with_tags(
        self,
        source_id: int,
        description: str,
        desc_vec: np.ndarray,
        snippet: str,
        tag_vectors: list[tuple[str, np.ndarray]],
        subject_role: str = "",
        object_role: str = "",
        interaction_dynamic: str = "",
        relational_dynamic: str = "",
        chapter_index: int = 0,
        exemplar_score: float = 0.0,
        invalidate_cache: bool = True,
    ) -> int:
        """Store an extracted knowledge entry, relational dynamics, and its semantic tag coordinates."""
        clean_snippet = snippet.strip()
        with atomic_transaction(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO ref_entries (
                    source_id, description, desc_embedding, snippet,
                    subject_role, object_role, interaction_dynamic, relational_dynamic,
                    chapter_index, exemplar_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    source_id,
                    description.strip(),
                    vector_to_blob(desc_vec),
                    clean_snippet,
                    subject_role.strip(),
                    object_role.strip(),
                    interaction_dynamic.strip(),
                    relational_dynamic.strip(),
                    chapter_index,
                    exemplar_score,
                ),
            )
            entry_id = int(cursor.lastrowid)
            for tag_item in tag_vectors:
                tag_name = tag_item[0]
                tag_vec = tag_item[1]
                axis_type = tag_item[2].strip() if len(tag_item) > 2 else "RHETORIC"
                conn.execute(
                    """
                    INSERT INTO ref_tags (entry_id, axis_type, tag_name, tag_embedding)
                    VALUES (?, ?, ?, ?);
                    """,
                    (entry_id, axis_type, tag_name.strip(), vector_to_blob(tag_vec)),
                )
            conn.execute(
                "UPDATE ref_sources SET entry_count = entry_count + 1 WHERE id = ?;",
                (source_id,),
            )
            if invalidate_cache:
                self._invalidate_cache()
            return entry_id

    def insert_entries_batch(
        self,
        source_id: int,
        entries_data: list[dict[str, Any]],
    ) -> list[int]:
        """Batch insert extracted knowledge entries in a single transaction for big data ingestion."""
        if not entries_data:
            return []
        entry_ids = []
        with atomic_transaction(self.db_path) as conn:
            for item in entries_data:
                cursor = conn.execute(
                    """
                    INSERT INTO ref_entries (
                        source_id, description, desc_embedding, snippet,
                        subject_role, object_role, interaction_dynamic, relational_dynamic,
                        chapter_index, exemplar_score
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        source_id,
                        item["description"].strip(),
                        vector_to_blob(item["desc_vec"]),
                        item["snippet"].strip(),
                        item.get("subject_role", "").strip(),
                        item.get("object_role", "").strip(),
                        item.get("interaction_dynamic", "").strip(),
                        item.get("relational_dynamic", "").strip(),
                        int(item.get("chapter_index", 0)),
                        float(item.get("exemplar_score", 0.0)),
                    ),
                )
                eid = int(cursor.lastrowid)
                entry_ids.append(eid)
                for tag_item in item.get("tag_vectors", []):
                    tag_name = tag_item[0]
                    tag_vec = tag_item[1]
                    axis_type = tag_item[2].strip() if len(tag_item) > 2 else "RHETORIC"
                    conn.execute(
                        """
                        INSERT INTO ref_tags (entry_id, axis_type, tag_name, tag_embedding)
                        VALUES (?, ?, ?, ?);
                        """,
                        (eid, axis_type, tag_name.strip(), vector_to_blob(tag_vec)),
                    )
            conn.execute(
                "UPDATE ref_sources SET entry_count = entry_count + ? WHERE id = ?;",
                (len(entries_data), source_id),
            )
        self._invalidate_cache()
        return entry_ids

    def search_dual_track(
        self,
        task_vec: np.ndarray,
        query_tag_vecs: list[np.ndarray],
        threshold: float = 0.72,
        top_k: int = 3,
        query_genre: str = "",
    ) -> list[dict[str, Any]]:
        """Execute dual-track similarity search (0.4*desc + 0.45*tag + 0.15*genre) with matrix acceleration."""
        cache = self._get_or_build_matrix_cache()
        if not cache:
            return []

        entries = cache["entries"]
        desc_matrix = cache["desc_matrix"]
        tag_map = cache["tag_map"]
        entry_tag_matrices = cache["entry_tag_matrices"]
        genre_matrix = cache["genre_matrix"]

        norm_task = blob_to_vector(vector_to_blob(task_vec))
        if norm_task.size != desc_matrix.shape[1]:
            # The query comes from a different embedding model than the indexed rows;
            # a dot product would raise and take down the whole retrieval step.
            logger.warning(
                "Reference search skipped: query dim %d != index dim %d.",
                norm_task.size,
                desc_matrix.shape[1],
            )
            return []
        norm_q_tags = [blob_to_vector(vector_to_blob(v)) for v in query_tag_vecs]
        stacked_q_tags = np.vstack(norm_q_tags) if norm_q_tags else None
        q_genre_vec = self._genre_vector(query_genre) if query_genre else None
        if q_genre_vec is not None and q_genre_vec.size != desc_matrix.shape[1]:
            logger.warning(
                "Genre weighting disabled: genre dim %d != index dim %d.",
                q_genre_vec.size,
                desc_matrix.shape[1],
            )
            q_genre_vec = None

        # 1. Vectorized batch cosine similarity for all descriptions: (N,)
        s_descs = np.dot(desc_matrix, norm_task)
        if q_genre_vec is not None:
            # Score the third (genre) track for every row in one pass.
            s_genres = np.dot(genre_matrix, q_genre_vec)
        else:
            s_genres = np.zeros(len(entries), dtype=np.float32)
        num_entries = len(entries)

        # 2. Big Data 2-Stage Candidate Slicing:
        # Theoretical lower bound on s_desc for s_final >= threshold:
        # s_final = 0.4 * s_desc + 0.45 * s_tag + 0.15 * s_genre <= 0.4 * s_desc + 0.6
        # => s_desc >= (threshold - 0.6) / 0.4
        min_desc_bound = (threshold - 0.6) / 0.4
        if num_entries > 500:
            candidate_cap = min(num_entries, max(top_k * 50, 1000))
            top_indices = np.argpartition(s_descs, -candidate_cap)[-candidate_cap:]
            candidate_indices = [idx for idx in top_indices if s_descs[idx] >= min_desc_bound]
        else:
            candidate_indices = [idx for idx in range(num_entries) if s_descs[idx] >= min_desc_bound]

        results: list[dict[str, Any]] = []
        tag_freq = cache.get("tag_freq", {})

        for idx in candidate_indices:
            row = entries[idx]
            eid = int(row["id"])
            s_desc = float(s_descs[idx])
            entry_tags = tag_map.get(eid, [])
            has_tag_track = stacked_q_tags is not None and eid in entry_tag_matrices

            if has_tag_track:
                # (M, 512) @ (512, T) = (M, T) pairwise similarities
                t_matrix = entry_tag_matrices[eid]
                sim_matrix = np.dot(stacked_q_tags, t_matrix.T)

                # Salient facet pooling: focus on strongest matching facets without frequency penalty.
                # All literary tags enjoy equal semantic rights regardless of appearance count.
                best_per_qtag = np.max(sim_matrix, axis=1)
                top_salient = np.sort(best_per_qtag)[::-1]
                k_salient = min(len(top_salient), max(1, (len(stacked_q_tags) + 1) // 2))
                s_tag = float(np.mean(top_salient[:k_salient]))
                s_genre = float(s_genres[idx])
                s_final = 0.4 * s_desc + 0.45 * s_tag + 0.15 * s_genre
            elif stacked_q_tags is None:
                # The query carries no tag vectors, so the tag track is undefined and
                # the pair degenerates to a pure description search.
                s_tag = s_desc
                s_genre = 0.0
                s_final = s_desc
            else:
                # Tagless rows are a data anomaly: extraction always writes >= 3 tags and
                # maintenance sweeps delete rows that lose all of them. Skip defensively.
                continue

            if s_final >= threshold:
                results.append(
                    {
                        "entry_id": eid,
                        "description": row["description"],
                        "snippet": row["snippet"],
                        "tags": [t[0] for t in entry_tags],
                        "subject_role": str(row["subject_role"] if "subject_role" in row.keys() else ""),
                        "object_role": str(row["object_role"] if "object_role" in row.keys() else ""),
                        "interaction_dynamic": str(row["interaction_dynamic"] if "interaction_dynamic" in row.keys() else ""),
                        "relational_dynamic": str(row["relational_dynamic"] if "relational_dynamic" in row.keys() else ""),
                        "score_desc": round(s_desc, 4),
                        "score_tag": round(s_tag, 4),
                        "score_genre": round(s_genre, 4),
                        "score_final": round(s_final, 4),
                    }
                )

        results.sort(key=lambda item: item["score_final"], reverse=True)
        if len(results) <= top_k:
            return results

        # Maximal Marginal Relevance (MMR) diversity selection to prevent redundant hits
        selected: list[dict[str, Any]] = [results[0]]
        candidates = results[1:]
        lambda_param = 0.75  # 75% relevance score, 25% diversity bonus

        while len(selected) < top_k and candidates:
            best_idx = 0
            best_mmr_score = -999.0
            for c_idx, cand in enumerate(candidates):
                rel_score = cand["score_final"]
                cand_tags_set = set(cand["tags"])
                max_redundancy = 0.0
                for sel in selected:
                    sel_tags_set = set(sel["tags"])
                    union_len = len(cand_tags_set | sel_tags_set)
                    overlap = len(cand_tags_set & sel_tags_set) / max(1, union_len)
                    if overlap > max_redundancy:
                        max_redundancy = overlap
                mmr_score = lambda_param * rel_score - (1.0 - lambda_param) * max_redundancy
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = c_idx
            selected.append(candidates.pop(best_idx))

        return selected

    def get_library_stats(self) -> dict[str, int]:
        """Return total source count, entry count, and tag count."""
        with read_connection(self.db_path) as conn:
            sources = conn.execute("SELECT COUNT(*) AS c FROM ref_sources;").fetchone()["c"]
            entries = conn.execute("SELECT COUNT(*) AS c FROM ref_entries;").fetchone()["c"]
            tags = conn.execute("SELECT COUNT(*) AS c FROM ref_tags;").fetchone()["c"]
            return {"sources": int(sources), "entries": int(entries), "tags": int(tags)}

    def discover_dynamic_genres(
        self, base_catalog: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Enrich genre hierarchy with AI-synthesized blueprint genres and structured 'Primary:Sub' tags."""
        catalog: dict[str, list[str]] = {k: list(v) for k, v in base_catalog.items()}
        with read_connection(self.db_path) as conn:
            tag_rows = conn.execute(
                "SELECT DISTINCT tag_name FROM ref_tags ORDER BY id DESC LIMIT 120;"
            ).fetchall()
            bp_rows = conn.execute(
                "SELECT DISTINCT genre_primary, genre_sub FROM ref_book_blueprints ORDER BY id DESC;"
            ).fetchall()
        for bp in bp_rows:
            gp = str(bp["genre_primary"]).strip()
            gs = str(bp["genre_sub"]).strip()
            if gp and gs:
                subs = catalog.setdefault(gp, [])
                if gs not in subs:
                    subs.append(gs)
        if not tag_rows:
            return catalog
        self._merge_corpus_tags_into_catalog(
            catalog,
            [str(r["tag_name"]).strip() for r in tag_rows if r["tag_name"]],
        )
        return catalog

    @staticmethod
    def _merge_corpus_tags_into_catalog(
        catalog: dict[str, list[str]], tags: list[str]
    ) -> None:
        """Map structured 'Primary:Sub' tags extracted by AI into the genre dictionary."""
        for tag in tags:
            for sep in (":", "：", "·", "/"):
                if sep in tag:
                    p_name, s_name = [part.strip() for part in tag.split(sep, 1)]
                    if p_name and s_name:
                        subs = catalog.setdefault(p_name, [])
                        if s_name not in subs:
                            subs.append(s_name)
                    break
            else:
                if 2 <= len(tag) <= 8:
                    derived = catalog.setdefault("書庫衍生題材", [])
                    if tag not in derived and len(derived) < 12:
                        derived.append(tag)

    def get_style_and_theme_samples(self, limit: int = 4) -> list[dict[str, str]]:
        """Fetch recent extracted style descriptions and snippets for AI wizard question synthesis."""
        with read_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT description, snippet FROM ref_entries ORDER BY id DESC LIMIT ?;",
                (limit,),
            ).fetchall()
            return [
                {"description": str(r["description"]), "snippet": str(r["snippet"])}
                for r in rows
            ]

    def list_sources_with_counts(self) -> list[dict[str, Any]]:
        """List all imported reference book sources with chunk, entry, blueprint, and pacing template status."""
        with read_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.file_sha256, s.file_name, s.imported_at, s.chunk_count, s.entry_count,
                       EXISTS(SELECT 1 FROM ref_pacing_templates p WHERE p.source_id = s.id) AS has_pacing,
                       EXISTS(SELECT 1 FROM ref_book_blueprints b WHERE b.source_id = s.id) AS has_blueprint,
                       COALESCE((SELECT b.total_chapters_analyzed FROM ref_book_blueprints b WHERE b.source_id = s.id LIMIT 1), 0) AS total_chapters
                FROM ref_sources s
                ORDER BY s.id DESC;
                """
            ).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "file_sha256": str(r["file_sha256"]),
                    "file_name": str(r["file_name"]),
                    "imported_at": str(r["imported_at"]),
                    "chunk_count": int(r["chunk_count"]),
                    "entry_count": int(r["entry_count"]),
                    "has_skills": int(r["entry_count"]) > 0,
                    "has_pacing": bool(r["has_pacing"]),
                    "has_blueprint": bool(r["has_blueprint"]),
                    "total_chapters": int(r["total_chapters"]),
                }
                for r in rows
            ]

    def delete_pacing_templates_by_source(self, source_id: int) -> int:
        """Delete any macro pacing templates associated with a specific source_id."""
        with atomic_transaction(self.db_path) as conn:
            cur = conn.execute("DELETE FROM ref_pacing_templates WHERE source_id = ?;", (source_id,))
            return int(cur.rowcount)

    def delete_source(self, source_id: int) -> bool:
        """Delete a reference book source and cascade delete its entries, tags, and pacing templates."""
        cache_dir = self.db_path.parent / "cache"
        for ckpt in (
            cache_dir / f"stage1_facts_{source_id}.json",
            cache_dir / f"stage2_chunks_{source_id}.json",
        ):
            try:
                ckpt.unlink(missing_ok=True)
            except OSError:
                pass

        with atomic_transaction(self.db_path) as conn:
            conn.execute("DELETE FROM ref_pacing_templates WHERE source_id = ?;", (source_id,))
            cursor = conn.execute("DELETE FROM ref_sources WHERE id = ?;", (source_id,))
            res = cursor.rowcount > 0
            if res:
                self._invalidate_cache()
            return res

    def query_entries(
        self,
        source_id: int | None = None,
        keyword: str = "",
        tag: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Query knowledge entries with multi-dimensional filtering, pagination, and tags."""
        where_clauses: list[str] = []
        params: list[Any] = []
        if source_id is not None:
            where_clauses.append("e.source_id = ?")
            params.append(source_id)
        if keyword.strip():
            where_clauses.append("(e.description LIKE ? OR e.snippet LIKE ?)")
            kw = f"%{keyword.strip()}%"
            params.extend([kw, kw])
        if tag.strip():
            where_clauses.append(
                "EXISTS (SELECT 1 FROM ref_tags t WHERE t.entry_id = e.id AND t.tag_name = ?)"
            )
            params.append(tag.strip())

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with read_connection(self.db_path) as conn:
            count_row = conn.execute(
                f"SELECT COUNT(*) AS total FROM ref_entries e {where_sql};", params
            ).fetchone()
            total_count = int(count_row["total"]) if count_row else 0

            fetch_params = list(params) + [limit, offset]
            entry_rows = conn.execute(
                f"""
                SELECT e.id, e.source_id, e.description, e.snippet, e.created_at, s.file_name,
                       COALESCE(e.subject_role, '') AS subject_role,
                       COALESCE(e.object_role, '') AS object_role,
                       COALESCE(e.interaction_dynamic, '') AS interaction_dynamic,
                       COALESCE(e.relational_dynamic, '') AS relational_dynamic,
                       COALESCE(e.chapter_index, 0) AS chapter_index,
                       COALESCE(e.exemplar_score, 0.0) AS exemplar_score
                FROM ref_entries e
                JOIN ref_sources s ON e.source_id = s.id
                {where_sql}
                ORDER BY e.id DESC
                LIMIT ? OFFSET ?;
                """,
                fetch_params,
            ).fetchall()

            if not entry_rows:
                return [], total_count

            entry_ids = [int(r["id"]) for r in entry_rows]
            placeholders = ",".join("?" for _ in entry_ids)
            tag_rows = conn.execute(
                f"SELECT entry_id, tag_name FROM ref_tags WHERE entry_id IN ({placeholders});",
                entry_ids,
            ).fetchall()

            tag_map: dict[int, list[str]] = {}
            for tr in tag_rows:
                tag_map.setdefault(int(tr["entry_id"]), []).append(str(tr["tag_name"]))

            entries = [
                {
                    "id": int(r["id"]),
                    "source_id": int(r["source_id"]),
                    "file_name": str(r["file_name"]),
                    "description": str(r["description"]),
                    "snippet": str(r["snippet"]),
                    "created_at": str(r["created_at"]),
                    "tags": tag_map.get(int(r["id"]), []),
                    "subject_role": str(r["subject_role"] if "subject_role" in r.keys() else ""),
                    "object_role": str(r["object_role"] if "object_role" in r.keys() else ""),
                    "interaction_dynamic": str(r["interaction_dynamic"] if "interaction_dynamic" in r.keys() else ""),
                    "relational_dynamic": str(r["relational_dynamic"] if "relational_dynamic" in r.keys() else ""),
                    "chapter_index": int(r["chapter_index"] if "chapter_index" in r.keys() else 0),
                    "exemplar_score": float(r["exemplar_score"] if "exemplar_score" in r.keys() else 0.0),
                }
                for r in entry_rows
            ]
            return entries, total_count

    def delete_entry(self, entry_id: int) -> bool:
        """Delete a single knowledge entry and decrement source entry_count."""
        with atomic_transaction(self.db_path) as conn:
            entry = conn.execute(
                "SELECT source_id FROM ref_entries WHERE id = ?;", (entry_id,)
            ).fetchone()
            if not entry:
                return False
            source_id = int(entry["source_id"])
            conn.execute("DELETE FROM ref_entries WHERE id = ?;", (entry_id,))
            conn.execute(
                "UPDATE ref_sources SET entry_count = MAX(0, entry_count - 1) WHERE id = ?;",
                (source_id,),
            )
            self._invalidate_cache()
            return True

    def add_tag_to_entry(
        self, entry_id: int, tag_name: str, tag_vec: np.ndarray, axis_type: str = "RHETORIC"
    ) -> bool:
        """Associate a new semantic tag with an existing entry."""
        clean_name = tag_name.strip()
        if not clean_name:
            return False
        with atomic_transaction(self.db_path) as conn:
            entry = conn.execute("SELECT id FROM ref_entries WHERE id = ?;", (entry_id,)).fetchone()
            if not entry:
                return False
            existing = conn.execute(
                "SELECT id FROM ref_tags WHERE entry_id = ? AND tag_name = ?;",
                (entry_id, clean_name),
            ).fetchone()
            if existing:
                return True
            conn.execute(
                """
                INSERT INTO ref_tags (entry_id, axis_type, tag_name, tag_embedding)
                VALUES (?, ?, ?, ?);
                """,
                (entry_id, axis_type.strip(), clean_name, vector_to_blob(tag_vec)),
            )
            self._invalidate_cache()
            return True

    def list_tags_with_frequency(self) -> list[dict[str, Any]]:
        """List all unique tags with their occurrence frequency."""
        with read_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT tag_name, COUNT(*) AS count
                FROM ref_tags
                GROUP BY tag_name
                ORDER BY count DESC, tag_name ASC;
                """
            ).fetchall()
            return [{"tag_name": str(r["tag_name"]), "count": int(r["count"])} for r in rows]

    def rename_tag_globally(self, old_tag: str, new_tag: str, new_vec: np.ndarray) -> int:
        """Rename an existing tag across the entire reference library and update its embedding."""
        old_clean = old_tag.strip()
        new_clean = new_tag.strip()
        if not old_clean or not new_clean or old_clean == new_clean:
            return 0
        new_blob = vector_to_blob(new_vec)
        with atomic_transaction(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE ref_tags
                SET tag_name = ?, tag_embedding = ?
                WHERE tag_name = ?;
                """,
                (new_clean, new_blob, old_clean),
            )
            res = cursor.rowcount
            if res > 0:
                self._invalidate_cache()
            return res

    def prune_marginalized_tags(
        self,
        min_frequency: int = 2,
        eps: float = 0.45,
    ) -> dict[str, Any]:
        """Analyze topological distribution without destructively deleting rare literary tags."""
        density_info = self.get_density_analysis(eps=eps, min_samples=2)
        outliers = density_info.get("outliers", [])
        return {
            "pruned_tags_count": 0,
            "pruned_tag_names": [],
            "total_outliers_identified": len(outliers),
            "deleted_tagless_entries": 0,
            "message": "已保留所有長尾特色文學標籤，確保文學資產完整性，不作破壞性刪除。",
        }

    def _fetch_distinct_tags_with_embeddings(
        self,
    ) -> tuple[list[str], list[int], list[np.ndarray]]:
        """Fetch distinct tags ordered by frequency, keeping vectors matching the dominant dimension."""
        with read_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT tag_name, COUNT(*) AS count, tag_embedding
                FROM ref_tags
                GROUP BY tag_name
                ORDER BY count DESC, tag_name ASC;
                """
            ).fetchall()

        if not rows:
            return [], [], []

        raw_items: list[tuple[str, int, np.ndarray]] = []
        dim_counts: Counter[int] = Counter()
        for r in rows:
            blob = r["tag_embedding"]
            if not blob:
                continue
            vec = blob_to_vector(blob)
            if vec.size == 0:
                continue
            raw_items.append((str(r["tag_name"]), int(r["count"]), vec))
            dim_counts[vec.size] += 1

        if not dim_counts:
            return [], [], []

        expected_dim = dim_counts.most_common(1)[0][0]
        names: list[str] = []
        counts: list[int] = []
        vectors: list[np.ndarray] = []
        for name, count, vec in raw_items:
            if vec.size == expected_dim:
                names.append(name)
                counts.append(count)
                vectors.append(vec)
        if len(names) < len(raw_items):
            logger.warning(
                "Dropped %d distinct tags whose embedding dimension differs from %d.",
                len(raw_items) - len(names),
                expected_dim,
            )
        return names, counts, vectors

    @staticmethod
    def _select_representative_tag_indices(
        counts: list[int],
        clusters: list[str],
        cluster_order: list[str],
        max_nodes: int,
    ) -> list[int]:
        """Select a bounded stratified subset covering clusters, high-frequency tags, and long-tail singletons."""
        total = len(counts)
        if total <= max_nodes:
            return list(range(total))

        chosen: set[int] = set()
        # 1. Guarantee coverage of every semantic cluster with its highest-frequency exemplars
        cluster_slots: dict[str, int] = {c: 0 for c in cluster_order}
        for idx, c in enumerate(clusters):
            if len(chosen) >= max_nodes:
                break
            if cluster_slots.get(c, 4) < 4:
                chosen.add(idx)
                cluster_slots[c] = cluster_slots.get(c, 0) + 1
                if all(v >= 4 for v in cluster_slots.values()):
                    break

        # 2. Reserve a slice for long-tail rare/orphan tags so outlier detection stays visible
        tail_budget = min(30, max(0, max_nodes // 8))
        head_cap = max(len(chosen), max_nodes - tail_budget)
        for idx in range(total):
            if len(chosen) >= head_cap:
                break
            chosen.add(idx)

        for idx in range(total - 1, -1, -1):
            if len(chosen) >= max_nodes:
                break
            if counts[idx] <= 2:
                chosen.add(idx)

        for idx in range(total):
            if len(chosen) >= max_nodes:
                break
            chosen.add(idx)

        return sorted(chosen, key=lambda idx: (-counts[idx], idx))

    @staticmethod
    def _deduplicate_entry_tags(conn: Any, tag_names: set[str]) -> None:
        """Deduplicate (entry_id, tag_name) pairs in batches for modified target tags."""
        if not tag_names:
            return
        ordered = sorted(tag_names)
        for i in range(0, len(ordered), 400):
            chunk = ordered[i : i + 400]
            placeholders = ",".join("?" for _ in chunk)
            conn.execute(
                f"""
                DELETE FROM ref_tags
                WHERE tag_name IN ({placeholders})
                  AND id NOT IN (
                      SELECT MIN(id)
                      FROM ref_tags
                      WHERE tag_name IN ({placeholders})
                      GROUP BY entry_id, tag_name
                  );
                """,
                chunk + chunk,
            )

    def apply_tag_rules(
        self, rules: list[TagRuleContract], embedder: Any
    ) -> dict[str, int]:
        """Apply AI tag maintenance rules (normalize, merge, drop) atomically with deduplication."""
        stats = {"normalized": 0, "merged": 0, "dropped": 0, "total_affected": 0}
        if not rules:
            return stats

        # Pre-compute embeddings once for distinct cleaned & Traditional target tags
        normalized_targets: dict[int, str] = {}
        for idx, r in enumerate(rules):
            if r.action in ("NORMALIZE", "MERGE"):
                cleaned_new = _clean_rule_tag_name(r.new_tag)
                if cleaned_new:
                    normalized_targets[idx] = convert_simplified_to_traditional(cleaned_new)

        new_tag_names = sorted(set(normalized_targets.values()))
        tag_vectors: dict[str, bytes] = {}
        if new_tag_names:
            vecs = embedder.embed_batch(new_tag_names)
            for name, vec in zip(new_tag_names, vecs):
                tag_vectors[name] = vector_to_blob(vec)

        modified_new_tags: set[str] = set()
        with atomic_transaction(self.db_path) as conn:
            for idx, rule in enumerate(rules):
                raw_old = rule.old_tag.strip()
                cleaned_old = _clean_rule_tag_name(raw_old)
                old_candidates = [raw_old] if raw_old else []
                if cleaned_old and cleaned_old != raw_old:
                    old_candidates.append(cleaned_old)
                if not old_candidates:
                    continue

                if rule.action == "DROP":
                    placeholders = ",".join("?" for _ in old_candidates)
                    c = conn.execute(
                        f"DELETE FROM ref_tags WHERE tag_name IN ({placeholders});",
                        old_candidates,
                    ).rowcount
                    if c > 0:
                        stats["dropped"] += c
                        stats["total_affected"] += c
                elif rule.action in ("NORMALIZE", "MERGE"):
                    new_name = normalized_targets.get(idx, "")
                    target_olds = [t for t in old_candidates if t != new_name]
                    if not new_name or not target_olds:
                        continue
                    blob = tag_vectors.get(new_name)
                    if not blob:
                        blob = vector_to_blob(embedder.embed_text(new_name))
                        tag_vectors[new_name] = blob
                    placeholders = ",".join("?" for _ in target_olds)
                    c = conn.execute(
                        f"""
                        UPDATE ref_tags
                        SET tag_name = ?, tag_embedding = ?
                        WHERE tag_name IN ({placeholders});
                        """,
                        [new_name, blob, *target_olds],
                    ).rowcount
                    if c > 0:
                        modified_new_tags.add(new_name)
                        stats["normalized" if rule.action == "NORMALIZE" else "merged"] += c
                        stats["total_affected"] += c

            self._deduplicate_entry_tags(conn, modified_new_tags)
            # Keep existing behavior: do not purge entries on tag maintenance
            stats["deleted_tagless_entries"] = 0

        self._invalidate_cache()
        return stats

    def get_density_analysis(
        self, eps: float = 0.45, min_samples: int = 2, max_tags: int = 600
    ) -> dict[str, Any]:
        """Perform bounded DBSCAN density analysis across representative library tags."""
        all_names, all_counts, all_vecs = self._fetch_distinct_tags_with_embeddings()
        if not all_names:
            return {"dense_clusters": [], "outliers": []}

        # Bound DBSCAN input size to avoid O(N^2) memory blowup on large libraries
        if len(all_names) > max_tags:
            head_n = max(1, max_tags - 150)
            head_indices = list(range(head_n))
            tail_pool = [i for i in range(head_n, len(all_names)) if all_counts[i] <= 2]
            if len(tail_pool) >= 150:
                tail_indices = tail_pool[-150:]
            else:
                extra_start = max(head_n, len(all_names) - (max_tags - head_n))
                tail_indices = list(range(extra_start, len(all_names)))
            selected = head_indices + [i for i in tail_indices if i >= head_n]
        else:
            selected = list(range(len(all_names)))

        names = [all_names[i] for i in selected]
        counts = [all_counts[i] for i in selected]
        vectors = np.stack([all_vecs[i] for i in selected])

        # Tighten eps on large Chinese tag spaces to prevent transitive mega-cluster percolation
        effective_eps = min(eps, 0.18) if (len(vectors) > 50 and eps >= 0.35) else eps
        labels = compute_density_clusters(vectors, eps=effective_eps, min_samples=min_samples)

        cluster_groups: dict[int, list[str]] = {}
        outliers: list[dict[str, Any]] = []
        for i, name in enumerate(names):
            lbl = int(labels[i])
            if lbl >= 0:
                cluster_groups.setdefault(lbl, []).append(name)
            else:
                outliers.append({"tag_name": name, "count": counts[i]})

        dense_clusters = [
            {"cluster_id": cid, "tags": tags}
            for cid, tags in sorted(
                cluster_groups.items(), key=lambda x: len(x[1]), reverse=True
            )
        ]
        return {"dense_clusters": dense_clusters, "outliers": outliers}

    def get_tag_clusters_and_stats(self, max_nodes: int = 350) -> dict[str, Any]:
        """Compute full-library tag metrics alongside 2D PCA coordinates and DBSCAN labels for representative nodes."""
        if max_nodes == 350 and self._tag_assets_cache is not None:
            return self._tag_assets_cache

        all_names, all_counts, all_vecs = self._fetch_distinct_tags_with_embeddings()
        if not all_names:
            return {
                "stats": {
                    "total_unique": 0,
                    "total_usages": 0,
                    "genre_tags_count": 0,
                    "technique_tags_count": 0,
                    "clusters_count": 0,
                    "outliers_count": 0,
                },
                "nodes": [],
                "clusters": {},
            }

        raw_clusters = [resolve_tag_cluster(n) for n in all_names]
        raw_cluster_counts = Counter(raw_clusters)
        if len(all_names) > 50:
            # Fold noisy single-occurrence prefix fragments into "其他" to keep legend legible
            kept_clusters: set[str] = {
                c for c, cnt in raw_cluster_counts.most_common(24)
                if cnt >= 2 or c in STANDARD_CLUSTER_COLORS
            }
            kept_clusters.add("通用技法")
            all_node_clusters = [c if c in kept_clusters else "其他" for c in raw_clusters]
        else:
            all_node_clusters = raw_clusters

        full_clusters_map = dict(Counter(all_node_clusters).most_common())
        selected = self._select_representative_tag_indices(
            all_counts, all_node_clusters, list(full_clusters_map.keys()), max_nodes
        )

        sample_names = [all_names[i] for i in selected]
        sample_counts = [all_counts[i] for i in selected]
        sample_clusters = [all_node_clusters[i] for i in selected]
        sample_vectors = np.stack([all_vecs[i] for i in selected])

        coords = project_2d_pca(sample_vectors)
        effective_eps = 0.18 if len(sample_vectors) > 50 else 0.45
        density_labels = compute_density_clusters(sample_vectors, eps=effective_eps, min_samples=2)

        nodes = [
            {
                "name": sample_names[i],
                "count": sample_counts[i],
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "cluster": sample_clusters[i],
                "color": get_cluster_color(sample_clusters[i]),
                "density_cluster": int(density_labels[i]),
                "is_outlier": bool(density_labels[i] == -1),
            }
            for i in range(len(sample_names))
        ]

        visible_clusters = {
            c: full_clusters_map[c]
            for c in full_clusters_map
            if c in set(sample_clusters)
        }
        genre_count = sum(1 for n in all_names if ":" in n or "：" in n)
        result = {
            "stats": {
                "total_unique": len(all_names),
                "total_usages": sum(all_counts),
                "genre_tags_count": genre_count,
                "technique_tags_count": len(all_names) - genre_count,
                "clusters_count": len(visible_clusters),
                "outliers_count": int(np.sum(density_labels == -1)),
            },
            "nodes": nodes,
            "clusters": visible_clusters,
        }
        if max_nodes == 350:
            self._tag_assets_cache = result
        return result

    def save_book_blueprint(
        self,
        source_id: int,
        blueprint: BookBlueprintSynthesisContract,
        factual_pacing_curve: list[dict[str, Any]],
        total_chapters: int,
    ) -> int:
        """Insert or replace permanent factual book blueprint for reference novel."""
        outline_json = json.dumps(blueprint.macro_outline, ensure_ascii=False)
        chars_json = json.dumps([c.model_dump() for c in blueprint.character_network], ensure_ascii=False)
        factions_json = json.dumps([f.model_dump() for f in blueprint.faction_geopolitics], ensure_ascii=False)
        hierarchy_json = json.dumps([h.model_dump() for h in blueprint.hierarchy_rules], ensure_ascii=False)
        laws_json = json.dumps(blueprint.world_laws, ensure_ascii=False)
        pacing_json = json.dumps(factual_pacing_curve, ensure_ascii=False)
        with atomic_transaction(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO ref_book_blueprints (
                    source_id, novel_title, genre_primary, genre_sub,
                    macro_outline_json, character_network_json, faction_geopolitics_json,
                    hierarchy_rules_json, world_laws_json, factual_pacing_curve, total_chapters_analyzed,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(source_id) DO UPDATE SET
                    novel_title = excluded.novel_title,
                    genre_primary = excluded.genre_primary,
                    genre_sub = excluded.genre_sub,
                    macro_outline_json = excluded.macro_outline_json,
                    character_network_json = excluded.character_network_json,
                    faction_geopolitics_json = excluded.faction_geopolitics_json,
                    hierarchy_rules_json = excluded.hierarchy_rules_json,
                    world_laws_json = excluded.world_laws_json,
                    factual_pacing_curve = excluded.factual_pacing_curve,
                    total_chapters_analyzed = excluded.total_chapters_analyzed,
                    updated_at = datetime('now');
                """,
                (
                    source_id,
                    blueprint.novel_title,
                    blueprint.genre_primary,
                    blueprint.genre_sub,
                    outline_json,
                    chars_json,
                    factions_json,
                    hierarchy_json,
                    laws_json,
                    pacing_json,
                    total_chapters,
                ),
            )
            blueprint_id = cur.lastrowid
            if not blueprint_id:
                row = conn.execute("SELECT id FROM ref_book_blueprints WHERE source_id = ?;", (source_id,)).fetchone()
                blueprint_id = int(row["id"]) if row else 0
            return blueprint_id

    def get_book_blueprint(self, source_id: int) -> dict[str, Any] | None:
        """Fetch book blueprint by source_id with parsed JSON structures."""
        with read_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM ref_book_blueprints WHERE source_id = ?;", (source_id,)
            ).fetchone()
            if not row:
                return None

            def _parse_json(text: str, default: Any) -> Any:
                try:
                    return json.loads(text) if text else default
                except Exception:
                    return default

            return {
                "id": int(row["id"]),
                "source_id": int(row["source_id"]),
                "novel_title": str(row["novel_title"]),
                "genre_primary": str(row["genre_primary"]),
                "genre_sub": str(row["genre_sub"]),
                "macro_outline": _parse_json(row["macro_outline_json"], {}),
                "character_network": _parse_json(row["character_network_json"], []),
                "faction_geopolitics": _parse_json(row["faction_geopolitics_json"], []),
                "hierarchy_rules": _parse_json(row["hierarchy_rules_json"], []),
                "world_laws": _parse_json(row["world_laws_json"], []) if "world_laws_json" in row.keys() else [],
                "factual_pacing_curve": _parse_json(row["factual_pacing_curve"], []),
                "total_chapters_analyzed": int(row["total_chapters_analyzed"]),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
            }

    def has_book_blueprint(self, source_id: int) -> bool:
        """Check whether a reference book already has a synthesized factual blueprint."""
        with read_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM ref_book_blueprints WHERE source_id = ?;", (source_id,)
            ).fetchone()
            return row is not None

    def save_pacing_template(
        self,
        template: MacroPacingTemplateContract,
        source_id: int | None = None,
    ) -> int:
        """Insert or replace a macro pacing template, its stages, and 100% chapter curve data."""
        tags_json = json.dumps(template.pacing_tags, ensure_ascii=False)
        total_chapters = getattr(template, "total_chapters_analyzed", 0)
        curve_data = getattr(template, "chapter_curve_data", [])
        curve_json = json.dumps(curve_data, ensure_ascii=False)
        with atomic_transaction(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO ref_pacing_templates 
                (source_id, name, genre_primary, genre_sub, archetype, pacing_tags, description, total_stages, total_chapters, chapter_curve_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    source_id=excluded.source_id,
                    genre_primary=excluded.genre_primary,
                    genre_sub=excluded.genre_sub,
                    archetype=excluded.archetype,
                    pacing_tags=excluded.pacing_tags,
                    description=excluded.description,
                    total_stages=excluded.total_stages,
                    total_chapters=excluded.total_chapters,
                    chapter_curve_data=excluded.chapter_curve_data
                """,
                (
                    source_id,
                    template.template_name,
                    template.genre_primary,
                    template.genre_sub,
                    template.narrative_archetype,
                    tags_json,
                    template.summary,
                    len(template.stages),
                    total_chapters,
                    curve_json,
                ),
            )
            template_id = cur.lastrowid
            if not template_id:
                row = conn.execute(
                    "SELECT id FROM ref_pacing_templates WHERE name = ?",
                    (template.template_name,),
                ).fetchone()
                template_id = int(row["id"]) if row else 0

            # Replace old stages
            conn.execute("DELETE FROM ref_pacing_stages WHERE template_id = ?", (template_id,))
            for st in template.stages:
                conn.execute(
                    """
                    INSERT INTO ref_pacing_stages
                    (template_id, stage_index, offset_percent, pacing_phase, target_tension, target_progress, target_density, dramatic_purpose)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        template_id,
                        st.stage_index,
                        st.offset_percent,
                        st.pacing_phase,
                        st.target_tension,
                        st.target_progress,
                        st.target_density,
                        st.dramatic_purpose,
                    ),
                )
            return template_id

    def get_pacing_template(self, template_id: int) -> dict[str, Any] | None:
        """Retrieve pacing template with full stage details."""
        with read_connection(self.db_path) as conn:
            tpl = conn.execute(
                "SELECT * FROM ref_pacing_templates WHERE id = ?", (template_id,)
            ).fetchone()
            if not tpl:
                return None
            stages = conn.execute(
                "SELECT * FROM ref_pacing_stages WHERE template_id = ? ORDER BY stage_index ASC",
                (template_id,),
            ).fetchall()
            tags = []
            try:
                tags = json.loads(tpl["pacing_tags"])
            except Exception:
                tags = []
            curve_data = []
            try:
                if "chapter_curve_data" in tpl.keys() and tpl["chapter_curve_data"]:
                    curve_data = json.loads(tpl["chapter_curve_data"])
            except Exception:
                curve_data = []
            total_chapters = int(tpl["total_chapters"]) if "total_chapters" in tpl.keys() and tpl["total_chapters"] else 0
            return {
                "id": tpl["id"],
                "source_id": tpl["source_id"],
                "name": tpl["name"],
                "genre_primary": tpl["genre_primary"],
                "genre_sub": tpl["genre_sub"],
                "archetype": tpl["archetype"],
                "pacing_tags": tags,
                "description": tpl["description"],
                "total_stages": tpl["total_stages"],
                "total_chapters": total_chapters,
                "chapter_curve_data": curve_data,
                "created_at": tpl["created_at"],
                "stages": [dict(s) for s in stages],
            }

    def list_pacing_templates(self) -> list[dict[str, Any]]:
        """List all pacing templates with basic metadata and stage previews."""
        with read_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM ref_pacing_templates ORDER BY id DESC"
            ).fetchall()
            result = []
            for r in rows:
                tid = r["id"]
                stages = conn.execute(
                    "SELECT * FROM ref_pacing_stages WHERE template_id = ? ORDER BY stage_index ASC",
                    (tid,),
                ).fetchall()
                tags = []
                try:
                    tags = json.loads(r["pacing_tags"])
                except Exception:
                    tags = []
                curve_data = []
                try:
                    if "chapter_curve_data" in r.keys() and r["chapter_curve_data"]:
                        curve_data = json.loads(r["chapter_curve_data"])
                except Exception:
                    curve_data = []
                total_chapters = int(r["total_chapters"]) if "total_chapters" in r.keys() and r["total_chapters"] else 0
                result.append(
                    {
                        "id": tid,
                        "source_id": r["source_id"],
                        "name": r["name"],
                        "genre_primary": r["genre_primary"],
                        "genre_sub": r["genre_sub"],
                        "archetype": r["archetype"],
                        "pacing_tags": tags,
                        "description": r["description"],
                        "total_stages": r["total_stages"],
                        "total_chapters": total_chapters,
                        "chapter_curve_data": curve_data,
                        "created_at": r["created_at"],
                        "stages": [dict(s) for s in stages],
                    }
                )
            return result

    def delete_pacing_template(self, template_id: int) -> bool:
        """Delete a pacing template and its stages."""
        with atomic_transaction(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM ref_pacing_templates WHERE id = ?", (template_id,)
            )
            return cur.rowcount > 0

    def override_pacing_stages(
        self, template_id: int, stages: list[dict[str, Any]]
    ) -> bool:
        """Update stage curves for an existing pacing template."""
        with atomic_transaction(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM ref_pacing_templates WHERE id = ?", (template_id,)
            ).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM ref_pacing_stages WHERE template_id = ?", (template_id,))
            for idx, st in enumerate(stages, start=1):
                conn.execute(
                    """
                    INSERT INTO ref_pacing_stages
                    (template_id, stage_index, offset_percent, pacing_phase, target_tension, target_progress, target_density, dramatic_purpose)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        template_id,
                        st.get("stage_index", idx),
                        float(st.get("offset_percent", 0.0)),
                        str(st.get("pacing_phase", "BUILDUP")),
                        int(st.get("target_tension", 3)),
                        int(st.get("target_progress", 3)),
                        int(st.get("target_density", 3)),
                        str(st.get("dramatic_purpose", "")),
                    ),
                )
            conn.execute(
                "UPDATE ref_pacing_templates SET total_stages = ? WHERE id = ?",
                (len(stages), template_id),
            )
            return True

    def match_pacing_template(
        self,
        genre_primary: str,
        genre_sub: str,
        keywords: list[str] | None = None,
        archetype: str | None = None,
    ) -> dict[str, Any] | None:
        """Three-tier scoring engine to recommend the most suitable pacing template."""
        all_tpls = self.list_pacing_templates()
        if not all_tpls:
            return None
        kw_set = set(keywords or [])
        scored: list[tuple[float, dict[str, Any]]] = []
        for tpl in all_tpls:
            score = 0.0
            if tpl["genre_primary"].strip().lower() == genre_primary.strip().lower():
                score += 4.0
            if genre_sub and tpl["genre_sub"].strip().lower() == genre_sub.strip().lower():
                score += 2.0
            if archetype and tpl["archetype"].strip().lower() == archetype.strip().lower():
                score += 2.0
            tpl_tags = set(tpl.get("pacing_tags", []))
            if kw_set and tpl_tags:
                jaccard = len(kw_set & tpl_tags) / len(kw_set | tpl_tags)
                score += 3.0 * jaccard
            scored.append((score, tpl))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def blend_pacing_templates(
        self,
        template_id_a: int,
        template_id_b: int,
        weight_a: float = 0.5,
        new_name: str = "",
    ) -> int:
        """Synthesize a hybrid pacing template from two parent archetypes."""
        tpl_a = self.get_pacing_template(template_id_a)
        tpl_b = self.get_pacing_template(template_id_b)
        if not tpl_a or not tpl_b:
            raise ValueError("Both parent templates must exist to blend.")
        weight_b = 1.0 - weight_a
        name = new_name or f"{tpl_a['name']} x {tpl_b['name']} (混血)"
        stages_a = tpl_a["stages"]
        stages_b = tpl_b["stages"]
        num_stages = max(len(stages_a), len(stages_b))
        blended_stages = []
        for i in range(num_stages):
            sa = stages_a[min(i, len(stages_a) - 1)]
            sb = stages_b[min(i, len(stages_b) - 1)]
            offset = round(sa["offset_percent"] * weight_a + sb["offset_percent"] * weight_b, 1)
            tension = max(1, min(5, round(sa["target_tension"] * weight_a + sb["target_tension"] * weight_b)))
            progress = max(1, min(5, round(sa["target_progress"] * weight_a + sb["target_progress"] * weight_b)))
            density = max(1, min(5, round(sa["target_density"] * weight_a + sb["target_density"] * weight_b)))
            phase = sa["pacing_phase"] if weight_a >= 0.5 else sb["pacing_phase"]
            purpose = f"{sa['dramatic_purpose']} 與 {sb['dramatic_purpose']} 融合"
            blended_stages.append(
                PacingStageTemplateContract(
                    stage_index=i + 1,
                    offset_percent=offset,
                    pacing_phase=phase,
                    target_tension=tension,
                    target_progress=progress,
                    target_density=density,
                    dramatic_purpose=purpose,
                )
            )
        all_tags = list(dict.fromkeys(tpl_a.get("pacing_tags", []) + tpl_b.get("pacing_tags", [])))
        cur_a = tpl_a.get("chapter_curve_data", [])
        cur_b = tpl_b.get("chapter_curve_data", [])
        blended_curve = []
        if cur_a or cur_b:
            src_a = cur_a if cur_a else cur_b
            src_b = cur_b if cur_b else cur_a
            total_pts = max(len(src_a), len(src_b))
            for i in range(total_pts):
                pa = src_a[min(i, len(src_a) - 1)]
                pb = src_b[min(i, len(src_b) - 1)]
                blended_curve.append({
                    "ch": i + 1,
                    "pct": round(pa.get("pct", 0) * weight_a + pb.get("pct", 0) * weight_b, 2),
                    "t": round(pa.get("t", 3.0) * weight_a + pb.get("t", 3.0) * weight_b, 2),
                    "p": round(pa.get("p", 3.0) * weight_a + pb.get("p", 3.0) * weight_b, 2),
                    "d": round(pa.get("d", 3.0) * weight_a + pb.get("d", 3.0) * weight_b, 2),
                    "title": f"混血節奏點 #{i + 1}",
                })

        hybrid_contract = MacroPacingTemplateContract(
            template_name=name,
            genre_primary=tpl_a["genre_primary"],
            genre_sub=tpl_a["genre_sub"],
            source_novel_title=f"{tpl_a['name']} & {tpl_b['name']}",
            total_chapters_analyzed=len(blended_curve),
            narrative_archetype=f"{tpl_a['archetype']}+{tpl_b['archetype']}",
            pacing_tags=all_tags,
            summary=f"混血範本：融合 {tpl_a['name']} ({weight_a*100:.0f}%) 與 {tpl_b['name']} ({weight_b*100:.0f}%)",
            stages=blended_stages,
            chapter_curve_data=blended_curve,
        )
        return self.save_pacing_template(hybrid_contract)
