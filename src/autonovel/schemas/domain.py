"""Domain models for project state, database entities, and API payloads."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PipelineState(str, Enum):
    """Lifecycle states of the single novel project."""

    WIZARD = "WIZARD"
    REVIEW_GATE = "REVIEW_GATE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"


class ChapterStatus(str, Enum):
    """Lifecycle states of a chapter in the two-phase atomic commit workflow."""

    UNWRITTEN = "UNWRITTEN"
    WRITING = "WRITING"
    STAGING = "STAGING"
    REVISING = "REVISING"
    COMPLETED = "COMPLETED"
    REVISED = "REVISED"


class VolumeStatus(str, Enum):
    """Lifecycle states of a novel volume."""

    PLANNED = "PLANNED"
    WRITING = "WRITING"
    COMPLETED = "COMPLETED"


class CharacterStatus(str, Enum):
    """Lifecycle states of a character."""

    ALIVE = "ALIVE"
    DEAD = "DEAD"
    ARCHIVED = "ARCHIVED"


class HierarchyChangeType(str, Enum):
    """Permitted state transition types for character rank or artifact hierarchy."""

    NORMAL_BREAKTHROUGH = "NORMAL_BREAKTHROUGH"
    INJURY_REGRESSION = "INJURY_REGRESSION"
    SEALED = "SEALED"
    RE_CULTIVATION = "RE_CULTIVATION"
    ARTIFACT_DAMAGED = "ARTIFACT_DAMAGED"
    FORTUITOUS_LEAP = "FORTUITOUS_LEAP"


class ForeshadowScope(str, Enum):
    """Expected resolution span tier for a foreshadowing item."""

    SHORT = "SHORT"
    VOLUME = "VOLUME"
    MAIN = "MAIN"


class ForeshadowStatus(str, Enum):
    """Resolution state of a foreshadowing item."""

    PENDING = "PENDING"
    RESOLVED = "RESOLVED"
    EXPIRED = "EXPIRED"


class AnomalyType(str, Enum):
    """Categories of automatically detected and resolved anomalies."""

    HARD_CONFLICT = "HARD_CONFLICT"
    SOFT_DRIFT = "SOFT_DRIFT"
    FORESHADOW_EXPIRED = "FORESHADOW_EXPIRED"
    PACING_ALERT = "PACING_ALERT"
    WATCHDOG_RESTART = "WATCHDOG_RESTART"


class SceneBeat(BaseModel):
    """Paragraph-level scene transition beat within a chapter outline."""

    beat_index: int = Field(..., ge=1, le=5)
    scene_tag: str
    pacing_instruction: str
    plot_objective: str
    involved_characters: list[str] = Field(default_factory=list)
    location_name: str = ""


class StageGateQuota(BaseModel):
    """Macro pacing stage gate for a 5,000,000+ word novel."""

    stage_index: int = Field(..., ge=1)
    stage_title: str
    target_word_quota: int = Field(..., ge=50000)
    max_allowed_rank_level: int = Field(..., ge=1)
    core_mystery_quota: str
    milestone_goal: str


class BlacklistConfig(BaseModel):
    """Three-tier adaptive vocabulary blacklist."""

    tier1_global_cliches: list[str] = Field(default_factory=list)
    tier2_genre_anachronisms: list[str] = Field(default_factory=list)
    tier3_user_custom: list[str] = Field(default_factory=list)

    def merged_terms(self) -> list[str]:
        """Return deduplicated list of all active forbidden terms."""
        seen: set[str] = set()
        ordered: list[str] = []
        for term in (
            self.tier1_global_cliches
            + self.tier2_genre_anachronisms
            + self.tier3_user_custom
        ):
            cleaned = term.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                ordered.append(cleaned)
        return ordered


class PreflightFileItem(BaseModel):
    """Result item from the 1-second reference book import pre-flight check."""

    file_path: str
    file_name: str
    file_sha256: str
    encoding: str
    status: str  # "NEW" | "DUPLICATE" | "CORRUPTED"
    error_reason: str = ""
    has_skills: bool = False
    entry_count: int = 0
    has_pacing: bool = False
    has_blueprint: bool = False


class ReferenceIngestItem(BaseModel):
    """Per-file action specification for reference library ingestion."""

    file_path: str
    action: str = "INGEST"  # "INGEST" | "REEXTRACT" | "SKIP"
    extract_skills: bool = True
    extract_pacing: bool = False


class ExportRequest(BaseModel):
    """User configuration for exporting chapters across 5 supported formats."""

    scope_type: str = "ALL"  # "ALL" | "VOLUME" | "RANGE"
    volume_num: int | None = None
    start_chapter: int | None = None
    end_chapter: int | None = None
    formats: list[str] = Field(
        default_factory=lambda: ["md", "txt", "html", "epub", "docx"]
    )
    merge_mode: str = "SINGLE_FILE"  # "SINGLE_FILE" | "PER_VOLUME"


class AcceptanceReport(BaseModel):
    """Quantitative verification report for end-of-volume and end-of-book checks."""

    hard_conflict_rate: float = 0.0
    timeline_consistency_rate: float = 0.0
    foreshadow_recovery_rate: float = 0.0
    pacing_stability_rate: float = 0.0
    mean_literary_score: float = 0.0
    passed: bool = False
    failed_reasons: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
