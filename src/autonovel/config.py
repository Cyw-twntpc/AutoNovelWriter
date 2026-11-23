"""Configuration constants, paths, and logging setup for AutoNovelWriter."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path

logger = logging.getLogger(__name__)


def env_float(name: str, default: float) -> float:
    """Parse a float environment variable, falling back to the default when malformed."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using default %s.", name, raw, default)
        return default


def env_int(name: str, default: int) -> int:
    """Parse an int environment variable, falling back to the default when malformed."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using default %s.", name, raw, default)
        return default


@dataclass(frozen=True)
class ContextBudgetConfig:
    """Strict 16k token budget allocation across prompt components."""

    total_context_tokens: int = 16000
    system_and_style_tokens: int = 1200
    context_router_settings_tokens: int = 2300
    reference_digest_tokens: int = 500
    scene_beats_tokens: int = 600
    previous_tail_tokens: int = 900
    generation_and_revision_buffer_tokens: int = 10500
    previous_tail_min_chars: int = 600
    previous_tail_max_chars: int = 800
    chapter_soft_min_chars: int = 2000
    chapter_soft_max_chars: int = 4200
    chapter_hard_max_chars: int = 4500


@dataclass(frozen=True)
class QualityThresholdConfig:
    """Quantitative thresholds for Gate A, Gate B, and end-of-volume/book acceptance."""

    single_chapter_pass_score: float = 80.0
    single_dimension_min_score: float = 65.0
    volume_acceptance_mean_score: float = 85.0
    dynamic_anchor_min_score: float = 85.0
    foreshadow_recovery_min_rate: float = 0.95
    pacing_stability_min_rate: float = 0.95
    max_gate_a_retries: int = 3
    max_gate_b_revisions: int = 1
    reference_similarity_threshold: float = 0.72
    reference_top_k: int = 3


@dataclass(frozen=True)
class RuntimeConfig:
    """Global runtime paths, model endpoints, and hardware guard settings."""

    workspace_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("AUTONOVEL_WORKSPACE", Path.cwd())
        ).resolve()
    )
    data_dir: Path = field(init=False)
    projects_dir: Path = field(init=False)
    projects_meta_path: Path = field(init=False)
    project_db_path: Path = field(init=False)
    reference_db_path: Path = field(init=False)
    snapshot_dir: Path = field(init=False)
    export_dir: Path = field(init=False)
    llama_server_url: str = field(
        default_factory=lambda: os.environ.get(
            "AUTONOVEL_LLAMA_URL", "http://127.0.0.1:8080"
        )
    )
    llama_bin_path: str = field(
        default_factory=lambda: os.environ.get("AUTONOVEL_LLAMA_BIN", "llama-server")
    )
    llama_model_path: str = field(
        default_factory=lambda: os.environ.get(
            "AUTONOVEL_MODEL_PATH", "models/Qwen2.5-7B-Instruct-Q5_K_M.gguf"
        )
    )
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    llm_timeout_sec: float = 600.0
    model_idle_timeout_sec: float = field(
        default_factory=lambda: env_float("AUTONOVEL_MODEL_IDLE_TIMEOUT_SEC", 600.0)
    )
    gpu_temp_cooldown_threshold_c: int = 83
    gpu_temp_cooldown_sleep_sec: float = 12.0
    deep_organize_interval_chapters: int = 10
    periodic_snapshot_interval_chapters: int = 10
    max_periodic_snapshots: int = 5
    logs_dir: Path = field(default_factory=Path)
    log_file_path: Path = field(default_factory=Path)
    max_log_bytes: int = 10 * 1024 * 1024  # 10 MB per file segment
    log_backup_count: int = 5  # Retain 5 rotated segments
    context_budget: ContextBudgetConfig = field(default_factory=ContextBudgetConfig)
    quality_thresholds: QualityThresholdConfig = field(
        default_factory=QualityThresholdConfig
    )

    def __post_init__(self) -> None:
        """Initialize derived filesystem paths and ensure directories exist."""
        data_dir = self.workspace_dir / "data"
        projects_dir = data_dir / "projects"
        projects_meta_path = data_dir / "projects.json"
        snap_dir = data_dir / "snapshots"
        exp_dir = data_dir / "exports"
        logs_dir = data_dir / "logs"
        log_file_path = logs_dir / "autonovel.log"

        object.__setattr__(self, "data_dir", data_dir)
        object.__setattr__(self, "projects_dir", projects_dir)
        object.__setattr__(self, "projects_meta_path", projects_meta_path)
        object.__setattr__(self, "project_db_path", data_dir / "novel_project.db")
        object.__setattr__(self, "reference_db_path", data_dir / "reference_library.db")
        object.__setattr__(self, "snapshot_dir", snap_dir)
        object.__setattr__(self, "export_dir", exp_dir)
        object.__setattr__(self, "logs_dir", logs_dir)
        object.__setattr__(self, "log_file_path", log_file_path)

        for directory in (data_dir, projects_dir, snap_dir, exp_dir, logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        setup_logging(self)


_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(config: RuntimeConfig | None = None) -> None:
    """Configure root logger with unified console and rotating file handler."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    # Attach Console StreamHandler if not present
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in root_logger.handlers
    )
    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

    # Check if a RotatingFileHandler is already active
    current_file_handler = next(
        (h for h in root_logger.handlers if isinstance(h, RotatingFileHandler)),
        None,
    )
    if config is None and current_file_handler is not None:
        return

    # Resolve target rotating log file path
    if config and config.log_file_path and str(config.log_file_path) != ".":
        log_file = config.log_file_path
        max_bytes = config.max_log_bytes
        backup_count = config.log_backup_count
    else:
        log_file = Path.cwd() / "data" / "logs" / "autonovel.log"
        max_bytes = 10 * 1024 * 1024
        backup_count = 5

    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        resolved_path = str(log_file.resolve())
        existing_handler = next(
            (
                h
                for h in root_logger.handlers
                if isinstance(h, RotatingFileHandler)
                and getattr(h, "baseFilename", "") == resolved_path
            ),
            None,
        )
        if not existing_handler:
            # Safely close and remove prior file handlers
            for h in list(root_logger.handlers):
                if isinstance(h, RotatingFileHandler):
                    root_logger.removeHandler(h)
                    h.close()

            file_handler = RotatingFileHandler(
                filename=str(log_file),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
                delay=True,
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)
    except Exception as exc:
        root_logger.warning("Could not initialize RotatingFileHandler: %s", exc)


def close_logging_handlers() -> None:
    """Flush and close all file handlers on application shutdown."""
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        if isinstance(h, RotatingFileHandler):
            root_logger.removeHandler(h)
            h.close()


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a structured logger using console and rotating file output."""
    setup_logging()
    return logging.getLogger(name)
