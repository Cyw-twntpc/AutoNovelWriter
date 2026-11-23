"""Tests for persistent rotating log file system, rotation mechanics, and API inspection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from autonovel.config import RuntimeConfig, close_logging_handlers, get_logger, setup_logging
from autonovel.web.app import create_app


def test_persistent_rotating_logger_creation_and_write(tmp_path: Path) -> None:
    """Verify logger writes physical log entries to autonovel.log with UTF-8 encoding."""
    cfg = RuntimeConfig(workspace_dir=tmp_path)
    assert cfg.logs_dir.exists()
    assert cfg.log_file_path == cfg.logs_dir / "autonovel.log"

    test_logger = get_logger("test_module")
    test_logger.info("This is a persistent test log entry: 萬界星穹傳")

    # Flush handlers
    for h in logging.getLogger().handlers:
        h.flush()

    assert cfg.log_file_path.exists()
    content = cfg.log_file_path.read_text(encoding="utf-8")
    assert "This is a persistent test log entry: 萬界星穹傳" in content
    assert "[INFO]" in content

    close_logging_handlers()


def test_rotating_file_mechanism_on_size_limit(tmp_path: Path) -> None:
    """Verify logs rotate into backup segments when exceeding max_log_bytes."""
    # Set small segment size to test rotation
    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        max_log_bytes=512,  # 512 bytes triggers quick rotation
        log_backup_count=3,
    )
    setup_logging(cfg)

    rotate_logger = get_logger("rotate_test")
    # Write enough lines to exceed 512 bytes multiple times
    for i in range(50):
        rotate_logger.info("Rotation log payload line %d: %s", i, "X" * 30)

    for h in logging.getLogger().handlers:
        h.flush()

    # Verify primary log and at least one rotated backup exist
    assert cfg.log_file_path.exists()
    rotated_backup = cfg.logs_dir / "autonovel.log.1"
    assert rotated_backup.exists()

    close_logging_handlers()


@pytest.mark.asyncio
async def test_system_logs_api_endpoint(tmp_path: Path, mock_llm_client: Any) -> None:
    """Verify /api/system/logs returns recent tail entries from the rotating log file."""
    cfg = RuntimeConfig(workspace_dir=tmp_path)
    app = create_app(cfg)
    app.state.daemon.llm_client = mock_llm_client

    logger = get_logger("api_log_test")
    logger.info("Entry Alpha for API log verification")
    logger.info("Entry Beta for API log verification")

    for h in logging.getLogger().handlers:
        h.flush()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/api/system/logs?lines=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "logs" in data
        assert data["total_lines"] >= 2
        joined_logs = "\n".join(data["logs"])
        assert "Entry Alpha for API log verification" in joined_logs
        assert "Entry Beta for API log verification" in joined_logs

    close_logging_handlers()
