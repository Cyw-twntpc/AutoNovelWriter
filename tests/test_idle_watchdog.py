"""Tests for the 10-minute frontend idle detection, activity heartbeat, and automatic model unload watchdog."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from autonovel.config import RuntimeConfig
from autonovel.engine.daemon import NovelWritingDaemon
from autonovel.web.app import create_app


@pytest.mark.asyncio
async def test_idle_watchdog_unloads_when_frontend_idle_and_model_not_in_use(tmp_path: Path) -> None:
    """Verify watchdog automatically stops llama-server when frontend idle exceeds threshold and model is free."""
    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        model_idle_timeout_sec=0.2,  # 200ms threshold for fast testing
    )
    daemon = NovelWritingDaemon(cfg)
    daemon.llama_manager.is_healthy = MagicMock(return_value=True)

    def _mock_stop() -> None:
        daemon.llama_manager.is_healthy.return_value = False

    daemon.llama_manager.stop_server = MagicMock(side_effect=_mock_stop)

    # Daemon has no user activity and model is not in use
    assert daemon.is_model_in_use() is False

    # Start idle watchdog with rapid 0.05s check interval
    task = daemon.start_idle_watchdog(check_interval_sec=0.05)

    # Wait past the 0.2s threshold
    await asyncio.sleep(0.35)

    # Verify stop_server was invoked to release GPU VRAM
    daemon.llama_manager.stop_server.assert_called_once()

    daemon.stop_idle_watchdog()
    await asyncio.sleep(0.01)
    assert task.done()


@pytest.mark.asyncio
async def test_idle_watchdog_does_not_unload_when_frontend_is_active(tmp_path: Path) -> None:
    """Verify frontend user activity pings prevent the idle watchdog from unloading the model."""
    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        model_idle_timeout_sec=0.3,
    )
    daemon = NovelWritingDaemon(cfg)
    daemon.llama_manager.is_healthy = MagicMock(return_value=True)
    daemon.llama_manager.stop_server = MagicMock()

    task = daemon.start_idle_watchdog(check_interval_sec=0.05)

    # Simulate ongoing frontend user activity every 80ms
    for _ in range(4):
        await asyncio.sleep(0.08)
        daemon.record_user_activity()

    # Total ~320ms passed, but idle duration never exceeded 80ms (< 300ms threshold)
    daemon.llama_manager.stop_server.assert_not_called()

    daemon.stop_idle_watchdog()
    await asyncio.sleep(0.01)
    assert task.done()


@pytest.mark.asyncio
async def test_idle_watchdog_does_not_unload_when_model_is_in_use(tmp_path: Path) -> None:
    """Verify model is NOT unloaded even if frontend is idle, provided model inference is active."""
    cfg = RuntimeConfig(
        workspace_dir=tmp_path,
        model_idle_timeout_sec=0.15,
    )
    daemon = NovelWritingDaemon(cfg)
    daemon.llama_manager.is_healthy = MagicMock(return_value=True)
    daemon.llama_manager.stop_server = MagicMock()

    # Mark LLM client as busy (e.g. streaming or generating novel chapter)
    daemon.llm_client._active_requests = 1
    assert daemon.is_model_in_use() is True

    task = daemon.start_idle_watchdog(check_interval_sec=0.05)

    # Wait 0.3s (idle exceeded 0.15s threshold)
    await asyncio.sleep(0.3)

    # Must NOT unload because model is currently in use!
    daemon.llama_manager.stop_server.assert_not_called()

    daemon.stop_idle_watchdog()
    await asyncio.sleep(0.01)
    assert task.done()


@pytest.mark.asyncio
async def test_heartbeat_api_and_activity_middleware(tmp_path: Path) -> None:
    """Verify /api/system/heartbeat and interactive routes update daemon's last user activity timestamp."""
    cfg = RuntimeConfig(workspace_dir=tmp_path)
    app = create_app(cfg)
    daemon = app.state.daemon

    # Artificially set activity timestamp to 500s ago
    daemon._last_user_activity_ts = time.time() - 500.0
    assert daemon.get_idle_seconds() >= 490.0

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Send heartbeat
        r_hb = await client.post("/api/system/heartbeat")
        assert r_hb.status_code == 200
        hb_data = r_hb.json()
        assert hb_data["status"] == "ok"
        assert hb_data["idle_seconds"] < 2.0
        assert daemon.get_idle_seconds() < 2.0

        # 2. Artificially set idle again and test interactive middleware tracking
        daemon._last_user_activity_ts = time.time() - 300.0
        assert daemon.get_idle_seconds() >= 290.0

        # Accessing interactive endpoint /api/wizard/genres updates last activity
        r_gen = await client.get("/api/wizard/genres")
        assert r_gen.status_code == 200
        assert daemon.get_idle_seconds() < 2.0

        # Background poll /api/warroom/dashboard does NOT reset the user activity timestamp
        daemon._last_user_activity_ts = time.time() - 200.0
        r_dash = await client.get("/api/warroom/dashboard")
        assert r_dash.status_code == 200
        assert daemon.get_idle_seconds() >= 195.0
