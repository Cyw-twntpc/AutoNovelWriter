"""Subprocess lifecycle and watchdog restart manager for llama-server (llama.cpp)."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from autonovel.config import RuntimeConfig, env_int, get_logger

logger = get_logger(__name__)


def resolve_llama_server_binary(cfg: RuntimeConfig) -> Path | None:
    """Resolve llama-server executable from PATH, workspace, or standard bin/tools directories."""
    # 1. PATH lookup
    path_hit = shutil.which(cfg.llama_bin_path)
    if path_hit:
        return Path(path_hit)

    # 2. Direct path
    direct = Path(cfg.llama_bin_path)
    if direct.is_file():
        return direct

    # 3. Workspace / local directories
    candidates = [
        cfg.workspace_dir / cfg.llama_bin_path,
        cfg.workspace_dir / f"{cfg.llama_bin_path}.exe",
        cfg.workspace_dir / "bin" / "llama-server.exe",
        cfg.workspace_dir / "tools" / "llama-server" / "llama-server.exe",
        cfg.workspace_dir / "llama-server.exe",
        Path("bin/llama-server.exe"),
        Path("tools/llama-server/llama-server.exe"),
        Path("llama-server.exe"),
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


class LlamaServerManager:
    """Controls native llama-server process with 16k Q8_0 KV cache and FlashAttention."""

    def __init__(self, config: RuntimeConfig) -> None:
        """Initialize process handle, health cache, and configuration."""
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._last_probe_ts: float = 0.0
        self._last_healthy: bool = False
        self._manually_unloaded: bool = False

    def build_command_args(self) -> list[str]:
        """Construct CLI arguments for 8GB VRAM Qwen2.5-7B-Instruct (Q5_K_M) execution."""
        parsed = urlparse(self.config.llama_server_url)
        host = parsed.hostname or "127.0.0.1"
        port = str(parsed.port or 8080)
        bin_path = resolve_llama_server_binary(self.config) or self.config.llama_bin_path
        model_path = Path(self.config.llama_model_path)
        if not model_path.exists():
            candidate = self.config.workspace_dir / model_path
            if candidate.exists():
                model_path = candidate
        return [
            str(bin_path),
            "-m",
            str(model_path),
            "--host",
            host,
            "--port",
            port,
            "-c",
            str(self.config.context_budget.total_context_tokens),
            "-ngl",
            "99",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "--flash-attn",
            "auto",
            "-np",
            str(env_int("AUTONOVEL_LLAMA_NP", 1)),
        ]

    def is_healthy(self) -> bool:
        """Probe llama-server /health endpoint with brief caching to avoid TCP retry stalls."""
        if self._manually_unloaded and self._process is None:
            self._last_healthy = False
            return False
        now = time.monotonic()
        if now - self._last_probe_ts < 5.0:
            return self._last_healthy
        self._last_probe_ts = now
        model_path = Path(self.config.llama_model_path)
        if not model_path.exists() and not (self.config.workspace_dir / model_path).exists() and self._process is None:
            self._last_healthy = False
            return False
        try:
            resp = httpx.get(f"{self.config.llama_server_url}/health", timeout=0.35)
            self._last_healthy = resp.status_code == 200
        except httpx.HTTPError:
            self._last_healthy = False
        return self._last_healthy

    def ensure_running(self, clear_manual_unload: bool = True) -> bool:
        """Start llama-server if not already healthy and binary/model are available."""
        if clear_manual_unload:
            self._manually_unloaded = False
        elif self._manually_unloaded:
            return False
        self._last_probe_ts = 0.0
        if self.is_healthy():
            return True
        bin_resolved = resolve_llama_server_binary(self.config)
        model_path = Path(self.config.llama_model_path)
        if not model_path.exists():
            candidate = self.config.workspace_dir / model_path
            if candidate.exists():
                model_path = candidate
        model_exists = model_path.exists()
        if not bin_resolved or not model_exists:
            logger.warning(
                "llama-server binary or GGUF model not found at '%s' / '%s'.",
                bin_resolved or self.config.llama_bin_path,
                model_path,
            )
            return False
        cmd = self.build_command_args()
        cmd[0] = str(bin_resolved)
        logger.info("Starting llama-server process: %s", " ".join(cmd))
        llama_log_file = None
        try:
            llama_log_path = self.config.logs_dir / "llama_server.log"
            llama_log_file = open(llama_log_path, "a", encoding="utf-8")
        except Exception:
            llama_log_file = subprocess.DEVNULL

        self._process = subprocess.Popen(
            cmd,
            stdout=llama_log_file,
            stderr=subprocess.STDOUT,
        )
        for _ in range(60):
            if self._process.poll() is not None:
                logger.error(
                    "llama-server process exited prematurely with code %s.",
                    self._process.returncode,
                )
                self._last_healthy = False
                return False
            self._last_probe_ts = 0.0
            if self.is_healthy():
                return True
            time.sleep(0.5)
        return self.is_healthy()

    def restart_server(self) -> bool:
        """Terminate unresponsive llama-server process and relaunch cleanly."""
        if self._manually_unloaded:
            logger.info("Skipping watchdog server restart because model was manually unloaded by user.")
            return False
        logger.warning("Watchdog triggered: restarting llama-server process.")
        self.stop_server(manual=False)
        self._last_probe_ts = 0.0
        return self.ensure_running(clear_manual_unload=False)

    async def ensure_running_async(self, clear_manual_unload: bool = True) -> bool:
        """Async wrapper: the start-up poll sleeps for up to 30s and must not block the loop."""
        return await asyncio.to_thread(self.ensure_running, clear_manual_unload)

    async def restart_server_async(self) -> bool:
        """Async wrapper for the blocking stop/relaunch sequence."""
        return await asyncio.to_thread(self.restart_server)

    def _is_port_open(self, host: str, port: int) -> bool:
        """Check whether a local TCP port is actively accepting connections."""
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            return False

    def _is_llama_process(self, pid: int) -> bool:
        """Return True only when the PID really is a llama-server image.

        Killing whatever happens to listen on port 8080 would take down unrelated
        local services, so the image name is verified before ``taskkill``.
        """
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("Unable to inspect PID %s before kill: %s", pid, exc)
            return False
        output = (result.stdout or "").strip()
        if not output.startswith('"'):
            return False
        image = output.split('","', 1)[0].strip('"').lower()
        expected = Path(self.config.llama_bin_path).stem.lower()
        return image.startswith("llama-server") or image == f"{expected}.exe" or image == expected

    def _kill_orphan_llama_on_port(self, host: str, port: int) -> None:
        """Find and terminate any orphan llama-server process bound to the configured port."""
        if not self._is_port_open(host, port):
            return
        if os.name == "nt":
            try:
                netstat = subprocess.run(
                    ["netstat", "-ano", "-p", "TCP"],
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                )
                pids: set[int] = set()
                port_pattern = re.compile(rf":{port}\s+.*LISTENING\s+(\d+)", re.IGNORECASE)
                for line in (netstat.stdout or "").splitlines():
                    m = port_pattern.search(line)
                    if m:
                        pid = int(m.group(1))
                        if pid > 0 and pid != os.getpid() and self._is_llama_process(pid):
                            pids.add(pid)
                for pid in pids:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                        timeout=3.0,
                    )
            except Exception as exc:
                logger.warning("Failed to terminate orphan llama-server on port %s: %s", port, exc)

    def stop_server(self, manual: bool = False) -> None:
        """Stop managed or orphan llama-server subprocess and release GPU VRAM."""
        if manual:
            self._manually_unloaded = True
        if self._process and self._process.poll() is None:
            pid = self._process.pid
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                try:
                    self._process.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    pass
            if os.name == "nt" and pid:
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                        timeout=2.0,
                    )
                except Exception:
                    pass
        self._process = None

        parsed = urlparse(self.config.llama_server_url)
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or 8080)
        self._kill_orphan_llama_on_port(host, port)

        self._last_healthy = False
        self._last_probe_ts = 0.0
