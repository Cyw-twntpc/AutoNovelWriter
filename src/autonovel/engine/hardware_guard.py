"""Windows anti-sleep execution state lock and NVML GPU telemetry/thermal guard."""

from __future__ import annotations

import asyncio
import ctypes
import subprocess
import sys
import time
from typing import Any

import psutil

from autonovel.config import RuntimeConfig, get_logger

logger = get_logger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

_NVML_INIT_ATTEMPTED = False
_NVML_AVAILABLE = False


class HardwareGuard:
    """Manages Windows anti-sleep state and RTX 4060 VRAM/thermal telemetry."""

    def __init__(self, config: RuntimeConfig) -> None:
        """Initialize hardware guard with thermal thresholds and telemetry cache."""
        self.config = config
        self._sleep_locked = False
        self._telemetry_cache: dict[str, Any] | None = None
        self._telemetry_cache_ts: float = 0.0
        self._telemetry_ttl_sec: float = 2.5

    def acquire_anti_sleep_lock(self) -> bool:
        """Prevent Windows OS from sleeping during background serialization."""
        if sys.platform != "win32":
            return False
        try:
            res = ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            )
            self._sleep_locked = bool(res)
            return self._sleep_locked
        except Exception as exc:
            logger.warning("Failed to set Windows execution state: %s", exc)
            return False

    def release_anti_sleep_lock(self) -> None:
        """Restore normal Windows power management state when paused or stopped."""
        if sys.platform == "win32" and self._sleep_locked:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            except Exception as exc:
                logger.warning("Failed to release Windows execution state: %s", exc)
        self._sleep_locked = False

    def get_hardware_telemetry(self) -> dict[str, Any]:
        """Query GPU VRAM, GPU temperature, and system RAM utilization with 2.5s caching."""
        now = time.monotonic()
        if (
            self._telemetry_cache is not None
            and (now - self._telemetry_cache_ts) < self._telemetry_ttl_sec
        ):
            cached = dict(self._telemetry_cache)
            cached["sleep_lock_active"] = self._sleep_locked
            return cached

        gpu_info = self._probe_nvidia_gpu()
        mem = psutil.virtual_memory()
        vram_pct = 0.0
        if gpu_info["vram_total_mb"] > 0:
            vram_pct = round(
                (gpu_info["vram_used_mb"] / gpu_info["vram_total_mb"]) * 100.0, 1
            )
        data = {
            "gpu_name": gpu_info.get("gpu_name", ""),
            "vram_used_mb": gpu_info["vram_used_mb"],
            "vram_total_mb": gpu_info["vram_total_mb"],
            "vram_percent": vram_pct,
            "gpu_temp_c": gpu_info["gpu_temp_c"],
            "ram_percent": mem.percent,
            "sleep_lock_active": self._sleep_locked,
        }
        self._telemetry_cache = data
        self._telemetry_cache_ts = now
        return data

    @classmethod
    def _probe_nvidia_gpu(cls) -> dict[str, Any]:
        """Query NVIDIA GPU stats via native NVML (pynvml), nvidia-smi CLI, or safe zero-state defaults."""
        global _NVML_INIT_ATTEMPTED, _NVML_AVAILABLE
        if not _NVML_INIT_ATTEMPTED:
            _NVML_INIT_ATTEMPTED = True
            try:
                import pynvml

                pynvml.nvmlInit()
                _NVML_AVAILABLE = True
            except Exception:
                _NVML_AVAILABLE = False

        if _NVML_AVAILABLE:
            try:
                import pynvml

                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                raw_name = pynvml.nvmlDeviceGetName(handle)
                gpu_name = (
                    raw_name.decode("utf-8", errors="ignore")
                    if isinstance(raw_name, bytes)
                    else str(raw_name)
                )
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp_c = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                return {
                    "gpu_name": gpu_name.strip(),
                    "vram_used_mb": int(mem_info.used // (1024 * 1024)),
                    "vram_total_mb": int(mem_info.total // (1024 * 1024)),
                    "gpu_temp_c": int(temp_c),
                }
            except Exception:
                pass
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=gpu_name,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                first_line = proc.stdout.strip().splitlines()[0]
                parts = [x.strip() for x in first_line.split(",")]
                if len(parts) >= 4:
                    return {
                        "gpu_name": parts[0],
                        "vram_used_mb": int(float(parts[1])),
                        "vram_total_mb": int(float(parts[2])),
                        "gpu_temp_c": int(float(parts[3])),
                    }
        except Exception:
            pass
        return {
            "gpu_name": "",
            "vram_used_mb": 0,
            "vram_total_mb": 0,
            "gpu_temp_c": 0,
        }

    async def check_and_cooldown_if_hot(self, resume_threshold_c: int = 76) -> bool:
        """Dynamic hysteresis thermal cooldown: if GPU temp >= 83°C, poll every 2s until temp <= 76°C.

        Returns True only when the GPU actually reached ``resume_threshold_c``; a timeout
        with the GPU still hot returns False.
        """
        self._telemetry_cache_ts = 0.0  # Force fresh probe
        telemetry = self.get_hardware_telemetry()
        temp_c = int(telemetry["gpu_temp_c"])
        if temp_c >= self.config.gpu_temp_cooldown_threshold_c:
            logger.warning(
                "GPU temperature %d°C reached thermal limit (%d°C). Halting inference and cooling down to %d°C...",
                temp_c,
                self.config.gpu_temp_cooldown_threshold_c,
                resume_threshold_c,
            )
            total_slept = 0.0
            max_wait_sec = 60.0
            while temp_c > resume_threshold_c and total_slept < max_wait_sec:
                await asyncio.sleep(2.0)
                total_slept += 2.0
                self._telemetry_cache_ts = 0.0  # Force fresh probe
                probe = self.get_hardware_telemetry()
                temp_c = int(probe["gpu_temp_c"])
                logger.debug(
                    "GPU cooling in progress: current %d°C, target <= %d°C (waited %.1fs)",
                    temp_c,
                    resume_threshold_c,
                    total_slept,
                )
            if temp_c > resume_threshold_c:
                # Timing out is not a recovery: report it honestly so callers can
                # decide to throttle instead of assuming the GPU has cooled.
                logger.warning(
                    "GPU still at %d°C after %.1fs cooldown (target <= %d°C); cooling did not complete.",
                    temp_c,
                    total_slept,
                    resume_threshold_c,
                )
                return False
            logger.info(
                "GPU temperature recovered to %d°C after %.1fs. Resuming inference pipeline.",
                temp_c,
                total_slept,
            )
            return True
        return False
