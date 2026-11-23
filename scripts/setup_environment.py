"""Environment check and automated model downloader for 8GB VRAM deployment and Web UI."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
from collections.abc import Callable, Generator
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
for path_str in (str(SRC_DIR), str(ROOT_DIR)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from autonovel.config import RuntimeConfig, get_logger  # noqa: E402

logger = get_logger("autonovel.setup")


_LLAMA_VERSION_CACHE: dict[str, str] = {}


def _get_llama_version(bin_path: Path | None) -> str:
    """Extract version identifier from llama-server binary."""
    if not bin_path or not bin_path.exists():
        return ""
    key = str(bin_path)
    if key in _LLAMA_VERSION_CACHE:
        return _LLAMA_VERSION_CACHE[key]
    try:
        proc = subprocess.run([str(bin_path), "--version"], capture_output=True, text=True, timeout=2)
        raw = (proc.stdout or proc.stderr).strip()
        if raw:
            line = raw.splitlines()[0]
            ver_str = line if "llama" in line.lower() else f"llama.cpp ({line})"
        else:
            ver_str = "llama.cpp Windows Binary"
    except Exception:
        ver_str = "llama.cpp Windows Binary"
    _LLAMA_VERSION_CACHE[key] = ver_str
    return ver_str


def inspect_environment(
    config: RuntimeConfig | None = None, daemon: Any = None
) -> dict[str, Any]:
    """Verify local GGUF model, llama-server binary, and CPU ONNX embedding model status."""
    cfg = config or (daemon.config if daemon else RuntimeConfig())
    bge_dir = cfg.workspace_dir / "models" / "bge-small-zh-v1.5"
    if not bge_dir.exists() and Path("models/bge-small-zh-v1.5").exists():
        bge_dir = Path("models/bge-small-zh-v1.5")
    bge_ready = (bge_dir / "tokenizer.json").exists() and (
        (bge_dir / "model.onnx").exists() or (bge_dir / "onnx" / "model.onnx").exists()
    )
    gguf_path = Path(cfg.llama_model_path)
    if not gguf_path.is_absolute() and not gguf_path.exists():
        candidate = cfg.workspace_dir / gguf_path
        if candidate.exists():
            gguf_path = candidate
    if not gguf_path.exists():
        # Fallback to any gguf in models directory
        models_dir = cfg.workspace_dir / "models" if hasattr(cfg, "workspace_dir") else Path("models")
        if models_dir.exists():
            found_ggufs = sorted(list(models_dir.glob("*.gguf")), key=lambda p: p.stat().st_size, reverse=True)
            if found_ggufs:
                gguf_path = found_ggufs[0]
    gguf_ready = gguf_path.exists()
    gguf_size_mb = round(gguf_path.stat().st_size / (1024 * 1024), 2) if gguf_ready else 0.0
    from autonovel.llm.llama_manager import resolve_llama_server_binary

    resolved_llama = resolve_llama_server_binary(cfg)
    llama_found = resolved_llama is not None
    llama_bin_display = str(resolved_llama) if resolved_llama else cfg.llama_bin_path
    embed_mode = "NGRAM_FALLBACK"
    llama_healthy = False
    hw_info: dict[str, Any] = {}
    if daemon is not None:
        llama_healthy = bool(daemon.llama_manager.is_healthy())
        hw_info = daemon.hardware_guard.get_hardware_telemetry()
        if hasattr(daemon.embedder, "_lazy_init_model"):
            daemon.embedder._lazy_init_model()
        if getattr(daemon.embedder, "_onnx_session", None) is not None:
            embed_mode = "ONNX_CPU"
        elif getattr(daemon.embedder, "_fastembed_model", None) is not None:
            embed_mode = "FASTEMBED_CPU"

    llama_ver_str = _get_llama_version(resolved_llama) if llama_found else ""
    if not llama_ver_str and llama_healthy:
        llama_ver_str = "llama.cpp (已啟用本地推論)"

    gguf_filename = gguf_path.name if gguf_ready else ""
    gguf_ver_str = f"{gguf_filename} ({gguf_size_mb:,.1f} MB)" if gguf_ready else ""
    bge_ver_str = f"BGE-Small-ZH v1.5 ({embed_mode})" if bge_ready else ""

    status = {
        "gguf_path": str(gguf_path),
        "gguf_filename": gguf_filename,
        "gguf_ready": gguf_ready,
        "gguf_size_mb": gguf_size_mb,
        "gguf_version": gguf_ver_str,
        "bge_onnx_dir": str(bge_dir),
        "bge_onnx_ready": bge_ready,
        "bge_version": bge_ver_str,
        "embedder_mode": embed_mode,
        "llama_server_bin": llama_bin_display,
        "llama_server_found": llama_found,
        "llama_server_healthy": llama_healthy,
        "llama_version": llama_ver_str,
        "hardware": hw_info,
    }
    logger.info("Environment inspection result: %s", status)
    return status


@contextlib.contextmanager
def hf_download_progress_hook(
    on_progress: Callable[[float, str], None] | None,
    description: str = "模型",
) -> Generator[None, None, None]:
    """Context manager to intercept huggingface_hub download chunks and broadcast live progress to Web UI."""
    if on_progress is None:
        yield
        return

    import huggingface_hub.file_download as fd

    orig_tqdm = fd.tqdm
    sink = io.StringIO()

    class _ProgressInterceptor(orig_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = False
            kwargs["file"] = sink
            super().__init__(*args, **kwargs)
            self._downloaded = kwargs.get("initial", 0)

        def update(self, n: int = 1) -> None:
            super().update(n)
            self._downloaded += n
            try:
                if self.total and self.total > 0:
                    pct = round(self._downloaded / self.total * 100, 1)
                    mb_down = round(self._downloaded / (1024 * 1024), 1)
                    mb_total = round(self.total / (1024 * 1024), 1)
                    msg = f"正在下載 {description}：{pct}% ({mb_down} / {mb_total} MB)"
                    on_progress(pct, msg)
                elif self._downloaded > 0:
                    mb_down = round(self._downloaded / (1024 * 1024), 1)
                    msg = f"正在下載 {description}：已下載 {mb_down} MB"
                    on_progress(0.0, msg)
            except Exception:
                pass

    fd.tqdm = _ProgressInterceptor
    try:
        yield
    finally:
        fd.tqdm = orig_tqdm


def download_bge_small_onnx(
    target_dir: Path | None = None,
    repo_id: str = "Xenova/bge-small-zh-v1.5",
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Download BGE small Chinese tokenizer.json and ONNX weights for CPU inference with live progress."""
    from huggingface_hub import hf_hub_download

    # BAAI/bge-small-zh-v1.5 only contains PyTorch/safetensors weights.
    # Xenova/bge-small-zh-v1.5 hosts the official ONNX model file.
    actual_repo = (
        "Xenova/bge-small-zh-v1.5"
        if (not repo_id or repo_id == "BAAI/bge-small-zh-v1.5")
        else repo_id
    )

    dest = target_dir or Path("models/bge-small-zh-v1.5")
    dest.mkdir(parents=True, exist_ok=True)
    with hf_download_progress_hook(on_progress, description="語意嵌入模型"):
        tok_path = hf_hub_download(
            repo_id=actual_repo,
            filename="tokenizer.json",
            local_dir=str(dest),
        )
        try:
            onnx_path = hf_hub_download(
                repo_id=actual_repo,
                filename="onnx/model.onnx",
                local_dir=str(dest),
            )
        except Exception:
            onnx_path = hf_hub_download(
                repo_id=actual_repo,
                filename="model.onnx",
                local_dir=str(dest),
            )
    logger.info("Downloaded BGE ONNX model to %s", dest)
    return {"tokenizer_path": str(tok_path), "onnx_path": str(onnx_path), "target_dir": str(dest)}


def download_gguf_model(
    target_dir: Path | None = None,
    repo_id: str = "bartowski/Qwen2.5-7B-Instruct-GGUF",
    filename: str = "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
    max_retries: int = 5,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Download GGUF model file from Hugging Face Hub into the local models directory with auto-resume and live progress."""
    import time
    from huggingface_hub import hf_hub_download

    dest = target_dir or Path("models")
    dest.mkdir(parents=True, exist_ok=True)

    with hf_download_progress_hook(on_progress, description="核心大語言模型 (LLM)"):
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Starting/resuming GGUF model download %s/%s (attempt %d/%d)...",
                    repo_id,
                    filename,
                    attempt,
                    max_retries,
                )
                gguf_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(dest),
                )
                logger.info("Downloaded GGUF model %s/%s to %s", repo_id, filename, gguf_file)
                return {"gguf_path": str(gguf_file), "repo_id": repo_id, "filename": filename}
            except Exception as exc:
                err_str = str(exc)
                if "404" in err_str or "EntryNotFoundError" in err_str or attempt >= max_retries:
                    raise
                logger.warning(
                    "GGUF download interrupted on attempt %d/%d (%s). Auto-resuming in 3s...",
                    attempt,
                    max_retries,
                    exc,
                )
                time.sleep(3.0)
    raise RuntimeError(f"GGUF 下載在嘗試 {max_retries} 次後依然失敗。")


def format_friendly_error_message(exc: Exception, prefix: str = "下載失敗") -> str:
    """Format a clean, user-friendly error summary without exposing technical stack traces."""
    err_str = str(exc)
    err_lower = err_str.lower()
    if "404" in err_str or "entry not found" in err_lower or "repository not found" in err_lower:
        return f"{prefix}：遠端倉庫未找到指定的模型檔案，請檢查模型名稱或版本設定。"
    if "401" in err_str or "403" in err_str or "unauthorized" in err_lower or "gated repo" in err_lower:
        return f"{prefix}：存取權限不足，該模型需要 Hugging Face 存取權杖 (Access Token)。"
    if (
        "connection" in err_lower
        or "timeout" in err_lower
        or "timed out" in err_lower
        or "network" in err_lower
        or "incompleteread" in err_lower
        or "chunkedencodingerror" in err_lower
    ):
        return f"{prefix}：網路傳輸中斷或逾時（已自動保留斷點暫存進度，可直接點擊重新下載續傳）。"
    if "no space left" in err_lower or "disk full" in err_lower:
        return f"{prefix}：本地硬碟空間不足，無法儲存模型檔案。"
    return f"{prefix}：遭遇未預期的錯誤，詳細技術資訊已記錄於伺服器日誌。"


class EnvironmentDownloadManager:
    """Manages non-blocking model downloads and broadcasts real-time progress to the Web UI."""

    def __init__(self) -> None:
        """Initialize download state tracker."""
        self.state: dict[str, Any] = {
            "active": False,
            "target": "",
            "message": "待命",
            "percent": 0.0,
            "error": "",
            "last_result": {},
        }
        self._task: asyncio.Task[dict[str, Any]] | None = None

    async def execute_download(
        self,
        daemon: Any,
        target: str,
        repo_id: str = "",
        filename: str = "",
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run BGE ONNX or GGUF download in a worker thread and hot-reload daemon components."""
        models_root = daemon.config.workspace_dir / "models"
        self.state = {
            "active": True,
            "target": target,
            "message": f"正在從 Hugging Face 下載 {target} 模型權重...",
            "percent": 15.0,
            "error": "",
            "last_result": {},
        }
        if on_event:
            on_event(dict(self.state))
        try:
            if target == "bge":
                bge_dir = models_root / "bge-small-zh-v1.5"

                def _bge_progress(pct: float, msg: str) -> None:
                    self.state.update({"target": "bge", "percent": round(pct, 1), "message": msg})
                    if on_event:
                        on_event(dict(self.state))

                res = await asyncio.to_thread(
                    download_bge_small_onnx,
                    bge_dir,
                    repo_id or "Xenova/bge-small-zh-v1.5",
                    _bge_progress,
                )
                daemon.embedder.load_onnx_dir(bge_dir)
            elif target == "gguf":
                def _gguf_progress(pct: float, msg: str) -> None:
                    self.state.update({"target": "gguf", "percent": round(pct, 1), "message": msg})
                    if on_event:
                        on_event(dict(self.state))

                res = await asyncio.to_thread(
                    download_gguf_model,
                    models_root,
                    repo_id or "bartowski/Qwen2.5-7B-Instruct-GGUF",
                    filename or "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
                    5,
                    _gguf_progress,
                )
            elif target == "llama":
                self.state.update(
                    {
                        "target": "llama",
                        "percent": 0.0,
                        "message": "正在自動偵測硬體並下載安裝本地推論引擎...",
                    }
                )
                if on_event:
                    on_event(dict(self.state))
                from scripts.download_llama_server import download_and_extract_llama_server

                def _llama_progress(pct: float, msg: str) -> None:
                    self.state.update({"target": "llama", "percent": round(pct, 1), "message": msg})
                    if on_event:
                        on_event(dict(self.state))

                bin_dir = daemon.config.workspace_dir / "bin"
                server_exe = await asyncio.to_thread(
                    download_and_extract_llama_server,
                    bin_dir,
                    False,
                    _llama_progress,
                )
                if daemon.llama_manager and daemon.config.llama_model_path:
                    self.state.update(
                        {"target": "llama", "percent": 100.0, "message": "正在啟動本地推論引擎服務..."}
                    )
                    if on_event:
                        on_event(dict(self.state))
                    await asyncio.to_thread(daemon.llama_manager.ensure_running)
                res = {"server_exe": str(server_exe)}
            else:
                raise ValueError(f"不支援的模型下載目標：{target}")
            self.state.update(
                {
                    "active": False,
                    "message": f"{target.upper()} 模型下載並掛載完成！",
                    "percent": 100.0,
                    "last_result": res,
                }
            )
            return res
        except Exception as exc:
            logger.exception("Model download failed: %s", exc)
            friendly_msg = format_friendly_error_message(exc, prefix=f"{target.upper()} 下載失敗")
            self.state.update(
                {
                    "active": False,
                    "message": friendly_msg,
                    "percent": 0.0,
                    "error": str(exc),
                }
            )
            raise
        finally:
            if on_event:
                on_event(dict(self.state))

    def start_background_download(
        self,
        daemon: Any,
        target: str,
        repo_id: str = "",
        filename: str = "",
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> asyncio.Task[dict[str, Any]]:
        """Schedule model download in background task without blocking HTTP response."""
        async def _runner() -> dict[str, Any]:
            try:
                return await self.execute_download(daemon, target, repo_id, filename, on_event)
            except Exception as e:
                logger.error("Background download failed: %s", e)
                return {"error": str(e)}

        self._task = asyncio.create_task(_runner())
        return self._task

    async def execute_deploy_all(
        self,
        daemon: Any,
        repo_id: str = "",
        filename: str = "",
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Check all environments, sequentially download missing models, and launch llama-server."""
        models_root = daemon.config.workspace_dir / "models"
        self.state = {
            "active": True,
            "target": "all",
            "message": "開始一鍵檢查與部署所有模型環境...",
            "percent": 5.0,
            "error": "",
            "last_result": {},
        }
        if on_event:
            on_event(dict(self.state))
        try:
            # Step 1: Check BGE ONNX
            env = inspect_environment(daemon.config, daemon=daemon)
            if not env.get("bge_onnx_ready", False):
                self.state.update({"target": "bge", "percent": 0.0, "message": "正在下載語意嵌入模型：0%..."})
                if on_event:
                    on_event(dict(self.state))

                def _bge_deploy_progress(pct: float, msg: str) -> None:
                    self.state.update({"target": "bge", "percent": round(pct, 1), "message": msg})
                    if on_event:
                        on_event(dict(self.state))

                bge_dir = models_root / "bge-small-zh-v1.5"
                await asyncio.to_thread(
                    download_bge_small_onnx,
                    bge_dir,
                    "Xenova/bge-small-zh-v1.5",
                    _bge_deploy_progress,
                )
                daemon.embedder.load_onnx_dir(bge_dir)

            # Step 2: Check Qwen GGUF
            env = inspect_environment(daemon.config, daemon=daemon)
            if not env.get("gguf_ready", False):
                self.state.update({"target": "gguf", "percent": 0.0, "message": "正在下載核心大語言模型 (LLM)：0%..."})
                if on_event:
                    on_event(dict(self.state))

                def _gguf_deploy_progress(pct: float, msg: str) -> None:
                    self.state.update({"target": "gguf", "percent": round(pct, 1), "message": msg})
                    if on_event:
                        on_event(dict(self.state))

                await asyncio.to_thread(
                    download_gguf_model,
                    models_root,
                    repo_id or "bartowski/Qwen2.5-7B-Instruct-GGUF",
                    filename or "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
                    5,
                    _gguf_deploy_progress,
                )

            # Step 3: Check and download llama-server binary if missing
            env = inspect_environment(daemon.config, daemon=daemon)
            if not env.get("llama_server_found", False):
                self.state.update(
                    {
                        "target": "llama",
                        "percent": 0.0,
                        "message": "正在自動偵測硬體並下載安裝本地推論引擎：0%...",
                    }
                )
                if on_event:
                    on_event(dict(self.state))
                from scripts.download_llama_server import download_and_extract_llama_server

                def _llama_deploy_progress(pct: float, msg: str) -> None:
                    self.state.update({"target": "llama", "percent": round(pct, 1), "message": msg})
                    if on_event:
                        on_event(dict(self.state))

                bin_dir = daemon.config.workspace_dir / "bin"
                await asyncio.to_thread(
                    download_and_extract_llama_server,
                    bin_dir,
                    False,
                    _llama_deploy_progress,
                )

            # Step 4: Check llama-server and launch if needed
            env = inspect_environment(daemon.config, daemon=daemon)
            if not env.get("llama_server_healthy", False) and env.get("llama_server_found", False) and env.get("gguf_ready", False):
                self.state.update({"target": "llama", "percent": 90.0, "message": "正在啟動本地推論引擎服務..."})
                if on_event:
                    on_event(dict(self.state))
                await asyncio.to_thread(daemon.llama_manager.ensure_running)

            env = inspect_environment(daemon.config, daemon=daemon)
            self.state.update(
                {
                    "active": False,
                    "message": "所有環境與模型部署檢查完成！",
                    "percent": 100.0,
                    "last_result": env,
                }
            )
            return env
        except Exception as exc:
            logger.exception("Sequential deployment failed: %s", exc)
            friendly_msg = format_friendly_error_message(exc, prefix="部署失敗")
            self.state.update(
                {
                    "active": False,
                    "message": friendly_msg,
                    "percent": 0.0,
                    "error": str(exc),
                }
            )
            raise
        finally:
            if on_event:
                on_event(dict(self.state))

    def start_background_deploy_all(
        self,
        daemon: Any,
        repo_id: str = "",
        filename: str = "",
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> asyncio.Task[dict[str, Any]]:
        """Schedule sequential all-in-one deployment in background without blocking HTTP response."""
        async def _runner() -> dict[str, Any]:
            try:
                return await self.execute_deploy_all(daemon, repo_id, filename, on_event)
            except Exception as e:
                logger.error("Background deploy all failed: %s", e)
                return {"error": str(e)}

        self._task = asyncio.create_task(_runner())
        return self._task


ENV_DOWNLOAD_MANAGER = EnvironmentDownloadManager()


def main() -> None:
    """CLI entry point for environment verification and optional ONNX model download."""
    parser = argparse.ArgumentParser(
        description="Check 8GB VRAM environment and optionally download bge-small-zh-v1.5 ONNX."
    )
    parser.add_argument(
        "--download-bge",
        action="store_true",
        help="Download BAAI/bge-small-zh-v1.5 ONNX files into models/bge-small-zh-v1.5",
    )
    args = parser.parse_args()
    inspect_environment()
    if args.download_bge:
        download_bge_small_onnx()


if __name__ == "__main__":
    main()
