"""Automated downloader and installer for llama.cpp llama-server Windows binary with dynamic CUDA auto-detection."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("autonovel.llama_downloader")

LLAMA_RELEASES_API = "https://api.github.com/repos/ggerganov/llama.cpp/releases?per_page=5"


def get_system_cuda_version() -> float | None:
    """Detect the maximum CUDA runtime version supported by local NVIDIA driver/toolkit."""
    # 1. Probe nvidia-smi driver reporting
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi], capture_output=True, text=True, timeout=3).stdout
            m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
            if m:
                ver = float(m.group(1))
                logger.info("Detected NVIDIA driver CUDA support: %s via nvidia-smi", ver)
                return ver
        except Exception as e:
            logger.debug("nvidia-smi query failed: %s", e)

    # 2. Probe installed NVIDIA GPU Computing Toolkit directories
    toolkit_base = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if toolkit_base.exists():
        versions: list[float] = []
        for d in toolkit_base.iterdir():
            m = re.match(r"v(\d+\.\d+)", d.name)
            if m:
                versions.append(float(m.group(1)))
        if versions:
            ver = max(versions)
            logger.info("Detected local CUDA Toolkit version: %s from filesystem", ver)
            return ver

    # 3. Probe nvcc compiler if available in PATH
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=3).stdout
            m = re.search(r"release\s+(\d+\.\d+)", out)
            if m:
                ver = float(m.group(1))
                logger.info("Detected CUDA compiler release: %s via nvcc", ver)
                return ver
        except Exception as e:
            logger.debug("nvcc query failed: %s", e)

    logger.info("No NVIDIA CUDA environment detected; will use CPU build.")
    return None


def fetch_latest_release_asset(force_cpu: bool = False) -> tuple[str, str, str]:
    """Dynamically query GitHub releases and select the optimal Windows build matching system CUDA version.

    Returns:
        tuple of (browser_download_url, asset_name, build_description)
    """
    sys_cuda = None if force_cpu else get_system_cuda_version()
    logger.info(
        "Querying GitHub for llama.cpp Windows releases (system CUDA: %s, force_cpu=%s)...",
        sys_cuda or "N/A (CPU)",
        force_cpu,
    )

    req = urllib.request.Request(LLAMA_RELEASES_API, headers={"User-Agent": "AutoNovelWriter/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        releases = json.loads(resp.read().decode("utf-8"))

    for rel in releases:
        tag_name = rel.get("tag_name", "unknown")
        assets = rel.get("assets", [])
        cuda_assets: dict[float, dict] = {}
        cpu_asset: dict | None = None

        for a in assets:
            name = a.get("name", "")
            # Skip non-zip or pure runtime dependency zips like cudart
            if not name.endswith(".zip") or name.startswith("cudart"):
                continue

            # Check CUDA Windows builds: e.g. llama-b11870-bin-win-cuda-12.4-x64.zip
            m = re.search(r"cuda-(\d+\.\d+)-x64\.zip", name)
            if m:
                cuda_assets[float(m.group(1))] = a
            elif "win-cpu-x64.zip" in name:
                cpu_asset = a

        # If system has CUDA and not forcing CPU, match the highest CUDA version <= system CUDA
        if sys_cuda is not None and cuda_assets:
            compatible_cudas = [v for v in cuda_assets if v <= sys_cuda]
            if compatible_cudas:
                best_cuda = max(compatible_cudas)
                chosen = cuda_assets[best_cuda]
                desc = f"CUDA {best_cuda} (相容本地 CUDA {sys_cuda})"
                logger.info(
                    "Selected release %s: %s [%s]",
                    tag_name,
                    chosen["name"],
                    desc,
                )
                return chosen["browser_download_url"], chosen["name"], desc
            logger.warning(
                "System CUDA %s is lower than available release CUDA versions %s. Falling back to CPU.",
                sys_cuda,
                list(cuda_assets.keys()),
            )

        # Fallback to general CPU x64 asset
        if cpu_asset:
            desc = "CPU (x64 AVX2 預設通用版)"
            logger.info("Selected release %s: %s [%s]", tag_name, cpu_asset["name"], desc)
            return cpu_asset["browser_download_url"], cpu_asset["name"], desc

    raise RuntimeError("無法在 GitHub Release 中找到相符的 llama.cpp Windows 執行檔壓縮包。")


def download_file_with_resume(
    url: str,
    target_path: Path,
    description: str = "",
    max_retries: int = 5,
    chunk_size: int = 1024 * 1024,
    on_progress: Callable[[float, str], None] | None = None,
) -> Path:
    """Download a remote file with HTTP Range resume support, retry resilience, and zip integrity validation.

    Writes to `<target_path>.incomplete` during download and atomically renames to `<target_path>` upon completion.
    """
    incomplete_path = target_path.with_name(target_path.name + ".incomplete")

    # If the target file already exists and is a valid zip, reuse it directly
    if target_path.exists() and target_path.suffix.lower() == ".zip" and zipfile.is_zipfile(target_path):
        logger.info("Target zip archive already exists and is valid: %s", target_path)
        return target_path

    for attempt in range(1, max_retries + 1):
        try:
            existing_bytes = incomplete_path.stat().st_size if incomplete_path.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": "AutoNovelWriter/1.0"})

            if existing_bytes > 0:
                req.add_header("Range", f"bytes={existing_bytes}-")
                logger.info(
                    "Resuming download of %s from offset %d bytes (%.1f MB) (attempt %d/%d)...",
                    target_path.name,
                    existing_bytes,
                    existing_bytes / (1024 * 1024),
                    attempt,
                    max_retries,
                )
            else:
                logger.info("Starting download of %s (attempt %d/%d)...", target_path.name, attempt, max_retries)

            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    status_code = response.getcode()
                    content_length = int(response.headers.get("Content-Length", 0))

                    if status_code == 206:
                        # Server honored HTTP Range request: append to existing partial file
                        cr = response.headers.get("Content-Range", "")
                        m = re.search(r"/(\d+)$", cr)
                        if m:
                            total_size = int(m.group(1))
                        else:
                            total_size = existing_bytes + content_length
                        open_mode = "ab"
                        downloaded = existing_bytes
                        logger.info(
                            "Server accepted HTTP Range (206 Partial Content): total %.1f MB, remaining %.1f MB",
                            total_size / (1024 * 1024),
                            content_length / (1024 * 1024),
                        )
                    elif status_code == 200:
                        # Server sent entire file from offset 0
                        total_size = content_length
                        open_mode = "wb"
                        downloaded = 0
                        if existing_bytes > 0:
                            logger.warning("Server responded with 200 OK instead of 206; restarting download from 0")
                    else:
                        raise RuntimeError(f"Unexpected HTTP status code: {status_code}")

                    with open(incomplete_path, open_mode) as out_file:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                pct = round(downloaded / total_size * 100, 1)
                                msg = f"下載推論引擎中 ({description})：{pct}%"
                                sys.stdout.write(
                                    f"\r{msg} ({round(downloaded / (1024*1024), 1)} / {round(total_size / (1024*1024), 1)} MB)"
                                )
                                sys.stdout.flush()
                                if on_progress:
                                    on_progress(pct, msg)

                    sys.stdout.write("\n")

            except urllib.error.HTTPError as http_err:
                # 416 Range Not Satisfiable: partial file might already be complete or invalid
                if http_err.code == 416:
                    logger.warning("HTTP 416 Range Not Satisfiable: checking if existing partial file is complete...")
                    if target_path.suffix.lower() == ".zip" and zipfile.is_zipfile(incomplete_path):
                        logger.info("Existing partial file is a complete valid zip archive.")
                        incomplete_path.replace(target_path)
                        return target_path
                    logger.warning("Existing partial file is corrupted; removing %s to re-download", incomplete_path)
                    incomplete_path.unlink(missing_ok=True)
                    continue
                raise

            # Post-download verification
            if not incomplete_path.exists():
                raise RuntimeError(f"Download file missing after stream loop: {incomplete_path}")

            if target_path.suffix.lower() == ".zip":
                if not zipfile.is_zipfile(incomplete_path):
                    logger.error("File %s is not a valid zip archive (incomplete or corrupted).", incomplete_path)
                    incomplete_path.unlink(missing_ok=True)
                    if attempt >= max_retries:
                        raise RuntimeError(f"下載的 llama-server 壓縮包驗證損壞，已重試 {max_retries} 次。")
                    time.sleep(2.0)
                    continue

            # Atomically rename .incomplete -> target_path
            incomplete_path.replace(target_path)
            logger.info("Download and archive verification successful: %s", target_path)
            return target_path

        except Exception as exc:
            logger.warning(
                "Download interrupted on attempt %d/%d (%s). Will retry in 2s...",
                attempt,
                max_retries,
                exc,
            )
            if attempt >= max_retries:
                raise RuntimeError(f"llama-server 安裝包下載失敗（已重試 {max_retries} 次）：{exc}") from exc
            time.sleep(2.0)

    raise RuntimeError(f"下載在嘗試 {max_retries} 次後失敗。")


def download_and_extract_llama_server(
    dest_dir: Path,
    force_cpu: bool = False,
    on_progress: Callable[[float, str], None] | None = None,
) -> Path:
    """Download the dynamically matched llama-server package with resume support, extract to dest_dir, and verify binary."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    download_url, asset_name, build_desc = fetch_latest_release_asset(force_cpu=force_cpu)
    zip_path = dest_dir / asset_name

    logger.info("Ensuring %s (%s) from %s with breakpoint resume...", asset_name, build_desc, download_url)
    download_file_with_resume(
        url=download_url,
        target_path=zip_path,
        description=build_desc,
        on_progress=on_progress,
    )

    logger.info("Extracting %s to %s ...", zip_path, dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    zip_path.unlink(missing_ok=True)
    server_exe = dest_dir / "llama-server.exe"
    if not server_exe.exists():
        found = list(dest_dir.glob("**/llama-server.exe"))
        if found:
            server_exe = found[0]

    logger.info("llama-server binary resolved to: %s", server_exe)
    try:
        ver_proc = subprocess.run([str(server_exe), "--version"], capture_output=True, text=True, timeout=5)
        logger.info("Verification succeeded: %s", (ver_proc.stdout or ver_proc.stderr).strip())
    except Exception as e:
        logger.warning("Could not execute --version check: %s", e)

    return server_exe


def main() -> None:
    """Parse CLI options and trigger installation."""
    parser = argparse.ArgumentParser(description="Install llama-server binary for AutoNovelWriter")
    parser.add_argument("--dest", type=Path, default=Path("bin"), help="Destination directory (default: bin)")
    parser.add_argument("--cpu-only", action="store_true", help="Download CPU build instead of dynamic CUDA build")
    args = parser.parse_args()

    target_dir = args.dest if args.dest.is_absolute() else Path.cwd() / args.dest
    download_and_extract_llama_server(target_dir, force_cpu=args.cpu_only)


if __name__ == "__main__":
    main()
