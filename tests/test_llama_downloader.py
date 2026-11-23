"""Tests for llama.cpp binary downloader and HTTP Range breakpoint resume mechanism."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import zipfile


from scripts.download_llama_server import (
    download_file_with_resume,
    get_system_cuda_version,
)


def _create_sample_zip_bytes(filename: str = "llama-server.exe", content: bytes = b"dummy exe binary") -> bytes:
    """Helper to create valid in-memory zip bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, content)
    return buf.getvalue()


def test_cuda_detection_mocked() -> None:
    """Ensure get_system_cuda_version parses nvidia-smi correctly."""
    mock_smi_out = "NVIDIA-SMI 572.70   Driver Version: 572.70   CUDA Version: 12.8"
    with patch("shutil.which", return_value="nvidia-smi"), patch(
        "subprocess.run", return_value=MagicMock(stdout=mock_smi_out)
    ):
        ver = get_system_cuda_version()
        assert ver == 12.8


def test_download_file_with_resume_clean_start(tmp_path: Path) -> None:
    """Test standard 200 OK single-run download."""
    zip_bytes = _create_sample_zip_bytes()
    target_zip = tmp_path / "llama-test.zip"

    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.getcode.return_value = 200
    mock_resp.headers = {"Content-Length": str(len(zip_bytes))}
    mock_resp.read.side_effect = [zip_bytes, b""]

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = download_file_with_resume("https://example.com/test.zip", target_zip, description="Test")
        assert res.exists()
        assert zipfile.is_zipfile(res)


def test_download_file_with_resume_partial_and_continue(tmp_path: Path) -> None:
    """Test breakpoint resume: partial download exists, sends Range, server returns 206, completes file."""
    zip_bytes = _create_sample_zip_bytes()
    part_size = len(zip_bytes) // 2
    part_1 = zip_bytes[:part_size]
    part_2 = zip_bytes[part_size:]

    target_zip = tmp_path / "llama-test.zip"
    incomplete_path = tmp_path / "llama-test.zip.incomplete"
    incomplete_path.write_bytes(part_1)

    captured_headers: dict[str, str] = {}

    def fake_urlopen(req, timeout=60):
        # Capture requested headers
        for k, v in req.headers.items():
            captured_headers[k.lower()] = v
        resp = MagicMock()
        resp.__enter__.return_value = resp
        resp.getcode.return_value = 206
        resp.headers = {
            "Content-Length": str(len(part_2)),
            "Content-Range": f"bytes {part_size}-{len(zip_bytes)-1}/{len(zip_bytes)}",
        }
        resp.read.side_effect = [part_2, b""]
        return resp

    progress_records: list[tuple[float, str]] = []

    def on_progress(pct: float, msg: str) -> None:
        progress_records.append((pct, msg))

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        res = download_file_with_resume(
            "https://example.com/test.zip",
            target_zip,
            description="Resume Test",
            on_progress=on_progress,
        )

        assert res.exists()
        assert not incomplete_path.exists()
        assert res.read_bytes() == zip_bytes
        assert zipfile.is_zipfile(res)
        # Check that Range header was sent with existing offset
        assert captured_headers.get("range") == f"bytes={part_size}-"
        # Progress records should reach 100%
        assert any(pct == 100.0 for pct, _ in progress_records)


def test_download_file_already_complete_skips(tmp_path: Path) -> None:
    """Test that if target zip already exists and is a valid zip, download is skipped."""
    zip_bytes = _create_sample_zip_bytes()
    target_zip = tmp_path / "llama-already-done.zip"
    target_zip.write_bytes(zip_bytes)

    with patch("urllib.request.urlopen") as mock_url:
        res = download_file_with_resume("https://example.com/test.zip", target_zip)
        assert res == target_zip
        # urllib.request.urlopen should never be called
        mock_url.assert_not_called()


def test_download_network_drop_and_auto_resume(tmp_path: Path) -> None:
    """Test that if connection drops during stream, retry automatically resumes from the saved partial bytes."""
    zip_bytes = _create_sample_zip_bytes()
    part_size = len(zip_bytes) // 2
    part_1 = zip_bytes[:part_size]
    part_2 = zip_bytes[part_size:]

    target_zip = tmp_path / "llama-drop-test.zip"
    call_count = 0
    captured_ranges: list[str | None] = []

    def failing_urlopen(req, timeout=60):
        nonlocal call_count
        call_count += 1
        captured_ranges.append(req.headers.get("Range"))

        resp = MagicMock()
        resp.__enter__.return_value = resp

        if call_count == 1:
            # First attempt: drops after yielding part_1
            resp.getcode.return_value = 200
            resp.headers = {"Content-Length": str(len(zip_bytes))}

            def buggy_read(chunk_size):
                if buggy_read.called:
                    raise ConnectionResetError("Simulated socket drop mid-stream")
                buggy_read.called = True
                return part_1

            buggy_read.called = False
            resp.read.side_effect = buggy_read
            return resp
        else:
            # Second attempt: resumes with HTTP 206
            resp.getcode.return_value = 206
            resp.headers = {
                "Content-Length": str(len(part_2)),
                "Content-Range": f"bytes {part_size}-{len(zip_bytes)-1}/{len(zip_bytes)}",
            }
            resp.read.side_effect = [part_2, b""]
            return resp

    with patch("urllib.request.urlopen", side_effect=failing_urlopen), patch("time.sleep"):
        res = download_file_with_resume(
            "https://example.com/test.zip",
            target_zip,
            description="Drop Test",
        )
        assert res.exists()
        assert res.read_bytes() == zip_bytes
        assert zipfile.is_zipfile(res)
        assert call_count == 2
        # First attempt sent no Range, second attempt sent Range with part_1 size
        assert captured_ranges[0] is None
        assert captured_ranges[1] == f"bytes={part_size}-"

