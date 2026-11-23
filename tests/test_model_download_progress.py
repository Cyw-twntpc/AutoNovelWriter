"""Tests for Hugging Face Hub model download real-time progress interceptor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts.setup_environment import (
    download_bge_small_onnx,
    download_gguf_model,
    hf_download_progress_hook,
)


def test_hf_download_progress_hook_intercepts_chunks() -> None:
    """Verify that hf_download_progress_hook captures chunk updates and passes percentage and MB."""
    import huggingface_hub.file_download as fd

    recorded: list[tuple[float, str]] = []

    def on_progress(pct: float, msg: str) -> None:
        recorded.append((pct, msg))

    with hf_download_progress_hook(on_progress, description="測試模型"):
        # Simulate http_get creating a tqdm progress bar
        bar = fd.tqdm(total=10 * 1024 * 1024, unit="B", unit_scale=True)
        bar.update(2 * 1024 * 1024)
        bar.update(3 * 1024 * 1024)
        bar.update(5 * 1024 * 1024)

    assert len(recorded) == 3
    assert recorded[0][0] == 20.0
    assert "2.0 / 10.0 MB" in recorded[0][1]
    assert recorded[1][0] == 50.0
    assert "5.0 / 10.0 MB" in recorded[1][1]
    assert recorded[2][0] == 100.0
    assert "10.0 / 10.0 MB" in recorded[2][1]


def test_download_gguf_model_with_progress(tmp_path: Path) -> None:
    """Verify download_gguf_model runs and passes on_progress callback correctly."""
    records: list[tuple[float, str]] = []

    def on_progress(pct: float, msg: str) -> None:
        records.append((pct, msg))

    def fake_download(*args, **kwargs):
        # Simulate chunk updates during download
        import huggingface_hub.file_download as fd

        bar = fd.tqdm(total=1000, unit="B")
        bar.update(500)
        bar.update(500)
        fake_file = tmp_path / "model.gguf"
        fake_file.write_bytes(b"dummy")
        return str(fake_file)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        res = download_gguf_model(
            target_dir=tmp_path,
            repo_id="test/repo",
            filename="model.gguf",
            on_progress=on_progress,
        )
        assert Path(res["gguf_path"]).exists()
        assert len(records) >= 2
        assert records[-1][0] == 100.0


def test_download_bge_small_onnx_with_progress(tmp_path: Path) -> None:
    """Verify download_bge_small_onnx runs and passes on_progress callback."""
    records: list[tuple[float, str]] = []

    def on_progress(pct: float, msg: str) -> None:
        records.append((pct, msg))

    def fake_download(*args, **kwargs):
        filename = kwargs.get("filename", "")
        import huggingface_hub.file_download as fd

        bar = fd.tqdm(total=200, unit="B")
        bar.update(200)
        out_f = tmp_path / filename
        out_f.parent.mkdir(parents=True, exist_ok=True)
        out_f.write_bytes(b"{}")
        return str(out_f)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        res = download_bge_small_onnx(target_dir=tmp_path, on_progress=on_progress)
        assert Path(res["tokenizer_path"]).exists()
        assert len(records) > 0
