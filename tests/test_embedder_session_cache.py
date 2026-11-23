"""Tests for CpuEmbedder session caching and load-failure diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from autonovel.llm import embedder as embedder_module
from autonovel.llm.embedder import CpuEmbedder


class _FakeTokenizer:
    """Minimal stand-in for tokenizers.Tokenizer used by the ONNX loader."""

    def __init__(self, path: str) -> None:
        self.path = path

    @classmethod
    def from_file(cls, path: str) -> "_FakeTokenizer":
        return cls(path)

    def enable_padding(self, **kwargs: object) -> None:
        """Record padding configuration without loading a real tokenizer."""

    def enable_truncation(self, **kwargs: object) -> None:
        """Record truncation configuration without loading a real tokenizer."""


def test_onnx_session_is_shared_across_embedder_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two CpuEmbedder instances must reuse one ONNX session instead of allocating twice."""
    created: list[tuple[str, tuple[str, ...]]] = []

    class _FakeInferenceSession:
        def __init__(self, path: str, providers: list[str] | None = None) -> None:
            created.append((path, tuple(providers or ())))

    monkeypatch.setitem(
        sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=_FakeInferenceSession)
    )
    monkeypatch.setitem(
        sys.modules, "tokenizers", SimpleNamespace(Tokenizer=_FakeTokenizer)
    )
    monkeypatch.setattr(embedder_module, "_ONNX_SESSION_CACHE", {})

    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.onnx").write_bytes(b"fake")

    first = CpuEmbedder(local_onnx_dir=tmp_path)
    second = CpuEmbedder(local_onnx_dir=tmp_path)

    assert first.load_onnx_dir(tmp_path) is True
    assert second.load_onnx_dir(tmp_path) is True
    assert first._onnx_session is second._onnx_session
    assert len(created) == 1


def test_embed_batch_distinguishes_load_failure_from_missing_model() -> None:
    """A failed model load must surface the original error, not a generic 'not loaded'."""
    missing = CpuEmbedder(local_onnx_dir=Path("does-not-exist"))
    missing._init_attempted = True
    with pytest.raises(RuntimeError, match="is not loaded"):
        missing.embed_batch(["測試"])

    broken = CpuEmbedder(local_onnx_dir=Path("does-not-exist"))
    broken._init_attempted = True
    broken._load_error = MemoryError("bad allocation")
    with pytest.raises(RuntimeError, match="failed to load") as excinfo:
        broken.embed_batch(["測試"])
    assert isinstance(excinfo.value.__cause__, MemoryError)
