"""CPU/RAM vector embedding engine (bge-small-zh-v1.5, 512-dim, 0 MB VRAM)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from autonovel.config import get_logger

logger = get_logger(__name__)

# Process-wide cache of loaded ONNX sessions keyed by (tokenizer, model) paths.
# Each CpuEmbedder() used to allocate its own session, which caused transient
# "bad allocation" failures under memory pressure and wasted RAM in tests.
_ONNX_SESSION_CACHE: dict[str, tuple[Any, Any]] = {}
_ONNX_SESSION_LOCK = threading.Lock()


def _load_onnx_session(tok_path: Path, onnx_path: Path) -> tuple[Any, Any]:
    """Return a shared (tokenizer, session) pair, loading it at most once per path."""
    key = f"{tok_path.resolve()}::{onnx_path.resolve()}"
    with _ONNX_SESSION_LOCK:
        cached = _ONNX_SESSION_CACHE.get(key)
        if cached is not None:
            return cached
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tok_path))
        tok.enable_padding(pad_id=0, pad_token="[PAD]")
        tok.enable_truncation(max_length=512)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        _ONNX_SESSION_CACHE[key] = (tok, sess)
        logger.info("Loaded native ONNX CPU embedding model from %s", onnx_path.parent)
        return tok, sess


class CpuEmbedder:
    """Generates 512-dim normalized semantic embeddings on CPU via ONNX Runtime or fastembed."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        dim: int = 512,
        local_onnx_dir: Path | None = None,
    ) -> None:
        """Initialize CPU embedding provider without allocating any GPU VRAM."""
        self.model_name = model_name
        self.dim = dim
        self.local_onnx_dir = local_onnx_dir or Path("models/bge-small-zh-v1.5")
        self._onnx_session: Any = None
        self._tokenizer: Any = None
        self._fastembed_model: Any = None
        self._init_attempted = False
        self._load_error: Exception | None = None

    def _lazy_init_model(self) -> None:
        """Attempt to load native ONNX Runtime CPU model or fastembed on first call."""
        if self._init_attempted:
            return
        self._init_attempted = True
        if self.load_onnx_dir(self.local_onnx_dir):
            return
        if self._try_load_hf_cached_onnx():
            return
        try:
            from fastembed import TextEmbedding

            self._fastembed_model = TextEmbedding(
                model_name=self.model_name, providers=["CPUExecutionProvider"]
            )
            logger.info("Loaded CPU fastembed model: %s", self.model_name)
        except Exception as exc:
            logger.info(
                "Using deterministic CPU character/word n-gram embedding engine (%s).",
                exc,
            )
            self._fastembed_model = None

    def load_onnx_dir(self, model_dir: Path) -> bool:
        """Load tokenizer.json and model.onnx from a local directory using CPUExecutionProvider."""
        tok_path = model_dir / "tokenizer.json"
        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            onnx_path = model_dir / "onnx" / "model.onnx"
        if not (tok_path.exists() and onnx_path.exists()):
            return False
        try:
            self._tokenizer, self._onnx_session = _load_onnx_session(tok_path, onnx_path)
            return True
        except Exception as exc:
            self._load_error = exc
            logger.warning("Failed loading ONNX model from %s: %s", model_dir, exc)
            return False

    def _try_load_hf_cached_onnx(self) -> bool:
        """Check local HuggingFace cache for bge-small-zh-v1.5 ONNX weights without network block."""
        try:
            from huggingface_hub import hf_hub_download

            tok_file = hf_hub_download(
                repo_id=self.model_name, filename="tokenizer.json", local_files_only=True
            )
            try:
                onnx_file = hf_hub_download(
                    repo_id=self.model_name, filename="onnx/model.onnx", local_files_only=True
                )
            except Exception as exc:
                # Fall back to the models published with the weights at repo root.
                logger.debug("No onnx/model.onnx in HuggingFace cache: %s", exc)
                onnx_file = hf_hub_download(
                    repo_id=self.model_name, filename="model.onnx", local_files_only=True
                )
            cache_dir = Path(tok_file).parent
            if Path(onnx_file).parent != cache_dir:
                return self._init_session_from_files(Path(tok_file), Path(onnx_file))
            return self.load_onnx_dir(cache_dir)
        except Exception as exc:
            logger.debug("No local HuggingFace ONNX cache for %s: %s", self.model_name, exc)
            return False

    def _init_session_from_files(self, tok_path: Path, onnx_path: Path) -> bool:
        """Initialize tokenizer and ONNX session from explicit file paths."""
        try:
            self._tokenizer, self._onnx_session = _load_onnx_session(tok_path, onnx_path)
            return True
        except Exception as exc:
            self._load_error = exc
            logger.warning(
                "Failed loading ONNX model from %s: %s", onnx_path.parent, exc
            )
            return False

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string into a normalized 512-dim float32 vector."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of strings into normalized 512-dim float32 vectors on CPU."""
        if not texts:
            return []
        self._lazy_init_model()
        if self._onnx_session is not None and self._tokenizer is not None:
            return self._run_onnx_batch(texts)
        if self._fastembed_model is not None:
            raw_vecs = list(self._fastembed_model.embed(texts))
            return [self._normalize_and_fit(np.asarray(v, dtype=np.float32)) for v in raw_vecs]
        self._raise_unavailable()

    def _raise_unavailable(self) -> NoReturn:
        """Raise a distinct error for a failed load vs. a missing model."""
        if self._load_error is not None:
            raise RuntimeError(
                "CPU semantic embedding model failed to load: "
                f"{type(self._load_error).__name__}: {self._load_error}"
            ) from self._load_error
        raise RuntimeError(
            "CPU semantic embedding model (ONNX or fastembed) is not loaded; "
            "statistical n-gram hashing is prohibited."
        )

    def _run_onnx_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Execute native ONNX CPU inference and extract CLS token embeddings."""
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
        input_names = {inp.name for inp in self._onnx_session.get_inputs()}
        feed: dict[str, Any] = {}
        if "input_ids" in input_names:
            feed["input_ids"] = input_ids
        if "attention_mask" in input_names:
            feed["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = token_type_ids
        outputs = self._onnx_session.run(None, feed)
        first_out = np.asarray(outputs[0], dtype=np.float32)
        cls_vecs = first_out[:, 0, :] if first_out.ndim == 3 else first_out
        return [self._normalize_and_fit(cls_vecs[i]) for i in range(len(texts))]

    def _normalize_and_fit(self, vec: np.ndarray) -> np.ndarray:
        """Ensure vector has exact target dimension and unit L2 norm."""
        flat = vec.reshape(-1).astype(np.float32)
        if flat.shape[0] > self.dim:
            flat = flat[: self.dim]
        elif flat.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[: flat.shape[0]] = flat
            flat = padded
        norm = float(np.linalg.norm(flat))
        return flat / norm if norm > 1e-8 else flat
