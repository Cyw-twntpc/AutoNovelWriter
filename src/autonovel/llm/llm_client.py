"""Async LLM client with GBNF JSON Schema constraints and dual-watchdog protection."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator, Callable
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from autonovel.config import RuntimeConfig, get_logger
from autonovel.llm.llama_manager import LlamaServerManager

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class WatchdogLoopError(RuntimeError):
    """Raised when the streaming watchdog detects repetitive text generation loops."""


class LLMResponseFormatError(RuntimeError):
    """Raised when llama-server returns truncated or non-schema structured output."""


def detect_repetition_loop(text: str, min_pattern_len: int = 18, min_repeats: int = 4) -> bool:
    """Detect consecutive repeating phrases or clauses at the tail of generated text."""
    tail = text[-600:]
    n = len(tail)
    if n < min_pattern_len * min_repeats:
        return False
    max_pat = n // min_repeats
    for pat_len in range(min_pattern_len, max_pat + 1):
        unit = tail[-pat_len:]
        if tail.endswith(unit * min_repeats):
            return True
    return False


class LLMClient:
    """Communicates with llama-server with timeout, loop watchdog, and GBNF schemas."""

    def __init__(
        self,
        config: RuntimeConfig,
        server_manager: LlamaServerManager | None = None,
    ) -> None:
        """Initialize client with runtime config and optional server manager."""
        self.config = config
        self.server_manager = server_manager or LlamaServerManager(config)
        self._active_requests: int = 0
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or initialize persistent HTTP client with keep-alive pooling."""
        if self._http_client is None or self._http_client.is_closed:
            timeout = httpx.Timeout(self.config.llm_timeout_sec, connect=10.0)
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
            self._http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
        return self._http_client

    async def aclose(self) -> None:
        """Close persistent HTTP client session."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    def close(self) -> None:
        """Synchronously close persistent HTTP client if possible."""
        if self._http_client is not None and not self._http_client.is_closed:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.aclose())
            except RuntimeError:
                pass

    def is_busy(self) -> bool:
        """Check if any LLM text generation or structured query is actively running."""
        return self._active_requests > 0

    async def generate_text_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4500,
        temperature: float = 0.72,
        seed: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Stream text generation while monitoring for 10-min timeout and repetition loops."""
        self._active_requests += 1
        try:
            actual_seed = seed if seed is not None else random.randint(1, 2_000_000_000)
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": actual_seed,
                "stream": True,
            }
            chunks: list[str] = []
            async for token in self._stream_chunks(payload):
                chunks.append(token)
                if on_token:
                    on_token(token)
                accumulated = "".join(chunks)
                if len(chunks) % 12 == 0 and detect_repetition_loop(accumulated):
                    await self.server_manager.restart_server_async()
                    raise WatchdogLoopError("Watchdog detected repetitive loop in generation stream.")
            return "".join(chunks).strip()
        finally:
            self._active_requests = max(0, self._active_requests - 1)

    async def _stream_chunks(self, payload: dict[str, object]) -> AsyncIterator[str]:
        """Execute streaming HTTP POST to /v1/chat/completions using persistent connection pool."""
        client = self._get_http_client()
        url = f"{self.config.llama_server_url}/v1/chat/completions"
        try:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    delta = self._parse_sse_line(line)
                    if delta:
                        yield delta
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            resp_detail = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    resp_detail = f" | Response body: {exc.response.text}"
                except Exception:
                    pass
            logger.error("LLM request failed or timed out: %s%s", exc, resp_detail)
            # Do not close the shared client here: other concurrent streams still use
            # its connection pool, and closing it would abort them.
            if not getattr(self.server_manager, "_manually_unloaded", False):
                await self.server_manager.restart_server_async()
            raise

    @staticmethod
    def _parse_sse_line(line: str) -> str:
        """Extract delta content string with robust non-JSON / log line fault tolerance."""
        stripped = line.strip()
        if not stripped.startswith("data:"):
            return ""
        data_str = stripped[5:].strip()
        if data_str == "[DONE]":
            return ""
        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            return ""
        choices = data.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("delta", {}).get("content") or "")

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[T],
        max_tokens: int = 3072,
        temperature: float = 0.25,
        seed: int | None = None,
    ) -> T:
        """Generate output constrained by GBNF JSON Schema using persistent connection pool."""
        self._active_requests += 1
        try:
            actual_seed = seed if seed is not None else random.randint(1, 2_000_000_000)
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.1,
                "seed": actual_seed,
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_model.__name__,
                        "strict": True,
                        "schema": schema_model.model_json_schema(),
                    },
                },
            }
            client = self._get_http_client()
            url = f"{self.config.llama_server_url}/v1/chat/completions"
            last_format_error: Exception | None = None
            for attempt in range(1, 3):
                try:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    body = resp.json()
                    raw_json = body["choices"][0]["message"]["content"]
                    return schema_model.model_validate_json(raw_json)
                except (ValidationError, json.JSONDecodeError, KeyError, IndexError) as exc:
                    # A truncated generation is a normal failure mode; retry once
                    # instead of surfacing an opaque 500 to the caller.
                    last_format_error = exc
                    logger.warning(
                        "Structured output malformed for %s (attempt %d/2): %s",
                        schema_model.__name__,
                        attempt,
                        exc,
                    )
                    continue
                except (httpx.TimeoutException, httpx.HTTPError) as exc:
                    resp_detail = ""
                    if hasattr(exc, "response") and exc.response is not None:
                        try:
                            resp_detail = f" | Response body: {exc.response.text}"
                        except Exception:
                            pass
                    logger.error("Structured LLM call failed: %s%s", exc, resp_detail)
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)
                    if status_code and 400 <= status_code < 500:
                        logger.warning(
                            "HTTP %s client error%s: skipping server restart.",
                            status_code,
                            resp_detail,
                        )
                    elif getattr(self.server_manager, "_manually_unloaded", False):
                        logger.info("Model was manually unloaded; skipping server auto-restart.")
                    else:
                        # Keep the shared HTTP client open for concurrent streams.
                        await self.server_manager.restart_server_async()
                    raise
            raise LLMResponseFormatError(
                f"{schema_model.__name__} 結構化輸出解析失敗（已重試 2 次）：{last_format_error}"
            ) from last_format_error
        finally:
            self._active_requests = max(0, self._active_requests - 1)
