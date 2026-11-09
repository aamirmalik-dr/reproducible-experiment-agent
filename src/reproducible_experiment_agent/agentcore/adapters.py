"""Model adapters: one protocol, two offline replay adapters, one live adapter.

The fixture and replay adapters exist for tests and demos. Nothing in this
package substitutes them for a live model silently: a live run is requested
explicitly and fails loudly when no endpoint is configured.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import BaseModel

ENV_BASE_URL = "AGENT_MODEL_BASE_URL"
ENV_MODEL = "AGENT_MODEL_NAME"
ENV_API_KEY = "AGENT_MODEL_API_KEY"


class AdapterError(RuntimeError):
    """The adapter could not produce a response."""


class LiveModelUnavailable(AdapterError):
    """A live model was required but no endpoint is configured."""


class ModelResponse(BaseModel):
    """Raw model text plus accounting. Token counts are None when unknown."""

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_s: float = 0.0


class ModelAdapter(Protocol):
    """Anything that turns chat messages into one text completion."""

    name: str
    is_live: bool

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelResponse:
        """Return one completion for the given messages."""
        ...


class FixtureAdapter:
    """Replays a scripted list of raw responses in order (offline, deterministic)."""

    is_live = False

    def __init__(self, responses: list[str | dict[str, Any]], name: str = "fixture-script") -> None:
        self.name = name
        self._responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.cursor = 0

    @classmethod
    def from_file(cls, path: Path, key: str) -> FixtureAdapter:
        """Load the script stored under ``key`` in a JSON mapping file."""
        scripts = json.loads(Path(path).read_text(encoding="utf-8"))
        if key not in scripts:
            raise AdapterError(f"no fixture script for {key!r} in {Path(path).name}")
        return cls(scripts[key], name=f"fixture-script:{Path(path).stem}")

    def seek(self, index: int) -> None:
        """Position the cursor, used when resuming from a checkpoint."""
        self.cursor = index

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelResponse:
        """Return the next scripted response."""
        if self.cursor >= len(self._responses):
            raise AdapterError("fixture script exhausted")
        text = self._responses[self.cursor]
        self.cursor += 1
        return ModelResponse(text=text, model=self.name)


class ReplayAdapter(FixtureAdapter):
    """Replays the raw model responses recorded in a saved JSONL trace."""

    @classmethod
    def from_trace(cls, path: Path) -> ReplayAdapter:
        """Collect every raw response from the step events of a trace."""
        responses: list[str] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            responses.extend(event.get("raw_responses", []))
        return cls(responses, name=f"replay:{Path(path).stem}")


class OpenAICompatibleAdapter:
    """Chat-completions client for any OpenAI-compatible endpoint.

    Works with local servers (Ollama, vLLM, LM Studio) and hosted providers.
    Plain chat completion is used instead of native tool calling so that small
    local models work with the same protocol. Transient failures (HTTP 429 and
    5xx, timeouts, connection errors and replies without choices) are retried
    a bounded number of times; anything else fails immediately.
    """

    is_live = True
    RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        temperature: float = 0.0,
        client: httpx.Client | None = None,
        max_retries: int = 2,
        backoff_s: float = 2.0,
    ) -> None:
        self.name = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._temperature = temperature
        self._client = client or httpx.Client(timeout=timeout_s)
        self.max_retries, self.backoff_s = max_retries, backoff_s
        self.retries = 0

    @classmethod
    def from_env(cls, model: str | None = None, **kwargs: Any) -> OpenAICompatibleAdapter:
        """Build from environment variables, or raise ``LiveModelUnavailable``.

        Args:
            model: Explicit model id. When given it overrides ``AGENT_MODEL_NAME``,
                so a frozen config can pin the exact model.
            **kwargs: Passed to the constructor.
        """
        base, model = os.environ.get(ENV_BASE_URL), model or os.environ.get(ENV_MODEL)
        if not base or not model:
            raise LiveModelUnavailable(
                f"live mode needs {ENV_BASE_URL} and {ENV_MODEL} (and {ENV_API_KEY} for "
                "hosted providers); none were found, and fixture fallback is disabled"
            )
        return cls(base, model, os.environ.get(ENV_API_KEY) or None, **kwargs)

    def _post_once(self, body: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        """One request. Returns ``(data, "")`` or ``(None, reason)`` if retryable."""
        try:
            resp = self._client.post(self._url, json=body, headers=self._headers)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            return None, type(exc).__name__
        if resp.status_code in self.RETRY_STATUS:
            return None, f"HTTP {resp.status_code}"
        try:
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AdapterError(f"model request failed: {type(exc).__name__}") from None
        if not isinstance(data, dict) or not data.get("choices"):
            return None, "reply without choices"
        return data, ""

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelResponse:
        """Send one chat-completions request and parse text, usage and cost."""
        body = {
            "model": self.name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self._temperature,
        }
        start = time.perf_counter()
        data, reason = self._post_once(body)
        for attempt in range(self.max_retries):
            if data is not None:
                break
            self.retries += 1
            time.sleep(self.backoff_s * (2**attempt))
            data, reason = self._post_once(body)
        if data is None:
            raise AdapterError(f"model request failed after retries: {reason}")
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise AdapterError("model request failed: unexpected reply shape") from None
        usage = data.get("usage") or {}
        cost = usage.get("cost")
        return ModelResponse(
            text=text,
            model=str(data.get("model", self.name)),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
            latency_s=time.perf_counter() - start,
        )
