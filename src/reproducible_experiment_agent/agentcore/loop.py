"""The observe, decide, validate, execute, checkpoint loop."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from .adapters import AdapterError, ModelAdapter
from .policy import Policy
from .schemas import BaseAgentState, Decision, HistoryEntry, PendingAction, Status, StopReason
from .trace import TraceWriter

S = TypeVar("S", bound=BaseAgentState)
Renderer = Callable[[BaseAgentState, list[HistoryEntry]], str]

REFUSED = ("unknown_tool", "not_allowed", "invalid_arguments")
RESPONSE_FORMAT = (
    'Reply with exactly one JSON object and nothing else: {"summary": "<one sentence>", '
    '"action": {"tool": "<tool name>", "arguments": {...}}}'
)


class CheckpointStore:
    """Atomic JSON checkpoints with one previous version kept as a fallback.

    Each save writes a temporary file, fsyncs it and renames it over the
    checkpoint, after moving the old checkpoint to ``<name>.prev``. Each file
    carries a SHA-256 of its payload, so a truncated or corrupted file is
    detected on load and the previous version is used instead.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.prev = self.path.with_name(self.path.name + ".prev")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: BaseAgentState) -> None:
        """Persist ``state`` atomically and bump its checkpoint version."""
        state.checkpoint_version += 1
        payload = state.model_dump_json()
        doc = {"sha256": hashlib.sha256(payload.encode()).hexdigest(), "state": payload}
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(doc, fh)
            fh.flush()
            os.fsync(fh.fileno())
        if self.path.exists():
            os.replace(self.path, self.prev)
        os.replace(tmp, self.path)

    def load(self, state_cls: type[S]) -> S | None:
        """Return the newest valid checkpoint, or None if none is readable."""
        for candidate in (self.path, self.prev):
            try:
                doc = json.loads(candidate.read_text(encoding="utf-8"))
                payload = doc["state"]
                if hashlib.sha256(payload.encode()).hexdigest() != doc["sha256"]:
                    continue
                return state_cls.model_validate_json(payload)
            except (OSError, ValueError, KeyError, TypeError, ValidationError):
                continue
        return None


def parse_decision(text: str, max_chars: int) -> tuple[Decision | None, str | None]:
    """Parse one model reply into a ``Decision``.

    The first complete JSON object in the reply is decoded; text after it
    (for example a stray closing brace) is ignored, because the object itself
    is unambiguous. Anything that is not a complete object is malformed.

    Returns:
        ``(decision, None)`` on success or ``(None, error message)``.
    """
    if len(text) > max_chars:
        return None, f"reply was {len(text)} characters, the limit is {max_chars}"
    body = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    start = body.find("{")
    if start < 0:
        return None, "no JSON object found"
    try:
        obj, _ = json.JSONDecoder().raw_decode(body, start)
        return Decision.model_validate(obj), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} at position {exc.pos}"
    except ValidationError as exc:
        first = exc.errors()[0]
        return None, f"schema error at {'.'.join(str(p) for p in first['loc'])}: {first['msg']}"


def render_recent(recent: list[HistoryEntry]) -> str:
    """Format recent tool results as clearly delimited untrusted data blocks."""
    blocks = []
    for h in recent:
        body = h.result.output if h.result.ok else {"error": h.result.error.model_dump()}
        blocks.append(
            f'<tool_result step="{h.step}" tool="{h.action.tool}" trust="untrusted-data">\n'
            f"{json.dumps(body, ensure_ascii=False)}\n</tool_result>"
        )
    return "\n".join(blocks) if blocks else "(no tool results yet)"


class AgentLoop:
    """Runs one episode with a model adapter under a policy.

    Args:
        adapter: Source of model replies.
        policy: Allowlist, validation, budgets and execution.
        system_prompt: Full system prompt, including the tool schemas.
        render: Builds the per-step user message from the state and the last
            ``recent_k`` history entries. It must show only observable state.
        checkpoint: Optional checkpoint store; written after every result.
        trace: Optional trace writer; one event per step.
        recent_k: Number of recent results shown to the model.
        max_output_chars: Replies longer than this are treated as malformed.
        max_completion_tokens: Per-call completion cap passed to the adapter.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        policy: Policy,
        system_prompt: str,
        render: Renderer,
        checkpoint: CheckpointStore | None = None,
        trace: TraceWriter | None = None,
        recent_k: int = 4,
        max_output_chars: int = 6000,
        max_completion_tokens: int = 700,
    ) -> None:
        self.adapter, self.policy, self.system_prompt = adapter, policy, system_prompt
        self.render, self.checkpoint, self.trace = render, checkpoint, trace
        self.recent_k, self.max_output_chars = recent_k, max_output_chars
        self.max_completion_tokens = max_completion_tokens

    def run(self, state: S) -> S:
        """Run until the episode completes, abstains, fails or hits a cap."""
        started, base_elapsed = time.monotonic(), state.elapsed_seconds
        if hasattr(self.adapter, "seek"):
            self.adapter.seek(state.model_calls)
        if state.pending is not None:
            self._execute_pending(state, raws=[], latency=None, resumed=True)
        while state.status == "running":
            state.elapsed_seconds = base_elapsed + time.monotonic() - started
            stop = self.policy.budget_stop(state)
            if stop is not None:
                self._finish(state, "stopped", stop)
                break
            try:
                decision, raws, latency, error = self._decide(state)
            except AdapterError as exc:
                self._event(state, {"event": "adapter_error", "message": str(exc)})
                self._finish(state, "failed", "adapter_error")
                break
            state.steps += 1
            if decision is None:
                state.malformed_actions += 1
                self._event(
                    state,
                    {
                        "event": "malformed",
                        "step": state.steps,
                        "error": error,
                        "raw_responses": raws,
                        "latency_s": latency,
                    },
                )
                self._save(state)
                continue
            state.pending = PendingAction(
                step=state.steps,
                summary=decision.summary,
                action=decision.action,
                idempotency_key=f"{state.episode_id}:{state.steps}",
            )
            self._save(state)  # write-ahead: the action is recorded before it runs
            self._execute_pending(state, raws=raws, latency=latency, resumed=False)
        state.elapsed_seconds = base_elapsed + time.monotonic() - started
        self._save(state)
        return state

    def _messages(self, state: BaseAgentState) -> list[dict[str, str]]:
        recent = state.history[-self.recent_k :] if self.recent_k > 0 else []
        user = self.render(state, recent) + "\n\n" + RESPONSE_FORMAT
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    def _call(self, state: BaseAgentState, messages: list[dict[str, str]]) -> tuple[str, float]:
        state.prompt_chars += sum(len(m["content"]) for m in messages)
        resp = self.adapter.complete(messages, self.max_completion_tokens)
        state.model_calls += 1
        state.token_usage.prompt += resp.prompt_tokens or 0
        state.token_usage.completion += resp.completion_tokens or 0
        state.reported_cost_usd += resp.cost_usd or 0.0
        return resp.text, resp.latency_s

    def _decide(
        self, state: BaseAgentState
    ) -> tuple[Decision | None, list[str], float, str | None]:
        messages = self._messages(state)
        text, latency = self._call(state, messages)
        decision, error = parse_decision(text, self.max_output_chars)
        raws = [text]
        if decision is None:  # one corrective re-prompt that includes the parse error
            state.reprompts += 1
            messages += [
                {"role": "assistant", "content": text[: self.max_output_chars]},
                {
                    "role": "user",
                    "content": f"Your reply could not be used: {error}. {RESPONSE_FORMAT}",
                },
            ]
            text, more = self._call(state, messages)
            raws.append(text)
            latency += more
            decision, error = parse_decision(text, self.max_output_chars)
        return decision, raws, latency, error

    def _execute_pending(
        self, state: S, raws: list[str], latency: float | None, resumed: bool
    ) -> None:
        pending = state.pending
        assert pending is not None
        done = next(
            (h for h in state.history if h.idempotency_key == pending.idempotency_key), None
        )
        if done is None:
            result, consumed = self.policy.execute(state, pending)
            state.remaining_tool_calls -= int(consumed)
            state.history.append(HistoryEntry(**pending.model_dump(), result=result))
        else:
            result = done.result
        spec = self.policy.registry.get(pending.action.tool)
        if result.ok and spec is not None and spec.terminal is not None:
            state.final = result.output
            self._finish(
                state, spec.terminal, "finished" if spec.terminal == "complete" else "abstained"
            )
        state.pending = None
        self._event(
            state,
            {
                "event": "step",
                "step": pending.step,
                "summary": pending.summary,
                "raw_responses": raws,
                "action": pending.action.model_dump(),
                "validation": "rejected" if result.error and result.error.code in REFUSED else "ok",
                "result": result.model_dump(),
                "resumed": resumed,
                "latency_s": latency,
                "tokens": state.token_usage.model_dump(),
                "remaining_tool_calls": state.remaining_tool_calls,
            },
        )
        self._save(state)

    def _finish(self, state: BaseAgentState, status: Status, reason: StopReason) -> None:
        state.status = status
        state.stop_reason = reason
        self._event(
            state, {"event": "stop", "status": status, "stop_reason": reason, "steps": state.steps}
        )

    def _event(self, state: BaseAgentState, event: dict) -> None:
        if self.trace is not None:
            self.trace.write({"episode_id": state.episode_id, **event})

    def _save(self, state: BaseAgentState) -> None:
        if self.checkpoint is not None:
            self.checkpoint.save(state)
