"""Tool registry, allowlist, argument validation and budget enforcement.

Every rule that limits what the agent can do lives here, outside the model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from .schemas import (
    Action,
    BaseAgentState,
    ErrorCode,
    PendingAction,
    StopReason,
    ToolError,
    ToolResult,
)


class ToolFailure(Exception):
    """Raised inside a tool or precheck to return a typed error to the model."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code: ErrorCode = code
        self.message = message


@dataclass(frozen=True)
class ToolSpec:
    """A finite, typed operation the model may request.

    Attributes:
        name: Tool name used in actions.
        description: One-line description shown to the model.
        args_model: Pydantic model that validates the arguments.
        fn: Callable ``(state, args) -> dict`` that performs the operation.
        precheck: Optional callable ``(state, args) -> None`` that raises
            ``ToolFailure`` to reject the call before ``fn`` runs.
        terminal: Status the episode takes when this tool succeeds, if any.
        side_effect: True if the tool changes anything outside the state.
    """

    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any, Any], dict[str, Any]]
    precheck: Callable[[Any, Any], None] | None = None
    terminal: Literal["complete", "abstained"] | None = None
    side_effect: bool = False


class ToolRegistry:
    """Holds tool specs and the allowlist for one episode configuration."""

    def __init__(self, specs: Iterable[ToolSpec], allowlist: Iterable[str] | None = None) -> None:
        self._specs = {s.name: s for s in specs}
        allowed = set(self._specs) if allowlist is None else set(allowlist)
        unknown = allowed - set(self._specs)
        if unknown:
            raise ValueError(f"allowlist names unknown tools: {sorted(unknown)}")
        self.allowed = frozenset(allowed)

    def get(self, name: str) -> ToolSpec | None:
        """Return the spec for ``name`` or None."""
        return self._specs.get(name)

    def describe(self) -> list[dict[str, Any]]:
        """JSON-schema description of the allowed tools, for the system prompt."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "arguments": s.args_model.model_json_schema(),
            }
            for name, s in sorted(self._specs.items())
            if name in self.allowed
        ]


class Budget(BaseModel):
    """Hard per-episode caps. ``None`` disables a cap."""

    max_steps: int = 20
    max_tool_calls: int = 12
    max_tokens: int | None = None
    max_wall_seconds: float | None = None


class Policy:
    """Validates and executes actions under a registry and a budget."""

    def __init__(self, registry: ToolRegistry, budget: Budget) -> None:
        self.registry = registry
        self.budget = budget

    def budget_stop(self, state: BaseAgentState) -> StopReason | None:
        """Return the stop reason if any cap is reached, else None."""
        b = self.budget
        if state.steps >= b.max_steps:
            return "max_steps"
        if state.remaining_tool_calls <= 0:
            return "max_tool_calls"
        if b.max_tokens is not None and state.token_usage.total >= b.max_tokens:
            return "max_tokens"
        if b.max_wall_seconds is not None and state.elapsed_seconds >= b.max_wall_seconds:
            return "max_wall_time"
        return None

    def validate(self, action: Action) -> tuple[ToolSpec, BaseModel] | ToolError:
        """Check the allowlist and argument schema without executing anything."""
        spec = self.registry.get(action.tool)
        if spec is None:
            return ToolError(code="unknown_tool", message=f"no tool named {action.tool!r}")
        if spec.name not in self.registry.allowed:
            return ToolError(code="not_allowed", message=f"tool {spec.name!r} is not enabled")
        try:
            args = spec.args_model.model_validate(action.arguments)
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
                for e in exc.errors()[:5]
            )
            return ToolError(code="invalid_arguments", message=detail[:400])
        return spec, args

    def execute(self, state: BaseAgentState, pending: PendingAction) -> tuple[ToolResult, bool]:
        """Run one pending action.

        Returns:
            The result and whether a tool call was consumed. Actions refused at
            validation do not consume a tool call; they still cost a step.
        """
        step, tool = pending.step, pending.action.tool
        checked = self.validate(pending.action)
        if isinstance(checked, ToolError):
            return ToolResult(step=step, tool=tool, ok=False, error=checked), False
        spec, args = checked
        if spec.side_effect:
            done = {h.idempotency_key: h.result for h in state.history}
            if pending.idempotency_key in done:
                return done[pending.idempotency_key], False
        try:
            if spec.precheck is not None:
                spec.precheck(state, args)
            output = spec.fn(state, args)
        except ToolFailure as exc:
            err = ToolError(code=exc.code, message=exc.message[:400])
            return ToolResult(step=step, tool=tool, ok=False, error=err), True
        except Exception as exc:  # noqa: BLE001, a tool bug must not crash the loop
            err = ToolError(code="tool_exception", message=type(exc).__name__)
            return ToolResult(step=step, tool=tool, ok=False, error=err), True
        return ToolResult(step=step, tool=tool, ok=True, output=output), True
