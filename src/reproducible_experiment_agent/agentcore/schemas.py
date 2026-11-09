"""Typed records shared by the agent loop, the policy layer and the tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ErrorCode = Literal[
    "unknown_tool",
    "not_allowed",
    "invalid_arguments",
    "not_found",
    "out_of_range",
    "precondition_failed",
    "budget_exhausted",
    "malformed_action",
    "tool_exception",
]
"""Closed set of error codes a tool call can produce."""

StopReason = Literal[
    "finished",
    "abstained",
    "max_steps",
    "max_tool_calls",
    "max_tokens",
    "max_wall_time",
    "adapter_error",
]

Status = Literal["running", "complete", "abstained", "failed", "stopped"]


class Action(BaseModel):
    """One tool invocation requested by the model."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """The single JSON object the model must return each step."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(max_length=400)
    action: Action


class ToolError(BaseModel):
    """A typed, recoverable failure returned to the model as an observation."""

    code: ErrorCode
    message: str


class ToolResult(BaseModel):
    """Outcome of executing (or refusing) one action."""

    step: int
    tool: str
    ok: bool
    output: dict[str, Any] | None = None
    error: ToolError | None = None


class TokenUsage(BaseModel):
    """Model token counts as reported by the adapter."""

    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt + self.completion


class PendingAction(BaseModel):
    """A validated action written ahead to the checkpoint before execution."""

    step: int
    summary: str
    action: Action
    idempotency_key: str


class HistoryEntry(PendingAction):
    """An executed (or refused) action together with its result."""

    result: ToolResult


class BaseAgentState(BaseModel):
    """Generic, serializable episode state. Projects subclass it.

    Only fields that the model may observe belong here. Gold answers and
    evaluator data never enter the state.
    """

    episode_id: str
    steps: int = 0
    remaining_tool_calls: int
    model_calls: int = 0
    malformed_actions: int = 0
    reprompts: int = 0
    prompt_chars: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    reported_cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    checkpoint_version: int = 0
    status: Status = "running"
    stop_reason: StopReason | None = None
    pending: PendingAction | None = None
    history: list[HistoryEntry] = Field(default_factory=list)
    final: dict[str, Any] | None = None

    def observable(self) -> dict[str, Any]:
        """Return the compact state summary shown to the model each step."""
        return {"steps_used": self.steps, "remaining_tool_calls": self.remaining_tool_calls}
