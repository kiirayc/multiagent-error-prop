"""Typed LangGraph state and the run-time Context.

State is kept plain/serializable (dicts, lists, primitives) so it can be dropped
straight into logs. Heavy, non-serializable collaborators (LLM client, config,
problem) live in a Context object passed to node closures — not in state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

from .config import Config
from .dataset import ProblemView
from .llm import LLMClient


@dataclass
class Context:
    """Everything the nodes need that is NOT graph state."""
    cfg: Config
    llm: LLMClient
    problem: ProblemView
    plan_gate_active: bool
    final_gate_active: bool


class IterationRecord(TypedDict):
    iteration: int
    phase: str                 # "initial" | "revise_tests" | "revise_final_gate"
    code: str
    visible_result: dict       # TestSetResult.to_dict()
    in_error_state: bool       # True iff visible tests not all passed
    num_failing: int


class GateEvent(TypedDict):
    gate: str                  # "plan" | "final"
    attempt: int
    verdict: str               # "pass" | "fail"
    feedback: str


class GraphState(TypedDict, total=False):
    # Artifacts
    plan: str
    code: str
    # Loop bookkeeping
    iteration: int             # number of visible-test executions so far
    phase: str                 # phase that produced the current code
    pending_feedback: str      # feedback fed to the next Coder revision
    # Results
    last_visible_result: dict  # most recent TestSetResult.to_dict()
    hidden_result: Optional[dict]
    # Gate bookkeeping
    plan_gate_retries: int
    final_gate_retries: int
    gate_events: list          # list[GateEvent]
    # History for offline metrics
    history: list              # list[IterationRecord]
    # Terminal bookkeeping
    termination: str           # why the episode ended
