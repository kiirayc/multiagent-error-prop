from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class SupervisionCondition(str, Enum):
    NONE = "none"
    PLAN_GATE = "plan_gate"
    FINAL_GATE = "final_gate"
    PLAN_AND_FINAL_GATE = "plan_and_final_gate"


class IOCase(BaseModel):
    input: str
    output: str


class TaskSpec(BaseModel):
    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    problem_statement: str
    input_format: str = ""
    output_format: str = ""
    constraints: str = ""
    visible_tests: list[IOCase]
    hidden_tests: list[IOCase] = Field(default_factory=list)


class TestFailure(BaseModel):
    test_index: int
    input: str
    expected: str
    actual: str | None = None
    error: str | None = None
    failure_type: Literal[
        "wrong_answer",
        "runtime_error",
        "timeout",
        "syntax_error",
    ]


class ExecutionResult(BaseModel):
    passed: bool
    num_tests: int
    num_failed: int
    failures: list[TestFailure] = Field(default_factory=list)
    syntax_error: str | None = None
    runtime_error: str | None = None
    timeout: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)