"""The three agent roles. All go through the same LLMClient and the same pinned
model. Prompts come from `prompts` (identical across conditions)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import prompts
from .execution import TestSetResult
from .llm import LLMClient
from .dataset import ProblemView


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull the last fenced code block; fall back to the raw text."""
    blocks = _CODE_FENCE.findall(text)
    if blocks:
        return blocks[-1].strip()  # final code treated as answer
    return text.strip()


# ---------------------------------------------------------------------------
# Planner / Coder
# ---------------------------------------------------------------------------

def run_planner(llm: LLMClient, problem: ProblemView) -> str:
    res = llm.complete("planner", prompts.planner_messages(problem))
    return res.text.strip()


def run_coder_initial(llm: LLMClient, problem: ProblemView, plan: str) -> str:
    res = llm.complete("coder", prompts.coder_initial_messages(problem, plan))
    return extract_code(res.text)


def run_coder_revise(
    llm: LLMClient, problem: ProblemView, plan: str, prev_code: str, feedback: str
) -> str:
    res = llm.complete(
        "coder", prompts.coder_revise_messages(problem, plan, prev_code, feedback))
    return extract_code(res.text)


# ---------------------------------------------------------------------------
# Reviewer (gates)
# ---------------------------------------------------------------------------

@dataclass
class ReviewOutcome:
    passed: bool
    feedback: str
    raw: str


_VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|FAIL)", re.IGNORECASE)
_FEEDBACK_RE = re.compile(r"FEEDBACK:\s*(.*)", re.IGNORECASE | re.DOTALL)


def _parse_review(text: str) -> ReviewOutcome:
    m = _VERDICT_RE.search(text)
    # Default to PASS if the reviewer's verdict is unparseable, so a malformed
    # review never silently blocks the pipeline (logged via raw text either way).
    passed = True   # NOTE this is a failed-open design
    if m:
        passed = m.group(1).upper() == "PASS"
    fb = _FEEDBACK_RE.search(text)
    feedback = fb.group(1).strip() if fb else text.strip()
    return ReviewOutcome(passed=passed, feedback=feedback, raw=text)


def run_reviewer_plan(llm: LLMClient, problem: ProblemView, plan: str) -> ReviewOutcome:
    res = llm.complete("reviewer", prompts.reviewer_plan_messages(problem, plan))
    return _parse_review(res.text)


def run_reviewer_code(llm: LLMClient, problem: ProblemView, code: str) -> ReviewOutcome:
    res = llm.complete("reviewer", prompts.reviewer_code_messages(problem, code))
    return _parse_review(res.text)


# ---------------------------------------------------------------------------
# Failing-test feedback formatting (for the intrinsic revise loop)
# ---------------------------------------------------------------------------

def format_test_feedback(result: TestSetResult, tests: list, max_examples: int = 3) -> str:
    """Build revision feedback from a visible-test result. Visible tests are public,
    so it is legitimate to show the model the failing inputs/expected outputs."""
    lines = [
        f"Status: {result.status}. "
        f"Passed {result.passed}/{result.total} visible tests. "
        f"Failing test indices: {result.failing_indices}.",
    ]
    if result.first_failure_metadata:
        msg = result.first_failure_metadata.get("error_message")
        if msg:
            lines.append(f"First failure: {msg}")
    shown = 0
    for idx in result.failing_indices:
        if shown >= max_examples:
            break
        if idx < len(tests):
            t = tests[idx]
            lines.append(
                f"- Test {idx}: input={t.input!r} expected_output={t.output!r}")
            shown += 1
    return "\n".join(lines)
