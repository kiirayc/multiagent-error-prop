"""Execution node: the single place code is ever checked against tests.

We REUSE the LiveCodeBench harness — `run_test` wrapped in a subprocess by
`check_correctness` (mirroring upstream's `multiprocessing.Process` pattern, which
is required because `reliability_guard()` irreversibly mutates the host process).
We never hand-roll a code executor, stdin/stdout parsing, or the checker.

The harness SHORT-CIRCUITS at the first failing test, so to recover a true
passed-count and the full set of failing indices (needed for the error-magnitude
metric) we invoke the harness once per individual test case (`per_test=True`).

The result is structured — never a boolean — exposing total / passed /
failing_indices / status, plus per-test detail and the harness's failure metadata.
"""

from __future__ import annotations

import json
import multiprocessing as _mp
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

# The LiveCodeBench harness runs `run_test` in a `multiprocessing.Process` and was
# written assuming the `fork` start method (Linux default). On macOS the default
# switched to `spawn`, which re-imports the entrypoint module in each child and
# breaks the harness. Force `fork` to restore the intended semantics. Entrypoint
# scripts must still guard top-level code under `if __name__ == "__main__"`.
try:
    _mp.set_start_method("fork", force=True)
except (RuntimeError, ValueError):
    pass

from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness


class ExecStatus(str, Enum):
    OK = "ok"                    # executed cleanly (test may still be wrong answer)
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    SYNTAX_ERROR = "syntax_error"


# Severity ordering for aggregating per-test statuses into a set-level status.
_SEVERITY = {
    ExecStatus.OK: 0,
    ExecStatus.TIMEOUT: 1,
    ExecStatus.RUNTIME_ERROR: 2,
    ExecStatus.SYNTAX_ERROR: 3,
}


@dataclass
class TestSetResult:
    """Structured outcome of running one candidate against one test set."""
    total: int
    passed: int
    failing_indices: list          # indices (0-based) of tests that did not pass
    status: str                    # set-level ExecStatus value (most severe)
    per_test_status: list          # ExecStatus value per test, in order
    per_test_passed: list          # bool per test, in order
    first_failure_metadata: Optional[dict] = None  # harness metadata for first fail
    wall_time_s: float = 0.0
    test_set: str = ""             # "visible" | "hidden"

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total

    @property
    def num_failing(self) -> int:
        return len(self.failing_indices)

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_result_code(code_val) -> tuple[bool, ExecStatus]:
    """Map a single harness result entry to (passed, status)."""
    # Normalize numpy scalars just in case.
    try:
        import numpy as np  # local import; harness already depends on numpy
        if isinstance(code_val, (np.bool_,)):   # converts NumPy boolean scalar into Python bool
            code_val = bool(code_val)
        elif isinstance(code_val, np.ndarray):  # extracts scalar from the NumPy array 
            code_val = code_val.item(0)
    except Exception:
        pass

    if code_val is True or code_val == 1:
        return True, ExecStatus.OK
    # Wrong answer: ran cleanly, output mismatched. NOTE the two graders encode WA
    # differently — grade_stdio appends -2, grade_call_based appends a bare False.
    if code_val is False or code_val == -2:
        return False, ExecStatus.OK
    if code_val in (-3, -1):           # time limit / global timeout
        return False, ExecStatus.TIMEOUT
    # -4 (runtime/compile error), -5 (test-runner error), anything else
    return False, ExecStatus.RUNTIME_ERROR


def _syntax_ok(code: str) -> bool:
    try:
        compile(code, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False
    except Exception:
        # Non-syntax compile problems (rare) are left for the harness to surface.
        return True


def _run_one_case(code: str, inp: str, out: str, fn_name, timeout: int):
    """Run a single test case through the harness subprocess. Returns (passed, status,
    metadata). Robust to the rare empty-metadata edge case in check_correctness."""
    sample = {
        "input_output": json.dumps(
            {"inputs": [inp], "outputs": [out], "fn_name": fn_name}
        )
    }
    try:
        res, metadata = check_correctness(sample, code, timeout=timeout, debug=False)
    except Exception as e:  # subprocess/manager edge cases -> treat as runtime error
        return False, ExecStatus.RUNTIME_ERROR, {
            "error": repr(e), "error_code": -5, "error_message": "TestRunnerError",
        }
    code_val = res[0] if isinstance(res, list) and res else -4  # -4 for runtime error
    passed, status = _classify_result_code(code_val)
    return passed, status, (metadata if isinstance(metadata, dict) else {})


def run_tests(
    code: str,
    tests: list,               # list of objects/dicts with .input/.output (see dataset.py)
    fn_name,                   # str for call-based, None for stdin
    timeout: int,
    test_set: str = "",
    per_test: bool = True,
) -> TestSetResult:
    """Run candidate `code` against `tests`, returning a structured TestSetResult.

    `tests` items must expose `input` and `output` (strings) — dataset.py normalizes
    LiveCodeBench `Test` objects into that shape.
    """
    start = time.time()
    inputs = [t.input for t in tests]
    outputs = [t.output for t in tests]
    n = len(inputs)

    # Syntax pre-check: the harness collapses SyntaxError into a generic runtime
    # error (-4), so we detect it here to expose the `syntax_error` status.
    if not _syntax_ok(code):
        return TestSetResult(
            total=n, passed=0, failing_indices=list(range(n)),
            status=ExecStatus.SYNTAX_ERROR.value,
            per_test_status=[ExecStatus.SYNTAX_ERROR.value] * n,
            per_test_passed=[False] * n,
            first_failure_metadata={"error_code": -4, "error_message": "SyntaxError"},
            wall_time_s=time.time() - start, test_set=test_set,
        )

    per_test_passed: list = []
    per_test_status: list = []
    first_failure_metadata: Optional[dict] = None

    if per_test:
        for inp, out in zip(inputs, outputs):
            passed, status, metadata = _run_one_case(code, inp, out, fn_name, timeout)
            per_test_passed.append(passed)
            per_test_status.append(status.value)
            if not passed and first_failure_metadata is None:
                first_failure_metadata = metadata
    else:
        # Full-set short-circuit path (config option). One harness call; on failure
        # we only know the prefix of passing tests. Kept for completeness.
        sample = {
            "input_output": json.dumps(
                {"inputs": inputs, "outputs": outputs, "fn_name": fn_name}
            )
        }
        try:
            res, metadata = check_correctness(sample, code, timeout=timeout, debug=False)
        except Exception as e:
            res, metadata = [-5] * n, {"error": repr(e), "error_message": "TestRunnerError"}
        for i in range(n):
            code_val = res[i] if i < len(res) else -2  # untested-after-failure -> fail
            passed, status = _classify_result_code(code_val)
            per_test_passed.append(passed)
            per_test_status.append(status.value)
            if not passed and first_failure_metadata is None:
                first_failure_metadata = metadata if isinstance(metadata, dict) else {}

    passed_count = sum(1 for p in per_test_passed if p)
    failing_indices = [i for i, p in enumerate(per_test_passed) if not p]
    set_status = max(
        (ExecStatus(s) for s in per_test_status),
        key=lambda s: _SEVERITY[s],
        default=ExecStatus.OK,
    ).value

    return TestSetResult(
        total=n, passed=passed_count, failing_indices=failing_indices,
        status=set_status, per_test_status=per_test_status,
        per_test_passed=per_test_passed,
        first_failure_metadata=first_failure_metadata,
        wall_time_s=time.time() - start, test_set=test_set,
    )
