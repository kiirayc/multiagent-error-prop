from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from src.schemas import ExecutionResult, IOCase, TestFailure

# Important note: this is okay for local research prototypes,
# but not secure against malicious generated code. Later, use Docker sandboxing.

def normalize_output(text: str) -> str:
    return text.strip()


def run_code(
    code: str,
    tests: list[IOCase],
    timeout_s: int = 2,
) -> ExecutionResult:
    failures: list[TestFailure] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / "solution.py"
        code_path.write_text(code, encoding="utf-8")

        for index, test in enumerate(tests):
            try:
                completed = subprocess.run(
                    [sys.executable, str(code_path)],
                    input=test.input,
                    text=True,
                    capture_output=True,
                    timeout=timeout_s,
                )

            except subprocess.TimeoutExpired:
                failures.append(
                    TestFailure(
                        test_index=index,
                        input=test.input,
                        expected=test.output,
                        actual=None,
                        error=f"Timed out after {timeout_s} seconds",
                        failure_type="timeout",
                    )
                )
                continue

            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                failure_type = (
                    "syntax_error"
                    if "SyntaxError" in stderr
                    else "runtime_error"
                )

                failures.append(
                    TestFailure(
                        test_index=index,
                        input=test.input,
                        expected=test.output,
                        actual=completed.stdout,
                        error=stderr,
                        failure_type=failure_type,
                    )
                )
                continue

            expected = normalize_output(test.output)
            actual = normalize_output(completed.stdout)

            if actual != expected:
                failures.append(
                    TestFailure(
                        test_index=index,
                        input=test.input,
                        expected=expected,
                        actual=actual,
                        error=None,
                        failure_type="wrong_answer",
                    )
                )

    return ExecutionResult(
        passed=len(failures) == 0,
        num_tests=len(tests),
        num_failed=len(failures),
        failures=failures,
        timeout=any(f.failure_type == "timeout" for f in failures),
        # TODO: Give the Reviewer a compact list of all failures for better agent feedback.
        syntax_error=next(
            (f.error for f in failures if f.failure_type == "syntax_error"),
            None,
        ),
        runtime_error=next(
            (f.error for f in failures if f.failure_type == "runtime_error"),
            None,
        ),
    )
