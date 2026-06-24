from src.harness import run_code
from src.schemas import IOCase


def test_correct_solution_passes():
    code = """
a, b = map(int, input().split())
print(a + b)
"""
    tests = [
        IOCase(input="2 3\n", output="5\n"),
        IOCase(input="-4 10\n", output="6\n"),
    ]

    result = run_code(code, tests)

    assert result.passed is True
    assert result.num_failed == 0


def test_wrong_answer_fails():
    code = """
a, b = map(int, input().split())
print(a - b)
"""
    tests = [
        IOCase(input="2 3\n", output="5\n"),
    ]

    result = run_code(code, tests)

    assert result.passed is False
    assert result.num_failed == 1
    assert result.failures[0].failure_type == "wrong_answer"


def test_runtime_error_fails():
    code = """
raise ValueError("boom")
"""
    tests = [
        IOCase(input="", output="anything\n"),
    ]

    result = run_code(code, tests)

    assert result.passed is False
    assert result.failures[0].failure_type == "runtime_error"


def test_timeout_fails():
    code = """
while True:
    pass
"""
    tests = [
        IOCase(input="", output="")
    ]

    result = run_code(code, tests, timeout_s=1)

    assert result.passed is False
    assert result.failures[0].failure_type == "timeout"