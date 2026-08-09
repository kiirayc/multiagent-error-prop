"""Versioned prompts. IDENTICAL across all conditions — prompts are a controlled
variable. Bump PROMPT_VERSION on any change so logs pin exactly which text was used.

Three roles:
  * Planner  : problem -> algorithmic plan
  * Coder    : plan (+ prior code + failing-test feedback) -> code
  * Reviewer : rubric evaluator at the gates -> pass/fail + feedback
"""

from __future__ import annotations

PROMPT_VERSION = "v3"

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are the Planner in a multi-agent competitive-programming \
system. Given a programming problem, produce a concise, concrete algorithmic plan \
that a separate Coder agent will implement. Do NOT write code. Focus on: the core \
algorithm and why it is correct, the key data structures, edge cases, and the \
input/output format and complexity. Be specific and implementable."""

PLANNER_USER = """Problem title: {title}

Problem statement:
{content}

Input/output format: {io_format}
{starter}
Write the algorithmic plan."""


def planner_messages(problem) -> list:
    io_format = (
        f"Call-based: implement the function `{problem.fn_name}`."
        if problem.is_call_based
        else "Standard input/output (read from stdin, write to stdout)."
    )
    starter = (
        f"\nStarter code to complete:\n```python\n{problem.starter_code}\n```\n"
        if problem.starter_code.strip()
        else ""
    )
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": PLANNER_USER.format(
            title=problem.title, content=problem.content,
            io_format=io_format, starter=starter)},
    ]


# ---------------------------------------------------------------------------
# Coder
# ---------------------------------------------------------------------------

CODER_SYSTEM = """You are the Coder in a multi-agent competitive-programming system. \
Implement a correct, efficient Python 3 solution following the given plan. Output \
ONLY a single Python code block — no prose. \
For standard-input problems, read from stdin and print to stdout. \
For call-based problems, define the required function/class exactly as specified \
(LeetCode-style `class Solution` when starter code indicates it)."""

CODER_USER_INITIAL = """Problem title: {title}

Problem statement:
{content}

Input/output format: {io_format}
{starter}
Algorithmic plan:
{plan}

Write the complete Python solution as a single code block."""

CODER_USER_REVISE = """Your previous solution failed some visible tests. Revise it.

Problem title: {title}

Problem statement:
{content}

Input/output format: {io_format}
{starter}
Algorithmic plan:
{plan}

Your previous code:
```python
{prev_code}
```

Execution feedback on visible tests:
{feedback}

Fix the issues and output the complete corrected Python solution as a single code \
block."""


def coder_initial_messages(problem, plan: str) -> list:
    io_format = (
        f"Call-based: implement the function `{problem.fn_name}`."
        if problem.is_call_based
        else "Standard input/output (read from stdin, write to stdout)."
    )
    starter = (
        f"\nStarter code to complete:\n```python\n{problem.starter_code}\n```\n"
        if problem.starter_code.strip()
        else ""
    )
    return [
        {"role": "system", "content": CODER_SYSTEM},
        {"role": "user", "content": CODER_USER_INITIAL.format(
            title=problem.title, content=problem.content,
            io_format=io_format, starter=starter, plan=plan)},
    ]


def coder_revise_messages(problem, plan: str, prev_code: str, feedback: str) -> list:
    io_format = (
        f"Call-based: implement the function `{problem.fn_name}`."
        if problem.is_call_based
        else "Standard input/output (read from stdin, write to stdout)."
    )
    starter = (
        f"\nStarter code to complete:\n```python\n{problem.starter_code}\n```\n"
        if problem.starter_code.strip()
        else ""
    )
    return [
        {"role": "system", "content": CODER_SYSTEM},
        {"role": "user", "content": CODER_USER_REVISE.format(
            title=problem.title, content=problem.content, io_format=io_format,
            starter=starter, plan=plan, prev_code=prev_code, feedback=feedback)},
    ]


# ---------------------------------------------------------------------------
# Reviewer (gates only)
# ---------------------------------------------------------------------------

REVIEWER_SYSTEM = """You are the Reviewer, an ADVERSARIAL quality gate in a \
multi-agent competitive-programming system. Your job is to catch a defect in the \
artifact (a plan or a solution) BEFORE it reaches the hidden test suite. These are \
"hard" problems: assume the artifact has a subtle flaw until you have positively \
verified otherwise. A lenient pass is a failure of YOUR job.

Do not skim. Work the rubric checks below concretely, in this order:
- COMPLEXITY: derive the largest inputs the constraints allow, and estimate the \
artifact's worst-case time and space. If it could exceed the limits on that input \
(as a rule of thumb, >~10^8 basic operations), that is a FAIL — likely Time Limit \
Exceeded. This is the single most common defect; check it first and explicitly.
- TRACE: work through one sample AND one adversarial edge case you construct \
yourself (empty / minimum / maximum size, ties, duplicates, negatives, zero, a \
single element, a no-solution input, potential overflow). Confirm the artifact \
actually produces the correct result — do not accept it merely because it "looks \
reasonable".
- CONTRACT: check the exact input/output contract — stdin/stdout vs the required \
function or class signature, the print format and separators, modulo, and the \
return type.
- RUNTIME SAFETY: look for undefined names, off-by-one and index/None errors, a \
wrong default or sentinel return, an unhandled "no answer" branch, and recursion \
depth.

Decision rule (be discriminating, not lenient):
- FAIL if you find ANY concrete defect, OR if a high-risk item — worst-case \
complexity against the constraints, or correctness on an edge case — cannot be \
POSITIVELY verified. Name the specific input or rubric item and the fix.
- PASS only if every rubric item positively checks out AND you could not construct \
a single failing input.

Respond in EXACTLY this format (the VERDICT line is mandatory and must read PASS or \
FAIL):

VERDICT: PASS or FAIL
FEEDBACK: <specific and actionable: name the failing input or rubric item and the \
fix, or, if passing, state what you verified and the worst-case complexity you \
derived>"""

PLAN_RUBRIC = """Rubric for an algorithmic plan (competitive programming):
1. CONCRETE ALGORITHM: names a specific, implementable approach — not a restatement \
of the problem and not a vague gesture ("use DP") that hides the hard step.
2. CORRECTNESS: the approach actually solves the problem, with a stated reason why; \
no hand-waved step that silently skips the crux.
3. COMPLEXITY vs CONSTRAINTS: the worst-case time/space implied by the plan fits the \
input limits. A brute-force or super-polynomial approach on large limits is a FAIL \
(Time Limit Exceeded risk).
4. EDGE CASES: identifies the tricky cases the statement implies (min/max size, \
empty, ties, no-solution, negatives, overflow, modulo).
5. I/O CONTRACT: matches the required format — function signature vs stdin/stdout, \
return type, and any modulo."""

CODE_RUBRIC = """Rubric for a completed solution (competitive programming):
1. CORRECT ALGORITHM: trace a sample and one adversarial edge case; the code must \
produce the right result, not merely read as plausible.
2. COMPLEXITY vs CONSTRAINTS: worst-case time/space must fit the limits on the \
largest allowed input — no accidental O(n^2)/exponential behavior where n is large \
(Time Limit Exceeded).
3. I/O CONTRACT: reads and writes exactly as required (stdin/stdout OR the exact \
function/class signature); correct print format, separators, return type, and modulo.
4. RUNTIME SAFETY: no undefined names, index/None errors, wrong default or sentinel \
return, unhandled "no answer" path, or recursion-depth blowup.
5. EDGE CASES: handles empty / minimum / maximum, duplicates, ties, negatives, zero, \
a single element, and overflow."""

REVIEWER_PLAN_USER = """{rubric}

Problem title: {title}
Problem statement:
{content}

Plan to review:
{plan}

Evaluate the plan against the rubric."""

REVIEWER_CODE_USER = """{rubric}

Problem title: {title}
Problem statement:
{content}

Solution to review:
```python
{code}
```

Evaluate the solution against the rubric."""


def reviewer_plan_messages(problem, plan: str) -> list:
    return [
        {"role": "system", "content": REVIEWER_SYSTEM},
        {"role": "user", "content": REVIEWER_PLAN_USER.format(
            rubric=PLAN_RUBRIC, title=problem.title,
            content=problem.content, plan=plan)},
    ]


def reviewer_code_messages(problem, code: str) -> list:
    return [
        {"role": "system", "content": REVIEWER_SYSTEM},
        {"role": "user", "content": REVIEWER_CODE_USER.format(
            rubric=CODE_RUBRIC, title=problem.title,
            content=problem.content, code=code)},
    ]
