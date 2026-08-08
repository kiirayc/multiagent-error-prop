# Pinned versions (reproducibility)

The contamination-free design depends on problem release dates relative to the
model's knowledge cutoff, so these pins are load-bearing — do not float them.

## LiveCodeBench harness

- Repo: https://github.com/LiveCodeBench/LiveCodeBench
- Pinned commit: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` (2025-07-15)
- No upstream git tags exist; we pin the commit SHA.
- Installed editable **without its deps** (`pip install -e ./LiveCodeBench --no-deps`)
  because its `pyproject` pulls `torch`/`vllm`/provider SDKs we do not use. We drive
  the model through our own LangGraph agents and only reuse the *execution harness*
  (`lcb_runner.evaluation.testing_util.run_test` via `check_correctness`) and the
  problem dataclass (`lcb_runner.benchmarks.code_generation.CodeGenerationProblem`).

## Dataset

- HuggingFace: `livecodebench/code_generation_lite`
- Release version: `release_v5` (May 2023–Jan 2025, 880 problems), set in config.
- Contamination filter: `contest_date >= dataset.contamination_start_date` (config),
  keeping only problems released after o3-mini's knowledge cutoff.
- **o3-mini knowledge cutoff: 2023-10-01** — verified 2026-07 against OpenAI's model
  docs (https://platform.openai.com/docs/models/o3-mini, redirecting to
  developers.openai.com; context window 200K, max output 100K).
- `contamination_start_date` default: **`2023-11-01`** — one month after the cutoff,
  a small buffer against cutoff imprecision while keeping the maximal
  contamination-free pool. Strict-cutoff alternative: `2023-10-01`. (Changing this
  changes the derived cache filename, so re-run `python -m mas.curate`.)

## Key package versions (venv)

- Python 3.13.2
- langgraph 1.2.9
- langchain-core 1.4.9
- openai (>=1.59.6)
- datasets 3.2.0+

Recompute exact versions with `pip freeze` inside `.venv`.
