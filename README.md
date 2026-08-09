# Multi-agent supervision-checkpoint experiment

Experimental pipeline for a study on **error propagation in multi-agent LLM coding
systems**. A single parameterized LangGraph multi-agent coder (Planner, Coder,
Reviewer) solves LiveCodeBench tasks with o3-mini across **4 supervision
conditions**, reusing LiveCodeBench's own execution harness to grade code.

Research question: how does removing structured external supervision checkpoints
affect the **rate, depth, and magnitude** of error propagation?

## Conditions (the only thing that varies)

One graph, parameterized by `condition`. The plan/final gates are the manipulated
variable; everything else is byte-for-byte identical across conditions.

| condition    | plan gate | final gate |
|--------------|-----------|------------|
| `none`       | off       | off        |
| `plan_gate`  | **on**    | off        |
| `final_gate` | off       | **on**     |
| `both`       | **on**    | **on**     |

The **intrinsic execute-and-revise loop** (Coder revising off failing *visible*
tests) is part of the base system and runs under **every** condition, including
`none`. It is *not* the supervision variable — do not conflate the two.

## Design invariants

- **Reuse the LiveCodeBench harness** for execution. We call the low-level
  per-problem runner (`run_test` via `check_correctness`), never the batch CLI.
- **Per-iteration, per-test results.** The harness short-circuits at the first
  failing test, so we invoke it **once per test case** to recover a true passed
  count + failing indices + status (`ok`/`runtime_error`/`timeout`/`syntax_error`).
- **Visible (public) tests drive the loop; hidden (private) tests are evaluated once
  at the end** for true final correctness. They are kept strictly separate.
- **Everything is config-driven** (`configs/base.yaml`). No hardcoded experiment
  parameters.
- **Metrics are computed offline** (`scripts/metrics.py`) from logs — never inside
  the run. Logs contain everything needed to recompute every metric.

## Pinned versions

See [PINNED_VERSIONS.md](PINNED_VERSIONS.md). LiveCodeBench is pinned to commit
`28fef95`; the dataset is `livecodebench/code_generation_lite` `release_v5`, filtered
by `contest_date` to problems released after the model's knowledge cutoff (the
contamination-free design depends on this). `datasets` is pinned to `3.2.0` (later
versions dropped the loading-script support this dataset needs).

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e ./LiveCodeBench --no-deps   # harness only; we skip its torch/vllm deps
git -C LiveCodeBench checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
export OPENAI_API_KEY=sk-...                # required to run episodes 
```

`lcb_runner` is a namespace package; `mas/__init__.py` puts the vendored
`LiveCodeBench/` checkout on `sys.path`, so imports work regardless of the editable
install.

## Curate the dataset (once)

Downloads the release (cached by HuggingFace), filters to the configured difficulty +
contamination window, decompresses only the kept problems, and writes a small local
JSON cache under `data/`. The run loop reads that cache — no HuggingFace round-trip.

```bash
python -m mas.curate --config configs/base.yaml
```

## Run

First milestone — one task end-to-end across all 4 conditions:

```bash
python -m mas.batch --config configs/base.yaml
```

Sweep specific conditions / trials / tasks:

```bash
python -m mas.batch --config configs/base.yaml --conditions none both --trials 3
python -m mas.batch --config configs/base.yaml --tasks <question_id_1> <question_id_2>
```

One structured log per episode is written to `logs/<episode_id>.json`, plus an
`episodes.jsonl` index.

## Metrics (offline)

```bash
python scripts/metrics.py --logs logs                 # per condition
python scripts/metrics.py --logs logs --by-difficulty # per condition × difficulty
python scripts/metrics.py --logs logs --json report.json
```

Reports per condition: final hidden pass rate, initial defect probability, recovery
rate, mean error-propagation depth, mean error magnitude, iterations, token usage,
and the mean visible failing-count trajectory over iterations.

## Layout

```
configs/base.yaml        # all experiment parameters (condition, model, caps, dataset)
mas/
  config.py              # YAML -> typed config; with_condition() sweep helper
  dataset.py             # curate-once cache + fast load; public/private kept separate
  curate.py              # one-time curation CLI
  execution.py           # per-test wrapper around the LiveCodeBench harness
  llm.py                 # OpenAI reasoning-model client + token accounting
  prompts.py             # versioned prompts, identical across conditions
  agents.py              # Planner / Coder / Reviewer
  state.py               # typed graph state + run-time Context
  graph.py               # the single parameterized LangGraph
  episode_log.py         # structured per-episode logging
  run_episode.py         # one task × condition × trial
  batch.py               # batch runner CLI
scripts/metrics.py       # offline metrics
LiveCodeBench/           # pinned harness checkout
```

## Model settings

o3-mini is a reasoning model: `temperature`/`top_p` are unsupported and not sent; it
uses `max_completion_tokens` and `reasoning_effort` (held constant across conditions).
Every call logs the real settings used, including reasoning-token counts.

## Safety

Generated code is untrusted and always executed in a subprocess via the harness
(`multiprocessing.Process`, forced `fork` start method) with the configured timeout.
`reliability_guard()` disables destructive calls inside that subprocess. Note the
harness authors' caveat: this is a hardening layer, not a true sandbox.
```
