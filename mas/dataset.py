"""Dataset loading, curation, and task selection.

The full LiveCodeBench release is expensive to materialize: HuggingFace downloads
all shards (one-time, cached) and constructing every `CodeGenerationProblem`
base64/zlib/pickle-decompresses its private tests. We do NOT want to pay that on
every run.

Strategy:
  * `curate()` runs ONCE. It loads the raw release, filters the *raw* rows by
    difficulty + contamination date BEFORE constructing the heavy dataclass (so we
    only decompress the problems we keep), selects the task subset, and writes a
    small self-contained JSON cache.
  * `load_problems()` reads that JSON cache — no HuggingFace, no `datasets`, no
    decompressing hundreds of problems. This is what the run loop uses.

Public vs private tests are kept strictly separate throughout: `public_tests`
drive the intrinsic revise loop; `private_tests` are evaluated once at the end.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

from .config import DatasetConfig


# ---------------------------------------------------------------------------
# Special-judge exclusions
# ---------------------------------------------------------------------------
# LiveCodeBench grades by EXACT-MATCH against a single stored expected output
# (`grade_stdio` line-compare / `grade_call_based` `==`). It has no special-judge
# / checker hook — the `Test` schema is just (input, output, testtype). Problems
# whose statement permits MULTIPLE valid outputs ("print any of them") therefore
# get false-negative "wrong answer" verdicts whenever a correct solution emits a
# valid-but-different answer than the one stored. That noise is decided by the
# model's tie-breaking, which the supervision conditions can shift — so it
# confounds the independent variable. We drop these problems at curation.
#
# Each id was confirmed by reading the statement's output contract (not a keyword
# match): the quoted phrase is the sentence that establishes output multiplicity.
# Keep this list explicit and versioned so every exclusion is auditable.
SPECIAL_JUDGE_EXCLUSIONS: dict = {
    "abc333_e": 'output any min-K action sequence — "you may print any of them"',
    "abc343_e": '"If multiple solutions exist, you may print any of them."',
    "abc366_g": '"If multiple solutions exist, any of them will be accepted."',
    "abc373_g": '"If there are multiple solutions, you may print any of them."',
    "arc181_c": '"If there are multiple ways to satisfy the conditions, any ... accepted."',
    "arc183_d": '"If there are multiple solutions, you may print any of them."',
    "arc185_c": '"If there are multiple solutions, you may print any of them."',
}


@dataclass
class TestCase:
    input: str
    output: str
    testtype: str        # "stdin" | "functional"


@dataclass
class ProblemView:
    question_id: str
    title: str
    content: str
    platform: str
    difficulty: str
    contest_date: str
    starter_code: str
    fn_name: Optional[str]        # call-based function name, or None for stdin
    public_tests: list            # list[TestCase]
    private_tests: list           # list[TestCase]

    @property
    def is_call_based(self) -> bool:
        return self.fn_name is not None


# ---------------------------------------------------------------------------
# Cache path derivation
# ---------------------------------------------------------------------------

def cache_path_for(cfg: DatasetConfig) -> str:
    if cfg.cache_path:
        return cfg.cache_path
    diff = "-".join(cfg.difficulty) if cfg.difficulty else "all"
    start = cfg.contamination_start_date or "any"
    # JSONL (one problem per line) so the run loop can stream + early-stop instead of
    # loading the whole (potentially large) cache to select a few tasks.
    fname = f"curated_{cfg.release_version}_{diff}_from-{start}.jsonl"
    return os.path.join("data", fname)


# ---------------------------------------------------------------------------
# Curation (run once)
# ---------------------------------------------------------------------------

def _passes_raw_filters(row: dict, cfg: DatasetConfig) -> bool:
    if cfg.difficulty and row["difficulty"] not in cfg.difficulty:
        return False
    date = datetime.fromisoformat(row["contest_date"])
    if cfg.contamination_start_date:
        if date < datetime.strptime(cfg.contamination_start_date, "%Y-%m-%d"):
            return False
    if cfg.contamination_end_date:
        if date > datetime.strptime(cfg.contamination_end_date, "%Y-%m-%d"):
            return False
    return True


def _view_from_problem(p) -> ProblemView:
    def conv(tlist):
        return [
            TestCase(input=t.input, output=t.output, testtype=t.testtype.value)
            for t in tlist
        ]
    return ProblemView(
        question_id=p.question_id,
        title=p.question_title,
        content=p.question_content,
        platform=p.platform.value,
        difficulty=p.difficulty.value,
        contest_date=p.contest_date.isoformat(),
        starter_code=p.starter_code or "",
        fn_name=p.metadata.get("func_name", None),
        public_tests=conv(p.public_test_cases),
        private_tests=conv(p.private_test_cases),
    )


def curate(cfg: DatasetConfig, out_path: Optional[str] = None) -> list[ProblemView]:
    """Load the raw release, filter, select, and write the JSON cache. Run once."""
    # Heavy imports live here so the run loop never pulls in `datasets`.
    from datasets import load_dataset
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    out_path = out_path or cache_path_for(cfg)
    print(f"[curate] loading raw release {cfg.release_version} from HuggingFace ...")
    raw = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag=cfg.release_version,
        trust_remote_code=True,
    )

    # Filter raw rows FIRST (cheap string comparisons) so we only construct/decompress
    # the problems we actually keep.
    # Cache the FULL difficulty/date-filtered pool (ignore max_tasks/task_ids here) so
    # one curation yields a reusable pool; per-run selection happens in load_problems.
    kept_rows = [row for row in raw if _passes_raw_filters(row, cfg)]
    print(f"[curate] {len(kept_rows)} problems pass difficulty/date filter "
          f"(of {len(raw)} total)")

    # Drop special-judge problems the exact-match harness cannot judge soundly.
    excluded = [r for r in kept_rows if r["question_id"] in SPECIAL_JUDGE_EXCLUSIONS]
    kept_rows = [r for r in kept_rows if r["question_id"] not in SPECIAL_JUDGE_EXCLUSIONS]
    if excluded:
        print(f"[curate] excluded {len(excluded)} special-judge problem(s) "
              f"(multiple valid outputs, unjudgeable by exact-match):")
        for r in excluded:
            print(f"           - {r['question_id']}: {SPECIAL_JUDGE_EXCLUSIONS[r['question_id']]}")

    problems = [CodeGenerationProblem(**row) for row in kept_rows]
    views = [_view_from_problem(p) for p in problems]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for v in views:
            f.write(json.dumps(asdict(v)) + "\n")
    print(f"[curate] wrote {len(views)} problems -> {out_path}")
    return views


# ---------------------------------------------------------------------------
# Fast load (run loop)
# ---------------------------------------------------------------------------

def _view_from_cache(d: dict) -> ProblemView:
    return ProblemView(
        question_id=d["question_id"], title=d["title"], content=d["content"],
        platform=d["platform"], difficulty=d["difficulty"],
        contest_date=d["contest_date"], starter_code=d["starter_code"],
        fn_name=d["fn_name"],
        public_tests=[TestCase(**t) for t in d["public_tests"]],
        private_tests=[TestCase(**t) for t in d["private_tests"]],
    )


def load_problems(cfg: DatasetConfig, auto_curate: bool = True) -> list[ProblemView]:
    """Stream curated problems from the local JSONL cache, selecting only what the
    config asks for and stopping early. If the cache is absent, curate once."""
    path = cache_path_for(cfg)
    if not os.path.exists(path):
        if not auto_curate:
            raise FileNotFoundError(
                f"No curated cache at {path}. Run `python -m mas.curate --config <cfg>`."
            )
        curate(cfg, path)

    wanted = set(cfg.task_ids) if cfg.task_ids else None
    limit = None if wanted else cfg.max_tasks

    views: list[ProblemView] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            # Cheap id peek to avoid materializing tests for problems we skip.
            d = json.loads(line)
            # Guard against a stale cache curated before the special-judge exclusions
            # existed: never surface an unjudgeable problem to the run loop.
            if d["question_id"] in SPECIAL_JUDGE_EXCLUSIONS:
                continue
            if wanted is not None:
                if d["question_id"] in wanted:
                    views.append(_view_from_cache(d))
                    if len(views) == len(wanted):
                        break
            else:
                views.append(_view_from_cache(d))
                if limit is not None and len(views) >= limit:
                    break
    return views
