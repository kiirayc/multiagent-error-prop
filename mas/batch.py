"""Batch runner CLI.

Runs episodes over a task set × condition(s) × trials, writing one structured log
per episode. Records abnormal terminations rather than crashing the batch.

    python -m mas.batch --config configs/base.yaml
    python -m mas.batch --config configs/base.yaml --conditions none both --trials 3
    python -m mas.batch --config configs/base.yaml --tasks <qid1> <qid2>
"""

from __future__ import annotations

import argparse
import random
import time

import os

from .config import Config, load_config, with_condition, CONDITIONS
from .dataset import load_problems
from .episode_log import write_episode_log, episode_log_path
from .run_episode import run_episode


def _select_problems(cfg: Config, task_ids):
    if task_ids:
        # Override config selection with an explicit CLI task list.
        import copy
        raw = copy.deepcopy(cfg.raw)
        raw["dataset"]["task_ids"] = list(task_ids)
        cfg = load_config_from_raw(raw)
    return load_problems(cfg.dataset), cfg


def load_config_from_raw(raw):
    from .config import _from_raw
    return _from_raw(raw)


def main():
    ap = argparse.ArgumentParser(description="Run multi-agent episodes in batch.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    help=f"Subset of {CONDITIONS}. Default: all four.")
    ap.add_argument("--trials", type=int, default=None,
                    help="Override config experiment.trials.")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Explicit question_ids (override dataset selection).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run episodes even if a log already exists. Default: skip "
                         "existing episode logs (resume-friendly).")
    args = ap.parse_args()

    base = load_config(args.config)
    for c in args.conditions:
        if c not in CONDITIONS:
            ap.error(f"unknown condition {c!r}; must be one of {CONDITIONS}")

    problems, base = _select_problems(base, args.tasks)
    trials = args.trials if args.trials is not None else base.experiment.trials
    random.seed(base.experiment.seed)

    print(f"Tasks: {[p.question_id for p in problems]}")
    print(f"Conditions: {args.conditions} | trials: {trials} | "
          f"log dir: {base.logging.dir}")

    n_ok = n_err = n_skip = 0
    for problem in problems:
        for condition in args.conditions:
            cfg = with_condition(base, condition)
            for trial in range(trials):
                episode_id = f"{problem.question_id}__{condition}__t{trial}"
                if not args.overwrite and os.path.exists(
                        episode_log_path(base.logging.dir, episode_id)):
                    n_skip += 1
                    print(f"[skip] {episode_id} (log exists; use --overwrite to redo)")
                    continue
                t0 = time.time()
                # Defense-in-depth: run_episode already records graph/model errors in
                # the log; this guard also survives failures in log-building itself so a
                # single bad episode never aborts the whole sweep.
                try:
                    log = run_episode(cfg, problem, trial)
                    path = write_episode_log(log, cfg.logging.dir)
                    hid = log.final_hidden_result or {}
                    if log.error:
                        n_err += 1
                    else:
                        n_ok += 1
                    status = "ERR" if log.error else "ok"
                    print(f"[{status}] {log.episode_id} "
                          f"hidden={hid.get('passed')}/{hid.get('total')} "
                          f"iters={len(log.history)} "
                          f"tok={log.token_totals.get('total_tokens')} "
                          f"{time.time()-t0:.1f}s -> {path}"
                          + (f"  ERROR: {log.error}" if log.error else ""))
                except Exception as e:
                    n_err += 1
                    print(f"[ERR] {problem.question_id}__{condition}__t{trial} "
                          f"batch-level failure: {type(e).__name__}: {e}")

    print(f"\nDone. episodes ok={n_ok} err={n_err} skipped={n_skip}")


if __name__ == "__main__":
    main()
