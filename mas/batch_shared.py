"""Shared-start batch runner.

Variant of `mas.batch` for isolating the gate effect from generation noise. For each
task it generates ONE baseline artifact with gates OFF — a plan and an initial code —
then runs all requested conditions from that IDENTICAL warm start (default 1 trial).

Because o3-mini has no sampling controls and code execution is deterministic, this
makes `none` and `final_gate` byte-identical up to the final gate, and lets the plan
gate branch off a shared plan: any difference in outcomes is attributable to a gate
firing, not to two conditions happening to generate different code.

    python -m mas.batch_shared --config configs/iter10_shared.yaml
    python -m mas.batch_shared --config configs/iter10_shared.yaml --conditions none both

The shared start for each task is recorded under `<log_dir>/_shared_start/<task>.json`
(plan, code, per-call token usage) so warm-start cost stays fully accountable offline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict

from . import agents
from .config import Config, load_config, with_condition, CONDITIONS
from .dataset import ProblemView, load_problems
from .episode_log import write_episode_log, episode_log_path
from .llm import LLMClient
from .run_episode import run_episode


def generate_shared_start(cfg: Config, problem: ProblemView):
    """Gates-OFF baseline: one plan + one initial code, shared by every condition.

    Returns (plan, code, llm) where `llm` carries the warm-start's own token usage.
    """
    llm = LLMClient(cfg.model)
    plan = agents.run_planner(llm, problem)
    code = agents.run_coder_initial(llm, problem, plan)
    return plan, code, llm


def _write_shared_start(log_dir: str, problem: ProblemView, plan: str, code: str,
                        llm: LLMClient, conditions, trials: int, wall_s: float) -> str:
    d = os.path.join(log_dir, "_shared_start")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{problem.question_id}.json")
    record = {
        "task_id": problem.question_id,
        "title": problem.title,
        "generated_with": {"plan_gate": False, "final_gate": False},
        "seeds_episodes": [f"{problem.question_id}__{c}__t{t}"
                           for c in conditions for t in range(trials)],
        "plan": plan,
        "code": code,
        "llm_calls": [asdict(c) for c in llm.calls],
        "token_totals": llm.token_totals(),
        "wall_time_s": wall_s,
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def _shared_start_path(log_dir: str, task_id: str) -> str:
    return os.path.join(log_dir, "_shared_start", f"{task_id}.json")


def main():
    ap = argparse.ArgumentParser(
        description="Run all conditions from one shared per-task warm start.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    help=f"Subset of {CONDITIONS}. Default: all four.")
    ap.add_argument("--trials", type=int, default=None,
                    help="Override config experiment.trials. Default: config value.")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Explicit question_ids (override dataset selection).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run episodes (and regenerate shared starts) even if logs "
                         "already exist. Default: skip existing (resume-friendly).")
    args = ap.parse_args()

    base = load_config(args.config)
    for c in args.conditions:
        if c not in CONDITIONS:
            ap.error(f"unknown condition {c!r}; must be one of {CONDITIONS}")

    if args.tasks:
        import copy
        from .config import _from_raw
        raw = copy.deepcopy(base.raw)
        raw["dataset"]["task_ids"] = list(args.tasks)
        base = _from_raw(raw)

    problems = load_problems(base.dataset)
    trials = args.trials if args.trials is not None else base.experiment.trials
    random.seed(base.experiment.seed)

    print(f"Tasks: {[p.question_id for p in problems]}")
    print(f"Conditions: {args.conditions} | trials: {trials} | "
          f"shared-start: ON | log dir: {base.logging.dir}")

    n_ok = n_err = n_skip = 0
    for problem in problems:
        # ---- 1) One shared warm start per task (gates off), reused or regenerated ----
        spath = _shared_start_path(base.logging.dir, problem.question_id)
        if not args.overwrite and os.path.exists(spath):
            rec = json.load(open(spath))
            plan, code = rec["plan"], rec["code"]
            print(f"[shared] {problem.question_id} reuse "
                  f"(tok={rec['token_totals'].get('total_tokens')})")
        else:
            t0 = time.time()
            try:
                plan, code, sllm = generate_shared_start(base, problem)
            except Exception as e:
                n_err += 1
                print(f"[ERR] {problem.question_id} shared-start failed: "
                      f"{type(e).__name__}: {e}")
                continue
            _write_shared_start(base.logging.dir, problem, plan, code, sllm,
                                args.conditions, trials, time.time() - t0)
            print(f"[shared] {problem.question_id} generated "
                  f"tok={sllm.token_totals().get('total_tokens')} "
                  f"code_len={len(code)} {time.time()-t0:.1f}s")

        # ---- 2) Every condition x trial branches from that identical start ----
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
                try:
                    log = run_episode(cfg, problem, trial,
                                      seed_plan=plan, seed_code=code)
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
                    print(f"[ERR] {episode_id} batch-level failure: "
                          f"{type(e).__name__}: {e}")

    print(f"\nDone. episodes ok={n_ok} err={n_err} skipped={n_skip}")


if __name__ == "__main__":
    main()
