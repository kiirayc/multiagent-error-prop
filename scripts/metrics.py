"""Offline metrics. Consumes episode logs; computes nothing during the run.

Reads a log directory of per-episode JSON files and reports, per condition (and
breakable by difficulty):

  * final_pass_rate        : fraction of episodes passing ALL hidden tests
  * initial_defect_prob    : fraction whose first code failed >=1 visible test
  * recovery_rate          : of episodes that entered a visible error state, the
                             fraction that recovered (final visible all-pass)
  * mean_propagation_depth : revisions from first error state to recovery
  * mean_error_magnitude   : visible failing count at first error minus at final
  * mean_iterations        : visible-test executions per episode
  * mean_tokens            : total model tokens per episode (+ reasoning tokens)
  * failing_trajectory     : mean visible failing-count at each iteration index

Usage:
    python scripts/metrics.py --logs logs
    python scripts/metrics.py --logs logs --by-difficulty --json out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean


def load_episodes(log_dir: str) -> list:
    episodes = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        with open(path) as f:
            episodes.append(json.load(f))
    return episodes


def _hidden_all_passed(ep) -> bool:
    h = ep.get("final_hidden_result") or {}
    return bool(h) and h.get("total", 0) > 0 and h.get("passed") == h.get("total")


def _first_error_index(history: list):
    for rec in history:
        if rec.get("in_error_state"):
            return rec["iteration"]
    return None


def _recovered(history: list) -> bool:
    return bool(history) and not history[-1].get("in_error_state", True)


def episode_metrics(ep) -> dict:
    """Per-episode raw quantities used to build condition aggregates."""
    history = ep.get("history", [])
    m = {
        "condition": ep.get("condition"),
        "difficulty": ep.get("problem", {}).get("difficulty", "unknown"),
        "hidden_pass": _hidden_all_passed(ep),
        "n_iterations": len(history),
        "total_tokens": ep.get("token_totals", {}).get("total_tokens", 0),
        "reasoning_tokens": ep.get("token_totals", {}).get("reasoning_tokens", 0),
        "aborted": ep.get("error") is not None,
    }
    if history:
        m["initial_defect"] = bool(history[0].get("in_error_state"))
        first_err = _first_error_index(history)
        m["entered_error"] = first_err is not None
        if first_err is not None:
            m["recovered"] = _recovered(history)
            # Propagation depth: iterations from first error to recovery (or end).
            first_err_fail = next(
                r["num_failing"] for r in history if r["iteration"] == first_err)
            final_fail = history[-1].get("num_failing", 0)
            m["error_magnitude"] = first_err_fail - final_fail
            if m["recovered"]:
                rec_idx = next(
                    r["iteration"] for r in history if not r["in_error_state"]
                    and r["iteration"] >= first_err)
                m["propagation_depth"] = rec_idx - first_err
            else:
                m["propagation_depth"] = history[-1]["iteration"] - first_err
    else:
        m["initial_defect"] = None
        m["entered_error"] = False
    m["failing_by_iter"] = {
        r["iteration"]: r["num_failing"] for r in history}
    return m


def _safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return round(mean(vals), 4) if vals else None


def aggregate(per_ep: list) -> dict:
    n = len(per_ep)
    entered = [m for m in per_ep if m.get("entered_error")]
    agg = {
        "n_episodes": n,
        "n_aborted": sum(1 for m in per_ep if m["aborted"]),
        "final_pass_rate": _safe_mean([1.0 if m["hidden_pass"] else 0.0 for m in per_ep]),
        "initial_defect_prob": _safe_mean(
            [1.0 if m.get("initial_defect") else 0.0
             for m in per_ep if m.get("initial_defect") is not None]),
        "recovery_rate": _safe_mean(
            [1.0 if m.get("recovered") else 0.0 for m in entered]) if entered else None,
        "mean_propagation_depth": _safe_mean(
            [m.get("propagation_depth") for m in entered]) if entered else None,
        "mean_error_magnitude": _safe_mean(
            [m.get("error_magnitude") for m in entered]) if entered else None,
        "mean_iterations": _safe_mean([m["n_iterations"] for m in per_ep]),
        "mean_total_tokens": _safe_mean([m["total_tokens"] for m in per_ep]),
        "mean_reasoning_tokens": _safe_mean([m["reasoning_tokens"] for m in per_ep]),
    }
    # Mean visible failing-count trajectory across iteration indices.
    traj = defaultdict(list)
    for m in per_ep:
        for it, nf in m["failing_by_iter"].items():
            traj[int(it)].append(nf)
    agg["failing_trajectory"] = {
        str(it): round(mean(v), 3) for it, v in sorted(traj.items())}
    return agg


def group_and_aggregate(per_ep: list, by_difficulty: bool) -> dict:
    groups: dict = defaultdict(list)
    for m in per_ep:
        key = m["condition"]
        if by_difficulty:
            key = f"{m['condition']}::{m['difficulty']}"
        groups[key].append(m)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--by-difficulty", action="store_true")
    ap.add_argument("--json", default=None, help="Write full report to this path.")
    args = ap.parse_args()

    episodes = load_episodes(args.logs)
    per_ep = [episode_metrics(ep) for ep in episodes]
    report = group_and_aggregate(per_ep, args.by_difficulty)

    print(json.dumps(report, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote report -> {args.json}")


if __name__ == "__main__":
    main()
