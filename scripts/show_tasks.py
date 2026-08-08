"""Browse the curated task cache: list tasks or dump one task's statement + tests.

The curated cache (data/curated_<...>.jsonl) has one problem per line. Private test
inputs can be very large, so this tool streams the file, prints public tests in full,
and shows private tests only on request (truncated).

    # list the first 10 tasks (what max_tasks:10 would select)
    python scripts/show_tasks.py --config configs/base.yaml --list --n 10

    # dump one task: statement, starter code, fn_name, and public tests
    python scripts/show_tasks.py --config configs/base.yaml --task abc374_f

    # also show 2 private tests (truncated)
    python scripts/show_tasks.py --config configs/base.yaml --task abc374_f --show-private 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mas.config import load_config
from mas.dataset import cache_path_for


def _iter_lines(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _trunc(s, n=800):
    s = str(s)
    return s if len(s) <= n else s[: n // 2] + f"\n  …(truncated {len(s)} chars)…\n" + s[-n // 2 :]


def cmd_list(path, n):
    print(f"Cache: {path}\n")
    print(f"{'#':>3}  {'question_id':<14} {'diff':<6} {'platform':<10} "
          f"{'call':<5} {'pub':>3} {'priv':>4} {'total':>5}  title")
    print("-" * 96)
    n_tasks = 0
    sum_pub = sum_priv = 0
    for i, d in enumerate(_iter_lines(path)):
        if n is not None and i >= n:
            break
        npub, npriv = len(d["public_tests"]), len(d["private_tests"])
        call = "yes" if d["fn_name"] else "no"
        print(f"{i:>3}  {d['question_id']:<14} {d['difficulty']:<6} {d['platform']:<10} "
              f"{call:<5} {npub:>3} {npriv:>4} {npub + npriv:>5}  {d['title'][:40]}")
        n_tasks += 1
        sum_pub += npub
        sum_priv += npriv
    print("-" * 96)
    print(f"{n_tasks} tasks | public={sum_pub} private={sum_priv} "
          f"total={sum_pub + sum_priv} | "
          f"avg per task: pub={sum_pub / n_tasks:.1f} priv={sum_priv / n_tasks:.1f} "
          f"total={(sum_pub + sum_priv) / n_tasks:.1f}"
          if n_tasks else "no tasks")
    if n is not None:
        print(f"(stopped at {n}; omit --n to list all)")


def cmd_task(path, qid, show_private):
    for d in _iter_lines(path):
        if d["question_id"] != qid:
            continue
        print("=" * 80)
        print(f"question_id : {d['question_id']}")
        print(f"title       : {d['title']}")
        print(f"platform    : {d['platform']}   difficulty: {d['difficulty']}   "
              f"contest_date: {d['contest_date']}")
        print(f"format      : {'call-based fn=' + d['fn_name'] if d['fn_name'] else 'stdin/stdout'}")
        print(f"tests       : {len(d['public_tests'])} public, {len(d['private_tests'])} private")
        if d["starter_code"].strip():
            print("\n--- starter_code ---")
            print(d["starter_code"])
        print("\n--- statement ---")
        print(d["content"])
        print("\n--- PUBLIC tests (drive the visible revise loop) ---")
        for j, t in enumerate(d["public_tests"]):
            print(f"[pub {j}] ({t['testtype']})")
            print(f"  input:    {_trunc(t['input'])!r}")
            print(f"  expected: {_trunc(t['output'])!r}")
        if show_private:
            print(f"\n--- PRIVATE tests (final hidden eval) — showing {show_private} ---")
            for j, t in enumerate(d["private_tests"][:show_private]):
                print(f"[priv {j}] ({t['testtype']})")
                print(f"  input:    {_trunc(t['input'])!r}")
                print(f"  expected: {_trunc(t['output'])!r}")
        return
    print(f"question_id {qid!r} not found in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--list", action="store_true", help="List tasks.")
    ap.add_argument("--n", type=int, default=None, help="Limit --list to first N.")
    ap.add_argument("--task", default=None, help="Dump one task by question_id.")
    ap.add_argument("--show-private", type=int, default=0,
                    help="With --task, also print this many (truncated) private tests.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = cache_path_for(cfg.dataset)
    if not os.path.exists(path):
        ap.error(f"cache not found: {path}. Run `python -m mas.curate --config {args.config}`.")

    if args.task:
        cmd_task(path, args.task, args.show_private)
    else:
        cmd_list(path, args.n if args.n is not None else (None if args.list else 10))


if __name__ == "__main__":
    main()
