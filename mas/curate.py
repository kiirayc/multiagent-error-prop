"""One-time dataset curation CLI.

Downloads the LiveCodeBench release (cached by HuggingFace), filters to the
configured difficulty + contamination window, decompresses only those problems,
and writes a small local JSON cache. Run this once; the run loop then reads the
cache with no HuggingFace round-trip.

    python -m mas.curate --config configs/base.yaml
"""

from __future__ import annotations

import argparse

from .config import load_config
from .dataset import curate, cache_path_for


def main():
    ap = argparse.ArgumentParser(description="Curate LiveCodeBench problems to cache.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--out", default=None, help="Override cache output path.")
    args = ap.parse_args()

    # Converts YAML into a Python config object
    cfg = load_config(args.config)
    out = args.out or cache_path_for(cfg.dataset)
    views = curate(cfg.dataset, out)

    print(f"\nCurated {len(views)} problems.")
    by_diff: dict = {}
    for v in views:
        by_diff[v.difficulty] = by_diff.get(v.difficulty, 0) + 1
    print("By difficulty:", by_diff)
    print("First few ids:", [v.question_id for v in views[:5]])


if __name__ == "__main__":
    main()
