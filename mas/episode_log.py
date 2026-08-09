"""Structured per-episode logging.

One episode = one task × one condition × one trial. Everything needed to recompute
every metric offline lives in the record — metrics are NEVER computed inside the run.
Each episode is written as one JSON file (and appended to a JSONL index).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from .config import Config
from .dataset import ProblemView
from .llm import LLMClient


@dataclass
class EpisodeLog:
    # Identity
    episode_id: str
    task_id: str
    condition: str
    trial: int
    seed: int
    prompt_version: str

    # Full config snapshot (so a log is self-describing / reproducible)
    config: dict

    # Problem provenance (no test payloads — keep logs lean; tests are in the cache)
    problem: dict

    # The real model settings actually used, and gate activation
    model_settings: dict
    gates_active: dict

    # Per-iteration history (code, visible result, error-state flags)
    history: list = field(default_factory=list)
    # Gate invocations + outcomes + feedback
    gate_events: list = field(default_factory=list)
    # Final results
    final_visible_result: Optional[dict] = None
    final_hidden_result: Optional[dict] = None
    # Plan text and the final submitted code
    plan: str = ""
    final_code: str = ""
    # Per-call token usage (list of LLMCall dicts) + totals
    llm_calls: list = field(default_factory=list)
    token_totals: dict = field(default_factory=dict)
    # Timing / termination
    wall_time_s: float = 0.0
    termination: str = ""
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def build_episode_log(
    cfg: Config,
    problem: ProblemView,
    trial: int,
    final_state: dict,
    llm: LLMClient,
    wall_time_s: float,
    error: Optional[str] = None,
) -> EpisodeLog:
    from . import prompts

    plan_active, final_active = cfg.gates_active()
    episode_id = f"{problem.question_id}__{cfg.experiment.condition}__t{trial}"

    hist = final_state.get("history", [])
    final_visible = hist[-1]["visible_result"] if hist else None

    return EpisodeLog(
        episode_id=episode_id,
        task_id=problem.question_id,
        condition=cfg.experiment.condition,
        trial=trial,
        seed=cfg.experiment.seed,
        prompt_version=prompts.PROMPT_VERSION,
        config=cfg.to_dict(),
        problem={
            "question_id": problem.question_id,
            "title": problem.title,
            "platform": problem.platform,
            "difficulty": problem.difficulty,
            "contest_date": problem.contest_date,
            "is_call_based": problem.is_call_based,
            "fn_name": problem.fn_name,
            "n_public_tests": len(problem.public_tests),
            "n_private_tests": len(problem.private_tests),
        },
        model_settings={
            "model": cfg.model.name,
            "reasoning_effort": cfg.model.reasoning_effort,
            "max_completion_tokens": cfg.model.max_completion_tokens,
            "supports_sampling": llm.supports_sampling,
        },
        gates_active={"plan_gate": plan_active, "final_gate": final_active},
        history=hist,
        gate_events=final_state.get("gate_events", []),
        final_visible_result=final_visible,
        final_hidden_result=final_state.get("hidden_result"),
        plan=final_state.get("plan", ""),
        final_code=final_state.get("code", ""),
        llm_calls=[asdict(c) for c in llm.calls],
        token_totals=llm.token_totals(),
        wall_time_s=wall_time_s,
        termination=final_state.get("termination", "") or ("error" if error else "completed"),
        error=error,
    )


def episode_log_path(log_dir: str, episode_id: str) -> str:
    return os.path.join(log_dir, f"{episode_id}.json")


def _upsert_index(index_path: str, entry: dict) -> None:
    """Rewrite episodes.jsonl so each episode_id appears exactly once (latest wins).
    Prevents duplicate index rows when an episode is re-run."""
    rows: dict = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "episode_id" in r:
                    rows[r["episode_id"]] = r
    rows[entry["episode_id"]] = entry  # insert or replace
    with open(index_path, "w") as f:
        for r in rows.values():
            f.write(json.dumps(r) + "\n")


def write_episode_log(log: EpisodeLog, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = episode_log_path(log_dir, log.episode_id)
    with open(path, "w") as f:
        f.write(log.to_json())
    # Upsert a one-line index entry (dedup by episode_id) for scanning / discovery.
    _upsert_index(os.path.join(log_dir, "episodes.jsonl"), {
        "episode_id": log.episode_id, "task_id": log.task_id,
        "condition": log.condition, "trial": log.trial,
        "termination": log.termination, "path": path,
        "hidden_passed": (log.final_hidden_result or {}).get("passed"),
        "hidden_total": (log.final_hidden_result or {}).get("total"),
    })
    return path
