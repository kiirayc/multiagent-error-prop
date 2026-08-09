"""Run a single episode: one task × one condition × one trial.

Builds the Context (LLM client, gate activation from the condition), compiles the
single parameterized graph, runs it, and returns a structured EpisodeLog.
"""

from __future__ import annotations

import time
from typing import Optional

from openai import OpenAI

from .config import Config
from .dataset import ProblemView
from .episode_log import EpisodeLog, build_episode_log
from .graph import build_graph
from .llm import LLMClient
from .state import Context

# Global safety cap on graph steps (defense-in-depth beyond the explicit loop/gate
# caps) so a misbehaving condition can never spin forever.
RECURSION_LIMIT = 60


def run_episode(
    cfg: Config,
    problem: ProblemView,
    trial: int,
    client: Optional[OpenAI] = None,
    seed_plan: Optional[str] = None,
    seed_code: Optional[str] = None,
) -> EpisodeLog:
    """Run one episode. When `seed_plan`/`seed_code` are given (shared warm-start),
    the graph starts from that identical plan+initial-code instead of generating its
    own, so every condition branches from the same baseline artifact."""
    plan_active, final_active = cfg.gates_active()
    llm = LLMClient(cfg.model, client=client)

    initial_state = {
        "iteration": 0,
        "plan_gate_retries": 0,
        "final_gate_retries": 0,
        "gate_events": [],
        "history": [],
        "hidden_result": None,
    }
    if seed_plan is not None:
        initial_state["plan"] = seed_plan
    if seed_code is not None:
        initial_state["code"] = seed_code

    start = time.time()
    error = None
    final_state = initial_state
    try:
        ctx = Context(
            cfg=cfg, llm=llm, problem=problem,
            plan_gate_active=plan_active, final_gate_active=final_active,
        )
        app = build_graph(ctx)
        final_state = app.invoke(initial_state, {"recursion_limit": RECURSION_LIMIT})
    except Exception as e:  # record abnormal termination rather than crashing the batch
        error = f"{type(e).__name__}: {e}"
        final_state = dict(final_state)
        final_state.setdefault("termination", "aborted")
    wall = time.time() - start

    return build_episode_log(
        cfg=cfg, problem=problem, trial=trial,
        final_state=final_state, llm=llm, wall_time_s=wall, error=error,
    )
