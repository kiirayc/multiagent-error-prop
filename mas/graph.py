"""The single parameterized LangGraph.

ONE graph serves all 4 conditions. The plan/final gates are the ONLY thing that
varies, and purely via `ctx.plan_gate_active` / `ctx.final_gate_active` (derived
from the condition). When a gate is inactive its node is a pass-through — no other
behavioral difference anywhere.

The intrinsic execute-and-revise loop (Coder revising off failing VISIBLE tests) is
part of the base system and runs under EVERY condition, including `none`. It is not
the supervision variable.

    START
      -> planner
      -> plan_gate      (pass-through unless plan_gate_active)
      -> coder
      -> execute_visible
      -> revise loop    (always on; bounded by max_iterations)
      -> final_gate     (pass-through unless final_gate_active)
      -> hidden_eval
      -> END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from . import agents
from .execution import run_tests
from .state import Context, GraphState


def build_graph(ctx: Context):
    # Shared state type
    g = StateGraph(GraphState)

    # ----------------------- nodes -----------------------
    def planner(state: GraphState) -> dict:
        # Respect a pre-seeded plan (shared warm-start) on the first entry so all
        # conditions branch from an identical plan. Regenerate only once a plan gate
        # has sent us back (retries>0), or when no plan was seeded (standard flow).
        if state.get("plan") and state.get("plan_gate_retries", 0) == 0:
            return {}
        plan = agents.run_planner(ctx.llm, ctx.problem)
        return {"plan": plan}

    def plan_gate(state: GraphState) -> dict:
        # Pass-through when inactive — recorded as skipped for auditability.
        if not ctx.plan_gate_active:
            return {}
        outcome = agents.run_reviewer_plan(ctx.llm, ctx.problem, state["plan"])
        events = list(state.get("gate_events", []))
        attempt = state.get("plan_gate_retries", 0)
        events.append({"gate": "plan", "attempt": attempt,
                       "verdict": "pass" if outcome.passed else "fail",
                       "feedback": outcome.feedback})
        upd = {"gate_events": events}
        if not outcome.passed:
            upd["plan_gate_retries"] = attempt + 1
            # A rejected plan invalidates any seeded/prior initial code: drop it so the
            # Coder regenerates from the revised plan. No-op in the standard flow, where
            # code is not yet set when the plan gate runs.
            upd["code"] = ""
        return upd

    def coder(state: GraphState) -> dict:
        phase = state.get("phase", "initial")
        if not state.get("code"):
            code = agents.run_coder_initial(ctx.llm, ctx.problem, state["plan"])
            return {"code": code, "phase": "initial"}
        # Code already present with no pending feedback => a shared warm-start initial
        # code (or already-current code). Pass it through untouched — do not revise.
        # In the standard flow the Coder is only re-entered with code present when a
        # revision was requested, so pending_feedback is always set there.
        feedback = state.get("pending_feedback", "")
        if not feedback:
            return {}
        code = agents.run_coder_revise(
            ctx.llm, ctx.problem, state["plan"], state["code"], feedback)
        return {"code": code, "phase": phase}

    def execute_visible(state: GraphState) -> dict:
        result = run_tests(
            state["code"], ctx.problem.public_tests,
            fn_name=ctx.problem.fn_name, timeout=ctx.cfg.execution.timeout,
            test_set="visible", per_test=ctx.cfg.execution.per_test,
        )
        iteration = state.get("iteration", 0) + 1
        rec = {
            "iteration": iteration,
            "phase": state.get("phase", "initial"),
            "code": state["code"],
            "visible_result": result.to_dict(),
            "in_error_state": not result.all_passed,
            "num_failing": result.num_failing,
        }
        history = list(state.get("history", []))
        history.append(rec)
        upd = {
            "iteration": iteration,
            "last_visible_result": result.to_dict(),
            "history": history,
        }
        # Prepare feedback in case we revise off failing tests next.
        if not result.all_passed:
            upd["pending_feedback"] = agents.format_test_feedback(
                result, ctx.problem.public_tests)
            upd["phase"] = "revise_tests"
        return upd

    def final_gate(state: GraphState) -> dict:
        if not ctx.final_gate_active:
            return {}
        outcome = agents.run_reviewer_code(ctx.llm, ctx.problem, state["code"])
        events = list(state.get("gate_events", []))
        attempt = state.get("final_gate_retries", 0)
        events.append({"gate": "final", "attempt": attempt,
                       "verdict": "pass" if outcome.passed else "fail",
                       "feedback": outcome.feedback})
        upd = {"gate_events": events}
        if not outcome.passed:
            upd["final_gate_retries"] = attempt + 1
            upd["pending_feedback"] = f"Reviewer feedback: {outcome.feedback}"
            upd["phase"] = "revise_final_gate"
        return upd

    def hidden_eval(state: GraphState) -> dict:
        result = run_tests(
            state["code"], ctx.problem.private_tests,
            fn_name=ctx.problem.fn_name, timeout=ctx.cfg.execution.timeout,
            test_set="hidden", per_test=ctx.cfg.execution.per_test,
        )
        return {"hidden_result": result.to_dict(),
                "termination": state.get("termination", "completed")}

    # ----------------------- edges -----------------------
    g.add_node("planner", planner)
    g.add_node("plan_gate", plan_gate)
    g.add_node("coder", coder)
    g.add_node("execute_visible", execute_visible)
    g.add_node("final_gate", final_gate)
    g.add_node("hidden_eval", hidden_eval)

    g.add_edge(START, "planner")
    g.add_edge("planner", "plan_gate")

    # Plan gate: fail (active only) -> back to Planner, bounded; else -> Coder.
    def after_plan_gate(state: GraphState) -> str:
        if not ctx.plan_gate_active:
            return "coder"
        events = state.get("gate_events", [])
        last = events[-1] if events else None
        failed = bool(last and last["gate"] == "plan" and last["verdict"] == "fail")
        # retries was incremented on the fail; allow reruns up to the cap.
        if failed and state.get("plan_gate_retries", 0) <= ctx.cfg.loop.plan_gate_max_retries:
            return "planner"
        return "coder"

    g.add_conditional_edges("plan_gate", after_plan_gate,
                            {"planner": "planner", "coder": "coder"})

    g.add_edge("coder", "execute_visible")

    # Intrinsic revise loop (ALWAYS on): fail + under cap -> Coder; else -> final_gate.
    def after_execute(state: GraphState) -> str:
        result = state["last_visible_result"]
        all_passed = result["passed"] == result["total"] and result["total"] > 0
        if not all_passed and state.get("iteration", 0) < ctx.cfg.loop.max_iterations:
            return "coder"
        return "final_gate"

    g.add_conditional_edges("execute_visible", after_execute,
                            {"coder": "coder", "final_gate": "final_gate"})

    # Final gate: fail (active only) -> back to Coder, bounded; else -> hidden_eval.
    def after_final_gate(state: GraphState) -> str:
        if not ctx.final_gate_active:
            return "hidden_eval"
        events = state.get("gate_events", [])
        last = events[-1] if events else None
        failed = bool(last and last["gate"] == "final" and last["verdict"] == "fail")
        if failed and state.get("final_gate_retries", 0) <= ctx.cfg.loop.final_gate_max_retries:
            return "coder"
        return "hidden_eval"

    g.add_conditional_edges("final_gate", after_final_gate,
                            {"coder": "coder", "hidden_eval": "hidden_eval"})

    g.add_edge("hidden_eval", END)

    return g.compile()
