"""OpenAI client wrapper for the reasoning model (o3-mini).

o3-mini is a reasoning model: `temperature`/`top_p` are NOT supported and must not
be sent; it uses `max_completion_tokens` (not `max_tokens`) and `reasoning_effort`.
We detect what is actually controllable, hold it constant across all conditions,
and log the REAL settings used on every call (including reasoning-token counts).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from .config import ModelConfig


@dataclass
class LLMCall:
    """Record of a single model call — everything needed for offline accounting."""
    role: str                       # planner | coder | reviewer
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    latency_s: float
    settings: dict                  # the REAL request settings actually sent
    finish_reason: Optional[str] = None


@dataclass
class LLMResult:
    text: str
    call: LLMCall


class LLMClient:
    def __init__(self, model_cfg: ModelConfig, client: Optional[OpenAI] = None):
        self.cfg = model_cfg
        # Lazily construct the OpenAI client so a missing API key surfaces only when a
        # call is actually made (inside the guarded graph run), never at construction —
        # keeping the batch resilient and token accounting always available.
        self._client = client
        # Detect which sampling controls this model family actually supports.
        self.supports_sampling = _model_supports_sampling(model_cfg.name)
        self.calls: list[LLMCall] = []

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def _request_settings(self) -> dict:
        """The exact, held-constant settings sent on every call. Reasoning models
        reject temperature/top_p, so we omit them and log that they are unsupported."""
        settings = {
            "model": self.cfg.name,
            "max_completion_tokens": self.cfg.max_completion_tokens,
        }
        if _model_is_reasoning(self.cfg.name):
            settings["reasoning_effort"] = self.cfg.reasoning_effort
        if self.supports_sampling:
            # Only sent for non-reasoning models; held constant if present.
            settings["temperature"] = 0.0
        return settings

    def complete(self, role: str, messages: list) -> LLMResult:
        settings = self._request_settings()
        start = time.time()
        resp = self.client.chat.completions.create(messages=messages, **settings)
        latency = time.time() - start

        usage = resp.usage
        reasoning_tokens = 0  # default reasoning token set to 0
        if usage is not None and getattr(usage, "completion_tokens_details", None):
            reasoning_tokens = getattr(
                usage.completion_tokens_details, "reasoning_tokens", 0) or 0

        call = LLMCall(
            role=role,
            model=self.cfg.name,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            reasoning_tokens=reasoning_tokens,
            total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
            latency_s=latency,
            settings={k: v for k, v in settings.items() if k != "model"},
            finish_reason=resp.choices[0].finish_reason,
        )
        self.calls.append(call)
        text = resp.choices[0].message.content or ""
        return LLMResult(text=text, call=call)

    def token_totals(self) -> dict:
        return {
            "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "completion_tokens": sum(c.completion_tokens for c in self.calls),
            "reasoning_tokens": sum(c.reasoning_tokens for c in self.calls),
            "total_tokens": sum(c.total_tokens for c in self.calls),
            "n_calls": len(self.calls),
        }


def _model_is_reasoning(name: str) -> bool:
    n = name.lower()
    return n.startswith("o1") or n.startswith("o3") or n.startswith("o4")


def _model_supports_sampling(name: str) -> bool:
    # Reasoning models (o1/o3/o4) reject temperature/top_p.
    return not _model_is_reasoning(name)
