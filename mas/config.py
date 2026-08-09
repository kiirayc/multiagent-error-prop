"""Config loading. YAML -> typed dataclasses. No experiment parameter is hardcoded
anywhere else; everything flows from here."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml

CONDITIONS = ("none", "plan_gate", "final_gate", "both")


@dataclass
class ModelConfig:
    provider: str
    name: str
    reasoning_effort: str
    max_completion_tokens: int


@dataclass
class LoopConfig:
    max_iterations: int
    plan_gate_max_retries: int
    final_gate_max_retries: int


@dataclass
class ExecutionConfig:
    timeout: int
    per_test: bool


@dataclass
class DatasetConfig:
    release_version: str
    contamination_start_date: Optional[str]
    contamination_end_date: Optional[str]
    task_ids: Optional[list]
    max_tasks: int
    difficulty: Optional[list] = None      # e.g. ["hard"]; None = all difficulties
    cache_path: Optional[str] = None       # explicit curated-cache path; else derived


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    condition: str
    trials: int


@dataclass
class LoggingConfig:
    dir: str


@dataclass
class Config:
    experiment: ExperimentConfig
    model: ModelConfig
    loop: LoopConfig
    execution: ExecutionConfig
    dataset: DatasetConfig
    logging: LoggingConfig
    # Raw dict retained verbatim so the full config can be embedded in every log.
    raw: dict = field(default_factory=dict)

    def gates_active(self) -> tuple[bool, bool]:
        """(plan_gate_active, final_gate_active) derived purely from condition."""
        c = self.experiment.condition
        return (c in ("plan_gate", "both"), c in ("final_gate", "both"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _from_raw(raw)


def _from_raw(raw: dict) -> Config:
    cfg = Config(
        experiment=ExperimentConfig(**raw["experiment"]),
        model=ModelConfig(**raw["model"]),
        loop=LoopConfig(**raw["loop"]),
        execution=ExecutionConfig(**raw["execution"]),
        dataset=DatasetConfig(**raw["dataset"]),
        logging=LoggingConfig(**raw["logging"]),
        raw=copy.deepcopy(raw),
    )
    if cfg.experiment.condition not in CONDITIONS:
        raise ValueError(
            f"condition must be one of {CONDITIONS}, got {cfg.experiment.condition!r}"
        )
    return cfg


def with_condition(cfg: Config, condition: str) -> Config:
    """Return a copy of cfg with a different condition. Used by the batch runner to
    sweep all 4 conditions from one base config — guaranteeing every other parameter
    is byte-for-byte identical across conditions."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    new_raw = copy.deepcopy(cfg.raw)
    new_raw["experiment"]["condition"] = condition
    return _from_raw(new_raw)
