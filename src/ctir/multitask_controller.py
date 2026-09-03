from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MultiTaskCandidateScore:
    beta: float
    new_descent: float
    old_harms: dict[str, float]
    normalized_old_harms: dict[str, float]
    update_distance_sq: float

    @property
    def worst_violation(self) -> float:
        return max((max(value, 0.0) for value in self.normalized_old_harms.values()), default=0.0)

    @property
    def all_old_tasks_safe(self) -> bool:
        return all(value <= 0.0 for value in self.old_harms.values())


@dataclass(frozen=True)
class MultiTaskControllerDecision:
    beta: float
    status: str
    new_descent_threshold: float
    worst_normalized_violation: float

    @property
    def jointly_feasible(self) -> bool:
        return self.status in {"forced_jointly_feasible", "closest_jointly_feasible"}


def select_multitask_global_beta(
    candidates: Iterable[MultiTaskCandidateScore],
    *,
    new_descent_ratio: float,
    force_beta: float | None = None,
) -> MultiTaskControllerDecision:
    """Select one global beta using per-task constraints and no task averaging."""
    scores = tuple(candidates)
    by_beta = {score.beta: score for score in scores}
    if len(by_beta) != len(scores) or 0.0 not in by_beta:
        raise ValueError("candidate betas must be unique and include beta=0")
    task_keys = set(by_beta[0.0].old_harms)
    if not task_keys or any(set(score.old_harms) != task_keys for score in scores):
        raise ValueError("all candidates must contain the same nonempty old-task set")
    if any(set(score.normalized_old_harms) != task_keys for score in scores):
        raise ValueError("normalized and raw old-task harms must have identical task keys")

    raw = by_beta[0.0]
    threshold = new_descent_ratio * raw.new_descent
    if force_beta is not None:
        chosen = by_beta[force_beta]
        feasible = chosen.new_descent >= threshold and chosen.all_old_tasks_safe
        return MultiTaskControllerDecision(
            beta=chosen.beta,
            status="forced_jointly_feasible" if feasible else "forced_constraint_violation",
            new_descent_threshold=threshold,
            worst_normalized_violation=chosen.worst_violation,
        )
    if raw.new_descent <= 0.0:
        return MultiTaskControllerDecision(
            beta=0.0,
            status="raw_new_descent_nonpositive",
            new_descent_threshold=threshold,
            worst_normalized_violation=raw.worst_violation,
        )
    safe_new = tuple(score for score in scores if score.new_descent >= threshold)
    chosen = min(
        safe_new,
        key=lambda score: (score.worst_violation, score.update_distance_sq, score.beta),
    )
    return MultiTaskControllerDecision(
        beta=chosen.beta,
        status="closest_jointly_feasible" if chosen.all_old_tasks_safe else "minimum_worst_task_violation",
        new_descent_threshold=threshold,
        worst_normalized_violation=chosen.worst_violation,
    )
