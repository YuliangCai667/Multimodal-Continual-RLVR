from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CandidateScore:
    beta: float
    new_descent: float
    old_harm: float
    update_distance_sq: float


@dataclass(frozen=True)
class ControllerDecision:
    beta: float
    status: str
    new_descent_threshold: float
    old_harm_violation: float

    @property
    def jointly_feasible(self) -> bool:
        return self.status in {"forced_jointly_feasible", "closest_jointly_feasible"}


def select_global_beta(
    candidates: Iterable[CandidateScore],
    *,
    new_descent_ratio: float,
    force_beta: float | None = None,
) -> ControllerDecision:
    """Choose the closest globally controlled isospectral candidate.

    The new-task descent floor is a hard constraint.  Within that set, the
    controller first minimizes positive old-task constraint violation
    ``max(old_harm, 0)``.  Once the old task is non-increasing to first order,
    it minimizes the actual distance from the raw optimizer update rather than
    rewarding increasingly negative old-task harm.
    """
    scores = tuple(candidates)
    by_beta = {score.beta: score for score in scores}
    if 0.0 not in by_beta:
        raise ValueError("controller candidates must contain beta=0")
    raw = by_beta[0.0]
    threshold = new_descent_ratio * raw.new_descent

    if force_beta is not None:
        chosen = by_beta[force_beta]
        jointly_feasible = chosen.new_descent >= threshold and chosen.old_harm <= 0.0
        return ControllerDecision(
            beta=chosen.beta,
            status="forced_jointly_feasible" if jointly_feasible else "forced_constraint_violation",
            new_descent_threshold=threshold,
            old_harm_violation=max(chosen.old_harm, 0.0),
        )

    # If the raw optimizer update is not a descent direction, a ratio to that
    # negative value has no useful meaning.  Preserve the raw update and expose
    # the condition in the log instead of allowing the controller to amplify it.
    if raw.new_descent <= 0.0:
        return ControllerDecision(
            beta=0.0,
            status="raw_new_descent_nonpositive",
            new_descent_threshold=threshold,
            old_harm_violation=max(raw.old_harm, 0.0),
        )

    safe = tuple(score for score in scores if score.new_descent >= threshold)
    chosen = min(
        safe,
        key=lambda score: (
            max(score.old_harm, 0.0),
            score.update_distance_sq,
            score.beta,
        ),
    )
    return ControllerDecision(
        beta=chosen.beta,
        status="closest_jointly_feasible" if chosen.old_harm <= 0.0 else "minimum_old_constraint_violation",
        new_descent_threshold=threshold,
        old_harm_violation=max(chosen.old_harm, 0.0),
    )
