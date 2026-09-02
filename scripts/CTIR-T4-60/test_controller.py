#!/usr/bin/env python3
from src.ctir.controller import CandidateScore, select_global_beta


def score(beta: float, new: float, old: float, distance: float) -> CandidateScore:
    return CandidateScore(beta, new, old, distance * distance)


def main() -> None:
    # Once several candidates satisfy both task constraints, choose the
    # nearest update, not the candidate with the most negative old harm.
    decision = select_global_beta(
        (
            score(0.0, 10.0, 2.0, 0.0),
            score(0.25, 9.8, -0.1, 0.2),
            score(0.50, 9.5, -4.0, 0.5),
        ),
        new_descent_ratio=0.9,
    )
    assert decision.beta == 0.25
    assert decision.status == "closest_jointly_feasible"

    # If the beta grid cannot reach old_harm <= 0, retain the candidate with
    # the smallest constraint violation while respecting new-task descent.
    decision = select_global_beta(
        (
            score(0.0, 10.0, 3.0, 0.0),
            score(0.25, 9.8, 1.0, 0.2),
            score(0.50, 8.9, -1.0, 0.5),
        ),
        new_descent_ratio=0.9,
    )
    assert decision.beta == 0.25
    assert decision.status == "minimum_old_constraint_violation"

    # A raw update that already protects the old task is the closest feasible
    # isospectral transport and therefore remains untouched.
    decision = select_global_beta(
        (
            score(0.0, 10.0, -0.2, 0.0),
            score(0.25, 9.8, -2.0, 0.2),
        ),
        new_descent_ratio=0.9,
    )
    assert decision.beta == 0.0

    # Do not reinterpret a non-descent raw step through a sign-sensitive ratio.
    decision = select_global_beta(
        (
            score(0.0, -1.0, 2.0, 0.0),
            score(0.25, 0.5, -1.0, 0.2),
        ),
        new_descent_ratio=0.9,
    )
    assert decision.beta == 0.0
    assert decision.status == "raw_new_descent_nonpositive"

    print("global closest-feasible controller: passed")


if __name__ == "__main__":
    main()
