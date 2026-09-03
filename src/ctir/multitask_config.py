from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultiTaskCTIRConfig:
    """Explicit configuration for the task-conditioned CTIR extension."""

    probe_index_path: str
    log_dir: str
    layer_start: int = 9
    layer_end: int = 26
    tangent_rank: int = 8
    raw_rank: int = 8
    refresh_interval: int = 5
    union_rtol: float = 1e-6
    new_descent_ratio: float = 0.90
    beta_candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    continual_start_step: int = 0
    force_beta: float | None = None
    exact_spectrum_check: bool = False
    stop_after_steps: int | None = None

    @classmethod
    def from_training_args(cls, args) -> "MultiTaskCTIRConfig":
        candidates = tuple(float(value.strip()) for value in args.ctir_multitask_beta_candidates.split(","))
        if not candidates or candidates[0] != 0.0:
            raise ValueError("ctir_multitask_beta_candidates must start with beta=0")
        if len(set(candidates)) != len(candidates) or any(beta < 0.0 or beta > 1.0 for beta in candidates):
            raise ValueError("ctir_multitask_beta_candidates must be unique values in [0, 1]")
        force_beta = None if args.ctir_multitask_force_beta is None else float(args.ctir_multitask_force_beta)
        if force_beta is not None and force_beta not in candidates:
            raise ValueError("ctir_multitask_force_beta must be a configured candidate")
        stop_after_steps = args.ctir_multitask_stop_after_steps
        if stop_after_steps is not None and stop_after_steps <= 0:
            raise ValueError("ctir_multitask_stop_after_steps must be positive")
        if args.ctir_multitask_probe_count != 32:
            raise ValueError("EXP-CTIR-MULTI-T1-T5 fixes 32 probes per protected task")
        if args.ctir_multitask_tangent_rank != 8 or args.ctir_multitask_raw_rank != 8:
            raise ValueError("EXP-CTIR-MULTI-T1-T5 fixes both task and raw ranks to 8")
        if args.ctir_multitask_union_rtol <= 0.0:
            raise ValueError("ctir_multitask_union_rtol must be positive")
        if args.ctir_multitask_continual_start_step < 0:
            raise ValueError("ctir_multitask_continual_start_step must be non-negative")
        return cls(
            probe_index_path=args.ctir_multitask_probe_index_path,
            log_dir=args.ctir_multitask_log_dir,
            layer_start=args.ctir_multitask_layer_start,
            layer_end=args.ctir_multitask_layer_end,
            tangent_rank=args.ctir_multitask_tangent_rank,
            raw_rank=args.ctir_multitask_raw_rank,
            refresh_interval=args.ctir_multitask_refresh_interval,
            union_rtol=args.ctir_multitask_union_rtol,
            new_descent_ratio=args.ctir_multitask_new_descent_ratio,
            beta_candidates=candidates,
            continual_start_step=args.ctir_multitask_continual_start_step,
            force_beta=force_beta,
            exact_spectrum_check=args.ctir_multitask_exact_spectrum_check,
            stop_after_steps=stop_after_steps,
        )
