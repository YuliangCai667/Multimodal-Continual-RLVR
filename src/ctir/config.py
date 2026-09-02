from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CTIRConfig:
    probe_path: str
    log_dir: str
    layer_start: int = 9
    layer_end: int = 26
    tangent_rank: int = 8
    raw_rank: int = 8
    refresh_interval: int = 5
    new_descent_ratio: float = 0.90
    beta_candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    force_beta: float | None = None
    exact_spectrum_check: bool = False
    stop_after_steps: int | None = None

    @classmethod
    def from_training_args(cls, args) -> "CTIRConfig":
        candidates = tuple(float(value.strip()) for value in args.ctir_beta_candidates.split(","))
        if not candidates or candidates[0] != 0.0 or any(beta < 0.0 or beta > 1.0 for beta in candidates):
            raise ValueError("ctir_beta_candidates must start with 0 and stay in [0, 1]")
        force_beta = None if args.ctir_force_beta is None else float(args.ctir_force_beta)
        if force_beta is not None and force_beta not in candidates:
            raise ValueError("ctir_force_beta must be one of ctir_beta_candidates")
        stop_after_steps = args.ctir_stop_after_steps
        if stop_after_steps is not None and stop_after_steps <= 0:
            raise ValueError("ctir_stop_after_steps must be positive")
        return cls(
            probe_path=args.ctir_probe_path,
            log_dir=args.ctir_log_dir,
            layer_start=args.ctir_layer_start,
            layer_end=args.ctir_layer_end,
            tangent_rank=args.ctir_tangent_rank,
            raw_rank=args.ctir_raw_rank,
            refresh_interval=args.ctir_refresh_interval,
            new_descent_ratio=args.ctir_new_descent_ratio,
            beta_candidates=candidates,
            force_beta=force_beta,
            exact_spectrum_check=args.ctir_exact_spectrum_check,
            stop_after_steps=stop_after_steps,
        )
