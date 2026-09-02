from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import pickle
import random
import zlib
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
from transformers import TrainerCallback

from .config import CTIRConfig
from .controller import CandidateScore, select_global_beta
from .metrics import JsonlLogger
from .orthogonal_redirect import (
    LowRankOrthogonalMap,
    apply_left,
    apply_right_transpose,
    randomized_svd,
    redirect_delta,
)
from .probe_dataset import load_navigation_probes
from .tangent_estimator import NavigationTangentEstimator, protected_parameter_name
from .tangent_state import TangentFrame, TangentState
from .zero3_utils import (
    broadcast_cpu_tensor,
    full_master_parameter,
    full_tensor_from_local_partition,
    local_gradient,
    local_master_parameter,
    set_full_master_and_low_precision,
    zero3_optimizer,
)


def _cpu_map(mapping: LowRankOrthogonalMap | None) -> LowRankOrthogonalMap | None:
    if mapping is None:
        return None
    return LowRankOrthogonalMap(
        mapping.source_basis.cpu(),
        mapping.normal_basis.cpu(),
        mapping.cosines.cpu(),
        mapping.sines.cpu(),
        mapping.frame_displacement,
    )


def _device_map(mapping: LowRankOrthogonalMap, device: torch.device) -> LowRankOrthogonalMap:
    return LowRankOrthogonalMap(
        mapping.source_basis.to(device),
        mapping.normal_basis.to(device),
        mapping.cosines.to(device),
        mapping.sines.to(device),
        mapping.frame_displacement,
    )


def _state_digest(value) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _local_rng_record() -> dict:
    device_index = torch.cuda.current_device()
    return {
        "rank": dist.get_rank(),
        "python_state_sha256": _state_digest(random.getstate()),
        "numpy_state_sha256": _state_digest(np.random.get_state()),
        "torch_cpu_state_sha256": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        "cuda_device": device_index,
        "cuda_initial_seed": torch.cuda.default_generators[device_index].initial_seed(),
        "torch_cuda_state_sha256": hashlib.sha256(
            torch.cuda.get_rng_state(device_index).cpu().numpy().tobytes()
        ).hexdigest(),
    }


@contextmanager
def _preserve_training_rng():
    """Keep probe-model work from changing the paired GRPO random stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    device_index = torch.cuda.current_device()
    try:
        with torch.random.fork_rng(devices=[device_index], enabled=True):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class ProtectedParameterInterceptor(TrainerCallback):
    """Intercept full FP32 AdamW master deltas at real optimizer boundaries."""

    def __init__(
        self,
        owner,
        config: CTIRConfig,
        model_path: str,
        prompt_path: str,
    ):
        self.owner = owner
        self.config = config
        self.model_path = model_path
        self.prompt_path = prompt_path
        self.logger = JsonlLogger(config.log_dir)
        self.tangent = TangentState()
        self.engine = None
        self.zero = None
        self.protected = {}
        self.before = {}
        self.new_gradients = {}
        self.estimator = None
        self._engine_step = None
        self._trainer_state = None
        self._trainer_control = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.engine = getattr(self.owner, "deepspeed", None) or getattr(self.owner, "model_wrapped", model)
        self.zero = zero3_optimizer(self.engine)
        self._trainer_state = state
        self._trainer_control = control
        module = self.engine.module
        self.protected = {
            name: parameter for name, parameter in module.named_parameters()
            if protected_parameter_name(name, self.config.layer_start, self.config.layer_end)
        }
        expected = (self.config.layer_end - self.config.layer_start + 1) * 7
        if len(self.protected) != expected or any(not hasattr(p, "ds_tensor") for p in self.protected.values()):
            raise RuntimeError(f"CTIR expected {expected} ZeRO-3 protected matrices, found {len(self.protected)}")
        probes = load_navigation_probes(self.config.probe_path)
        if int(probes["count"]) != 32 or int(probes["seed"]) != 142:
            raise RuntimeError("Formal CTIR v1 requires 32 Navigation probes frozen with seed 142")
        device = torch.device("cuda", torch.cuda.current_device())
        local_rng = _local_rng_record()
        rank_rng = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(local_rng, rank_rng, dst=0)
        if dist.get_rank() == 0:
            with _preserve_training_rng():
                self.estimator = NavigationTangentEstimator(
                    self.model_path,
                    probes["probes"],
                    self.prompt_path,
                    self.config.tangent_rank,
                    self.config.layer_start,
                    self.config.layer_end,
                    device,
                )
            resolved = {
                "ctir": asdict(self.config),
                "algorithm": {
                    "transport": "identity-connected principal-plane direct rotation",
                    "spectrum_object": "complete FP32 AdamW master-parameter delta",
                    "raw_rank_role": "dominant-frame control only; no delta truncation",
                    "controller_family": "global beta grid",
                    "controller_objective": (
                        "lexicographic positive-old-constraint violation then "
                        "full-update Frobenius distance"
                    ),
                    "old_harm_threshold": 0.0,
                    "probe_rng_isolated_from_training": True,
                },
                "training": {
                    "seed": args.seed,
                    "data_seed": args.data_seed,
                    "full_determinism": args.full_determinism,
                    "world_size": args.world_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "resolved_training_arguments": args.to_dict(),
                },
                "rng": {
                    "per_rank": rank_rng,
                    "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                },
                "rng_note": "Captured immediately at Trainer on_train_begin, before loading the rank-0 probe model.",
                "model_path": str(Path(self.model_path).resolve()),
                "probe_manifest": str(Path(self.config.probe_path).resolve()),
            }
            target = Path(self.config.log_dir) / "resolved_training_config.json"
            target.write_text(json.dumps(resolved, ensure_ascii=False, indent=2, default=list) + "\n", encoding="utf-8")
        dist.barrier()
        # Accelerate's DeepSpeed wrapper calls the real engine.step() from
        # accelerator.backward(); Trainer's optimizer callbacks surround a
        # separate no-op optimizer wrapper. Intercept the actual boundary.
        self._engine_step = self.engine.step
        self.engine.step = self._intercepted_engine_step

    def _intercepted_engine_step(self, lr_kwargs=None):
        if not self.engine.is_gradient_accumulation_boundary():
            return self._engine_step(lr_kwargs)
        step = int(self._trainer_state.global_step)
        self._before_engine_step(step)
        result = self._engine_step(lr_kwargs)
        self._after_engine_step(step)
        return result

    def on_train_end(self, args, state, control, **kwargs):
        if self.engine is not None and self._engine_step is not None:
            self.engine.step = self._engine_step

    @torch.no_grad()
    def _sync_geometry_model(self) -> None:
        assert self.engine is not None and self.zero is not None
        for name, parameter in self.engine.module.named_parameters():
            if not parameter.requires_grad:
                continue
            full = full_master_parameter(self.zero, parameter)
            if dist.get_rank() == 0:
                self.estimator.copy_parameter(name, full)
            del full
        dist.barrier()

    def _refresh_tangent(self, step: int) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        if step > 0:
            self._sync_geometry_model()
        if dist.get_rank() == 0:
            with _preserve_training_rng():
                gradients, frames, rows, _ = self.estimator.estimate(
                    step,
                    self.tangent.frames,
                    self.tangent.initial_frames,
                )
            for row in rows:
                self.logger.append("tangent_refresh.jsonl", row)
        else:
            gradients = frames = None

        received_gradients = {}
        received_frames = {}
        for name, parameter in self.protected.items():
            shape = tuple(parameter.ds_shape)
            gradient = broadcast_cpu_tensor(
                gradients[name] if dist.get_rank() == 0 else None,
                shape,
                device,
            )
            left_shape = (shape[0], self.config.tangent_rank)
            right_shape = (shape[1], self.config.tangent_rank)
            left = broadcast_cpu_tensor(
                frames[name].left if dist.get_rank() == 0 else None,
                left_shape,
                device,
            )
            right = broadcast_cpu_tensor(
                frames[name].right if dist.get_rank() == 0 else None,
                right_shape,
                device,
            )
            singular_values = broadcast_cpu_tensor(
                frames[name].singular_values if dist.get_rank() == 0 else None,
                (self.config.tangent_rank,),
                device,
            )
            local_size = self.zero.get_local_fp32_param(parameter).numel()
            local_start = dist.get_rank(group=self.zero.dp_process_group) * local_size
            received_gradients[name] = gradient.reshape(-1)[local_start:local_start + local_size].clone()
            received_frames[name] = TangentFrame(left, singular_values, right)

        self.tangent.previous_frames = self.tangent.frames
        self.tangent.frames = received_frames
        self.tangent.gradients = received_gradients
        if self.tangent.initial_frames is None:
            self.tangent.initial_frames = received_frames
        self.tangent.refresh_step = step
        dist.barrier()

    def _before_engine_step(self, step: int) -> None:
        self.before.clear()
        self.new_gradients.clear()
        for name, parameter in self.protected.items():
            self.before[name] = local_master_parameter(self.zero, parameter)
            self.new_gradients[name] = local_gradient(self.zero, parameter)
        if step % self.config.refresh_interval == 0:
            self._refresh_tangent(step)
        if self.tangent.frames is None or self.tangent.gradients is None:
            raise RuntimeError("CTIR optimizer step has no current Navigation tangent")

    def _reward_summary(self) -> dict[str, float]:
        summary = {}
        metrics = getattr(self.owner, "_metrics", {}).get("train", {})
        for key, values in metrics.items():
            if (key == "reward" or key == "reward_std" or key.startswith("rewards/")) and values:
                summary[key] = float(values[-1])
        return summary

    def _after_engine_step(self, step: int) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        beta_values = self.config.beta_candidates
        totals = {
            beta: {
                "new": 0.0,
                "old": 0.0,
                "update_distance_sq": 0.0,
                "map_operator_sum": 0.0,
                "map_operator_max": 0.0,
                "principal_angle_sum": 0.0,
                "principal_angle_max": 0.0,
                "frame_displacement_sum": 0.0,
            }
            for beta in beta_values
        }
        maps: dict[str, dict[float, tuple[LowRankOrthogonalMap | None, LowRankOrthogonalMap | None]]] = {}
        raw_norm_sq = 0.0
        matrix_count = len(self.protected)

        for name, parameter in self.protected.items():
            after_local = local_master_parameter(self.zero, parameter)
            delta_local = after_local - self.before[name]
            self.before[name] = delta_local
            delta = full_tensor_from_local_partition(self.zero, parameter, delta_local)
            new_gradient = full_tensor_from_local_partition(self.zero, parameter, self.new_gradients[name])
            old_gradient = full_tensor_from_local_partition(self.zero, parameter, self.tangent.gradients[name])
            frame = self.tangent.frames[name]
            raw_left, _, raw_right = randomized_svd(
                delta,
                self.config.raw_rank,
                seed=step * 1_000_003 + zlib.crc32(name.encode("utf-8")),
            )
            old_left = frame.left.to(device)
            old_right = frame.right.to(device)
            raw_norm_sq += float(torch.sum(delta * delta, dtype=torch.float64).item())
            maps[name] = {}
            for beta in beta_values:
                redirected, left_map, right_map = redirect_delta(
                    delta, raw_left, raw_right, old_left, old_right, beta,
                )
                totals[beta]["new"] += -float(torch.sum(new_gradient * redirected, dtype=torch.float64).item())
                totals[beta]["old"] += float(torch.sum(old_gradient * redirected, dtype=torch.float64).item())
                if left_map is not None and right_map is not None:
                    difference = redirected - delta
                    totals[beta]["update_distance_sq"] += float(
                        torch.sum(difference * difference, dtype=torch.float64).item()
                    )
                    del difference
                    operator_distance = 0.5 * (
                        left_map.operator_distance + right_map.operator_distance
                    )
                    frame_displacement = 0.5 * (
                        left_map.frame_displacement + right_map.frame_displacement
                    )
                    totals[beta]["map_operator_sum"] += operator_distance
                    totals[beta]["map_operator_max"] = max(
                        totals[beta]["map_operator_max"], operator_distance
                    )
                    principal_angle = 0.5 * (
                        left_map.max_principal_angle + right_map.max_principal_angle
                    )
                    totals[beta]["principal_angle_sum"] += principal_angle
                    totals[beta]["principal_angle_max"] = max(
                        totals[beta]["principal_angle_max"], principal_angle
                    )
                    totals[beta]["frame_displacement_sum"] += frame_displacement
                maps[name][beta] = (_cpu_map(left_map), _cpu_map(right_map))
                del redirected
            del after_local, delta_local, delta, new_gradient, old_gradient, raw_left, raw_right, old_left, old_right

        raw_new = totals[0.0]["new"]
        decision = select_global_beta(
            (
                CandidateScore(
                    beta=beta,
                    new_descent=totals[beta]["new"],
                    old_harm=totals[beta]["old"],
                    update_distance_sq=totals[beta]["update_distance_sq"],
                )
                for beta in beta_values
            ),
            new_descent_ratio=self.config.new_descent_ratio,
            force_beta=self.config.force_beta,
        )
        chosen_beta = decision.beta
        chosen_tensor = torch.tensor(beta_values.index(chosen_beta) if dist.get_rank() == 0 else 0, device=device)
        dist.broadcast(chosen_tensor, src=0)
        chosen_beta = beta_values[int(chosen_tensor.item())]

        ctir_norm_sq = 0.0
        spectrum_certificates = []
        beta0_error_sq = 0.0
        exact_check_suffixes = (
            "layers.9.self_attn.q_proj.weight",
            "layers.9.mlp.down_proj.weight",
        )
        for name, parameter in self.protected.items():
            delta = full_tensor_from_local_partition(self.zero, parameter, self.before[name])
            after = full_master_parameter(self.zero, parameter)
            if chosen_beta == 0.0:
                redirected = delta
                orthogonality_error = 0.0
            else:
                left_map_cpu, right_map_cpu = maps[name][chosen_beta]
                left_map = _device_map(left_map_cpu, device)
                right_map = _device_map(right_map_cpu, device)
                redirected = apply_right_transpose(apply_left(left_map, delta), right_map)
                orthogonality_error = max(left_map.orthogonality_error, right_map.orthogonality_error)
            delta_norm = torch.linalg.vector_norm(delta.double())
            redirected_norm = torch.linalg.vector_norm(redirected.double())
            ctir_norm_sq += float((redirected_norm * redirected_norm).item())
            # A zero-LR warmup step contains only zero matrices.  Their
            # isospectrality error is exactly zero, not |0 / eps - 1| = 1.
            norm_error = (
                float(torch.abs(redirected_norm / delta_norm - 1.0).item())
                if float(delta_norm.item()) > 0.0
                else 0.0
            )
            spectrum_certificates.append(max(norm_error, orthogonality_error))
            if chosen_beta == 0.0:
                beta0_error_sq += float(torch.sum((redirected - delta) ** 2, dtype=torch.float64).item())
            if (
                self.config.exact_spectrum_check
                and step + 1 == (self.config.stop_after_steps or 1)
                and chosen_beta > 0.0
                and name.endswith(exact_check_suffixes)
            ):
                if dist.get_rank() == 0:
                    # FP32 SVD itself has a ~1e-4 comparison floor for these
                    # dense, clustered spectra. Measure the FP32 updates in
                    # FP64 so the gate tests the redirect rather than LAPACK
                    # precision noise.
                    raw_singular_values = torch.linalg.svdvals(delta.double())
                    ctir_singular_values = torch.linalg.svdvals(redirected.double())
                    exact_error = float(
                        torch.linalg.vector_norm(ctir_singular_values - raw_singular_values)
                        / torch.linalg.vector_norm(raw_singular_values).clamp_min(1e-30)
                    )
                    self.logger.append("correctness.jsonl", {
                        "test": "exact_full_spectrum",
                        "local_step": step + 1,
                        "protected_matrix": name,
                        "chosen_beta": chosen_beta,
                        "full_singular_value_count": int(raw_singular_values.numel()),
                        "measurement_dtype": "float64",
                        "spectrum_relative_error": exact_error,
                        "frob_ratio": float(redirected_norm / delta_norm.clamp_min(1e-30)),
                    })
                    del raw_singular_values, ctir_singular_values
                dist.barrier()
            # beta=0 is a strict no-op: leave both raw FP32 master and BF16 shard untouched.
            if chosen_beta != 0.0:
                set_full_master_and_low_precision(self.zero, parameter, after + (redirected - delta))
            del after, delta, redirected

        raw_norm = math.sqrt(raw_norm_sq)
        ctir_norm = math.sqrt(ctir_norm_sq)
        update_distance = math.sqrt(totals[chosen_beta]["update_distance_sq"])
        raw_old = totals[0.0]["old"]
        ctir_old = totals[chosen_beta]["old"]
        ctir_new = totals[chosen_beta]["new"]
        ratio = ctir_new / raw_new if raw_new != 0.0 else float("nan")
        update_cosine = (
            (raw_norm * raw_norm + ctir_norm * ctir_norm - update_distance * update_distance)
            / (2.0 * raw_norm * ctir_norm)
            if raw_norm and ctir_norm
            else 1.0
        )
        candidate_diagnostics = {
            str(beta): {
                "old_harm": totals[beta]["old"],
                "positive_old_constraint_violation": max(totals[beta]["old"], 0.0),
                "new_descent": totals[beta]["new"],
                "new_descent_ratio": totals[beta]["new"] / raw_new if raw_new != 0.0 else float("nan"),
                "new_constraint_satisfied": (
                    raw_new > 0.0 and totals[beta]["new"] >= decision.new_descent_threshold
                ),
                "old_constraint_satisfied": totals[beta]["old"] <= 0.0,
                "relative_update_distance": (
                    math.sqrt(totals[beta]["update_distance_sq"]) / raw_norm if raw_norm else 0.0
                ),
            }
            for beta in beta_values
        }
        losses = getattr(self.owner, "_ctir_step_losses", [])
        row = {
            "local_step": step + 1,
            "global_continual_step": 901 + step,
            "grpo_loss": sum(losses) / len(losses) if losses else None,
            "reward_summary": self._reward_summary(),
            "chosen_beta": chosen_beta,
            "raw_update_fro_norm": raw_norm,
            "ctir_update_fro_norm": ctir_norm,
            "frob_ratio": ctir_norm / raw_norm if raw_norm else float("nan"),
            "update_distance_fro_norm": update_distance,
            "relative_update_distance": update_distance / raw_norm if raw_norm else 0.0,
            "raw_redirected_update_cosine": max(-1.0, min(1.0, update_cosine)),
            "raw_old_harm": raw_old,
            "ctir_old_harm": ctir_old,
            "old_harm_reduction": raw_old - ctir_old,
            "raw_new_descent": raw_new,
            "ctir_new_descent": ctir_new,
            "new_descent_ratio": ratio,
            "controller": {
                "family": "global_beta_grid",
                "objective": "lexicographic positive-old-violation then full-update Frobenius distance",
                "status": decision.status,
                "jointly_feasible": decision.jointly_feasible,
                "old_harm_threshold": 0.0,
                "old_harm_violation": decision.old_harm_violation,
                "new_descent_threshold": decision.new_descent_threshold,
                "new_descent_ratio_floor": self.config.new_descent_ratio,
                "candidates": candidate_diagnostics,
            },
            "mean_rotation_strength": totals[chosen_beta]["map_operator_sum"] / matrix_count,
            "max_rotation_strength": totals[chosen_beta]["map_operator_max"],
            "rotation_strength_kind": "mean/max left-right direct-rotation operator distance ||R-I||_2",
            "mean_max_principal_angle_radians": totals[chosen_beta]["principal_angle_sum"] / matrix_count,
            "max_principal_angle_radians": totals[chosen_beta]["principal_angle_max"],
            "mean_frame_displacement": totals[chosen_beta]["frame_displacement_sum"] / matrix_count,
            "max_spectrum_error": max(spectrum_certificates),
            "mean_spectrum_error": sum(spectrum_certificates) / len(spectrum_certificates),
            "spectrum_error_kind": "direct-rotation basis/trigonometric residual plus Frobenius certificate; exact FP64 SVD is correctness-gate only",
            "tangent_age": self.tangent.age_at(step),
            "beta0_relative_error": math.sqrt(beta0_error_sq) / raw_norm if raw_norm else 0.0,
        }
        if dist.get_rank() == 0:
            self.logger.append("step_metrics.jsonl", row)
            if chosen_beta == 0.0:
                self.logger.append("correctness.jsonl", {
                    "test": "beta0_equivalence",
                    "local_step": step + 1,
                    "relative_error": row["beta0_relative_error"],
                    "implementation": "strict no-op after raw optimizer.step; no FP32/BF16 rewrite",
                })
            print(
                f"[CTIR step={step + 1}] beta={chosen_beta:.2f} old={raw_old:+.6e}->{ctir_old:+.6e} "
                f"new_ratio={ratio:.5f} frob_ratio={row['frob_ratio']:.8f}",
                flush=True,
            )
        self.owner._ctir_step_losses.clear()
        self.before.clear()
        self.new_gradients.clear()
        maps.clear()
        gc.collect()
        torch.cuda.empty_cache()
        if self.config.stop_after_steps is not None and step + 1 >= self.config.stop_after_steps:
            self._trainer_control.should_training_stop = True
