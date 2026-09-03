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

import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainerCallback

from .metrics import JsonlLogger
from .multitask_config import MultiTaskCTIRConfig
from .multitask_controller import MultiTaskCandidateScore, select_multitask_global_beta
from .multitask_probe_dataset import load_probe_index
from .multitask_tangent_estimator import MultiTaskTangentEstimator
from .multitask_tangent_state import MultiTaskTangentState
from .orthogonal_redirect import (
    LowRankOrthogonalMap,
    apply_left,
    apply_right_transpose,
    randomized_svd,
    redirect_delta,
)
from .subspace_union import rank_revealing_union
from .tangent_estimator import protected_parameter_name
from .tangent_state import TangentFrame
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


def commit_redirected_update(optimizer, parameter, delta, redirected, beta: float) -> bool:
    """Commit a nonzero-beta correction; beta=0 performs no optimizer write."""
    if beta == 0.0:
        return False
    after = full_master_parameter(optimizer, parameter)
    set_full_master_and_low_precision(optimizer, parameter, after + (redirected - delta))
    return True


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
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    device_index = torch.cuda.current_device()
    try:
        with torch.random.fork_rng(devices=[device_index], enabled=True):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class MultiTaskProtectedParameterInterceptor(TrainerCallback):
    """Redirect complete post-AdamW FP32 master deltas at true engine steps."""

    def __init__(self, owner, config: MultiTaskCTIRConfig, model_path: str, prompt_path: str):
        self.owner = owner
        self.config = config
        self.model_path = model_path
        self.prompt_path = prompt_path
        self.logger = JsonlLogger(config.log_dir)
        self.tangent = MultiTaskTangentState()
        self.engine = None
        self.zero = None
        self.protected: dict[str, torch.nn.Parameter] = {}
        self.before: dict[str, torch.Tensor] = {}
        self.new_gradients: dict[str, torch.Tensor] = {}
        self.estimator: MultiTaskTangentEstimator | None = None
        self.index: dict | None = None
        self.manifests: list[dict] = []
        self.tasks: tuple[str, ...] = ()
        self._engine_step = None
        self._trainer_state = None
        self._trainer_control = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.engine = getattr(self.owner, "deepspeed", None) or getattr(self.owner, "model_wrapped", model)
        self.zero = zero3_optimizer(self.engine)
        self._trainer_state = state
        self._trainer_control = control
        self.protected = {
            name: parameter
            for name, parameter in self.engine.module.named_parameters()
            if protected_parameter_name(name, self.config.layer_start, self.config.layer_end)
        }
        expected = (self.config.layer_end - self.config.layer_start + 1) * 7
        if len(self.protected) != expected or any(not hasattr(parameter, "ds_tensor") for parameter in self.protected.values()):
            raise RuntimeError(f"Multi-task CTIR expected {expected} ZeRO-3 protected matrices, found {len(self.protected)}")
        self.index, self.manifests = load_probe_index(self.config.probe_index_path)
        self.tasks = tuple(manifest["task"] for manifest in self.manifests)

        device = torch.device("cuda", torch.cuda.current_device())
        local_rng = _local_rng_record()
        rank_rng = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(local_rng, rank_rng, dst=0)
        if dist.get_rank() == 0:
            with _preserve_training_rng():
                self.estimator = MultiTaskTangentEstimator(
                    self.model_path,
                    self.manifests,
                    self.prompt_path,
                    self.config.tangent_rank,
                    self.config.layer_start,
                    self.config.layer_end,
                    device,
                )
            task_specs = [
                {
                    key: manifest[key]
                    for key in (
                        "task", "prompt_key", "target", "count", "seed", "source_path",
                        "image_root", "eligible_record_count", "excluded_empty_target_indices",
                        "selected_source_indices", "manifest_path",
                    )
                }
                for manifest in self.manifests
            ]
            resolved = {
                "experiment_id": "EXP-CTIR-MULTI-T1-T5-001",
                "ctir_multitask": asdict(self.config),
                "probe_index": str(Path(self.config.probe_index_path).resolve()),
                "protected_task_specs": task_specs,
                "algorithm": {
                    "transport": "one global identity-connected principal-plane direct rotation per beta",
                    "spectrum_object": "complete post-AdamW FP32 master-parameter delta",
                    "raw_rank_role": "rank-8 direction control only; full update including spectral tail is transported",
                    "old_task_geometry": "separate gradients and rank-8 frames; FP64 rank-revealing union",
                    "old_task_constraint": "H_k <= 0 independently for every old task",
                    "normalization": "Hhat_k=H_k/(||G_k||_F*||delta_raw||_F+eps)",
                    "controller": "lexicographic worst positive Hhat_k, update distance, smaller beta",
                    "new_task_floor": self.config.new_descent_ratio,
                    "probe_rng_isolated_from_training": True,
                },
                "training": {
                    "seed": args.seed,
                    "data_seed": args.data_seed,
                    "full_determinism": args.full_determinism,
                    "world_size": args.world_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "nominal_global_batch": (
                        args.world_size * args.gradient_accumulation_steps * args.per_device_train_batch_size
                    ),
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
                "model_path": str(Path(self.model_path).resolve()),
            }
            target = Path(self.config.log_dir) / "resolved_training_config.json"
            target.write_text(json.dumps(resolved, ensure_ascii=False, indent=2, default=list) + "\n", encoding="utf-8")
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        self._engine_step = self.engine.step
        self.engine.step = self._intercepted_engine_step

    def _intercepted_engine_step(self, *args, **kwargs):
        if not self.engine.is_gradient_accumulation_boundary():
            return self._engine_step(*args, **kwargs)
        step = int(self._trainer_state.global_step)
        self._before_engine_step(step)
        result = self._engine_step(*args, **kwargs)
        self._after_engine_step(step)
        return result

    def on_train_end(self, args, state, control, **kwargs):
        if self.engine is not None and self._engine_step is not None:
            self.engine.step = self._engine_step

    @torch.no_grad()
    def _sync_geometry_model(self) -> None:
        """Synchronize every trainable parameter once before task-wise probes."""
        if dist.get_rank() == 0 and self.estimator is None:
            raise RuntimeError("rank 0 has no multi-task geometry model")
        synchronized = 0
        for name, parameter in self.engine.module.named_parameters():
            if not parameter.requires_grad:
                continue
            full = full_master_parameter(self.zero, parameter)
            if dist.get_rank() == 0:
                self.estimator.copy_parameter(name, full)
            synchronized += 1
            del full
        if synchronized == 0:
            raise RuntimeError("No trainable parameters were synchronized into the geometry model")
        dist.barrier()

    def _local_partition(self, full: torch.Tensor, local_size: int) -> torch.Tensor:
        rank = dist.get_rank(group=self.zero.dp_process_group)
        start = rank * local_size
        valid = max(0, min(local_size, full.numel() - start))
        local = torch.zeros(local_size, dtype=torch.float32)
        if valid:
            local[:valid].copy_(full.reshape(-1)[start:start + valid].cpu())
        return local

    def _refresh_tangents(self, step: int) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        self._sync_geometry_model()
        previous_all = self.tangent.task_frames or {}
        initial_all = self.tangent.initial_task_frames or {}
        received_task_gradients: dict[str, dict[str, torch.Tensor]] = {}
        received_task_frames: dict[str, dict[str, TangentFrame]] = {}

        for task in self.tasks:
            if dist.get_rank() == 0:
                with _preserve_training_rng():
                    gradients, frames, rows, _ = self.estimator.estimate_task(
                        task,
                        step,
                        previous_all.get(task),
                        initial_all.get(task),
                    )
                for row in rows:
                    self.logger.append("tangent_refresh.jsonl", row)
            else:
                gradients = frames = None

            received_gradients: dict[str, torch.Tensor] = {}
            received_frames: dict[str, TangentFrame] = {}
            for name, parameter in self.protected.items():
                shape = tuple(parameter.ds_shape)
                gradient = broadcast_cpu_tensor(
                    gradients[name] if dist.get_rank() == 0 else None,
                    shape,
                    device,
                )
                left = broadcast_cpu_tensor(
                    frames[name].left if dist.get_rank() == 0 else None,
                    (shape[0], self.config.tangent_rank),
                    device,
                )
                right = broadcast_cpu_tensor(
                    frames[name].right if dist.get_rank() == 0 else None,
                    (shape[1], self.config.tangent_rank),
                    device,
                )
                singular_values = broadcast_cpu_tensor(
                    frames[name].singular_values if dist.get_rank() == 0 else None,
                    (self.config.tangent_rank,),
                    device,
                )
                local_size = self.zero.get_local_fp32_param(parameter).numel()
                received_gradients[name] = self._local_partition(gradient, local_size)
                received_frames[name] = TangentFrame(left, singular_values, right)
                del gradient, left, right, singular_values
            received_task_gradients[task] = received_gradients
            received_task_frames[task] = received_frames
            if dist.get_rank() == 0:
                del gradients, frames
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()

        self.tangent.previous_task_frames = self.tangent.task_frames
        self.tangent.task_frames = received_task_frames
        self.tangent.task_gradients = received_task_gradients
        if self.tangent.initial_task_frames is None:
            self.tangent.initial_task_frames = received_task_frames
        self.tangent.refresh_step = step

    def _before_engine_step(self, step: int) -> None:
        self.before.clear()
        self.new_gradients.clear()
        for name, parameter in self.protected.items():
            self.before[name] = local_master_parameter(self.zero, parameter)
            self.new_gradients[name] = local_gradient(self.zero, parameter)
        if step % self.config.refresh_interval == 0:
            self._refresh_tangents(step)
        if self.tangent.task_frames is None or self.tangent.task_gradients is None:
            raise RuntimeError("Multi-task CTIR optimizer step has no current task tangents")

    def _reward_summary(self) -> dict[str, float]:
        summary = {}
        metrics = getattr(self.owner, "_metrics", {}).get("train", {})
        for key, values in metrics.items():
            if (key == "reward" or key == "reward_std" or key.startswith("rewards/")) and values:
                summary[key] = float(values[-1])
        return summary

    def _all_reduce_float(self, value: float, device: torch.device) -> float:
        tensor = torch.tensor(value, device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.zero.dp_process_group)
        return float(tensor.item())

    def _candidate_local_view(self, full: torch.Tensor, local_size: int) -> torch.Tensor:
        rank = dist.get_rank(group=self.zero.dp_process_group)
        start = rank * local_size
        valid = max(0, min(local_size, full.numel() - start))
        return full.reshape(-1)[start:start + valid]

    def _after_engine_step(self, step: int) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        beta_values = self.config.beta_candidates
        totals = {
            beta: {
                "new_local": 0.0,
                "old_local": {task: 0.0 for task in self.tasks},
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
        union_rows: list[dict] = []
        old_gradient_norm_sq_local = {task: 0.0 for task in self.tasks}
        raw_norm_sq = 0.0
        matrix_count = len(self.protected)

        for name, parameter in self.protected.items():
            after_local = local_master_parameter(self.zero, parameter)
            delta_local = after_local - self.before[name]
            self.before[name] = delta_local
            delta = full_tensor_from_local_partition(self.zero, parameter, delta_local)
            local_size = delta_local.numel()
            new_local = self.new_gradients[name].to(device)
            old_locals = {
                task: self.tangent.task_gradients[task][name].to(device)
                for task in self.tasks
            }
            for task, gradient in old_locals.items():
                old_gradient_norm_sq_local[task] += float(torch.sum(gradient.double().square()).item())

            raw_left, _, raw_right = randomized_svd(
                delta,
                self.config.raw_rank,
                seed=step * 1_000_003 + zlib.crc32(name.encode("utf-8")),
            )
            left_union = rank_revealing_union(
                [self.tangent.task_frames[task][name].left.to(device) for task in self.tasks],
                rtol=self.config.union_rtol,
            )
            right_union = rank_revealing_union(
                [self.tangent.task_frames[task][name].right.to(device) for task in self.tasks],
                rtol=self.config.union_rtol,
            )
            raw_norm_sq += float(torch.sum(delta.double().square()).item())
            maps[name] = {}
            union_rows.append({
                "local_step": step + 1,
                "protected_matrix": name,
                "task_count": len(self.tasks),
                "per_task_rank": self.config.tangent_rank,
                "left_input_columns": left_union.input_columns,
                "left_effective_rank": left_union.effective_rank,
                "left_singular_values": [float(value) for value in left_union.singular_values.cpu()],
                "right_input_columns": right_union.input_columns,
                "right_effective_rank": right_union.effective_rank,
                "right_singular_values": [float(value) for value in right_union.singular_values.cpu()],
                "union_rtol": self.config.union_rtol,
            })

            for beta in beta_values:
                redirected, left_map, right_map = redirect_delta(
                    delta,
                    raw_left,
                    raw_right,
                    left_union.basis,
                    right_union.basis,
                    beta,
                )
                candidate_local = self._candidate_local_view(redirected, local_size)
                valid = candidate_local.numel()
                totals[beta]["new_local"] -= float(torch.sum(new_local[:valid].double() * candidate_local.double()).item())
                for task, gradient in old_locals.items():
                    totals[beta]["old_local"][task] += float(
                        torch.sum(gradient[:valid].double() * candidate_local.double()).item()
                    )
                if left_map is not None and right_map is not None:
                    difference = redirected - delta
                    totals[beta]["update_distance_sq"] += float(torch.sum(difference.double().square()).item())
                    operator_distance = 0.5 * (left_map.operator_distance + right_map.operator_distance)
                    principal_angle = 0.5 * (left_map.max_principal_angle + right_map.max_principal_angle)
                    frame_displacement = 0.5 * (left_map.frame_displacement + right_map.frame_displacement)
                    totals[beta]["map_operator_sum"] += operator_distance
                    totals[beta]["map_operator_max"] = max(totals[beta]["map_operator_max"], operator_distance)
                    totals[beta]["principal_angle_sum"] += principal_angle
                    totals[beta]["principal_angle_max"] = max(totals[beta]["principal_angle_max"], principal_angle)
                    totals[beta]["frame_displacement_sum"] += frame_displacement
                    del difference
                maps[name][beta] = (_cpu_map(left_map), _cpu_map(right_map))
                del candidate_local, redirected
            del after_local, delta_local, delta, new_local, old_locals
            del raw_left, raw_right, left_union, right_union

        for beta in beta_values:
            totals[beta]["new"] = self._all_reduce_float(totals[beta]["new_local"], device)
            totals[beta]["old"] = {
                task: self._all_reduce_float(totals[beta]["old_local"][task], device)
                for task in self.tasks
            }
        old_gradient_norms = {
            task: math.sqrt(self._all_reduce_float(old_gradient_norm_sq_local[task], device))
            for task in self.tasks
        }
        raw_norm = math.sqrt(raw_norm_sq)
        eps = 1e-30
        for beta in beta_values:
            totals[beta]["normalized_old"] = {
                task: totals[beta]["old"][task] / (old_gradient_norms[task] * raw_norm + eps)
                for task in self.tasks
            }

        candidates = tuple(
            MultiTaskCandidateScore(
                beta=beta,
                new_descent=totals[beta]["new"],
                old_harms=totals[beta]["old"],
                normalized_old_harms=totals[beta]["normalized_old"],
                update_distance_sq=totals[beta]["update_distance_sq"],
            )
            for beta in beta_values
        )
        decision = select_multitask_global_beta(
            candidates,
            new_descent_ratio=self.config.new_descent_ratio,
            force_beta=self.config.force_beta,
        )
        chosen_index = beta_values.index(decision.beta) if dist.get_rank() == 0 else 0
        chosen_tensor = torch.tensor(chosen_index, device=device, dtype=torch.int64)
        dist.broadcast(chosen_tensor, src=0)
        chosen_beta = beta_values[int(chosen_tensor.item())]

        ctir_norm_sq = 0.0
        spectrum_certificates: list[float] = []
        beta0_write_count = 0
        exact_check_suffixes = (
            "layers.9.self_attn.q_proj.weight",
            # A second, smaller attention matrix keeps the required full FP64
            # SVD gate meaningful without creating the MLP matrix's much
            # larger LAPACK workspace on an 80-GB rank-0 GPU.
            "layers.9.self_attn.k_proj.weight",
        )
        for name, parameter in self.protected.items():
            delta = full_tensor_from_local_partition(self.zero, parameter, self.before[name])
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
            ctir_norm_sq += float(redirected_norm.square().item())
            norm_error = (
                float(torch.abs(redirected_norm / delta_norm - 1.0).item())
                if float(delta_norm.item()) > 0.0 else 0.0
            )
            spectrum_certificates.append(max(norm_error, orthogonality_error))

            should_exact_check = (
                self.config.exact_spectrum_check
                and step + 1 == (self.config.stop_after_steps or 1)
                and name.endswith(exact_check_suffixes)
            )
            if should_exact_check:
                if dist.get_rank() == 0:
                    raw_singular_values = torch.linalg.svdvals(delta.double())
                    ctir_singular_values = torch.linalg.svdvals(redirected.double())
                    exact_error = float(
                        torch.linalg.vector_norm(ctir_singular_values - raw_singular_values)
                        / torch.linalg.vector_norm(raw_singular_values).clamp_min(1e-30)
                    )
                    self.logger.append("correctness.jsonl", {
                        "test": "sampled_exact_full_spectrum",
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

            if commit_redirected_update(self.zero, parameter, delta, redirected, chosen_beta):
                beta0_write_count += 1
            del delta, redirected

        ctir_norm = math.sqrt(ctir_norm_sq)
        update_distance = math.sqrt(totals[chosen_beta]["update_distance_sq"])
        raw_new = totals[0.0]["new"]
        ctir_new = totals[chosen_beta]["new"]
        update_cosine = (
            (raw_norm * raw_norm + ctir_norm * ctir_norm - update_distance * update_distance)
            / (2.0 * raw_norm * ctir_norm)
            if raw_norm and ctir_norm else 1.0
        )
        candidate_diagnostics = {
            str(beta): {
                "new_descent": totals[beta]["new"],
                "new_descent_ratio": totals[beta]["new"] / raw_new if raw_new != 0.0 else float("nan"),
                "new_constraint_satisfied": raw_new > 0.0 and totals[beta]["new"] >= decision.new_descent_threshold,
                "old_harms": totals[beta]["old"],
                "normalized_old_harms": totals[beta]["normalized_old"],
                "per_task_old_constraint_satisfied": {
                    task: totals[beta]["old"][task] <= 0.0 for task in self.tasks
                },
                "worst_positive_normalized_old_violation": max(
                    max(value, 0.0) for value in totals[beta]["normalized_old"].values()
                ),
                "update_distance_fro_norm": math.sqrt(totals[beta]["update_distance_sq"]),
                "relative_update_distance": (
                    math.sqrt(totals[beta]["update_distance_sq"]) / raw_norm if raw_norm else 0.0
                ),
            }
            for beta in beta_values
        }
        losses = getattr(self.owner, "_ctir_multitask_step_losses", [])
        row = {
            "local_step": step + 1,
            "global_continual_step": self.config.continual_start_step + step + 1,
            "protected_tasks": list(self.tasks),
            "grpo_loss": sum(losses) / len(losses) if losses else None,
            "reward_summary": self._reward_summary(),
            "chosen_beta": chosen_beta,
            "raw_update_fro_norm": raw_norm,
            "ctir_update_fro_norm": ctir_norm,
            "frob_ratio": ctir_norm / raw_norm if raw_norm else float("nan"),
            "update_distance_fro_norm": update_distance,
            "relative_update_distance": update_distance / raw_norm if raw_norm else 0.0,
            "raw_redirected_update_cosine": max(-1.0, min(1.0, update_cosine)),
            "old_gradient_fro_norms": old_gradient_norms,
            "raw_old_harms": totals[0.0]["old"],
            "ctir_old_harms": totals[chosen_beta]["old"],
            "raw_normalized_old_harms": totals[0.0]["normalized_old"],
            "ctir_normalized_old_harms": totals[chosen_beta]["normalized_old"],
            "raw_new_descent": raw_new,
            "ctir_new_descent": ctir_new,
            "new_descent_ratio": ctir_new / raw_new if raw_new != 0.0 else float("nan"),
            "controller": {
                "family": "one_global_beta_grid_for_all_tasks_and_matrices",
                "objective": "lexicographic worst positive normalized per-task harm, update distance, smaller beta",
                "status": decision.status,
                "jointly_feasible": decision.jointly_feasible,
                "per_task_old_harm_threshold": 0.0,
                "worst_normalized_violation": decision.worst_normalized_violation,
                "new_descent_threshold": decision.new_descent_threshold,
                "new_descent_ratio_floor": self.config.new_descent_ratio,
                "candidates": candidate_diagnostics,
            },
            "mean_rotation_strength": totals[chosen_beta]["map_operator_sum"] / matrix_count,
            "max_rotation_strength": totals[chosen_beta]["map_operator_max"],
            "mean_max_principal_angle_radians": totals[chosen_beta]["principal_angle_sum"] / matrix_count,
            "max_principal_angle_radians": totals[chosen_beta]["principal_angle_max"],
            "mean_frame_displacement": totals[chosen_beta]["frame_displacement_sum"] / matrix_count,
            "max_spectrum_certificate": max(spectrum_certificates),
            "mean_spectrum_certificate": sum(spectrum_certificates) / len(spectrum_certificates),
            "tangent_age": self.tangent.age_at(step),
            "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        if dist.get_rank() == 0:
            for union_row in union_rows:
                self.logger.append("union_rank.jsonl", union_row)
            self.logger.append("step_metrics.jsonl", row)
            if chosen_beta == 0.0:
                self.logger.append("correctness.jsonl", {
                    "test": "beta0_strict_noop",
                    "local_step": step + 1,
                    "post_optimizer_write_calls": beta0_write_count,
                    "implementation": "no set_full_hp_param and no BF16 shard rewrite",
                })
            print(
                f"[multi-CTIR step={step + 1}] beta={chosen_beta:.2f} "
                f"worst_Hhat+={decision.worst_normalized_violation:.6e} "
                f"new_ratio={row['new_descent_ratio']:.5f} frob_ratio={row['frob_ratio']:.8f}",
                flush=True,
            )
        self.owner._ctir_multitask_step_losses.clear()
        self.before.clear()
        self.new_gradients.clear()
        maps.clear()
        gc.collect()
        torch.cuda.empty_cache()
        if self.config.stop_after_steps is not None and step + 1 >= self.config.stop_after_steps:
            self._trainer_control.should_training_stop = True
