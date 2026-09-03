from __future__ import annotations

import gc
import zlib
from contextlib import contextmanager
from pathlib import Path

import torch

from .multitask_geometry_loss import task_geometry_loss
from .orthogonal_redirect import frame_alignment, randomized_svd
from .tangent_estimator import protected_parameter_name
from .tangent_state import TangentFrame


@contextmanager
def _independent_from_pretrained_context():
    """Keep the rank-0 full probe model outside the active ZeRO-3 context."""
    import transformers.integrations.deepspeed as transformers_deepspeed

    previous = transformers_deepspeed._hf_deepspeed_config_weak_ref
    transformers_deepspeed._hf_deepspeed_config_weak_ref = None
    try:
        yield
    finally:
        transformers_deepspeed._hf_deepspeed_config_weak_ref = previous


class MultiTaskTangentEstimator:
    """One rank-0 full model, differentiated against one old task at a time."""

    def __init__(
        self,
        model_path: str,
        manifests: list[dict],
        prompt_path: str,
        rank: int,
        layer_start: int,
        layer_end: int,
        device: torch.device,
    ):
        import yaml
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.rank = rank
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.device = device
        self.manifests = {manifest["task"]: manifest for manifest in manifests}
        with Path(prompt_path).open(encoding="utf-8") as handle:
            prompt_templates = yaml.safe_load(handle)
        self.prompt_templates = {}
        for task, manifest in self.manifests.items():
            prompt_key = manifest["prompt_key"]
            if prompt_key not in prompt_templates:
                raise KeyError(f"Prompt key {prompt_key!r} for {task} is absent from {prompt_path}")
            self.prompt_templates[task] = prompt_templates[prompt_key]

        with _independent_from_pretrained_context():
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model.to(device)
        self.model.eval()
        self.model.config.use_cache = False
        self.all_parameters = dict(self.model.named_parameters())
        self.parameters = {}
        for name, parameter in self.all_parameters.items():
            selected = protected_parameter_name(name, layer_start, layer_end)
            parameter.requires_grad_(selected)
            if selected:
                self.parameters[name] = parameter
        expected = (layer_end - layer_start + 1) * 7
        if len(self.parameters) != expected:
            raise RuntimeError(f"Expected {expected} protected matrices, found {len(self.parameters)}")

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(self.manifests)

    @torch.no_grad()
    def copy_parameter(self, name: str, value: torch.Tensor) -> None:
        target = self.all_parameters.get(name)
        if target is None:
            raise KeyError(f"Geometry model is missing trainable parameter {name}")
        target.copy_(value.to(device=self.device, dtype=target.dtype))

    def estimate_task(
        self,
        task: str,
        step: int,
        previous: dict[str, TangentFrame] | None,
        initial: dict[str, TangentFrame] | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, TangentFrame], list[dict], float]:
        """Differentiate one equal-weight task mean, then clear every gradient."""
        manifest = self.manifests[task]
        probes = manifest["probes"]
        self.model.zero_grad(set_to_none=True)
        losses: list[float] = []
        for position, probe in enumerate(probes, 1):
            loss = task_geometry_loss(
                self.model,
                self.processor,
                probe,
                self.prompt_templates[task],
            )
            (loss / len(probes)).backward()
            losses.append(float(loss.detach().cpu()))
            del loss
            if position % 8 == 0:
                print(f"[multi-CTIR tangent task={task} step={step}] probes {position}/{len(probes)}", flush=True)

        gradients: dict[str, torch.Tensor] = {}
        frames: dict[str, TangentFrame] = {}
        rows: list[dict] = []
        probe_loss = sum(losses) / len(losses)
        for name, parameter in self.parameters.items():
            if parameter.grad is None:
                raise RuntimeError(f"Missing {task} geometry gradient for {name}")
            gradient = parameter.grad.detach().float()
            seed = int(manifest["seed"]) + zlib.crc32(name.encode("utf-8"))
            left, singular_values, right = randomized_svd(gradient, self.rank, seed=seed)
            frame = TangentFrame(left.cpu(), singular_values.cpu(), right.cpu())
            gradients[name] = gradient.cpu()
            frames[name] = frame
            threshold = float(singular_values[0].item()) * 1e-6 if singular_values.numel() else 0.0
            rows.append({
                "step": step,
                "task": task,
                "probe_loss": probe_loss,
                "probe_count": len(probes),
                "probe_seed": int(manifest["seed"]),
                "target_field": manifest["target"],
                "protected_matrix": name,
                "tangent_singular_values": [float(value) for value in singular_values.cpu()],
                "effective_rank": int((singular_values > threshold).sum().item()),
                "Q_L_alignment_vs_previous": (
                    frame_alignment(frame.left, previous[name].left) if previous and name in previous else None
                ),
                "Q_R_alignment_vs_previous": (
                    frame_alignment(frame.right, previous[name].right) if previous and name in previous else None
                ),
                "Q_L_alignment_vs_step0": (
                    frame_alignment(frame.left, initial[name].left) if initial and name in initial else 1.0
                ),
                "Q_R_alignment_vs_step0": (
                    frame_alignment(frame.right, initial[name].right) if initial and name in initial else 1.0
                ),
            })
            parameter.grad = None
        self.model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        return gradients, frames, rows, probe_loss
