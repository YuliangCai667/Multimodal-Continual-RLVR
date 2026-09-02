from __future__ import annotations

import gc
import json
import zlib
from contextlib import contextmanager
from pathlib import Path

import torch

from .geometry_loss import navigation_geometry_loss
from .orthogonal_redirect import frame_alignment, randomized_svd
from .tangent_state import TangentFrame


PROTECTED_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


@contextmanager
def _independent_from_pretrained_context():
    """Keep the rank-0 probe model outside Trainer's active ZeRO-3 context."""
    import transformers.integrations.deepspeed as transformers_deepspeed

    previous = transformers_deepspeed._hf_deepspeed_config_weak_ref
    transformers_deepspeed._hf_deepspeed_config_weak_ref = None
    try:
        yield
    finally:
        transformers_deepspeed._hf_deepspeed_config_weak_ref = previous


def protected_parameter_name(name: str, layer_start: int, layer_end: int) -> bool:
    prefix = "model.language_model.layers."
    if not name.startswith(prefix) or not name.endswith(PROTECTED_SUFFIXES):
        return False
    layer_text = name[len(prefix):].split(".", 1)[0]
    return layer_text.isdigit() and layer_start <= int(layer_text) <= layer_end


class NavigationTangentEstimator:
    """A rank-0 full model keeps old-task differentiation separate from GRPO grads."""

    def __init__(
        self,
        model_path: str,
        probes: list[dict],
        prompt_path: str,
        rank: int,
        layer_start: int,
        layer_end: int,
        device: torch.device,
    ):
        import yaml
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.probes = probes
        self.rank = rank
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.device = device
        with Path(prompt_path).open(encoding="utf-8") as handle:
            self.prompt_template = yaml.safe_load(handle)["Navigation"]
        # Deliberately matches the Figure-B loading path.
        with _independent_from_pretrained_context():
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
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
        expected = (layer_end - layer_start + 1) * len(PROTECTED_SUFFIXES)
        if len(self.parameters) != expected:
            raise RuntimeError(f"Expected {expected} protected matrices, found {len(self.parameters)}")

    @torch.no_grad()
    def copy_parameter(self, name: str, value: torch.Tensor) -> None:
        target = self.all_parameters.get(name)
        if target is not None:
            target.copy_(value.to(device=self.device, dtype=target.dtype))

    def estimate(
        self,
        step: int,
        previous: dict[str, TangentFrame] | None,
        initial: dict[str, TangentFrame] | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, TangentFrame], list[dict], float]:
        self.model.zero_grad(set_to_none=True)
        losses = []
        for position, probe in enumerate(self.probes, 1):
            loss = navigation_geometry_loss(self.model, self.processor, probe, self.prompt_template)
            (loss / len(self.probes)).backward()
            losses.append(float(loss.detach().cpu()))
            del loss
            if position % 8 == 0:
                print(f"[CTIR tangent step={step}] probes {position}/{len(self.probes)}", flush=True)

        gradients: dict[str, torch.Tensor] = {}
        frames: dict[str, TangentFrame] = {}
        rows = []
        for name, parameter in self.parameters.items():
            if parameter.grad is None:
                raise RuntimeError(f"Missing Navigation geometry gradient for {name}")
            gradient = parameter.grad.detach().float()
            seed = 142 + zlib.crc32(name.encode("utf-8"))
            left, singular_values, right = randomized_svd(gradient, self.rank, seed=seed)
            frame = TangentFrame(left.cpu(), singular_values.cpu(), right.cpu())
            gradients[name] = gradient.cpu()
            frames[name] = frame
            threshold = float(singular_values[0].item()) * 1e-6 if singular_values.numel() else 0.0
            rows.append({
                "step": step,
                "navigation_probe_loss": sum(losses) / len(losses),
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
        gc.collect()
        torch.cuda.empty_cache()
        return gradients, frames, rows, sum(losses) / len(losses)

    @property
    def processor(self):
        if not hasattr(self, "_processor"):
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.model.name_or_path, local_files_only=True)
        return self._processor
