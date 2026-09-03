from __future__ import annotations

from typing import Any

import torch

from .geometry_loss import prepare_teacher_forced_inputs


def task_geometry_loss(
    model,
    processor,
    probe: dict[str, Any],
    prompt_template: str,
) -> torch.Tensor:
    """Per-sample length-normalized assistant-only causal-LM NLL."""
    inputs, labels = prepare_teacher_forced_inputs(
        processor,
        question=probe["question"],
        target_completion=probe["target_completion"],
        image_path=probe["image_path"],
        prompt_template=prompt_template,
        device=next(model.parameters()).device,
    )
    output = model(**inputs, labels=labels, use_cache=False, return_dict=True)
    return output.loss
