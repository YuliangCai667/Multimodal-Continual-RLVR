from __future__ import annotations

from dataclasses import dataclass

import torch

from .tangent_state import TangentFrame


TaskFrames = dict[str, dict[str, TangentFrame]]
TaskGradients = dict[str, dict[str, torch.Tensor]]


@dataclass
class MultiTaskTangentState:
    refresh_step: int = -1
    task_frames: TaskFrames | None = None
    task_gradients: TaskGradients | None = None
    initial_task_frames: TaskFrames | None = None
    previous_task_frames: TaskFrames | None = None

    def age_at(self, step: int) -> int:
        return step - self.refresh_step
