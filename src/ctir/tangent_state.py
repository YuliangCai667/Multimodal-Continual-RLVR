from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TangentFrame:
    left: torch.Tensor
    singular_values: torch.Tensor
    right: torch.Tensor


@dataclass
class TangentState:
    refresh_step: int = -1
    frames: dict[str, TangentFrame] | None = None
    gradients: dict[str, torch.Tensor] | None = None
    initial_frames: dict[str, TangentFrame] | None = None
    previous_frames: dict[str, TangentFrame] | None = None

    @property
    def age(self) -> int:
        raise RuntimeError("Use age_at(step) so the optimizer-step index is explicit")

    def age_at(self, step: int) -> int:
        return step - self.refresh_step
