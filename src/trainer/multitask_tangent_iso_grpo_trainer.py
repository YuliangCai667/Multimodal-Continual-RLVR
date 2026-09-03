from __future__ import annotations

from trl import GRPOTrainer

from src.ctir.multitask_optimizer_interceptor import MultiTaskProtectedParameterInterceptor


class MultiTaskTangentIsoGRPOTrainer(GRPOTrainer):
    """Baseline GRPO plus multi-task CTIR once per real optimizer step."""

    def __init__(self, *, ctir_config, ctir_model_path: str, ctir_prompt_path: str, **kwargs):
        self._ctir_multitask_step_losses: list[float] = []
        super().__init__(**kwargs)
        self.add_callback(MultiTaskProtectedParameterInterceptor(
            self,
            ctir_config,
            ctir_model_path,
            ctir_prompt_path,
        ))

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)
        self._ctir_multitask_step_losses.append(float(loss.detach().cpu()))
        return loss
