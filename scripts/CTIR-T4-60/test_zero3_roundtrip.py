#!/usr/bin/env python3
import torch
import torch.distributed as dist
from torch import nn

import deepspeed

from src.ctir.zero3_utils import (
    full_master_parameter,
    local_gradient,
    local_master_parameter,
    set_full_master_and_low_precision,
    zero3_optimizer,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)

    def forward(self, values):
        return self.proj(values).float().square().mean()


def main() -> None:
    deepspeed.init_distributed()
    torch.manual_seed(142)
    model = TinyModel().cuda()
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
        },
    }
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    zero = zero3_optimizer(engine)
    parameter = engine.module.proj.weight
    before = local_master_parameter(zero, parameter)
    loss = engine(torch.randn(2, 16, device=engine.device, dtype=torch.bfloat16))
    engine.backward(loss)
    assert local_gradient(zero, parameter).abs().sum() > 0
    engine.step()
    raw = full_master_parameter(zero, parameter)
    assert torch.linalg.vector_norm(raw - before.to(raw.device).view_as(raw)) > 0

    redirected = raw.T.contiguous()
    set_full_master_and_low_precision(zero, parameter, redirected)
    master_error = torch.linalg.vector_norm(full_master_parameter(zero, parameter) - redirected)
    shard_expected = redirected.to(torch.bfloat16).flatten().float().cpu()
    shard_error = torch.linalg.vector_norm(parameter.ds_tensor.float() - shard_expected)
    if dist.get_rank() == 0:
        print({"master_error": float(master_error), "bf16_shard_error": float(shard_error)})
    assert float(master_error) == 0.0
    assert float(shard_error) == 0.0


if __name__ == "__main__":
    main()
