from __future__ import annotations

import torch
import torch.distributed as dist


def zero3_optimizer(model):
    optimizer = getattr(model, "optimizer", None)
    required = ("get_full_hp_param", "set_full_hp_param", "get_fp32_grad_for_param")
    if optimizer is None or not all(hasattr(optimizer, name) for name in required):
        raise RuntimeError("CTIR requires a DeepSpeed ZeRO-3 optimizer with full-HP parameter APIs")
    return optimizer


def full_master_parameter(optimizer, parameter: torch.nn.Parameter) -> torch.Tensor:
    value = optimizer.get_full_hp_param(parameter)
    if value is None or tuple(value.shape) != tuple(parameter.ds_shape):
        raise RuntimeError("ZeRO-3 failed to gather a complete FP32 master parameter")
    return value.detach().float()


def full_gradient(optimizer, parameter: torch.nn.Parameter) -> torch.Tensor:
    value = optimizer.get_fp32_grad_for_param(parameter)
    if value is None or tuple(value.shape) != tuple(parameter.ds_shape):
        raise RuntimeError("ZeRO-3 failed to gather a complete FP32 gradient")
    return value.detach().float()


def local_master_parameter(optimizer, parameter: torch.nn.Parameter) -> torch.Tensor:
    value = optimizer.get_local_fp32_param(parameter)
    if value is None:
        raise RuntimeError("ZeRO-3 failed to expose the local FP32 master partition")
    return value.detach().float().cpu().clone()


def local_gradient(optimizer, parameter: torch.nn.Parameter) -> torch.Tensor:
    value = optimizer.get_local_fp32_grad_for_param(parameter)
    if value is None:
        raise RuntimeError("ZeRO-3 failed to expose the local FP32 gradient partition")
    return value.detach().float().cpu().clone()


def full_tensor_from_local_partition(
    optimizer,
    parameter: torch.nn.Parameter,
    local_value: torch.Tensor,
) -> torch.Tensor:
    expected = optimizer.get_local_fp32_param(parameter).numel()
    if local_value.numel() != expected:
        raise RuntimeError(f"Local partition has {local_value.numel()} values; expected {expected}")
    return optimizer._fp32_state_allgather(parameter, local_value).detach().float()


def set_full_master_and_low_precision(
    optimizer,
    parameter: torch.nn.Parameter,
    value: torch.Tensor,
) -> None:
    """Write the same full result into the FP32 Adam master and BF16 ZeRO partition."""
    optimizer.set_full_hp_param(value, parameter)
    rank = dist.get_rank(group=optimizer.dp_process_group)
    local = parameter.ds_tensor.view(-1)
    start = rank * local.numel()
    valid = max(0, min(local.numel(), parameter.ds_numel - start))
    if valid:
        local[:valid].copy_(value.reshape(-1)[start:start + valid].to(local.dtype))


def broadcast_cpu_tensor(value: torch.Tensor | None, shape: torch.Size | tuple[int, ...], device: torch.device) -> torch.Tensor:
    if dist.get_rank() == 0:
        if value is None:
            raise ValueError("rank 0 must supply the broadcast tensor")
        # Randomized-SVD right frames are transposed views; NCCL requires a
        # contiguous input buffer even when rank 0 is only broadcasting it.
        buffer = value.to(device=device, dtype=torch.float32).contiguous()
    else:
        buffer = torch.empty(shape, device=device, dtype=torch.float32)
    dist.broadcast(buffer, src=0)
    return buffer.cpu()
