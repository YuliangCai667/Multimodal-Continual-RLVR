from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class UnionBasis:
    basis: torch.Tensor
    singular_values: torch.Tensor
    input_columns: int
    effective_rank: int


def rank_revealing_union(frames: list[torch.Tensor] | tuple[torch.Tensor, ...], *, rtol: float) -> UnionBasis:
    """Return a permutation-invariant orthonormal basis for a frame union.

    The SVD is intentionally performed in FP64.  Columns below ``rtol`` times
    the largest singular value are removed, so duplicate task directions do
    not artificially increase the protected union rank.
    """
    if not frames:
        raise ValueError("rank_revealing_union requires at least one frame")
    rows = frames[0].shape[0]
    if any(frame.ndim != 2 or frame.shape[0] != rows for frame in frames):
        raise ValueError("all union frames must be two-dimensional with matching row counts")
    concatenated = torch.cat([frame.double() for frame in frames], dim=1)
    left, singular_values, _ = torch.linalg.svd(concatenated, full_matrices=False)
    threshold = singular_values[0] * rtol if singular_values.numel() else torch.tensor(0.0)
    effective_rank = int((singular_values > threshold).sum().item())
    return UnionBasis(
        basis=left[:, :effective_rank].to(dtype=frames[0].dtype),
        singular_values=singular_values,
        input_columns=int(concatenated.shape[1]),
        effective_rank=effective_rank,
    )
