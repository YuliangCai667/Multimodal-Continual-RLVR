from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class LowRankOrthogonalMap:
    """Identity-connected direct rotation on orthogonal principal planes."""

    source_basis: torch.Tensor
    normal_basis: torch.Tensor
    cosines: torch.Tensor
    sines: torch.Tensor
    frame_displacement: float

    @property
    def operator_distance(self) -> float:
        """Spectral distance ``||R - I||_2`` without materializing ``R``."""
        if self.cosines.numel() == 0:
            return 0.0
        return float(torch.sqrt(2.0 * (1.0 - self.cosines.double()).clamp_min(0.0)).max().item())

    @property
    def frobenius_distance(self) -> float:
        """Frobenius distance ``||R - I||_F`` on the active planes."""
        if self.cosines.numel() == 0:
            return 0.0
        squared = 4.0 * (1.0 - self.cosines.double()).clamp_min(0.0).sum()
        return float(torch.sqrt(squared).item())

    @property
    def max_principal_angle(self) -> float:
        if self.cosines.numel() == 0:
            return 0.0
        return math.acos(float(self.cosines.double().min().clamp(-1.0, 1.0).item()))

    @property
    def orthogonality_error(self) -> float:
        """Spectral-norm upper bound for ``||R.T @ R - I||_2``."""
        if self.cosines.numel() == 0:
            return 0.0
        basis = torch.cat((self.source_basis.double(), self.normal_basis.double()), dim=1)
        gram = basis.T @ basis
        cosine_offsets = torch.diag(self.cosines.double() - 1.0)
        sines = torch.diag(self.sines.double())
        top = torch.cat((cosine_offsets, -sines), dim=1)
        bottom = torch.cat((sines, cosine_offsets), dim=1)
        rotation_offset = torch.cat((top, bottom), dim=0)
        residual = (
            rotation_offset
            + rotation_offset.T
            + rotation_offset.T @ gram @ rotation_offset
        )
        bound = torch.linalg.matrix_norm(gram, ord=2) * torch.linalg.matrix_norm(residual, ord=2)
        return float(bound.item())


def _orthonormal_frame(frame: torch.Tensor) -> torch.Tensor:
    """Return a numerically orthonormal FP64 basis for a frame's span.

    Randomized SVD and distributed FP32 transport only determine the control
    subspace; small Gram errors in their concrete frames must not leak into
    the supposedly orthogonal full-update transport.  The diagonal sign fix
    keeps the representative local to the supplied frame.
    """
    basis, triangular = torch.linalg.qr(frame.double(), mode="reduced")
    diagonal = torch.diagonal(triangular)
    signs = torch.where(diagonal < 0.0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return basis * signs[None, :]


def randomized_svd(
    matrix: torch.Tensor,
    rank: int,
    *,
    seed: int,
    oversample: int = 4,
    power_iterations: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic randomized truncated SVD; all returned tensors are FP32."""
    matrix = matrix.float()
    q = min(rank + oversample, min(matrix.shape))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    omega = torch.randn(matrix.shape[1], q, generator=generator, dtype=torch.float32).to(matrix.device)
    basis, _ = torch.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(power_iterations):
        basis, _ = torch.linalg.qr(matrix @ (matrix.T @ basis), mode="reduced")
    small = basis.T @ matrix
    left_small, singular_values, right_h = torch.linalg.svd(small, full_matrices=False)
    kept = min(rank, singular_values.numel())
    return basis @ left_small[:, :kept], singular_values[:kept], right_h[:kept].T


def target_frame(source: torch.Tensor, sensitive: torch.Tensor, beta: float) -> torch.Tensor:
    if beta == 0.0 or sensitive.numel() == 0:
        return source
    dtype = source.dtype
    source = _orthonormal_frame(source)
    sensitive = _orthonormal_frame(sensitive)
    projected = source - beta * sensitive @ (sensitive.T @ source)
    span, _ = torch.linalg.qr(projected, mode="reduced")
    # Resolve QR signs/rotations by choosing the frame in this span closest to source.
    left, _, right_h = torch.linalg.svd(span.T @ source, full_matrices=False)
    return (span @ (left @ right_h)).to(dtype)


def orthogonal_map(source: torch.Tensor, target: torch.Tensor) -> LowRankOrthogonalMap:
    """Construct the direct rotation mapping an aligned source frame to target.

    ``target_frame`` Procrustes-aligns the two frames, so their cross-Gram
    matrix is symmetric positive semidefinite.  The resulting map rotates only
    the principal planes spanned by the two frames and is the identity on their
    orthogonal complement.  In particular, it approaches the identity as the
    target approaches the source; a Householder lift does not have this local
    property.
    """
    dtype = source.dtype
    source64 = _orthonormal_frame(source)
    target64 = _orthonormal_frame(target)

    # Distributed FP32 round trips can slightly disturb the Procrustes gauge
    # chosen by target_frame.  Restore it in FP64 before extracting principal
    # planes so the compact map is genuinely orthogonal, not merely close.
    left, _, right_h = torch.linalg.svd(target64.T @ source64, full_matrices=False)
    target64 = target64 @ (left @ right_h)
    overlap = source64.T @ target64
    symmetric_overlap = 0.5 * (overlap + overlap.T)
    cosines, principal_axes = torch.linalg.eigh(symmetric_overlap)
    order = torch.argsort(cosines, descending=True)
    cosines = cosines[order].clamp(-1.0, 1.0)
    principal_axes = principal_axes[:, order]
    if cosines.numel() and float(cosines.min()) < -1e-6:
        raise ValueError("direct rotation requires Procrustes-aligned frames")

    principal_source = source64 @ principal_axes
    principal_target = target64 @ principal_axes
    cosines = torch.sum(principal_source * principal_target, dim=0).clamp(-1.0, 1.0)
    residual = principal_target - principal_source * cosines[None, :]
    sines = torch.linalg.vector_norm(residual, dim=0)

    # Frames originate in FP32.  Angles below this floor contain no reliable
    # direction for the complementary principal vector and are effectively the
    # identity at update precision.
    active = sines > 1e-7
    principal_source = principal_source[:, active]
    residual = residual[:, active]
    cosines = cosines[active]
    sines = sines[active]
    normal_basis = residual / sines[None, :]
    trig_norm = torch.sqrt(cosines.square() + sines.square())
    cosines = cosines / trig_norm
    sines = sines / trig_norm

    return LowRankOrthogonalMap(
        source_basis=principal_source.to(dtype),
        normal_basis=normal_basis.to(dtype),
        cosines=cosines.to(dtype),
        sines=sines.to(dtype),
        frame_displacement=float(torch.linalg.matrix_norm(target64 - source64).item()),
    )


def apply_left(mapping: LowRankOrthogonalMap, matrix: torch.Tensor) -> torch.Tensor:
    if mapping.cosines.numel() == 0:
        return matrix
    source_coordinates = mapping.source_basis.T @ matrix
    normal_coordinates = mapping.normal_basis.T @ matrix
    cosine_offsets = mapping.cosines[:, None] - 1.0
    sines = mapping.sines[:, None]
    source_change = cosine_offsets * source_coordinates - sines * normal_coordinates
    normal_change = sines * source_coordinates + cosine_offsets * normal_coordinates
    return matrix + mapping.source_basis @ source_change + mapping.normal_basis @ normal_change


def apply_right_transpose(matrix: torch.Tensor, mapping: LowRankOrthogonalMap) -> torch.Tensor:
    if mapping.cosines.numel() == 0:
        return matrix
    source_coordinates = matrix @ mapping.source_basis
    normal_coordinates = matrix @ mapping.normal_basis
    cosine_offsets = mapping.cosines[None, :] - 1.0
    sines = mapping.sines[None, :]
    source_change = cosine_offsets * source_coordinates - sines * normal_coordinates
    normal_change = sines * source_coordinates + cosine_offsets * normal_coordinates
    return matrix + source_change @ mapping.source_basis.T + normal_change @ mapping.normal_basis.T


def redirect_delta(
    delta: torch.Tensor,
    raw_left: torch.Tensor,
    raw_right: torch.Tensor,
    old_left: torch.Tensor,
    old_right: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, LowRankOrthogonalMap | None, LowRankOrthogonalMap | None]:
    if beta == 0.0:
        return delta, None, None
    left_map = orthogonal_map(raw_left, target_frame(raw_left, old_left, beta))
    right_map = orthogonal_map(raw_right, target_frame(raw_right, old_right, beta))
    redirected = apply_right_transpose(apply_left(left_map, delta), right_map)
    return redirected, left_map, right_map


def frame_alignment(first: torch.Tensor, second: torch.Tensor) -> float:
    """Mean squared canonical cosine; 1 means identical subspaces."""
    if first.numel() == 0 or second.numel() == 0:
        return float("nan")
    singular_values = torch.linalg.svdvals(first.T.float() @ second.float())
    return float((singular_values.square().mean()).item())
