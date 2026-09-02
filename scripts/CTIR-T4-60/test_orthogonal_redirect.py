#!/usr/bin/env python3
import math

import torch

from src.ctir.orthogonal_redirect import (
    apply_left,
    orthogonal_map,
    randomized_svd,
    redirect_delta,
    target_frame,
)


def main() -> None:
    for shape in ((96, 64), (64, 128)):
        torch.manual_seed(7)
        delta = torch.randn(*shape)
        old_gradient = torch.randn_like(delta)
        raw_left, _, raw_right = randomized_svd(delta, 8, seed=9)
        old_left, _, old_right = randomized_svd(old_gradient, 8, seed=11)
        raw_spectrum = torch.linalg.svdvals(delta.double())
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
            redirected, left_map, right_map = redirect_delta(
                delta, raw_left, raw_right, old_left, old_right, beta,
            )
            # Gate measurement matches EXP-CTIR-002: FP32 updates, FP64 SVD.
            error = (
                torch.linalg.vector_norm(
                    torch.linalg.svdvals(redirected.double()) - raw_spectrum
                )
                / torch.linalg.vector_norm(raw_spectrum)
            )
            frob_ratio = torch.linalg.vector_norm(redirected.double()) / torch.linalg.vector_norm(delta.double())
            assert float(error) < 1e-5, (shape, beta, float(error))
            assert abs(float(frob_ratio) - 1.0) < 1e-5, (shape, beta, float(frob_ratio))
            if beta == 0.0:
                assert torch.equal(redirected, delta)
            else:
                assert left_map is not None and right_map is not None
                assert left_map.orthogonality_error < 1e-5
                assert right_map.orthogonality_error < 1e-5
                left_target = target_frame(raw_left, old_left, beta)
                assert torch.allclose(apply_left(left_map, raw_left), left_target, atol=2e-5, rtol=2e-5)
        print(f"{shape}: passed")

    # A small requested frame motion must induce a small, orientation-preserving
    # direct rotation and leave directions outside its principal plane fixed.
    angle = 1e-3
    source = torch.eye(8, 2)
    target = source.clone()
    target[:, 0] = math.cos(angle) * source[:, 0]
    target[2, 0] = math.sin(angle)
    mapping = orthogonal_map(source, target)
    assert mapping.operator_distance < 2e-3
    assert torch.allclose(apply_left(mapping, source), target, atol=1e-6, rtol=1e-6)
    outside = torch.eye(8)[:, 3:]
    assert torch.equal(apply_left(mapping, outside), outside)
    print("local direct rotation: passed")

    # FP32/distributed frame transport can leave small Gram errors.  Those
    # errors must not turn the full-update map into a non-orthogonal operator.
    torch.manual_seed(19)
    union = torch.linalg.qr(torch.randn(512, 16, dtype=torch.float64))[0]
    angles = torch.tensor(
        [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-2, 1e-1, 3e-1],
        dtype=torch.float64,
    )
    source_mixing = torch.eye(8, dtype=torch.float64) + 3e-4 * torch.randn(8, 8, dtype=torch.float64)
    target_mixing = torch.eye(8, dtype=torch.float64) + 3e-4 * torch.randn(8, 8, dtype=torch.float64)
    noisy_source = (union[:, :8] @ source_mixing).float()
    noisy_target = (
        (union[:, :8] * torch.cos(angles) + union[:, 8:] * torch.sin(angles)) @ target_mixing
    ).float()
    mapping = orthogonal_map(noisy_source, noisy_target)
    assert mapping.orthogonality_error < 1e-5, mapping.orthogonality_error
    print("FP32 frame re-orthogonalization: passed")


if __name__ == "__main__":
    main()
