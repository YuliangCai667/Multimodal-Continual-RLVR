from __future__ import annotations

import math
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.ctir.controller import CandidateScore, select_global_beta
from src.ctir.multitask_controller import MultiTaskCandidateScore, select_multitask_global_beta
from src.ctir.multitask_optimizer_interceptor import commit_redirected_update
from src.ctir.multitask_probe_dataset import freeze_task_probes
from src.ctir.orthogonal_redirect import randomized_svd, redirect_delta
from src.ctir.subspace_union import rank_revealing_union


class MultiTaskCTIRContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_single_navigation_regression_candidates_choice_and_delta(self):
        delta = torch.randn(19, 13)
        old_left, _ = torch.linalg.qr(torch.randn(19, 4), mode="reduced")
        old_right, _ = torch.linalg.qr(torch.randn(13, 4), mode="reduced")
        raw_left, _, raw_right = randomized_svd(delta, 4, seed=991)
        left_union = rank_revealing_union([old_left], rtol=1e-6).basis
        right_union = rank_revealing_union([old_right], rtol=1e-6).basis
        for beta in (0.0, 0.25, 0.5, 1.0):
            single, _, _ = redirect_delta(delta, raw_left, raw_right, old_left, old_right, beta)
            multi, _, _ = redirect_delta(delta, raw_left, raw_right, left_union, right_union, beta)
            self.assertTrue(torch.allclose(single, multi, atol=2e-5, rtol=2e-5), beta)

        values = (
            (0.0, 10.0, 2.0, 0.0),
            (0.25, 9.8, 0.4, 0.2),
            (0.5, 9.4, -0.1, 0.8),
            (0.75, 8.8, -0.4, 1.6),
        )
        single_decision = select_global_beta(
            (CandidateScore(beta, new, old, distance) for beta, new, old, distance in values),
            new_descent_ratio=0.9,
        )
        multi_decision = select_multitask_global_beta(
            (
                MultiTaskCandidateScore(beta, new, {"Navigation": old}, {"Navigation": old / 5.0}, distance)
                for beta, new, old, distance in values
            ),
            new_descent_ratio=0.9,
        )
        self.assertEqual(single_decision.beta, 0.5)
        self.assertEqual(multi_decision.beta, single_decision.beta)

    def test_gradient_cancellation_cannot_hide_one_task_harm(self):
        decision = select_multitask_global_beta(
            (
                MultiTaskCandidateScore(0.0, 10.0, {"A": 1.0, "B": -1.0}, {"A": 0.2, "B": -0.2}, 0.0),
                MultiTaskCandidateScore(0.5, 9.5, {"A": -0.1, "B": -0.1}, {"A": -0.02, "B": -0.02}, 1.0),
            ),
            new_descent_ratio=0.9,
        )
        self.assertEqual(decision.beta, 0.5)
        self.assertTrue(decision.jointly_feasible)

    def test_duplicate_and_near_duplicate_frames_do_not_inflate_union_rank(self):
        basis, _ = torch.linalg.qr(torch.randn(31, 5, dtype=torch.float64), mode="reduced")
        nearly_duplicate = basis + 1e-10 * torch.randn_like(basis)
        union = rank_revealing_union((basis.float(), basis.float(), nearly_duplicate.float()), rtol=1e-6)
        self.assertEqual(union.input_columns, 15)
        self.assertEqual(union.effective_rank, 5)

    def test_task_permutation_invariance(self):
        a, _ = torch.linalg.qr(torch.randn(23, 3), mode="reduced")
        b, _ = torch.linalg.qr(torch.randn(23, 3), mode="reduced")
        first = rank_revealing_union([a, b], rtol=1e-6).basis.double()
        second = rank_revealing_union([b, a], rtol=1e-6).basis.double()
        self.assertTrue(torch.allclose(first @ first.T, second @ second.T, atol=2e-6, rtol=2e-6))
        candidates_first = (
            MultiTaskCandidateScore(0.0, 3.0, {"A": 1.0, "B": -2.0}, {"A": 0.4, "B": -0.2}, 0.0),
            MultiTaskCandidateScore(1.0, 2.9, {"A": -1.0, "B": -0.1}, {"A": -0.4, "B": -0.01}, 2.0),
        )
        candidates_second = tuple(
            MultiTaskCandidateScore(
                score.beta,
                score.new_descent,
                dict(reversed(tuple(score.old_harms.items()))),
                dict(reversed(tuple(score.normalized_old_harms.items()))),
                score.update_distance_sq,
            )
            for score in candidates_first
        )
        self.assertEqual(
            select_multitask_global_beta(candidates_first, new_descent_ratio=0.9).beta,
            select_multitask_global_beta(candidates_second, new_descent_ratio=0.9).beta,
        )

    def test_beta_zero_is_a_strict_commit_noop(self):
        with mock.patch("src.ctir.multitask_optimizer_interceptor.full_master_parameter") as gather, \
             mock.patch("src.ctir.multitask_optimizer_interceptor.set_full_master_and_low_precision") as write:
            committed = commit_redirected_update(object(), object(), torch.ones(2, 2), torch.ones(2, 2), 0.0)
        self.assertFalse(committed)
        gather.assert_not_called()
        write.assert_not_called()

    def test_full_fp64_spectrum_is_preserved(self):
        delta = torch.randn(37, 29)
        raw_left, _, raw_right = randomized_svd(delta, 8, seed=19)
        sensitive_left, _ = torch.linalg.qr(torch.randn(37, 16), mode="reduced")
        sensitive_right, _ = torch.linalg.qr(torch.randn(29, 16), mode="reduced")
        redirected, left_map, right_map = redirect_delta(
            delta, raw_left, raw_right, sensitive_left, sensitive_right, 0.75
        )
        raw_s = torch.linalg.svdvals(delta.double())
        redirected_s = torch.linalg.svdvals(redirected.double())
        error = torch.linalg.vector_norm(raw_s - redirected_s) / torch.linalg.vector_norm(raw_s)
        self.assertLess(float(error), 2e-6)
        self.assertLess(max(left_map.orthogonality_error, right_map.orthogonality_error), 2e-6)

    def test_probe_freeze_rejects_a_missing_selected_image(self):
        with tempfile.TemporaryDirectory(prefix="ctir-probe-image-") as raw_tmp:
            root = Path(raw_tmp)
            image_root = root / "images"
            image_root.mkdir()
            source = root / "data.json"
            records = [
                {
                    "id": index,
                    "image": f"{index}.png",
                    "conversations": [
                        {"value": f"question {index}"},
                        {"value": f"answer {index}"},
                    ],
                }
                for index in range(32)
            ]
            source.write_text(json.dumps(records), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                freeze_task_probes(
                    source,
                    image_root,
                    root / "manifest.json",
                    task="MedBookVQA",
                    prompt_key="MedBookVQA",
                    target_field="conversations[1].value",
                )

    def test_cluster_script_contract(self):
        project = Path(os.environ["CTIR_PACKAGE_ROOT"])
        overlay = project / "repo_overlay" if (project / "repo_overlay").is_dir() else project
        stage = (overlay / "scripts/CTIR-MULTI-T1-T5/train_stage_slurm.sh").read_text()
        launch = (overlay / "scripts/CTIR-MULTI-T1-T5/launch_slurm.sh").read_text()
        submit = (overlay / "scripts/submit_ctir_multitask_h100x_4gpu.sh").read_text()
        config = (overlay / "configs/ctir_multitask_t1_t5_h100x_4gpu.yaml").read_text()
        self.assertIn("--max_steps \"$MAX_STEPS\"", stage)
        self.assertIn("MAX_STEPS=300", stage)
        self.assertIn("--ctir_multitask_stop_after_steps 2", stage)
        self.assertIn("--per_device_train_batch_size 8", stage)
        self.assertIn("--gradient_accumulation_steps 4", stage)
        self.assertIn("MRCL_WORLD_SIZE=4", submit)
        self.assertNotIn("SCRIPT_DIR=", submit)
        self.assertIn('exec bash "$BUNDLE_ROOT/CPO/scripts/CTIR-MULTI-T1-T5/launch_slurm.sh"', submit)
        self.assertIn(
            'export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"',
            launch,
        )
        self.assertIn("nominal_global_batch: 128", config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
