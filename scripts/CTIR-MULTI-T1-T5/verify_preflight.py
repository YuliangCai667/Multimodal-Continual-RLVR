#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()
    config = json.loads((args.log_dir / "resolved_training_config.json").read_text(encoding="utf-8"))
    training = config["training"]
    ctir = config["ctir_multitask"]
    assert training["world_size"] == 4, training
    assert training["per_device_train_batch_size"] == 8, training
    assert training["gradient_accumulation_steps"] == 4, training
    assert training["nominal_global_batch"] == 128, training
    assert training["resolved_training_arguments"]["max_steps"] == 300, training
    assert ctir["stop_after_steps"] == 2, ctir
    assert len(config["protected_task_specs"]) == 4, config["protected_task_specs"]

    metrics = rows(args.log_dir / "step_metrics.jsonl")
    assert [row["local_step"] for row in metrics] == [1, 2], metrics
    assert all(row["chosen_beta"] == 1.0 for row in metrics), metrics
    checks = rows(args.log_dir / "correctness.jsonl")
    spectrum = [row for row in checks if row["test"] == "sampled_exact_full_spectrum"]
    assert len(spectrum) == 2, spectrum
    assert all(row["measurement_dtype"] == "float64" for row in spectrum), spectrum
    assert all(row["spectrum_relative_error"] < 5e-6 for row in spectrum), spectrum
    print("H100-80GB multi-task CTIR preflight passed: 4 tasks, 4 GPUs, batch 8x4 accumulation, two optimizer steps")


if __name__ == "__main__":
    main()
