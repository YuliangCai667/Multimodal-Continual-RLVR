#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ctir.multitask_probe_dataset import freeze_task_probes, write_probe_index


TASK_SPECS = (
    ("MedBookVQA", "MedBookVQA", "conversations[1].value"),
    ("Navigation", "Navigation", "full_text_only_thought"),
    ("We-Math2", "Math", "conversations[1].value"),
    ("Puzzle", "Puzzle", "full_text_only_thought"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()

    manifest_paths = []
    summary = {"count_per_task": 32, "seed_per_task": 142, "tasks": {}}
    for task, prompt_key, target_field in TASK_SPECS:
        output = output_root / "manifests" / f"{task}.json"
        payload = freeze_task_probes(
            data_root / task / "jsons" / "train" / "data.json",
            data_root / task / "images",
            output,
            task=task,
            prompt_key=prompt_key,
            target_field=target_field,
            count=32,
            seed=142,
        )
        manifest_paths.append(output)
        summary["tasks"][task] = {
            "manifest": str(output),
            "target": target_field,
            "eligible": payload["eligible_record_count"],
            "excluded_empty_target_indices": payload["excluded_empty_target_indices"],
            "selected_source_indices": payload["selected_source_indices"],
        }

    for stage_number in range(2, 6):
        stage = f"T{stage_number}"
        write_probe_index(
            output_root / "indexes" / f"{stage}.json",
            stage=stage,
            manifest_paths=manifest_paths[:stage_number - 1],
        )
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "probe_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
