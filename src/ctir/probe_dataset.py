from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def freeze_navigation_probes(
    source_path: str | Path,
    image_root: str | Path,
    output_path: str | Path,
    *,
    count: int = 32,
    seed: int = 142,
) -> dict[str, Any]:
    source_path = Path(source_path).resolve()
    image_root = Path(image_root).resolve()
    with source_path.open(encoding="utf-8") as handle:
        source = json.load(handle)
    indices = sorted(random.Random(seed).sample(range(len(source)), count))
    probes = []
    for index in indices:
        record = source[index]
        completion = record.get("full_text_only_thought")
        if not completion:
            raise ValueError(f"Navigation source index {index} lacks full_text_only_thought")
        probes.append({
            "source_index": index,
            "sample_id": record.get("id", record.get("question_id")),
            "question": record["conversations"][0]["value"],
            "target_completion": completion,
            "image": record["image"],
            "image_path": str(image_root / record["image"]),
        })
    payload = {
        "task": "Navigation",
        "split": "train",
        "seed": seed,
        "count": count,
        "source_path": str(source_path),
        "selection": "uniform seeded sample without replacement, then source-index order",
        "target": "full_text_only_thought",
        "loss": "Figure-B teacher-forced causal-LM NLL averaged over valid assistant target tokens",
        "probes": probes,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output_path)
    return payload


def load_navigation_probes(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("task") != "Navigation" or payload.get("split") != "train":
        raise ValueError("CTIR probes must be frozen from the Navigation train split")
    if len(payload.get("probes", [])) != int(payload.get("count", -1)):
        raise ValueError("Navigation probe count does not match the frozen manifest")
    return payload
