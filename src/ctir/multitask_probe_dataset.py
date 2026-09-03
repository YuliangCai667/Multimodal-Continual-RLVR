from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


SUPPORTED_TARGETS = {
    "full_text_only_thought",
    "conversations[1].value",
}


def _target(record: dict[str, Any], target_field: str) -> str | None:
    if target_field == "full_text_only_thought":
        value = record.get("full_text_only_thought")
    elif target_field == "conversations[1].value":
        conversations = record.get("conversations", [])
        value = conversations[1].get("value") if len(conversations) > 1 else None
    else:
        raise ValueError(f"Unsupported target field: {target_field}")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def freeze_task_probes(
    source_path: str | Path,
    image_root: str | Path,
    output_path: str | Path,
    *,
    task: str,
    prompt_key: str,
    target_field: str,
    count: int = 32,
    seed: int = 142,
) -> dict[str, Any]:
    """Freeze a task's valid training probes before any model is loaded."""
    if target_field not in SUPPORTED_TARGETS:
        raise ValueError(f"target_field must be one of {sorted(SUPPORTED_TARGETS)}")
    source_path = Path(source_path).resolve()
    image_root = Path(image_root).resolve()
    with source_path.open(encoding="utf-8") as handle:
        source = json.load(handle)

    eligible: list[int] = []
    excluded: list[int] = []
    for index, record in enumerate(source):
        if _target(record, target_field) is None:
            excluded.append(index)
        else:
            eligible.append(index)
    if len(eligible) < count:
        raise ValueError(f"{task} has only {len(eligible)} valid targets for {count} probes")
    selected = sorted(random.Random(seed).sample(eligible, count))
    probes = []
    for index in selected:
        record = source[index]
        image = record.get("image")
        if not isinstance(image, str) or not image:
            raise ValueError(f"{task} source index {index} does not have one image path")
        image_path = image_root / image
        if not image_path.is_file():
            raise FileNotFoundError(f"{task} source index {index} image is missing: {image_path}")
        conversations = record.get("conversations", [])
        if not conversations or not isinstance(conversations[0].get("value"), str):
            raise ValueError(f"{task} source index {index} lacks conversations[0].value")
        probes.append({
            "source_index": index,
            "sample_id": record.get("id", record.get("question_id")),
            "question": conversations[0]["value"],
            "target_completion": _target(record, target_field),
            "image": image,
            "image_path": str(image_path),
        })
    payload = {
        "schema": "ctir_multitask_probe_manifest_v1",
        "task": task,
        "prompt_key": prompt_key,
        "split": "train",
        "seed": seed,
        "count": count,
        "source_path": str(source_path),
        "image_root": str(image_root),
        "source_record_count": len(source),
        "eligible_record_count": len(eligible),
        "excluded_empty_target_indices": excluded,
        "selected_source_indices": selected,
        "selection": "uniform seeded sample without replacement over nonempty eligible targets, then source-index order",
        "target": target_field,
        "loss": "teacher-forced causal-LM NLL: per-sample valid assistant-token mean, then equal task mean",
        "probes": probes,
    }
    _atomic_json(Path(output_path), payload)
    return payload


def write_probe_index(output_path: str | Path, *, stage: str, manifest_paths: list[str | Path]) -> dict[str, Any]:
    manifests = [str(Path(path).resolve()) for path in manifest_paths]
    payload = {
        "schema": "ctir_multitask_probe_index_v1",
        "stage": stage,
        "task_weighting": "equal independent constraints; no aggregate old-task loss",
        "manifest_paths": manifests,
    }
    _atomic_json(Path(output_path), payload)
    return payload


def load_probe_index(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index_path = Path(path).resolve()
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    if index.get("schema") != "ctir_multitask_probe_index_v1":
        raise ValueError("Invalid multi-task CTIR probe-index schema")
    manifests = []
    seen = set()
    for raw_path in index.get("manifest_paths", []):
        manifest_path = Path(raw_path)
        if not manifest_path.is_absolute():
            manifest_path = index_path.parent / manifest_path
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema") != "ctir_multitask_probe_manifest_v1":
            raise ValueError(f"Invalid multi-task CTIR manifest: {manifest_path}")
        task = manifest.get("task")
        if not task or task in seen:
            raise ValueError(f"Duplicate or missing task in probe index: {task}")
        if int(manifest.get("count", -1)) != 32 or int(manifest.get("seed", -1)) != 142:
            raise ValueError(f"Formal multi-task CTIR requires 32 probes at seed 142 for {task}")
        if len(manifest.get("probes", [])) != 32:
            raise ValueError(f"Probe manifest count mismatch for {task}")
        seen.add(task)
        manifest["manifest_path"] = str(manifest_path.resolve())
        manifests.append(manifest)
    if not manifests:
        raise ValueError("Multi-task CTIR requires at least one protected old task")
    return index, manifests
