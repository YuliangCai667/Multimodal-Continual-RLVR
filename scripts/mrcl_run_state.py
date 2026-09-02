#!/usr/bin/env python3
"""Maintain lightweight metadata and status for the official MRCL 4B run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "checkpoints" / "Qwen3-VL-4B" / "GRPO-CL"
TRAINING_ROOT = RUN_ROOT / "training"
STATUS_PATH = ROOT / "status.json"
METADATA_PATH = ROOT / "run_metadata.json"
MODEL_PATH = pathlib.Path(
    os.environ.get("BASE_MODEL", "/home/caiyuliang/models/Qwen3-VL-4B-Instruct")
)
DATASET_PATH = pathlib.Path(
    os.environ.get("BASE_PATH", "/home/caiyuliang/datasets/MRCL")
)
SEQUENCE = ["MedBookVQA", "Navigation", "We-Math2", "Puzzle", "FinMME"]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json(path: pathlib.Path, default: dict) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def atomic_write(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp = pathlib.Path(handle.name)
    tmp.replace(path)


def output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def hf_revision(path: pathlib.Path) -> str:
    revisions: set[str] = set()
    metadata_root = path / ".cache" / "huggingface" / "download"
    if metadata_root.exists():
        for item in metadata_root.rglob("*.metadata"):
            try:
                first_line = item.read_text().splitlines()[0].strip()
            except (OSError, IndexError, UnicodeDecodeError):
                continue
            if len(first_line) == 40:
                revisions.add(first_line)
    return ",".join(sorted(revisions)) if revisions else "unavailable"


def package_versions(python_path: str, packages: list[str]) -> dict[str, str]:
    code = (
        "import importlib.metadata,json;"
        f"p={packages!r};"
        "print(json.dumps({x:importlib.metadata.version(x) for x in p}))"
    )
    try:
        return json.loads(subprocess.check_output([python_path, "-c", code], text=True))
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return {}


def latest_checkpoint(task: str | None = None) -> str | None:
    roots = [TRAINING_ROOT / task] if task else list(TRAINING_ROOT.glob("*"))
    candidates: list[tuple[float, pathlib.Path]] = []
    for root in roots:
        if not root.exists():
            continue
        for checkpoint in root.glob("checkpoint-*"):
            if checkpoint.is_dir():
                candidates.append((checkpoint.stat().st_mtime, checkpoint))
    if not candidates:
        return None
    return str(max(candidates)[1].relative_to(ROOT))


def initialize() -> None:
    launcher_pid = int(os.environ.get("MRCL_LAUNCHER_PID", os.getpid()))
    metadata = load_json(METADATA_PATH, {})
    metadata.update(
        {
            "repo_commit": output(["git", "rev-parse", "HEAD"]),
            "repo_dirty": bool(output(["git", "status", "--short"])),
            "model_path": str(MODEL_PATH),
            "model_revision": hf_revision(MODEL_PATH),
            "dataset_path": str(DATASET_PATH),
            "dataset_revision": hf_revision(DATASET_PATH),
            "sequence": SEQUENCE,
            "steps": {"MedBookVQA": 300, "Navigation": 300, "We-Math2": 300, "Puzzle": 300, "FinMME": 300},
            "checkpoint_interval_steps": 30,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "nvidia_smi": output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]),
            "training_environment": package_versions(
                os.environ.get(
                    "TRAIN_PYTHON",
                    "/home/caiyuliang/anaconda3/envs/trlQwen/bin/python",
                ),
                ["torch", "torchvision", "torchaudio", "transformers", "trl", "deepspeed", "flash-attn"],
            ),
            "evaluation_environment": package_versions(
                os.environ.get(
                    "EVAL_PYTHON",
                    "/home/caiyuliang/anaconda3/envs/vllmQwen/bin/python",
                ),
                ["torch", "torchvision", "torchaudio", "vllm"],
            ),
            "hardware_reproduction_difference": "4 x NVIDIA RTX PRO 6000 Blackwell 96GB instead of 8 x NVIDIA H100",
            "hardware_batch_effect": {
                "official_gpu_count": 8,
                "actual_gpu_count": 4,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "nominal_global_batch_official": 256,
                "nominal_global_batch_actual": 128,
                "note": "Global batch is halved solely by GPU count; no LR or algorithm hyperparameter compensation was applied",
            },
            "official_config_changes": [
                "BASE_MODEL and BASE_PATH set to local snapshots",
                "CUDA_VISIBLE_DEVICES fixed to 0,1,2,3 by the thin launcher",
                "training.log uses tee -a so restart history is retained",
                "lightweight metadata/status hooks added; GRPO hyperparameters unchanged",
            ],
            "environment_compatibility_fixes": [
                "flash-attn 2.8.3 built locally with CUDA 12.8 because its published wheel requires a newer host GLIBC",
            ],
            "resume_behavior": "train_grpo.py resumes automatically from the newest checkpoint-* in the task output directory",
        }
    )
    launches = metadata.setdefault("launches", [])
    launches.append({"time": now(), "pid": launcher_pid})
    atomic_write(METADATA_PATH, metadata)

    status = load_json(STATUS_PATH, {})
    status.update(
        {
            "state": "running",
            "start_time": status.get("start_time", now()),
            "updated_at": now(),
            "launcher_pid": launcher_pid,
            "current_task": None,
            "current_task_id": None,
            "finished_tasks": status.get("finished_tasks", []),
            "last_detected_checkpoint": latest_checkpoint(),
        }
    )
    atomic_write(STATUS_PATH, status)


def update_stage(action: str, task: str, task_id: int) -> None:
    status = load_json(STATUS_PATH, {})
    status.update(
        {
            "state": "running",
            "updated_at": now(),
            "current_task": task,
            "current_task_id": task_id,
            "last_detected_checkpoint": latest_checkpoint(task),
        }
    )
    if action == "stage-complete":
        finished = status.setdefault("finished_tasks", [])
        if task not in finished:
            finished.append(task)
    atomic_write(STATUS_PATH, status)


def finish(state: str, exit_code: int | None) -> None:
    status = load_json(STATUS_PATH, {})
    status.update(
        {
            "state": state,
            "updated_at": now(),
            "end_time": now(),
            "exit_code": exit_code,
            "last_detected_checkpoint": latest_checkpoint(status.get("current_task")),
        }
    )
    atomic_write(STATUS_PATH, status)


def mark_launch_success(step: int) -> None:
    status = load_json(STATUS_PATH, {})
    status.update(
        {
            "launch_success": True,
            "launch_success_time": now(),
            "current_step": step,
            "updated_at": now(),
        }
    )
    atomic_write(STATUS_PATH, status)


def mark_checkpoint(step: int) -> None:
    status = load_json(STATUS_PATH, {})
    status.update(
        {
            "current_step": step,
            "first_checkpoint_confirmed": True,
            "first_checkpoint_confirmed_time": now(),
            "last_detected_checkpoint": latest_checkpoint(status.get("current_task")),
            "updated_at": now(),
        }
    )
    atomic_write(STATUS_PATH, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "initialize",
            "stage-start",
            "stage-complete",
            "launch-success",
            "checkpoint-confirmed",
            "run-complete",
            "run-failed",
        ],
    )
    parser.add_argument("--task")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()

    if args.action == "initialize":
        initialize()
    elif args.action in {"stage-start", "stage-complete"}:
        if not args.task or args.task_id is None:
            parser.error("stage actions require --task and --task-id")
        update_stage(args.action, args.task, args.task_id)
    elif args.action in {"launch-success", "checkpoint-confirmed"}:
        if args.step is None:
            parser.error(f"{args.action} requires --step")
        if args.action == "launch-success":
            mark_launch_success(args.step)
        else:
            mark_checkpoint(args.step)
    elif args.action == "run-complete":
        finish("completed", 0)
    else:
        finish("failed", args.exit_code)


if __name__ == "__main__":
    main()
