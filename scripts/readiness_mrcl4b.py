#!/usr/bin/env python3
"""Single lightweight readiness gate for the official MRCL 4B full run."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = pathlib.Path("/home/caiyuliang/models/Qwen3-VL-4B-Instruct")
DATA = pathlib.Path("/home/caiyuliang/datasets/MRCL")
TASKS = ["MedBookVQA", "Navigation", "We-Math2", "Puzzle", "FinMME"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    print(f"OK: {message}")


def run_import_check(python: str, statement: str, label: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    subprocess.run([python, "-c", statement], cwd=ROOT, env=env, check=True)
    print(f"OK: {label}")


def image_candidates(task_root: pathlib.Path, image_value: object) -> list[pathlib.Path]:
    values = image_value if isinstance(image_value, list) else [image_value]
    candidates: list[pathlib.Path] = []
    for value in values:
        if not isinstance(value, str):
            continue
        value_path = pathlib.Path(value)
        candidates.extend([task_root / "images" / value_path, task_root / value_path, DATA / value_path])
    return candidates


def main() -> None:
    require(os.getuid() != 0, "formal run user is non-root")
    require((ROOT / ".git").is_dir(), "official repository exists")
    require((ROOT / "scripts/GRPO-CL-4B/run_cl.sh").is_file(), "run_cl.sh exists")
    require((ROOT / "scripts/GRPO-CL-4B/grpo.sh").is_file(), "grpo.sh exists")

    require((MODEL / "config.json").is_file(), "model config exists")
    require((MODEL / "preprocessor_config.json").is_file(), "model processor config exists")
    require(any(MODEL.glob("tokenizer*")), "model tokenizer files exist")
    require((MODEL / "model.safetensors.index.json").is_file() or any(MODEL.glob("*.safetensors")), "model weights exist")

    for task in TASKS:
        task_root = DATA / task
        train_path = task_root / "jsons/train/data.json"
        test_path = task_root / "jsons/test/data.json"
        require((task_root / "images").is_dir(), f"{task} images directory exists")
        require(train_path.is_file() and test_path.is_file(), f"{task} official train/test JSON exists")
        records = json.loads(train_path.read_text())
        require(isinstance(records, list) and bool(records), f"{task} train JSON has records")
        sample = records[0]
        require("image" in sample, f"{task} sample has image field")
        require(any(path.is_file() for path in image_candidates(task_root, sample["image"])), f"{task} sample image resolves")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(visible == "0,1,2,3", "CUDA_VISIBLE_DEVICES is exactly 0,1,2,3")
    gpu_names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    require(len(gpu_names) == 4, "four GPUs are visible on the host")

    run_import_check(
        "/home/caiyuliang/anaconda3/envs/trlQwen/bin/python",
        "import torch,transformers,deepspeed,flash_attn; import train.train_grpo",
        "training imports",
    )
    run_import_check(
        "/home/caiyuliang/anaconda3/envs/vllmQwen/bin/python",
        "import vllm",
        "evaluation import",
    )
    print("READY: official MRCL Qwen3-VL-4B GRPO full sequence")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"NOT READY: {exc}", file=sys.stderr)
        raise
