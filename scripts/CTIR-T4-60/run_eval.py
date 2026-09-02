#!/usr/bin/env python3
"""Four-GPU official-evaluator sweep for CTIR T4 checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS = ("MedBookVQA", "Navigation", "We-Math2", "Puzzle")
SHANGHAI = timezone(timedelta(hours=8))


def _now() -> str:
    return datetime.now(SHANGHAI).isoformat(timespec="seconds")


def _read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _score(result_dir: Path) -> tuple[float, float, int]:
    records = _read_json(result_dir / "evaluation_results.json")
    total = len(records)
    correct = float(sum(float(record.get("correct", 0.0)) for record in records))
    return (correct / total if total else 0.0), correct, total


def checkpoints() -> list[dict]:
    puzzle = ROOT / "checkpoints/Qwen3-VL-4B/CTIR-T4-60/training/Puzzle"
    t3 = ROOT / "checkpoints/Qwen3-VL-4B/GRPO-CL/training/We-Math2"
    rows = [{
        "local_step": 0,
        "alias": "step0",
        "path": t3,
        "tasks": ["Puzzle"],
        "reuse": {
            "MedBookVQA": ROOT / "analysis/dynamics/raw_eval/T3_step300/MedBookVQA",
            "Navigation": ROOT / "analysis/dynamics/raw_eval/T3_step300/Navigation",
            "We-Math2": ROOT / "analysis/dynamics/raw_eval/T3_step300/We-Math2",
        },
    }]
    for step in (10, 20, 30, 40, 50, 60):
        rows.append({
            "local_step": step,
            "alias": f"step{step}",
            "path": puzzle / f"checkpoint-{step}",
            "tasks": list(TASKS),
            "reuse": {},
        })
    return rows


def result_dir(eval_root: Path, alias: str, task: str) -> Path:
    return eval_root / alias / task


def metrics_current(target: Path, checkpoint_path: Path) -> bool:
    metric_path = target / "metrics.json"
    if not metric_path.is_file():
        return False
    metric = _read_json(metric_path)
    return metric.get("checkpoint_path") == str(checkpoint_path.resolve()) and (target / "evaluation_results.json").is_file()


def reuse_existing(eval_root: Path, row: dict) -> None:
    for task, source in row.get("reuse", {}).items():
        source = Path(source)
        target = result_dir(eval_root, row["alias"], task)
        if metrics_current(target, row["path"]):
            continue
        required = ["merge.jsonl", "evaluation_results.json", "evaluation_stats.txt"]
        missing = [name for name in required if not (source / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Cannot reuse {source}: missing {missing}")
        target.mkdir(parents=True, exist_ok=True)
        for name in required:
            data = (source / name).read_bytes()
            (target / name).write_bytes(data)
        metric = _read_json(source / "metrics.json") if (source / "metrics.json").is_file() else {}
        score, correct, total = _score(target)
        _write_json(target / "metrics.json", {
            "alias": row["alias"],
            "checkpoint_path": str(row["path"].resolve()),
            "correct": correct,
            "eval_task": task,
            "generation": {"repetition_penalty": 1.05, "temperature": 0.0, "top_p": 1.0},
            "local_step": row["local_step"],
            "reused_from": str(source),
            "score": score,
            "source": f"reused:{source}",
            "timestamp": _now(),
            "total": total,
            "train_task": "Puzzle" if row["local_step"] else "We-Math2",
        })
        print(f"REUSE {row['alias']}/{task} score={score:.4f} from {source}", flush=True)


def worker(args: argparse.Namespace) -> None:
    from src.eval.inference import VLMInference
    import yaml

    prompts = yaml.safe_load((ROOT / "src/dataset/prompts_2.yaml").read_text(encoding="utf-8"))
    data_root = Path(args.data_root)
    selected = [name for name in args.eval_tasks.split(",") if name]
    first = selected[0]
    output = result_dir(Path(args.eval_root), args.alias, first) / "shards"
    inferencer = VLMInference(
        base_model_path=args.checkpoint_path,
        test_file=str(data_root / first / "jsons/test/data.json"),
        output_dir=str(output),
        media_dir=str(data_root / first / "images"),
        tensor_parallel_size=1,
        batch_size=args.batch_size,
        prompt_config=prompts,
        max_completion_length=2048,
        shard_rank=args.shard_rank,
        num_shards=args.num_shards,
    )
    for name in selected:
        output = result_dir(Path(args.eval_root), args.alias, name) / "shards"
        output.mkdir(parents=True, exist_ok=True)
        inferencer.test_file = str(data_root / name / "jsons/test/data.json")
        inferencer.media_dir = str(data_root / name / "images")
        inferencer.output_dir = str(output)
        inferencer.run_inference_on_json()


def evaluate_checkpoint(row: dict, eval_root: Path, data_root: Path, gpu_ids: list[str], batch_size: int) -> None:
    if not row["path"].is_dir():
        raise FileNotFoundError(f"Missing checkpoint: {row['path']}")
    reuse_existing(eval_root, row)
    pending = [task for task in row["tasks"] if not metrics_current(result_dir(eval_root, row["alias"], task), row["path"])]
    if not pending:
        print(f"SKIP {row['alias']}: all requested tasks are current", flush=True)
        return
    print(f"EVAL {row['alias']}: {', '.join(pending)} on {len(gpu_ids)} GPUs", flush=True)
    for task in pending:
        (result_dir(eval_root, row["alias"], task) / "shards").mkdir(parents=True, exist_ok=True)
    processes = []
    for rank, gpu in enumerate(gpu_ids):
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": gpu,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MRCL_GPU_MEMORY_UTILIZATION": os.environ.get("MRCL_GPU_MEMORY_UTILIZATION", "0.6"),
        })
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker",
            "--alias", row["alias"],
            "--checkpoint-path", str(row["path"]),
            "--eval-root", str(eval_root),
            "--data-root", str(data_root),
            "--eval-tasks", ",".join(pending),
            "--shard-rank", str(rank),
            "--num-shards", str(len(gpu_ids)),
            "--batch-size", str(batch_size),
        ]
        processes.append(subprocess.Popen(command, cwd=ROOT, env=env))
    failures = [process.wait() for process in processes]
    # vLLM teardown can leave several tens of GiB allocated for a few seconds;
    # the next checkpoint's 0.9 (or even 0.6) memory gate then fails. Wait for
    # the driver to release before merge/next start.
    time.sleep(20)
    if any(failures):
        raise RuntimeError(f"Evaluation workers failed for {row['alias']}: {failures}")
    for task in pending:
        target = result_dir(eval_root, row["alias"], task)
        shard_dir = target / "shards"
        merge = target / "merge.jsonl"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/merge_eval_shards.py"),
            "--shard-dir", str(shard_dir),
            "--output", str(merge),
            "--num-shards", str(len(gpu_ids)),
        ], cwd=ROOT, check=True)
        subprocess.run([
            sys.executable, str(ROOT / "src/eval/eval.py"),
            "--dataset_name", task,
            "--merged_file", str(merge),
            "--output_dir", str(target),
        ], cwd=ROOT, check=True)
        score, correct, total = _score(target)
        _write_json(target / "metrics.json", {
            "alias": row["alias"],
            "checkpoint_path": str(row["path"].resolve()),
            "correct": correct,
            "eval_task": task,
            "generation": {"repetition_penalty": 1.05, "temperature": 0.0, "top_p": 1.0},
            "local_step": row["local_step"],
            "score": score,
            "source": "generated",
            "timestamp": _now(),
            "total": total,
            "train_task": "Puzzle" if row["local_step"] else "We-Math2",
        })
        print(f"DONE {row['alias']}/{task} score={score:.4f} ({correct}/{total})", flush=True)


def write_performance_csv(eval_root: Path) -> Path:
    baseline = {
        0: {
            "MedBookVQA": 0.8183361629881154,
            "Navigation": 0.6764705882352942,
            "We-Math2": 0.6253687315634219,
            "Puzzle": None,
        },
        30: {
            "MedBookVQA": 0.7911714770797963,
            "Navigation": 0.59375,
            "We-Math2": 0.6029319746133905,
            "Puzzle": 0.3675,
        },
        60: {
            "MedBookVQA": 0.7962648556876061,
            "Navigation": 0.5147058823529411,
            "We-Math2": 0.6012335746849021,
            "Puzzle": 0.46,
        },
    }
    rows = []
    for item in checkpoints():
        for task in TASKS:
            target = result_dir(eval_root, item["alias"], task)
            metric_path = target / "metrics.json"
            if not metric_path.is_file():
                continue
            metric = _read_json(metric_path)
            step = item["local_step"]
            hist = baseline.get(step, {}).get(task)
            row = {
                "local_step": step,
                "global_continual_step": 900 + step,
                "alias": item["alias"],
                "eval_task": task,
                "ctir_score": metric["score"],
                "ctir_correct": metric["correct"],
                "ctir_total": metric["total"],
                "historical_baseline_score": hist if hist is not None else "",
                "delta_vs_historical": (metric["score"] - hist) if hist is not None else "",
                "source": metric.get("source", ""),
                "checkpoint_path": metric.get("checkpoint_path", ""),
            }
            rows.append(row)
    csv_path = eval_root / "performance.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "local_step", "eval_task", "ctir_score",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)", flush=True)
    return csv_path


def gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    override = os.environ.get("MRCL_GPU_IDS", "0,1,2,3").strip()
    return [item.strip() for item in override.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--alias", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--eval-root", default=str(ROOT / "experiments/ctir_t4_60/eval"))
    parser.add_argument("--data-root", default="/home/caiyuliang/datasets/MRCL")
    parser.add_argument("--eval-tasks", default="")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    eval_root = Path(args.eval_root)
    data_root = Path(args.data_root)
    gpus = gpu_ids()
    if len(gpus) < 1:
        raise ValueError("No GPUs specified via CUDA_VISIBLE_DEVICES or MRCL_GPU_IDS")
    print(f"CTIR eval start {_now()} gpus={gpus}", flush=True)
    for row in checkpoints():
        evaluate_checkpoint(row, eval_root, data_root, gpus, args.batch_size)
    write_performance_csv(eval_root)
    print(f"CTIR eval finished {_now()}", flush=True)


if __name__ == "__main__":
    main()
