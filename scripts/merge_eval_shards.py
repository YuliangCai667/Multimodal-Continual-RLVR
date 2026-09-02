#!/usr/bin/env python3
"""Merge contiguous evaluation shards atomically in original sample order."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be positive")

    parts = [
        args.shard_dir / f"part-{rank:05d}-of-{args.num_shards:05d}.jsonl"
        for rank in range(args.num_shards)
    ]
    missing = [str(path) for path in parts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation shard(s): {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as handle:
        temporary = pathlib.Path(handle.name)
        for part in parts:
            with part.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {part}:{line_number}: {exc}"
                        ) from exc
                    handle.write(line if line.endswith("\n") else line + "\n")
                    row_count += 1

    temporary.replace(args.output)
    print(f"Merged {len(parts)} shards and {row_count} rows into {args.output}")


if __name__ == "__main__":
    main()
