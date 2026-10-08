#!/usr/bin/env python3
"""Build a validated train+val init-video manifest for self evolution."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def prepare(
    *,
    task: str,
    train_manifest: Path,
    val_manifest: Path,
    output: Path,
    expected_train: int = 30,
    expected_val: int = 20,
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split, manifest, expected in (
        ("train", train_manifest, expected_train),
        ("val", val_manifest, expected_val),
    ):
        rows = read_jsonl(manifest)
        if len(rows) != expected:
            raise ValueError(f"{split} expected {expected} rows, found {len(rows)}")
        for row in rows:
            case_id = row.get("case_id")
            if row.get("task") != task or row.get("dataset_split") != split:
                raise ValueError(f"task/split mismatch in {manifest}: {case_id}")
            if not isinstance(case_id, str) or not case_id.startswith(f"{split}_{task}_episode_"):
                raise ValueError(f"invalid split-qualified case_id in {manifest}: {case_id}")
            if case_id in seen:
                raise ValueError(f"duplicate case_id: {case_id}")
            seen.add(case_id)
            video_path = row.get("video_path")
            if video_path is not None and not Path(video_path).is_file():
                raise ValueError(f"missing video for {case_id}: {video_path}")
            combined.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=30)
    parser.add_argument("--expected-val", type=int, default=20)
    args = parser.parse_args()
    rows = prepare(
        task=args.task,
        train_manifest=args.train_manifest.resolve(),
        val_manifest=args.val_manifest.resolve(),
        output=args.output.resolve(),
        expected_train=args.expected_train,
        expected_val=args.expected_val,
    )
    print(f"Wrote {len(rows)} validated cases to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
