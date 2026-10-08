#!/usr/bin/env python3
"""Build a frozen 16-task executor vocabulary from released pi0.5 training labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.dataset as pyarrow_dataset


TASKS = [
    "PatternLock",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoPlaceButton",
    "VideoUnmask",
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "PickHighlight",
    "MoveCube",
    "InsertPeg",
    "RouteStick",
    "BinFill",
    "VideoPlaceOrder",
    "VideoRepick",
    "VideoUnmaskSwap",
]


def normalize_template(value: str) -> str:
    value = value.lower().strip().rstrip(".")
    value = re.sub(r"<\|box_start\|>\(\d+,\d+\)<\|box_end\|>", "<point>", value)
    value = re.sub(r"<\s*-?\d+\s*,\s*-?\d+\s*>", "<point>", value)
    value = re.sub(r"<>|<point_[0-9]+>", "<point>", value)
    return re.sub(r"\s+", " ", value)


def numbered_template(value: str) -> str:
    index = 0

    def replace(_: re.Match[str]) -> str:
        nonlocal index
        index += 1
        return f"<point_{index}>"

    return re.sub(r"<point>", replace, value)


def skill_id(template: str) -> str:
    return "skill_" + hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_episode_goals(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        goals = row.get("tasks") or []
        if goals:
            result[int(row["episode_index"])] = str(goals[0])
    return result


def build(dataset_root: Path) -> dict[str, Any]:
    columns = [
        "episode_index",
        "frame_index",
        "is_demo",
        "simple_subgoal",
        "grounded_subgoal",
        "simple_subgoal_online",
        "grounded_subgoal_online",
    ]
    table = pyarrow_dataset.dataset(dataset_root / "data", format="parquet").to_table(
        columns=columns
    )
    frame = table.to_pandas()
    frame = frame[~frame["is_demo"]].sort_values(["episode_index", "frame_index"])
    goals = load_episode_goals(dataset_root / "meta" / "episodes.jsonl")

    counts: Counter[str] = Counter()
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    labels: dict[str, set[str]] = defaultdict(set)
    task_goals: dict[str, set[str]] = defaultdict(set)
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_transition_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    for row in frame.itertuples(index=False):
        task = TASKS[int(row.episode_index) // 100]
        task_goals[task].add(goals.get(int(row.episode_index), ""))
        for grounded_name, simple_name in (
            ("grounded_subgoal", "simple_subgoal"),
            ("grounded_subgoal_online", "simple_subgoal_online"),
        ):
            grounded = getattr(row, grounded_name)
            if not isinstance(grounded, str) or not grounded.strip():
                continue
            template = normalize_template(grounded)
            counts[template] += 1
            task_counts[task][template] += 1
            simple = getattr(row, simple_name)
            if isinstance(simple, str) and simple.strip():
                labels[template].add(normalize_template(simple))

    for episode, rows in frame.groupby("episode_index", sort=True):
        task = TASKS[int(episode) // 100]
        sequence: list[str] = []
        for value in rows["grounded_subgoal"]:
            if not isinstance(value, str) or not value.strip():
                continue
            template = normalize_template(value)
            if not sequence or sequence[-1] != template:
                sequence.append(template)
        for previous, following in zip(sequence, sequence[1:]):
            transition_counts[previous][following] += 1
            task_transition_counts[task][previous][following] += 1

    predecessors: dict[str, Counter[str]] = defaultdict(Counter)
    for previous, following_counts in transition_counts.items():
        for following, count in following_counts.items():
            predecessors[following][previous] += count

    entries = []
    for template in sorted(counts):
        tasks = sorted(task for task in TASKS if task_counts[task][template])
        entries.append(
            {
                "template_id": skill_id(template),
                "normalized_template": template,
                "canonical_template": numbered_template(template),
                "point_arity": template.count("<point>"),
                "simple_labels": sorted(labels[template]),
                "tasks": tasks,
                "task_goal_examples": {
                    task: sorted(value for value in task_goals[task] if value)[:8]
                    for task in tasks
                },
                "training_frame_count": counts[template],
                "task_frame_counts": {
                    task: task_counts[task][template] for task in tasks
                },
                "successors": {
                    skill_id(value): count
                    for value, count in transition_counts[template].most_common()
                },
                "predecessors": {
                    skill_id(value): count
                    for value, count in predecessors[template].most_common()
                },
                "task_successors": {
                    task: {
                        skill_id(value): count
                        for value, count in task_transition_counts[task][template].most_common()
                    }
                    for task in tasks
                    if task_transition_counts[task][template]
                },
            }
        )

    metadata_files = [
        dataset_root / "meta" / "info.json",
        dataset_root / "meta" / "tasks.jsonl",
        dataset_root / "meta" / "episodes.jsonl",
    ]
    return {
        "version": "robomme-executor-vocabulary-v1",
        "source": {
            "description": "Released grounded-level FT pi0.5 execution labels",
            "dataset_root_hint": "datasets/robomme",
            "metadata_sha256": {
                str(path.relative_to(dataset_root)): file_sha256(path)
                for path in metadata_files
            },
            "execution_rows": int(len(frame)),
        },
        "tasks": TASKS,
        "entry_count": len(entries),
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.dataset_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "entries": result["entry_count"]}))


if __name__ == "__main__":
    main()
