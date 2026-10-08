#!/usr/bin/env python3
"""Select informative anonymous train episodes for active failure discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from planner import (
    DEFAULT_MODEL,
    require_chatgpt_login,
    run_codex_inference,
    sanitized_codex_environment,
    write_json,
    write_text,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT = (
    SCRIPT_DIR / "prompts" / "self_evolve_v2_experiment_planner_system.txt"
)
DEFAULT_SCHEMA = (
    SCRIPT_DIR / "schemas" / "self_evolve_v2_experiment_plan.schema.json"
)
FORBIDDEN_TEXT = re.compile(r"(?i)\b(?:episode|seed|test|val(?:idation)?)\b")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evenly_spaced(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    indices = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in indices]


def candidate_pool(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for row in sorted(rows, key=lambda value: int(value["episode"])):
        groups.setdefault(str(row.get("difficulty", "unknown")).lower(), []).append(row)
    allocation = {"easy": limit // 2, "medium": limit // 4, "hard": limit // 4}
    while sum(allocation.values()) < limit:
        allocation["hard"] += 1
    selected: list[dict[str, Any]] = []
    for difficulty in ("easy", "medium", "hard"):
        selected.extend(evenly_spaced(groups.get(difficulty, []), allocation[difficulty]))
    if len(selected) < limit:
        seen = {str(row["case_id"]) for row in selected}
        selected.extend(
            row for row in rows if str(row["case_id"]) not in seen
        )
    return selected[:limit]


def make_contact_sheet(video_path: Path, output_path: Path) -> None:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    if output_path.is_file():
        return
    reader = imageio.get_reader(video_path)
    try:
        count = reader.count_frames()
        indices = sorted({max(0, round((count - 1) * fraction)) for fraction in (0.2, 0.5, 0.8)})
        images = [np.asarray(reader.get_data(index)) for index in indices]
    finally:
        reader.close()
    if not images:
        raise ValueError(f"No frames in {video_path}")
    height = min(image.shape[0] for image in images)
    images = [image[:height] for image in images]
    sheet = np.concatenate(images, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(output_path, quality=92)


def init_analyzer_video_path(card: dict[str, Any]) -> str | None:
    """Prevent selection-only execution observations from becoming init demos."""
    return str(card["video_path"]) if card["expert_demo_available"] else None


def render_prompt(task: str, cards: list[dict[str, Any]], batch_size: int) -> str:
    lines = [
        f"Task: {task}",
        f"Select exactly {batch_size} candidates.",
        "Attachments are ordered exactly as the candidate cards below.",
        "",
    ]
    for card in cards:
        lines.extend(
            [
                f"{card['candidate_id']}",
                f"- difficulty: {card['difficulty']}",
                f"- task instruction: {card['task_description']}",
                "",
            ]
        )
    return "\n".join(lines)


def validate_plan(plan: dict[str, Any], cards: list[dict[str, Any]], batch_size: int) -> None:
    selected = plan.get("selected")
    if not isinstance(selected, list) or len(selected) != batch_size:
        raise ValueError(f"Experiment planner must select exactly {batch_size} candidates")
    allowed = {str(card["candidate_id"]) for card in cards}
    ids = [str(item.get("candidate_id")) for item in selected]
    if len(ids) != len(set(ids)) or not set(ids).issubset(allowed):
        raise ValueError(f"Invalid or duplicate candidate selection: {ids}")
    for item in selected:
        for field in ("failure_hypothesis", "observable_rationale"):
            text = str(item.get(field, ""))
            if FORBIDDEN_TEXT.search(text):
                raise ValueError(
                    f"Experiment plan leaks forbidden identity/split language in {field}: {text!r}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-limit", type=int, default=16)
    parser.add_argument("--exclude-case-id", action="append", default=[])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high", "xhigh"), default="high"
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--output-schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.round < 1 or args.batch_size < 1 or args.candidate_limit < args.batch_size:
        parser.error("round/batch sizes must be positive and candidate-limit >= batch-size")

    excluded = set(args.exclude_case_id)
    rows = [
        row
        for row in load_jsonl(args.manifest.resolve())
        if row.get("dataset_split") == "train"
        and row.get("task") == args.task
        and row.get("case_id") not in excluded
    ]
    pool = candidate_pool(rows, args.candidate_limit)
    if len(pool) < args.batch_size:
        raise ValueError(f"Only {len(pool)} eligible train candidates remain")
    output_dir = args.output_dir.resolve()
    cards = []
    image_paths = []
    for index, row in enumerate(pool, start=1):
        candidate_id = f"C{index:02d}"
        reference_value = row.get("reference_video_path") or row.get("video_path")
        if not reference_value:
            raise ValueError(f"Candidate {row['case_id']} has no visual selection reference")
        video_path = Path(str(reference_value)).resolve()
        contact_path = output_dir / "contact_sheets" / f"{candidate_id}.jpg"
        make_contact_sheet(video_path, contact_path)
        cards.append(
            {
                "candidate_id": candidate_id,
                "case_id": row["case_id"],
                "episode": int(row["episode"]),
                "difficulty": row.get("difficulty"),
                "task_description": row["task_description"],
                "video_sha256": row.get("video_sha256"),
                "expert_demo_available": bool(row.get("expert_demo_available", True)),
                "reference_kind": row.get("reference_kind", "expert_demonstration"),
                "reference_sha256": row.get("reference_sha256") or row.get("video_sha256"),
                "video_path": str(video_path),
                "contact_sheet": str(contact_path),
            }
        )
        image_paths.append(contact_path)

    prompt = render_prompt(args.task, cards, args.batch_size)
    cache_key = sha256_text(
        json.dumps(
            {
                "version": "self-evolve-v2-experiment-planner-v1",
                "task": args.task,
                "round": args.round,
                "batch_size": args.batch_size,
                "cards": [
                    {
                        key: value
                        for key, value in card.items()
                        if key not in {"case_id", "episode", "contact_sheet", "video_path"}
                    }
                    for card in cards
                ],
                "prompt": prompt,
                "system_prompt": args.system_prompt.read_text(encoding="utf-8"),
                "schema": args.output_schema.read_text(encoding="utf-8"),
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("cache_key") == cache_key:
            print(plan_path)
            return
        raise ValueError(f"Refusing incompatible cached plan: {plan_path}")
    write_text(output_dir / "prompt.txt", prompt)
    write_json(output_dir / "candidate_map.json", {"candidates": cards})
    if args.dry_run:
        print(json.dumps({"cache_key": cache_key, "candidates": cards}, ensure_ascii=False))
        return

    environment = sanitized_codex_environment()
    require_chatgpt_login(args.codex_bin, environment)
    result = run_codex_inference(
        codex_bin=args.codex_bin,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        prompt=(args.system_prompt.read_text(encoding="utf-8") + "\n\n" + prompt),
        output_schema=args.output_schema,
        image_paths=image_paths,
        destination=output_dir / "codex",
        environment=environment,
        timeout_seconds=args.timeout_seconds,
    )
    plan = result["response"]
    validate_plan(plan, cards, args.batch_size)
    mapping = {card["candidate_id"]: card for card in cards}
    selected_internal = [
        {
            **item,
            "case_id": mapping[item["candidate_id"]]["case_id"],
            "episode": mapping[item["candidate_id"]]["episode"],
            "difficulty": mapping[item["candidate_id"]]["difficulty"],
        }
        for item in plan["selected"]
    ]
    write_json(
        plan_path,
        {
            "schema_version": "self-evolve-v2-experiment-plan-v1",
            "cache_key": cache_key,
            "task": args.task,
            "dataset_split": "train",
            "round": args.round,
            "strategy_summary": plan["strategy_summary"],
            "selected": selected_internal,
            "usage": result.get("usage", {}),
        },
    )
    selected_ids = {item["candidate_id"] for item in selected_internal}
    selected_rows = [
        {
            "case_id": card["case_id"],
            "dataset_split": "train",
            "task": args.task,
            "episode": card["episode"],
            "task_description": card["task_description"],
            # Only true expert demonstrations are sent to the init analyzer.
            # Selection-only execution observations must remain absent here.
            "video_path": init_analyzer_video_path(card),
        }
        for card in cards
        if card["candidate_id"] in selected_ids
    ]
    write_text(
        output_dir / "selected_init_manifest.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected_rows),
    )
    print(plan_path)


if __name__ == "__main__":
    main()
