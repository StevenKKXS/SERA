#!/usr/bin/env python3
"""Capture split-qualified RoboMME initial demonstrations without running a policy.

The simulator's DemonstrationWrapper emits the episode-specific expert demonstration
at reset. This utility persists that observable video once so active experiment
planning and init-video interpretation can reuse it without another simulator reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_episode_ids(value: str) -> list[int]:
    episodes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not episodes or len(episodes) != len(set(episodes)) or min(episodes) < 0:
        raise ValueError("--episode-ids must contain unique nonnegative integers")
    return episodes


def select_reference_frames(images: list[Any]) -> tuple[list[Any], bool]:
    """Return selection reference frames and whether they are an expert demo."""
    if not images:
        raise ValueError("Simulator returned no image observations")
    demo_frames = list(images[:-1])
    return (demo_frames, True) if demo_frames else ([images[-1]], False)


def normalize_cached_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Read v1 expert-demo summaries under the richer reference schema."""
    normalized = dict(summary)
    if "expert_demo_available" in normalized:
        return normalized
    video_path = normalized.get("video_path")
    video_sha256 = normalized.get("video_sha256")
    if not video_path or not video_sha256:
        raise ValueError("Legacy cache is missing its expert demonstration video")
    normalized.update(
        {
            "expert_demo_available": True,
            "reference_kind": "expert_demonstration",
            "reference_video_path": video_path,
            "reference_sha256": video_sha256,
        }
    )
    return normalized


def main() -> None:
    import imageio.v2 as imageio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-split", choices=("train", "val"), required=True)
    parser.add_argument("--episode-ids", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=1300)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    try:
        episode_ids = parse_episode_ids(args.episode_ids)
    except ValueError as exc:
        parser.error(str(exc))

    repo_root = args.repo_root.resolve()
    benchmark_root = args.benchmark_root.resolve()
    output_dir = args.output_dir.resolve()
    sys.path.insert(0, str(repo_root / "examples" / "robomme"))
    sys.path.insert(0, str(benchmark_root / "src"))
    rows: list[dict[str, Any]] = []
    runner = None
    try:
        for episode in episode_ids:
            case_id = f"{args.dataset_split}_{args.task}_episode_{episode:02d}"
            case_dir = output_dir / args.dataset_split / args.task / f"episode_{episode:02d}"
            summary_path = case_dir / "summary.json"
            if summary_path.is_file():
                previous = json.loads(summary_path.read_text(encoding="utf-8"))
                if (
                    previous.get("complete") is True
                    and previous.get("case_id") == case_id
                    and previous.get("dataset_split") == args.dataset_split
                ):
                    rows.append(normalize_cached_summary(previous))
                    print(f"[cache] {case_id}", flush=True)
                    continue
                raise ValueError(f"Refusing incompatible cached case: {summary_path}")
            if case_dir.exists():
                incomplete = (
                    output_dir
                    / args.dataset_split
                    / args.task
                    / "_incomplete"
                    / f"episode_{episode:02d}_{uuid.uuid4().hex}"
                )
                incomplete.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(case_dir), str(incomplete))

            if runner is None:
                from env_runner import EnvRunner

                runner = EnvRunner(
                    args.task,
                    str(output_dir / args.dataset_split / args.task),
                    max_steps=args.max_steps,
                    dataset_split=args.dataset_split,
                )
            runner.make_env(episode)
            pretraj = runner.get_init_obs()
            reference_frames, expert_demo_available = select_reference_frames(
                list(pretraj["images"])
            )
            # Some RoboMME tasks (notably the counting family) do not provide an
            # expert demonstration.  Keep the execution initial observation only
            # as an experiment-selection reference; never expose it to the init
            # analyzer as if it were ground-truth demonstration footage.
            case_dir.mkdir(parents=True, exist_ok=False)
            reference_video_path = case_dir / (
                "initial_demo.mp4"
                if expert_demo_available
                else "execution_initial_observation.mp4"
            )
            imageio.mimsave(reference_video_path, reference_frames, fps=args.fps)
            summary = {
                "complete": True,
                "schema_version": "robomme-init-demo-cache-v1",
                "case_id": case_id,
                "dataset_split": args.dataset_split,
                "task": args.task,
                "episode": episode,
                "difficulty": str(runner.difficulty),
                "task_description": str(pretraj["task_goal"]),
                "expert_demo_available": expert_demo_available,
                "video_path": (
                    str(reference_video_path) if expert_demo_available else None
                ),
                "video_sha256": (
                    sha256_file(reference_video_path)
                    if expert_demo_available
                    else None
                ),
                "reference_kind": (
                    "expert_demonstration"
                    if expert_demo_available
                    else "execution_initial_observation"
                ),
                "reference_video_path": str(reference_video_path),
                "reference_sha256": sha256_file(reference_video_path),
                "frame_count": len(reference_frames),
                "fps": args.fps,
            }
            write_json(summary_path, summary)
            rows.append(summary)
            print(f"[captured] {case_id}", flush=True)
    finally:
        if runner is not None:
            runner.close_env()

    manifest_path = output_dir / args.dataset_split / args.task / "manifest.jsonl"
    content = "".join(
        json.dumps(
            {
                "case_id": row["case_id"],
                "dataset_split": row["dataset_split"],
                "task": row["task"],
                "episode": row["episode"],
                "difficulty": row["difficulty"],
                "task_description": row["task_description"],
                "video_path": row["video_path"],
                "video_sha256": row["video_sha256"],
                "expert_demo_available": row["expert_demo_available"],
                "reference_kind": row["reference_kind"],
                "reference_video_path": row["reference_video_path"],
                "reference_sha256": row["reference_sha256"],
            },
            ensure_ascii=False,
        )
        + "\n"
        for row in sorted(rows, key=lambda item: int(item["episode"]))
    )
    temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(f"Wrote {len(rows)} split-qualified cases to {manifest_path}")


if __name__ == "__main__":
    main()
