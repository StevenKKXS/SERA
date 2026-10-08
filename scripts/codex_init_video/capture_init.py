#!/usr/bin/env python3
"""Prepare RoboMME test-episode demonstration videos and an init-video manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def capture_cases(tasks, num_episodes, output_dir, runner_factory, save_video):
    """Capture reset observations; never step a policy or include the execution frame."""
    rows = []
    for task in tasks:
        runner = runner_factory(task)
        try:
            if num_episodes > runner.num_episodes:
                raise ValueError(f"{task} has only {runner.num_episodes} test episodes")
            for episode in range(num_episodes):
                try:
                    runner.make_env(episode)
                    observation = runner.get_init_obs()
                    if not observation["images"]:
                        raise ValueError(f"No initial observations for {task} episode {episode}")
                    frames = list(observation["images"][:-1])
                    case_id = f"{task}_episode_{episode:02d}"
                    relative_video = Path("videos") / f"{case_id}.mp4"
                    if frames:
                        target = output_dir / relative_video
                        target.parent.mkdir(parents=True, exist_ok=True)
                        save_video(target, frames, fps=30)
                    rows.append({
                        "case_id": case_id,
                        "task": task,
                        "episode": episode,
                        "task_description": observation["task_goal"],
                        "video_path": relative_video.as_posix() if frames else None,
                    })
                finally:
                    runner.close_env()
        finally:
            runner.close_env()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="MoveCube", help="Comma-separated task names")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks or len(tasks) != len(set(tasks)) or args.num_episodes < 1:
        parser.error("Provide distinct tasks and a positive episode count")
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "examples" / "robomme"))
    sys.path.insert(0, str(repo / "third_party" / "robomme_benchmark" / "src"))
    from env_runner import EnvRunner
    import imageio.v2 as imageio

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    if manifest.exists():
        parser.error(f"Manifest already exists; use another output directory: {manifest}")
    rows = capture_cases(
        tasks, args.num_episodes, output,
        lambda task: EnvRunner(task, str(output / "simulator" / task)),
        imageio.mimsave,
    )
    temporary = manifest.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    os.replace(temporary, manifest)
    print(manifest)


if __name__ == "__main__":
    main()
