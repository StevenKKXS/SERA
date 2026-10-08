"""Shared utilities for RoboMME HDF5 dataset scripts.

Used by build_robomme_dataset.py and the VLM subgoal dataset builders
(memer, qwenvl).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import h5py


def first_execution_step(episode_data: "h5py.Group") -> int:
    """Index of first timestep where is_video_demo is False."""
    step = 0
    while episode_data[f"timestep_{step}"]["info"]["is_video_demo"][()]:
        step += 1
    return step


def get_episode_indices(
    data: "h5py.File", max_episodes: int | None = None
) -> list[int]:
    """Sorted episode indices from an HDF5 file; optionally capped."""
    indices = sorted(
        int(k.split("_")[1])
        for k in data.keys()
        if k.startswith("episode_")
    )
    if max_episodes is not None:
        indices = indices[:max_episodes]
    return indices


def get_timestep_indices(episode_data: "h5py.Group") -> list[int]:
    """Sorted timestep indices for an episode."""
    return sorted(
        int(k.split("_")[-1])
        for k in episode_data.keys()
        if k.startswith("timestep_")
    )


def get_task_goal(episode_data: "h5py.Group", lower: bool = False) -> str:
    """Task goal string from episode setup."""
    goal = episode_data["setup"]["task_goal"][()][0].decode()
    return goal.lower() if lower else goal


def get_env_id_from_filename(filename: str) -> str:
    """Extract env ID from H5 filename (e.g. 'data_ButtonUnmask.h5' -> 'ButtonUnmask')."""
    return filename.split(".")[0].split("_")[-1]


def resolve_subgoal(
    raw: str, last: str | None, sentinel: str = "complete"
) -> str:
    """Use last valid subgoal when current is sentinel (e.g. 'complete')."""
    return last if last is not None and sentinel in raw else raw


def preprocess_grounded_subgoal(subgoal: str) -> tuple[str, list]:
    """Extract bbox from 'at <y, x>' and replace with 'at <bbox>'. Returns (text, bbox)."""
    matches = re.findall(r"at <(\d+), (\d+)>", subgoal)
    bbox = (
        [[int(float(m[0])), int(float(m[1]))] for m in matches]
        if matches
        else []
    )
    text = re.sub(r"at <(\d+), (\d+)>", "at <bbox>", subgoal)
    return text, bbox


def add_noise_to_bbox(bbox: list) -> list:
    """Add small random noise to bbox coordinates (y, x) in [0, 255]."""
    return [
        [
            min(max(y + np.random.randint(-2, 2), 0), 255),
            min(max(x + np.random.randint(-2, 2), 0), 255),
        ]
        for (y, x) in bbox
    ]


def wrap_history_subgoals(subgoals: list) -> str:
    """Format subgoal list as '1. subgoal1; 2. subgoal2; ...'."""
    return "; ".join([f"{i + 1}. {s}" for i, s in enumerate(subgoals)])


def remove_redundant_keyframes(
    keyframe_idxs: list[int], exec_start_idx: int, threshold: int = 10
) -> list[int]:
    """Merge keyframes that are within threshold steps; keep only execution keyframes."""
    exec_keyframe_idxs = [i for i in keyframe_idxs if i >= exec_start_idx]
    if not exec_keyframe_idxs:
        return []
    new_keyframe_idxs = [exec_keyframe_idxs[0]]
    for i in range(1, len(exec_keyframe_idxs)):
        if abs(new_keyframe_idxs[-1] - exec_keyframe_idxs[i]) <= threshold:
            new_keyframe_idxs[-1] = exec_keyframe_idxs[i]
        else:
            new_keyframe_idxs.append(exec_keyframe_idxs[i])
    return sorted(new_keyframe_idxs)
