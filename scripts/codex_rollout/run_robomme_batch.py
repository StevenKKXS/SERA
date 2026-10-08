#!/usr/bin/env python3
"""Resumable RoboMME evaluation using Codex plus grounded-level FT pi0.5."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np


TERMINAL_RESULTS = (
    "success",
    "fail",
    "error",
    "timeout",
    "stagnation_early_stop",
    "planner_round_limit",
)


def is_planner_json_error(exc: Exception) -> bool:
    """Return whether a remote planner failure is specifically JSON decoding."""
    return "JSONDecodeError" in f"{type(exc).__name__}: {exc}"


def should_stop_for_semantic_stagnation(
    planner_record: dict[str, Any], limit: int, result: str
) -> bool:
    if limit <= 0 or result in TERMINAL_RESULTS:
        return False
    count = int(planner_record.get("consecutive_no_progress_rounds_after") or 0)
    return count >= limit


def should_stop_for_planner_round_limit(
    completed_rounds: int, limit: int, result: str
) -> bool:
    return limit > 0 and completed_rounds >= limit and result not in TERMINAL_RESULTS


def method_name(memory_mode: str) -> str:
    if memory_mode == "text_progress":
        return "codex_text_progress_planner_plus_grounded_level_ft_pi05"
    return "codex_planner_plus_grounded_level_ft_pi05"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def init_case_id(dataset_split: str, task: str, episode: int) -> str:
    """Return a split-qualified init-video identity.

    Historical test artifacts predate split isolation and use the unqualified
    identity. New train/val/test artifacts always use the qualified form.
    """
    return f"{dataset_split}_{task}_episode_{episode:02d}"


def load_detailed_task_description(
    init_results_dir: Path, task: str, episode: int, dataset_split: str = "test"
) -> dict[str, Any]:
    case_id = init_case_id(dataset_split, task, episode)
    case_dir = init_results_dir / "cases" / case_id
    if dataset_split == "test" and not case_dir.is_dir():
        # Read-only compatibility for the already frozen v8.5 test cache.
        case_id = f"{task}_episode_{episode:02d}"
        case_dir = init_results_dir / "cases" / case_id
    response_path = case_dir / "response.json"
    summary_path = case_dir / "case_summary.json"
    if not response_path.is_file():
        raise FileNotFoundError(
            f"Missing init-video response for {case_id}: {response_path}. "
            "Run scripts/codex_init_video/run_init_video.py first."
        )
    response = json.loads(response_path.read_text(encoding="utf-8"))
    description = response.get("detailed_task_description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Invalid detailed_task_description in {response_path}")
    planner_guidance = response.get("planner_guidance", {})
    if not isinstance(planner_guidance, dict):
        raise ValueError(f"Invalid planner_guidance in {response_path}")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("task") != task or int(summary.get("episode", -1)) != episode:
            raise ValueError(f"Init-video identity mismatch in {summary_path}")
        if summary.get("complete") is not True:
            raise ValueError(f"Init-video case is incomplete: {summary_path}")
        recorded_split = summary.get("dataset_split")
        if recorded_split is None and dataset_split != "test":
            raise ValueError(
                f"Split-qualified init-video metadata is required for {dataset_split}: "
                f"{summary_path}"
            )
        if recorded_split is not None and recorded_split != dataset_split:
            raise ValueError(
                f"Init-video split mismatch in {summary_path}: "
                f"expected {dataset_split}, got {recorded_split}"
            )
    return {
        "case_id": case_id,
        "dataset_split": dataset_split,
        "case_dir": str(case_dir.resolve()),
        "response_path": str(response_path.resolve()),
        "detailed_task_description": description.strip(),
        "planner_guidance": planner_guidance,
    }


def usage_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        usage = row.get("codex_usage") or row.get("codex_usage_total")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def save_rollout_video(path: Path, frames: list[np.ndarray]) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=10)


def subsample_recent_observations(
    front_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray],
    frame_ids: list[int],
    stride: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Subsample synchronized planner observations while retaining the newest frame."""
    if stride < 1:
        raise ValueError("planner frame stride must be positive")
    if len(front_frames) != len(wrist_frames) or len(front_frames) != len(frame_ids):
        raise ValueError("recent front, wrist, and frame-id lists must have equal length")
    if not frame_ids or stride == 1:
        return front_frames, wrist_frames, frame_ids

    # Anchor sampling at the newest observation.  For a full 16-frame chunk and
    # stride 2 this selects frames 2, 4, ..., 16 in one-based chunk numbering.
    start = (len(frame_ids) - 1) % stride
    selected = range(start, len(frame_ids), stride)
    return (
        [front_frames[index] for index in selected],
        [wrist_frames[index] for index in selected],
        [frame_ids[index] for index in selected],
    )


def _diagnostic_value(value: Any) -> Any:
    """Convert simulator values to JSON without exposing them to evaluated models."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return array.tolist()


def capture_env_diagnostics(env_runner: Any) -> dict[str, Any]:
    """Read privileged simulator state for an offline-only execution audit."""
    env = env_runner.env.unwrapped
    info = getattr(env_runner, "info", {}) or {}
    pickup_threshold = (
        0.15
        if env_runner.env_id
        in {"VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"}
        else None
    )
    button_threshold = 0.005
    colors = list(getattr(env, "color_names", []))
    bins = []
    for index, actor in enumerate(getattr(env, "spawned_bins", [])):
        position = np.asarray(_diagnostic_value(actor.pose.p), dtype=float).reshape(-1, 3)[0]
        grasping = None
        try:
            grasping = bool(_diagnostic_value(env.agent.is_grasping(actor)))
        except Exception:
            # Diagnostic collection must never perturb the evaluated policy.
            pass
        row = {
            "index": index,
            "name": getattr(actor, "name", f"bin_{index}"),
            "hidden_cube_color": colors[index] if index < len(colors) else None,
            "position_xyz": [float(value) for value in position],
            "z": float(position[2]),
            "is_grasping": grasping,
        }
        if pickup_threshold is not None:
            row["above_pickup_threshold"] = bool(position[2] > pickup_threshold)
        bins.append(row)
    buttons = []
    seen_buttons: set[int] = set()
    button_candidates: list[tuple[str, Any]] = []
    for name in ("button_left", "button_right", "button"):
        actor = getattr(env, name, None)
        if actor is not None:
            button_candidates.append((name, actor))
    for index, actor in enumerate(getattr(env, "button_list", []) or []):
        button_candidates.append((f"button_list_{index}", actor))
    for name, actor in button_candidates:
        identity = id(actor)
        if identity in seen_buttons:
            continue
        seen_buttons.add(identity)
        try:
            qpos = np.asarray(_diagnostic_value(actor.get_qpos()), dtype=float).reshape(-1)
            depth = float(-qpos[0])
        except Exception:
            continue
        buttons.append(
            {
                "name": name,
                "depth": depth,
                "pressed": bool(depth > button_threshold),
            }
        )
    return {
        "env_timestep": int(getattr(env, "timestep", -1)),
        "current_task_index": int(getattr(env, "current_task_index", -1)),
        "current_task_name": getattr(env, "current_task_name_online", None),
        "current_subgoal": getattr(env, "current_subgoal_segment_online", None),
        "pickup_height_threshold": pickup_threshold,
        "button_depth_threshold": button_threshold if buttons else None,
        "status": info.get("status"),
        "simple_subgoal_online": info.get("simple_subgoal_online"),
        "grounded_subgoal_online": info.get("grounded_subgoal_online"),
        "bins": bins,
        "buttons": buttons,
    }


def run_episode(
    *,
    args: argparse.Namespace,
    planner: Any,
    policy_client: Any,
    env_runner: Any,
    pack_buffer: Any,
    episode_id: int,
    episode_dir: Path,
    init_case: dict[str, Any],
    save_video: bool,
    max_steps: int | None = None,
) -> dict[str, Any]:
    episode_max_steps = args.max_steps if max_steps is None else max_steps
    planner_memory_mode = getattr(args, "planner_memory_mode", "keyframes")
    reset = policy_client.reset()
    while not reset.get("reset_finished", False):
        time.sleep(0.1)

    episode_started = time.perf_counter()
    session_id = f"{args.session_prefix}-{env_runner.env_id}-ep{episode_id:02d}-{uuid.uuid4().hex[:8]}"
    env_runner.make_env(episode_id)
    pretraj = env_runner.get_init_obs()
    task_goal = pretraj["task_goal"]
    current_front = pretraj["images"][-1]
    current_wrist = pretraj["wrist_images"][-1]
    current_state = pretraj["states"][-1]

    # Preserve the released grounded-level FT pi0.5 history interface. Only the
    # Codex planner omits the initial demonstration images.
    policy_images = list(pretraj["images"])
    policy_states = list(pretraj["states"])
    exec_start_idx = len(policy_images) - 1

    recent_front = [current_front.copy()]
    recent_wrist = [current_wrist.copy()]
    recent_ids = [0]
    rollout_frame_cursor = 0
    video_frames = (
        [np.concatenate([current_front, current_wrist], axis=1)] if save_video else []
    )
    timing_rows: list[dict[str, Any]] = []
    planner_records: list[dict[str, Any]] = []
    step_count = 0
    result = "unknown"
    planner_started = False
    last_grounded_subtask: str | None = None
    last_planner_record: dict[str, Any] | None = None
    consecutive_planner_json_failures = 0
    maximum_consecutive_planner_json_failures = 0
    total_planner_json_failures = 0
    planner_json_fallback_chunks = 0

    try:
        planner.start_episode(
            session_id=session_id,
            run_dir=episode_dir / "codex_planner",
            detailed_task_description=init_case["detailed_task_description"],
            task=env_runner.env_id,
            episode=episode_id,
            init_case_id=init_case["case_id"],
            planner_guidance=init_case.get("planner_guidance", {}),
        )
        planner_started = True
        while step_count < episode_max_steps:
            round_started = time.perf_counter()
            planner_call_started = time.perf_counter()
            source_recent_frame_ids = list(recent_ids)
            planner_front, planner_wrist, planner_frame_ids = (
                subsample_recent_observations(
                    recent_front,
                    recent_wrist,
                    recent_ids,
                    args.planner_frame_stride,
                )
            )
            planner_json_fallback = False
            planner_json_error: str | None = None
            try:
                grounded_subtask, planner_record = planner.plan(
                    front_frames=planner_front,
                    wrist_frames=planner_wrist,
                    frame_ids=planner_frame_ids,
                )
                planner_http_wall_seconds = time.perf_counter() - planner_call_started
                consecutive_planner_json_failures = 0
                last_grounded_subtask = grounded_subtask
                last_planner_record = planner_record
            except Exception as exc:
                planner_http_wall_seconds = time.perf_counter() - planner_call_started
                max_json_failures = int(
                    getattr(args, "max_consecutive_planner_json_failures", 0)
                )
                if max_json_failures <= 0 or not is_planner_json_error(exc):
                    raise
                planner_json_error = f"{type(exc).__name__}: {exc}"
                consecutive_planner_json_failures += 1
                total_planner_json_failures += 1
                maximum_consecutive_planner_json_failures = max(
                    maximum_consecutive_planner_json_failures,
                    consecutive_planner_json_failures,
                )
                failure_row = {
                    "event": "planner_json_failure",
                    "task": env_runner.env_id,
                    "episode": episode_id,
                    "step": step_count,
                    "planner_input_frame_ids": planner_frame_ids,
                    "consecutive_failures": consecutive_planner_json_failures,
                    "failure_limit": max_json_failures,
                    "has_previous_subtask": last_grounded_subtask is not None,
                    "previous_grounded_subtask": last_grounded_subtask,
                    "planner_http_wall_seconds": planner_http_wall_seconds,
                    "error": planner_json_error,
                }
                with (episode_dir / "planner_json_failures.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(json.dumps(failure_row, ensure_ascii=False) + "\n")
                print(json.dumps(failure_row, ensure_ascii=False), flush=True)
                if consecutive_planner_json_failures >= max_json_failures:
                    result = "planner_json_failure"
                    break
                if last_grounded_subtask is None or last_planner_record is None:
                    # Round 0 has no previous executor command to repeat. Retry the
                    # same observation without advancing simulator time.
                    continue
                grounded_subtask = last_grounded_subtask
                planner_json_fallback = True
                planner_json_fallback_chunks += 1
                previous_response = last_planner_record.get("response", {})
                planner_record = {
                    "round_index": last_planner_record.get("round_index"),
                    "recent_rollout_frame_ids": planner_frame_ids,
                    "response": {
                        "stage_summary": previous_response.get("stage_summary"),
                        "recent_outcome": (
                            "Planner JSON was unavailable; repeated the previous "
                            "grounded subtask for one executor chunk."
                        ),
                        "recent_status": "planner_json_fallback",
                        "updated_progress": previous_response.get("updated_progress"),
                    },
                    "prompt_profile": last_planner_record.get("prompt_profile"),
                    "skill_key": last_planner_record.get("skill_key"),
                    "instruction_strategy": "repeat_after_planner_json_failure",
                    "completion_gate": {"applicable": False},
                    "added_keyframes": [],
                    "codex_usage": None,
                    "codex_wall_seconds": planner_http_wall_seconds,
                    "grounded_subtask_for_vla": grounded_subtask,
                    "planner_json_fallback": True,
                    "planner_json_error": planner_json_error,
                    "consecutive_planner_json_failures": (
                        consecutive_planner_json_failures
                    ),
                }
            planner_records.append(planner_record)

            buffer_started = time.perf_counter()
            response = policy_client.add_buffer(
                pack_buffer(policy_images, policy_states, exec_start_idx)
            )
            while not response.get("add_buffer_finished", False):
                time.sleep(0.1)
            add_buffer_seconds = time.perf_counter() - buffer_started

            element = {
                "observation/image": current_front,
                "observation/wrist_image": current_wrist,
                "observation/state": current_state,
                "prompt": task_goal,
                "grounded_subgoal": grounded_subtask,
            }
            policy_started = time.perf_counter()
            # One grounded text command is submitted to pi0.5 once per planner
            # round.  The returned low-level action sequence is one executor
            # action chunk, whose maximum size is args.obs_horizon.  Completion
            # reinforcement counts repeated command submissions/chunks; its K
            # does not change this slice or mean K simulator steps.
            actions = policy_client.infer(element)["actions"][: args.obs_horizon]
            policy_seconds = time.perf_counter() - policy_started

            recent_front, recent_wrist, recent_ids = [], [], []
            policy_images, policy_states = [], []
            exec_start_idx = 0
            env_step_seconds = 0.0
            executed = 0
            for action_index, action in enumerate(actions):
                diagnostic_before = (
                    capture_env_diagnostics(env_runner)
                    if getattr(args, "record_env_diagnostics", False)
                    else None
                )
                step_started = time.perf_counter()
                obs, stop, result = env_runner.step(np.asarray(action))
                env_step_seconds += time.perf_counter() - step_started
                step_count += 1
                rollout_frame_cursor += 1
                executed += 1
                if getattr(args, "record_env_diagnostics", False):
                    diagnostic_row = {
                        "step": step_count,
                        "round_index": planner_record["round_index"],
                        "action_index_in_chunk": action_index,
                        "grounded_subtask": grounded_subtask,
                        "before": diagnostic_before,
                        "after": capture_env_diagnostics(env_runner),
                        "environment_result": result,
                        "stop": bool(stop),
                    }
                    with (episode_dir / "env_diagnostics.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(json.dumps(diagnostic_row, ensure_ascii=False) + "\n")
                if obs[0] is None:
                    stop = True
                else:
                    current_front, current_wrist, current_state = obs
                    recent_front.append(current_front.copy())
                    recent_wrist.append(current_wrist.copy())
                    recent_ids.append(rollout_frame_cursor)
                    policy_images.append(current_front.copy())
                    policy_states.append(current_state.copy())
                    if save_video:
                        video_frames.append(
                            np.concatenate([current_front, current_wrist], axis=1)
                        )
                if stop or step_count >= episode_max_steps:
                    break

            semantic_stagnation_limit = int(
                getattr(args, "semantic_stagnation_stop", 0)
            )
            semantic_stagnation_early_stop = should_stop_for_semantic_stagnation(
                planner_record, semantic_stagnation_limit, result
            )
            if semantic_stagnation_early_stop:
                result = "stagnation_early_stop"
            planner_round_limit = int(getattr(args, "max_planner_rounds", 0))
            planner_round_limit_stop = should_stop_for_planner_round_limit(
                len(planner_records), planner_round_limit, result
            )
            if planner_round_limit_stop:
                result = "planner_round_limit"
            row = {
                "round_index": planner_record["round_index"],
                "input_frame_ids": planner_record["recent_rollout_frame_ids"],
                "source_recent_frame_ids": source_recent_frame_ids,
                "source_recent_frame_count": len(source_recent_frame_ids),
                "planner_input_frame_count": len(planner_frame_ids),
                "planner_frame_stride": args.planner_frame_stride,
                "end_step": step_count,
                "executed_steps": executed,
                "grounded_subtask": grounded_subtask,
                "planner_prompt_profile": planner_record.get("prompt_profile"),
                "stage_summary": planner_record["response"].get("stage_summary"),
                "recent_outcome": planner_record["response"].get("recent_outcome"),
                "recent_status": planner_record["response"].get("recent_status"),
                "updated_progress": planner_record["response"].get("updated_progress"),
                "consecutive_no_progress_rounds_before": planner_record.get(
                    "consecutive_no_progress_rounds_before"
                ),
                "consecutive_no_progress_rounds_after": planner_record.get(
                    "consecutive_no_progress_rounds_after"
                ),
                "rethinking_required_this_turn": planner_record.get(
                    "rethinking_required_this_turn"
                ),
                "planner_mode": planner_record.get("planner_mode"),
                "skill_key": planner_record.get("skill_key"),
                "instruction_strategy": planner_record.get("instruction_strategy"),
                "completion_check": planner_record.get("response", {}).get(
                    "completion_check"
                ),
                "completion_gate": planner_record.get("completion_gate"),
                "refine_check": planner_record.get("response", {}).get(
                    "refine_check"
                ),
                "experience_gate": planner_record.get("experience_gate"),
                "episode_local_lesson_count": planner_record.get(
                    "episode_local_lesson_count"
                ),
                "added_keyframes": planner_record.get("added_keyframes", []),
                "codex_usage": planner_record.get("codex_usage"),
                "codex_cli_seconds": planner_record["codex_wall_seconds"],
                "planner_http_wall_seconds": planner_http_wall_seconds,
                "add_buffer_seconds": add_buffer_seconds,
                "vla_infer_seconds": policy_seconds,
                "env_step_seconds": env_step_seconds,
                "round_wall_seconds": time.perf_counter() - round_started,
                "environment_status": result,
                "planner_json_fallback": planner_json_fallback,
                "planner_json_error": planner_json_error,
                "consecutive_planner_json_failures": (
                    consecutive_planner_json_failures
                ),
                "semantic_stagnation_stop_limit": semantic_stagnation_limit,
                "semantic_stagnation_early_stop": semantic_stagnation_early_stop,
                "planner_round_limit": planner_round_limit,
                "planner_round_limit_stop": planner_round_limit_stop,
            }
            timing_rows.append(row)
            with (episode_dir / "timings.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"task": env_runner.env_id, "episode": episode_id, **row}, ensure_ascii=False), flush=True)
            if result in TERMINAL_RESULTS or not recent_front:
                break
    finally:
        env_runner.close_env()
        if planner_started:
            try:
                planner.end_episode()
            except Exception:
                traceback.print_exc()

    if step_count >= episode_max_steps and result not in TERMINAL_RESULTS:
        result = "timeout"
    if save_video and video_frames:
        save_rollout_video(
            episode_dir / "videos" / f"{env_runner.env_id}_ep{episode_id:02d}.mp4",
            video_frames,
        )

    planner_times = [row["codex_cli_seconds"] for row in timing_rows]
    keyframe_index_path = episode_dir / "codex_planner" / "keyframes" / "index.json"
    keyframes = json.loads(keyframe_index_path.read_text()) if keyframe_index_path.is_file() else []
    summary = {
        "complete": True,
        "method": method_name(planner_memory_mode),
        "planner_memory_mode": planner_memory_mode,
        "planner_prompt_profile": (
            planner_records[-1].get("prompt_profile") if planner_records else None
        ),
        "planner_behavior_id": (
            planner_records[-1].get("rollout_behavior_id")
            if planner_records else None
        ),
        "rethinking_threshold": (
            planner_records[-1].get("rethinking_threshold") if planner_records else None
        ),
        "rethinking_trigger_count": sum(
            record.get("rethinking_required_this_turn") is True
            for record in planner_records
        ),
        "instruction_exploration_trigger_count": sum(
            record.get("planner_mode") == "explore_instruction"
            for record in planner_records
        ),
        "action_success_criterion_turn_count": sum(
            record.get("completion_gate", {}).get("applicable") is True
            for record in planner_records
        ),
        "completion_gate_block_count": sum(
            record.get("completion_gate", {}).get("semantic_switch_blocked") is True
            or record.get("completion_gate", {}).get("completed_status_blocked") is True
            for record in planner_records
        ),
        "experience_gate_applicable_turn_count": sum(
            record.get("experience_gate", {}).get("applicable") is True
            for record in planner_records
        ),
        "experience_gate_block_count": sum(
            record.get("experience_gate", {}).get("transition_blocked") is True
            for record in planner_records
        ),
        "experience_gate_satisfied_release_count": sum(
            record.get("experience_gate", {}).get(
                "released_due_to_satisfied_evidence"
            )
            is True
            for record in planner_records
        ),
        "experience_gate_budget_release_count": sum(
            record.get("experience_gate", {}).get("released_due_to_budget") is True
            for record in planner_records
        ),
        "experience_ids_used": sorted(
            {
                str(record["experience_gate"]["experience_id"])
                for record in planner_records
                if record.get("experience_gate", {}).get("applicable") is True
            }
        ),
        "final_episode_local_lesson_count": (
            planner_records[-1].get("episode_local_lesson_count")
            if planner_records
            else 0
        ),
        "task": env_runner.env_id,
        "dataset_split": getattr(env_runner, "dataset_split", "test"),
        "task_goal_for_pi05": task_goal,
        "init_case_id": init_case["case_id"],
        "detailed_task_description_for_planner": init_case["detailed_task_description"],
        "planner_guidance_for_planner": init_case.get("planner_guidance", {}),
        "initial_demo_images_sent_to_planner": False,
        "initial_demo_history_preserved_for_pi05": True,
        "episode": episode_id,
        "result": result,
        "success": result == "success",
        "obs_horizon": args.obs_horizon,
        "planner_frame_stride": args.planner_frame_stride,
        "max_steps": episode_max_steps,
        "steps_executed": step_count,
        "rounds": len(timing_rows),
        "episode_wall_seconds": time.perf_counter() - episode_started,
        "planner_total_seconds": sum(planner_times),
        "planner_mean_seconds_per_round": float(np.mean(planner_times)) if planner_times else None,
        "codex_usage_total": usage_totals(planner_records),
        "vla_total_seconds": sum(row["vla_infer_seconds"] for row in timing_rows),
        "env_step_total_seconds": sum(row["env_step_seconds"] for row in timing_rows),
        "keyframe_count": len(keyframes),
        "final_progress": (
            planner_records[-1]["response"].get("updated_progress")
            if planner_records
            else None
        ),
        "last_planner_progress_before_terminal_action": (
            planner_records[-1]["response"].get("updated_progress")
            if planner_records
            else None
        ),
        "terminal_after_grounded_subtask": (
            planner_records[-1].get("grounded_subtask_for_vla")
            if planner_records
            else None
        ),
        "video_saved": save_video,
        "env_diagnostics_recorded": getattr(args, "record_env_diagnostics", False),
        "planner_json_failure_limit": int(
            getattr(args, "max_consecutive_planner_json_failures", 0)
        ),
        "semantic_stagnation_stop_limit": int(
            getattr(args, "semantic_stagnation_stop", 0)
        ),
        "semantic_stagnation_early_stop": result == "stagnation_early_stop",
        "planner_round_limit": int(getattr(args, "max_planner_rounds", 0)),
        "planner_round_limit_stop": result == "planner_round_limit",
        "planner_json_failure_count": total_planner_json_failures,
        "planner_json_fallback_chunk_count": planner_json_fallback_chunks,
        "maximum_consecutive_planner_json_failures": (
            maximum_consecutive_planner_json_failures
        ),
    }
    write_json(episode_dir / "summary.json", summary)
    return summary


def archive_failed_episode_attempt(
    *, run_dir: Path, episode_dir: Path, task: str, episode_id: int
) -> Path:
    failed_root = run_dir / "_failed_attempts"
    failed_root.mkdir(exist_ok=True)
    attempt = 1
    failed_dir = failed_root / f"{task}_episode_{episode_id:02d}_attempt_{attempt:03d}"
    while failed_dir.exists():
        attempt += 1
        failed_dir = failed_root / (
            f"{task}_episode_{episode_id:02d}_attempt_{attempt:03d}"
        )
    shutil.move(str(episode_dir), str(failed_dir))
    return failed_dir


def build_work_items(args: argparse.Namespace) -> tuple[list[str], list[tuple[str, int]]]:
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    if not tasks:
        raise ValueError("--tasks must contain at least one task")
    if args.episode_ids:
        episode_ids = [
            int(value.strip()) for value in args.episode_ids.split(",") if value.strip()
        ]
        if not episode_ids or len(set(episode_ids)) != len(episode_ids):
            raise ValueError("--episode-ids must contain unique episode integers")
        if any(value < 0 for value in episode_ids):
            raise ValueError("--episode-ids must be nonnegative")
    else:
        episode_ids = list(range(args.num_episodes))
    all_work_items = [(task, episode) for task in tasks for episode in episode_ids]
    if args.flat_shard:
        work_items = all_work_items[args.episode_start :: args.episode_stride]
    elif args.episode_ids:
        selected_episode_ids = episode_ids[
            args.episode_start :: args.episode_stride
        ]
        work_items = [
            (task, episode) for task in tasks for episode in selected_episode_ids
        ]
    else:
        selected_episode_ids = range(
            args.episode_start, args.num_episodes, args.episode_stride
        )
        work_items = [
            (task, episode) for task in tasks for episode in selected_episode_ids
        ]
    return tasks, work_items


def parse_task_max_steps(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in (part.strip() for part in value.split(",")):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "--task-max-steps entries must use TASK=STEPS syntax"
            )
        task, raw_steps = (part.strip() for part in item.split("=", 1))
        if not task or task in result:
            raise ValueError("--task-max-steps contains an empty or duplicate task")
        try:
            steps = int(raw_steps)
        except ValueError as exc:
            raise ValueError("--task-max-steps values must be integers") from exc
        if steps < 1:
            raise ValueError("--task-max-steps values must be positive")
        result[task] = steps
    return result


def parse_video_episodes(value: str, num_episodes: int) -> set[int]:
    """Parse a video subset while allowing a compact all-episodes selector."""
    normalized = value.strip().lower()
    if normalized == "all":
        return set(range(num_episodes))
    try:
        episodes = {
            int(item.strip()) for item in value.split(",") if item.strip()
        }
    except ValueError as exc:
        raise ValueError(
            "--video-episodes must be 'all' or comma-separated integers"
        ) from exc
    if any(episode < 0 for episode in episodes):
        raise ValueError("--video-episodes must be nonnegative")
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--init-results-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--planner-url", required=True)
    parser.add_argument("--planner-token-file", type=Path, required=True)
    parser.add_argument("--pi05-host", default="127.0.0.1")
    parser.add_argument("--pi05-port", type=int, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument(
        "--dataset-split",
        choices=("train", "val", "test"),
        default="test",
        help="RoboMME metadata split. The value is audited and isolates init caches.",
    )
    parser.add_argument(
        "--evolution-stage",
        choices=("baseline", "explore", "candidate", "frozen", "evaluate"),
        default="evaluate",
        help="Audited self-evolution stage; test only permits frozen/evaluate.",
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument(
        "--episode-ids",
        default="",
        help="Optional comma-separated episode subset before sharding, for example 1,21.",
    )
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-stride", type=int, default=1)
    parser.add_argument("--flat-shard", action="store_true")
    parser.add_argument("--obs-horizon", type=int, default=16, choices=(16,))
    parser.add_argument(
        "--planner-frame-stride",
        type=int,
        default=2,
        help=(
            "Send every Nth synchronized front/wrist observation to Codex, "
            "anchored at the newest frame (default: 2)."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument(
        "--task-max-steps",
        default="",
        help=(
            "Optional per-task overrides as comma-separated TASK=STEPS entries, "
            "for example VideoUnmask=500."
        ),
    )
    parser.add_argument("--video-episodes", default="0")
    parser.add_argument(
        "--record-env-diagnostics",
        action="store_true",
        help=(
            "Record privileged simulator state to env_diagnostics.jsonl for "
            "offline auditing only; it is never sent to the planner or policy."
        ),
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--session-prefix", default="codex-grounded-v1")
    parser.add_argument("--planner-timeout-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--max-consecutive-planner-json-failures",
        type=int,
        default=0,
        help=(
            "When positive, repeat the previous grounded subtask for one executor "
            "chunk after a planner JSONDecodeError and terminate after this many "
            "consecutive failures. Before the first valid command, retry without "
            "advancing the environment (default: 0, disabled)."
        ),
    )
    parser.add_argument(
        "--semantic-stagnation-stop",
        type=int,
        default=0,
        help=(
            "When positive, terminate after this many consecutive planner rounds "
            "with observable no-progress. Disabled by default so mature profiles "
            "retain their prior behavior."
        ),
    )
    parser.add_argument(
        "--max-planner-rounds",
        type=int,
        default=0,
        help=(
            "Optional hard cap on planner/executor rounds per episode. Disabled "
            "by default so mature profiles retain their prior behavior."
        ),
    )
    parser.add_argument(
        "--planner-memory-mode",
        choices=("keyframes", "text_progress", "codex_workflow"),
        default="keyframes",
    )
    parser.add_argument("--rerun-short-timeouts", action="store_true")
    parser.add_argument(
        "--max-episode-attempts",
        type=int,
        default=2,
        help=(
            "Maximum attempts for an episode after infrastructure or planner "
            "exceptions (default: 2). Simulator terminal failures are not retried."
        ),
    )
    parser.add_argument(
        "--upload-episodes",
        action="store_true",
        help="Upload each completed/error episode through the authenticated planner service.",
    )
    args = parser.parse_args()

    if args.run_id and not args.run_id.replace("_", "").replace("-", "").isalnum():
        parser.error("--run-id may only contain letters, digits, underscores, and hyphens")
    if args.episode_start < 0 or args.episode_stride < 1 or args.num_episodes < 1:
        parser.error("episode bounds must be nonnegative and stride/num-episodes positive")
    if args.planner_frame_stride < 1:
        parser.error("--planner-frame-stride must be positive")
    if args.max_episode_attempts < 1:
        parser.error("--max-episode-attempts must be positive")
    if args.max_consecutive_planner_json_failures < 0:
        parser.error("--max-consecutive-planner-json-failures must be nonnegative")
    if args.semantic_stagnation_stop < 0:
        parser.error("--semantic-stagnation-stop must be nonnegative")
    if args.max_planner_rounds < 0:
        parser.error("--max-planner-rounds must be nonnegative")
    if args.dataset_split == "test" and args.evolution_stage not in {"frozen", "evaluate"}:
        parser.error(
            "test is evaluation-only: --evolution-stage must be frozen or evaluate"
        )
    if args.dataset_split != "test" and args.evolution_stage == "evaluate":
        parser.error(
            "train/val require an explicit evolution stage; evaluation defaults are "
            "reserved for test to prevent accidental split mixing"
        )

    repo_root = args.repo_root.resolve()
    benchmark_root = args.benchmark_root.resolve()
    init_results_dir = args.init_results_dir.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo_root / "scripts" / "codex_rollout"))
    sys.path.insert(0, str(repo_root / "examples" / "robomme"))
    sys.path.insert(0, str(benchmark_root / "src"))

    tasks, work_items = build_work_items(args)
    try:
        task_max_steps = parse_task_max_steps(args.task_max_steps)
    except ValueError as exc:
        parser.error(str(exc))
    unknown_task_overrides = sorted(set(task_max_steps) - set(tasks))
    if unknown_task_overrides:
        parser.error(
            "--task-max-steps contains tasks absent from --tasks: "
            + ",".join(unknown_task_overrides)
        )
    init_cases = {
        (task, episode): load_detailed_task_description(
            init_results_dir, task, episode, args.dataset_split
        )
        for task, episode in work_items
    }
    from env_runner import EnvRunner
    from openpi_client import websocket_client_policy
    from remote_planner import RemoteCodexPlanner
    from utils import pack_buffer

    planner = RemoteCodexPlanner(
        args.planner_url,
        args.planner_timeout_seconds,
        args.planner_token_file.resolve(),
    )
    policy_client = websocket_client_policy.MMEVLAWebsocketClientPolicy(
        args.pi05_host, args.pi05_port
    )
    try:
        video_episodes = parse_video_episodes(args.video_episodes, args.num_episodes)
    except ValueError as exc:
        parser.error(str(exc))
    run_suffix = f"_{args.run_id}" if args.run_id else ""
    progress_path = run_dir / f"progress{run_suffix}.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    progress["config"] = {
        "method": method_name(args.planner_memory_mode),
        "planner_memory_mode": args.planner_memory_mode,
        "tasks": tasks,
        "dataset_split": args.dataset_split,
        "evolution_stage": args.evolution_stage,
        "num_episodes": args.num_episodes,
        "episode_ids": args.episode_ids,
        "episode_start": args.episode_start,
        "episode_stride": args.episode_stride,
        "flat_shard": args.flat_shard,
        "work_items": [{"task": task, "episode": episode} for task, episode in work_items],
        "obs_horizon": args.obs_horizon,
        "planner_frame_stride": args.planner_frame_stride,
        "max_steps": args.max_steps,
        "task_max_steps": task_max_steps,
        "planner_url": args.planner_url,
        "planner_http_authentication": "bearer token",
        "pi05_host": args.pi05_host,
        "pi05_port": args.pi05_port,
        "init_results_dir": str(init_results_dir),
        "grounded_level_ft_only": True,
        "initial_demo_images_sent_to_planner": False,
        "initial_demo_history_preserved_for_pi05": True,
        "record_env_diagnostics": args.record_env_diagnostics,
        "max_episode_attempts": args.max_episode_attempts,
    }
    progress.setdefault("episodes", {})
    write_json(progress_path, progress)

    runners: dict[str, Any] = {}
    for task, episode_id in work_items:
        episode_max_steps = task_max_steps.get(task, args.max_steps)
        progress["episodes"].setdefault(task, {})
        if args.upload_episodes:
            uploaded_summary = planner.get_episode_summary(
                task=task, episode=episode_id
            )
            if uploaded_summary is not None:
                progress["episodes"][task][str(episode_id)] = uploaded_summary
                print(
                    f"[remote-resume] skip {task} episode {episode_id}",
                    flush=True,
                )
                continue
        episode_dir = run_dir / task / f"episode_{episode_id:02d}"
        summary_path = episode_dir / "summary.json"
        previous: dict[str, Any] = {}
        if summary_path.is_file():
            try:
                previous = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError):
                previous = {}
            rerun_short_timeout = (
                args.rerun_short_timeouts
                and previous.get("complete") is True
                and previous.get("result") == "timeout"
                and int(previous.get("steps_executed", episode_max_steps))
                < episode_max_steps
            )
            if previous.get("complete") is True and not rerun_short_timeout:
                progress["episodes"][task][str(episode_id)] = previous
                print(f"[resume] skip {task} episode {episode_id}", flush=True)
                continue
        if episode_dir.exists():
            archive_failed_episode_attempt(
                run_dir=run_dir,
                episode_dir=episode_dir,
                task=task,
                episode_id=episode_id,
            )
        episode_dir.mkdir(parents=True)
        runner = runners.setdefault(
            task,
            EnvRunner(
                task,
                run_dir / task,
                max_steps=episode_max_steps,
                dataset_split=args.dataset_split,
            ),
        )
        summary: dict[str, Any] = {}
        for episode_attempt in range(1, args.max_episode_attempts + 1):
            try:
                summary = run_episode(
                    args=args,
                    planner=planner,
                    policy_client=policy_client,
                    env_runner=runner,
                    pack_buffer=pack_buffer,
                    episode_id=episode_id,
                    episode_dir=episode_dir,
                    init_case=init_cases[(task, episode_id)],
                    save_video=episode_id in video_episodes,
                    max_steps=episode_max_steps,
                )
                summary["episode_attempt"] = episode_attempt
                summary["max_episode_attempts"] = args.max_episode_attempts
                write_json(episode_dir / "summary.json", summary)
                break
            except Exception as exc:
                runner.close_env()
                try:
                    planner.end_episode()
                except Exception:
                    traceback.print_exc()
                summary = {
                    "complete": False,
                    "method": method_name(args.planner_memory_mode),
                    "planner_memory_mode": args.planner_memory_mode,
                    "task": task,
                    "episode": episode_id,
                    "result": "error",
                    "episode_attempt": episode_attempt,
                    "max_episode_attempts": args.max_episode_attempts,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                write_json(episode_dir / "error.json", summary)
                print(json.dumps(summary, ensure_ascii=False), flush=True)
                if episode_attempt >= args.max_episode_attempts:
                    break
                archived = archive_failed_episode_attempt(
                    run_dir=run_dir,
                    episode_dir=episode_dir,
                    task=task,
                    episode_id=episode_id,
                )
                print(
                    json.dumps(
                        {
                            "event": "episode_retry",
                            "task": task,
                            "episode": episode_id,
                            "failed_attempt": episode_attempt,
                            "archived_to": str(archived),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                episode_dir.mkdir(parents=True)
        progress["episodes"][task][str(episode_id)] = summary
        if args.upload_episodes:
            upload_result = planner.upload_episode(
                episode_dir=episode_dir,
                task=task,
                episode=episode_id,
            )
            print(
                json.dumps({"episode_upload": upload_result}, ensure_ascii=False),
                flush=True,
            )
        write_json(progress_path, progress)

    completed = [
        value
        for task, episode in work_items
        if isinstance((value := progress["episodes"].get(task, {}).get(str(episode))), dict)
        and value.get("complete") is True
    ]
    final = {
        "complete": len(completed) == len(work_items),
        "dataset_split": args.dataset_split,
        "evolution_stage": args.evolution_stage,
        "expected_episodes": len(work_items),
        "completed_episodes": len(completed),
        "success_rate": {
            task: (
                float(
                    np.mean(
                        [
                            progress["episodes"][task][str(episode)]["success"]
                            for item_task, episode in work_items
                            if item_task == task
                            and progress["episodes"][task].get(str(episode), {}).get("complete")
                        ]
                    )
                )
                if any(
                    item_task == task
                    and progress["episodes"][task].get(str(episode), {}).get("complete")
                    for item_task, episode in work_items
                )
                else None
            )
            for task in tasks
        },
        "codex_usage_total": usage_totals(completed),
    }
    write_json(run_dir / f"final_summary{run_suffix}.json", final)
    print(json.dumps(final, ensure_ascii=False), flush=True)
    if not final["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
