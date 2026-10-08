#!/usr/bin/env python3
"""Stateful visual-memory planner backed by the local Codex CLI login."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


HARNESS_VERSION = "codex-rollout-planner-v1"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "high"
CAPACITY_RETRY_LIMIT = 4
CAPACITY_RETRY_DELAY_SECONDS = 15.0
AUTH_ENV_VARS_TO_REMOVE = ("CODEX_API_KEY", "OPENAI_API_KEY")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT = SCRIPT_DIR / "prompts" / "planner_system_grounded_v1.txt"
DEFAULT_USER_PROMPT = SCRIPT_DIR / "prompts" / "planner_user_rollout_v1.txt"
DEFAULT_OUTPUT_SCHEMA = SCRIPT_DIR / "schemas" / "planner_rollout_output_v1.schema.json"


@dataclass(frozen=True)
class Keyframe:
    frame_id: str
    source_frame_id: str
    rollout_frame: int
    view: str
    path: Path
    description: str
    reason: str
    selected_round: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def sanitized_codex_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in AUTH_ENV_VARS_TO_REMOVE:
        environment.pop(variable, None)
    return environment


def get_codex_version(codex_bin: str, environment: dict[str, str]) -> str:
    result = subprocess.run(
        [codex_bin, "--version"], capture_output=True, text=True, env=environment
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to run {codex_bin} --version: {result.stderr.strip()}")
    return result.stdout.strip()


def require_chatgpt_login(codex_bin: str, environment: dict[str, str]) -> str:
    result = subprocess.run(
        [codex_bin, "login", "status"], capture_output=True, text=True, env=environment
    )
    status = "\n".join(
        value for value in (result.stdout.strip(), result.stderr.strip()) if value
    )
    if result.returncode != 0 or "chatgpt" not in status.lower():
        raise RuntimeError(
            "Codex is not logged in with ChatGPT. Run `codex login` first. "
            f"Reported status: {status or '<empty>'}"
        )
    return status


def parse_codex_events(events_path: Path) -> tuple[str | None, dict[str, int] | None]:
    thread_id: str | None = None
    usage: dict[str, int] | None = None
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
        elif event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = {str(key): int(value) for key, value in event["usage"].items()}
    return thread_id, usage


def run_codex_inference(
    *,
    codex_bin: str,
    model: str,
    reasoning_effort: str,
    prompt: str,
    output_schema: Path,
    image_paths: list[Path],
    destination: Path,
    environment: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("Rollout planning requires at least one attached image")
    destination.mkdir(parents=True, exist_ok=True)
    events_path = destination / "codex_events.jsonl"
    stderr_path = destination / "codex_stderr.log"
    response_text_path = destination / "response.txt"
    command = [
        codex_bin,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--cd",
        "__ISOLATED_WORKDIR__",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--model",
        model,
        "--config",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--output-schema",
        str(output_schema.resolve()),
        "--json",
        "--output-last-message",
        str(response_text_path.resolve()),
        "--image",
        *(str(path.resolve()) for path in image_paths),
        "--",
        "-",
    ]
    started_at = utc_now()
    started = time.monotonic()
    capacity_retry_count = 0
    with tempfile.TemporaryDirectory(prefix="codex_rollout_") as isolated_dir:
        command[command.index("__ISOLATED_WORKDIR__")] = isolated_dir
        while True:
            try:
                with events_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
                    "w", encoding="utf-8"
                ) as stderr_handle:
                    result = subprocess.run(
                        command,
                        input=prompt,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        text=True,
                        env=environment,
                        timeout=timeout_seconds,
                    )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"Codex timed out after {timeout_seconds:.1f}s") from exc
            if result.returncode == 0:
                break
            failure_text = "\n".join(
                (
                    events_path.read_text(encoding="utf-8", errors="replace"),
                    stderr_path.read_text(encoding="utf-8", errors="replace"),
                )
            )
            capacity_failure = "selected model is at capacity" in failure_text.lower()
            if not capacity_failure or capacity_retry_count >= CAPACITY_RETRY_LIMIT:
                break
            attempt_number = capacity_retry_count + 1
            shutil.copy2(
                events_path,
                destination / f"codex_events.capacity_attempt_{attempt_number:02d}.jsonl",
            )
            shutil.copy2(
                stderr_path,
                destination / f"codex_stderr.capacity_attempt_{attempt_number:02d}.log",
            )
            capacity_retry_count += 1
            time.sleep(CAPACITY_RETRY_DELAY_SECONDS)
    wall_seconds = time.monotonic() - started
    thread_id, usage = parse_codex_events(events_path)
    if result.returncode != 0:
        raise RuntimeError(
            f"Codex exited with status {result.returncode}; see {stderr_path} and {events_path}"
        )
    if not response_text_path.is_file():
        raise RuntimeError("Codex completed without writing response.txt")
    try:
        response = json.loads(response_text_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Codex response is not valid JSON: {response_text_path}") from exc
    return {
        "response": response,
        "thread_id": thread_id,
        "usage": usage,
        "started_at": started_at,
        "wall_seconds": wall_seconds,
        "exit_code": result.returncode,
        "capacity_retry_count": capacity_retry_count,
    }


def validate_response(
    response: Any,
    *,
    allowed_recent_ids: set[str],
    allowed_recent_front_ids: set[str],
) -> None:
    if not isinstance(response, dict):
        raise ValueError("Planner response must be a JSON object")
    expected = {"stage_summary", "current_subtask_template", "points", "new_keyframes"}
    if set(response) != expected:
        raise ValueError(f"Planner response fields must be exactly {sorted(expected)}")
    for key, limit in (("stage_summary", 240), ("current_subtask_template", 180)):
        if not isinstance(response[key], str) or not response[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
        if len(response[key]) > limit:
            raise ValueError(f"{key} exceeds {limit} characters")

    points = response["points"]
    if not isinstance(points, list) or len(points) > 2:
        raise ValueError("points must be a list containing at most two entries")
    point_ids: set[str] = set()
    point_fields = {"id", "object_description", "source_frame_id", "x_norm", "y_norm"}
    for index, point in enumerate(points):
        if not isinstance(point, dict) or set(point) != point_fields:
            raise ValueError(f"points[{index}] has invalid fields")
        point_id = point["id"]
        if point_id not in {"point_1", "point_2"} or point_id in point_ids:
            raise ValueError(f"points[{index}].id is invalid or duplicated: {point_id}")
        point_ids.add(point_id)
        if point["source_frame_id"] not in allowed_recent_front_ids:
            raise ValueError(
                f"points[{index}].source_frame_id is not an attached recent front frame"
            )
        if not isinstance(point["object_description"], str) or not point[
            "object_description"
        ].strip():
            raise ValueError(f"points[{index}].object_description must be non-empty")
        for coordinate in ("x_norm", "y_norm"):
            value = point[coordinate]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 999:
                raise ValueError(f"points[{index}].{coordinate} must be an integer in [0, 999]")

    template_point_ids = set(re.findall(r"<(point_[0-9]+)>", response["current_subtask_template"]))
    if template_point_ids != point_ids:
        raise ValueError(
            "current_subtask_template placeholders must match points exactly: "
            f"template={sorted(template_point_ids)}, points={sorted(point_ids)}"
        )

    keyframes = response["new_keyframes"]
    if not isinstance(keyframes, list) or len(keyframes) > 2:
        raise ValueError("new_keyframes must be a list containing at most two entries")
    seen_keyframes: set[str] = set()
    for index, keyframe in enumerate(keyframes):
        if not isinstance(keyframe, dict) or set(keyframe) != {
            "frame_id",
            "description",
            "reason",
        }:
            raise ValueError(f"new_keyframes[{index}] has invalid fields")
        frame_id = keyframe["frame_id"]
        if frame_id not in allowed_recent_ids or frame_id in seen_keyframes:
            raise ValueError(
                f"new_keyframes[{index}].frame_id is unattached or duplicated: {frame_id}"
            )
        seen_keyframes.add(frame_id)
        for field in ("description", "reason"):
            if not isinstance(keyframe[field], str) or not keyframe[field].strip():
                raise ValueError(f"new_keyframes[{index}].{field} must be non-empty")


def normalize_response(response: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Apply lossless-for-execution repairs while preserving an audit trail.

    A model may emit a valid point object and then omit its placeholder from the
    final subtask template.  Such a point has no effect on the downstream VLA,
    so deterministically dropping it is safer than failing the whole episode.
    Missing points for referenced placeholders remain validation errors.
    """
    if not isinstance(response, dict):
        return response, []
    template = response.get("current_subtask_template")
    points = response.get("points")
    if not isinstance(template, str) or not isinstance(points, list):
        return response, []

    referenced = set(re.findall(r"<(point_[0-9]+)>", template))
    retained: list[Any] = []
    dropped_ids: list[str] = []
    for point in points:
        point_id = point.get("id") if isinstance(point, dict) else None
        if isinstance(point_id, str) and point_id not in referenced:
            dropped_ids.append(point_id)
        else:
            retained.append(point)
    if not dropped_ids:
        return response, []

    normalized = dict(response)
    normalized["points"] = retained
    return normalized, [
        {
            "kind": "drop_unreferenced_points",
            "point_ids": dropped_ids,
            "reason": "The downstream subtask template did not reference these points.",
        }
    ]


def normalized_to_pixel(value: int, size: int) -> int:
    if size < 1:
        raise ValueError(f"Invalid image dimension: {size}")
    return int(round(value * (size - 1) / 999.0))


def render_grounded_subtask(
    response: dict[str, Any], front_shapes: dict[str, tuple[int, int]]
) -> tuple[str, list[dict[str, Any]]]:
    rendered = response["current_subtask_template"]
    rendered_points: list[dict[str, Any]] = []
    for point in response["points"]:
        height, width = front_shapes[point["source_frame_id"]]
        x_px = normalized_to_pixel(point["x_norm"], width)
        y_px = normalized_to_pixel(point["y_norm"], height)
        rendered = rendered.replace(f"<{point['id']}>", f"<{y_px}, {x_px}>")
        rendered_points.append({**point, "x_px": x_px, "y_px": y_px, "image_width": width, "image_height": height})
    if re.search(r"<point_[0-9]+>", rendered):
        raise ValueError(f"Unresolved point placeholder in rendered subtask: {rendered}")
    return rendered, rendered_points


class CodexRolloutPlanner:
    """Maintain visual keyframes while running one isolated Codex turn per chunk."""

    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT,
        user_prompt_path: Path = DEFAULT_USER_PROMPT,
        output_schema_path: Path = DEFAULT_OUTPUT_SCHEMA,
        timeout_seconds: float = 900.0,
        keep_recent_inputs: bool = False,
        verify_login: bool = True,
    ) -> None:
        self.codex_bin = codex_bin
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.system_prompt_path = system_prompt_path.resolve()
        self.user_prompt_path = user_prompt_path.resolve()
        self.output_schema_path = output_schema_path.resolve()
        self.system_prompt = self.system_prompt_path.read_text(encoding="utf-8")
        self.user_prompt_template = self.user_prompt_path.read_text(encoding="utf-8")
        self.timeout_seconds = timeout_seconds
        self.keep_recent_inputs = keep_recent_inputs
        self.environment = sanitized_codex_environment()
        self.codex_version = get_codex_version(codex_bin, self.environment)
        self.login_status = (
            require_chatgpt_login(codex_bin, self.environment) if verify_login else "not checked"
        )
        self.run_dir: Path | None = None
        self.detailed_task_description = ""
        self.round_index = 0
        self.keyframes: list[Keyframe] = []

    def start_episode(
        self,
        *,
        run_dir: Path,
        detailed_task_description: str,
        task: str,
        episode: int,
        init_case_id: str,
    ) -> None:
        description = detailed_task_description.strip()
        if not description:
            raise ValueError("detailed_task_description must be non-empty")
        run_dir = run_dir.resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty planner run directory: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.detailed_task_description = description
        self.round_index = 0
        self.keyframes = []
        (run_dir / "keyframes").mkdir()
        (run_dir / "rounds").mkdir()
        provenance = {
            "harness_version": HARNESS_VERSION,
            "task": task,
            "episode": episode,
            "init_case_id": init_case_id,
            "detailed_task_description": description,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "codex_version": self.codex_version,
            "authentication": "local ChatGPT login",
            "ephemeral_codex_turns": True,
            "initial_demo_images_sent_to_planner": False,
            "retain_all_selected_keyframes": True,
            "recent_frame_filtering": False,
            "created_at": utc_now(),
        }
        write_json(run_dir / "episode_config.json", provenance)
        shutil.copy2(self.system_prompt_path, run_dir / "planner_system_prompt.txt")
        shutil.copy2(self.user_prompt_path, run_dir / "planner_user_prompt_template.txt")
        shutil.copy2(self.output_schema_path, run_dir / "planner_output_schema.json")
        self._write_keyframe_index()

    def _require_started(self) -> Path:
        if self.run_dir is None:
            raise RuntimeError("start_episode must be called before plan")
        return self.run_dir

    def _write_keyframe_index(self) -> None:
        run_dir = self._require_started()
        write_json(
            run_dir / "keyframes" / "index.json",
            [
                {
                    "frame_id": value.frame_id,
                    "source_frame_id": value.source_frame_id,
                    "rollout_frame": value.rollout_frame,
                    "view": value.view,
                    "file": str(value.path.relative_to(run_dir)),
                    "description": value.description,
                    "reason": value.reason,
                    "selected_round": value.selected_round,
                }
                for value in self.keyframes
            ],
        )

    def _render_prompt(self, image_manifest: str) -> str:
        try:
            user_prompt = self.user_prompt_template.format(
                detailed_task_description=self.detailed_task_description,
                image_manifest=image_manifest,
            )
        except KeyError as exc:
            raise ValueError(f"Unsupported placeholder in rollout user prompt: {exc}") from exc
        return f"{self.system_prompt.strip()}\n\n{user_prompt.strip()}\n"

    def plan(
        self,
        *,
        front_frames: list[np.ndarray],
        wrist_frames: list[np.ndarray],
        frame_ids: list[int],
    ) -> tuple[str, dict[str, Any]]:
        run_dir = self._require_started()
        if not front_frames or len(front_frames) != len(wrist_frames) or len(front_frames) != len(frame_ids):
            raise ValueError("front_frames, wrist_frames, and frame_ids must have the same non-zero length")
        if len(set(frame_ids)) != len(frame_ids) or any(value < 0 for value in frame_ids):
            raise ValueError("frame_ids must be unique non-negative integers")
        if frame_ids != sorted(frame_ids):
            raise ValueError("frame_ids must be chronological")

        round_index = self.round_index
        round_dir = run_dir / "rounds" / f"round_{round_index:04d}"
        if round_dir.exists():
            raise FileExistsError(f"Planner round directory already exists: {round_dir}")
        inputs_dir = round_dir / "recent_inputs"
        inputs_dir.mkdir(parents=True)

        recent_paths: dict[str, Path] = {}
        front_shapes: dict[str, tuple[int, int]] = {}
        for rollout_frame, front, wrist in zip(frame_ids, front_frames, wrist_frames, strict=True):
            front_array = np.asarray(front, dtype=np.uint8)
            wrist_array = np.asarray(wrist, dtype=np.uint8)
            if front_array.ndim != 3 or front_array.shape[2] != 3:
                raise ValueError(f"Front frame {rollout_frame} is not an RGB image")
            if wrist_array.ndim != 3 or wrist_array.shape[2] != 3:
                raise ValueError(f"Wrist frame {rollout_frame} is not an RGB image")
            front_id = f"r_f{rollout_frame:04d}"
            wrist_id = f"r_w{rollout_frame:04d}"
            front_path = inputs_dir / f"{front_id}.png"
            wrist_path = inputs_dir / f"{wrist_id}.png"
            Image.fromarray(front_array).save(front_path)
            Image.fromarray(wrist_array).save(wrist_path)
            recent_paths[front_id] = front_path
            recent_paths[wrist_id] = wrist_path
            front_shapes[front_id] = (int(front_array.shape[0]), int(front_array.shape[1]))

        attachment_rows: list[dict[str, Any]] = []
        image_paths: list[Path] = []
        manifest_lines: list[str] = []
        for keyframe in self.keyframes:
            image_paths.append(keyframe.path)
            manifest_lines.append(
                f"{keyframe.frame_id}: retained {keyframe.view} keyframe from rollout frame "
                f"{keyframe.rollout_frame}; description: {keyframe.description}; reason: {keyframe.reason}"
            )
            attachment_rows.append(
                {
                    "frame_id": keyframe.frame_id,
                    "kind": "historical_keyframe",
                    "view": keyframe.view,
                    "rollout_frame": keyframe.rollout_frame,
                    "file": str(keyframe.path.relative_to(run_dir)),
                }
            )
        for rollout_frame in frame_ids:
            for view in ("front", "wrist"):
                short = "f" if view == "front" else "w"
                frame_id = f"r_{short}{rollout_frame:04d}"
                path = recent_paths[frame_id]
                image_paths.append(path)
                manifest_lines.append(
                    f"{frame_id}: recent {view} observation at rollout frame {rollout_frame}"
                )
                attachment_rows.append(
                    {
                        "frame_id": frame_id,
                        "kind": "recent_observation",
                        "view": view,
                        "rollout_frame": rollout_frame,
                        "file": str(path.relative_to(run_dir)),
                    }
                )

        image_manifest = "\n".join(manifest_lines)
        prompt = self._render_prompt(image_manifest)
        write_text(round_dir / "prompt.txt", prompt)
        write_json(
            round_dir / "input_manifest.json",
            {
                "round_index": round_index,
                "recent_rollout_frame_ids": frame_ids,
                "historical_keyframe_count": len(self.keyframes),
                "recent_pair_count": len(frame_ids),
                "attachment_count": len(image_paths),
                "attachments": attachment_rows,
            },
        )

        result = run_codex_inference(
            codex_bin=self.codex_bin,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            prompt=prompt,
            output_schema=self.output_schema_path,
            image_paths=image_paths,
            destination=round_dir,
            environment=self.environment,
            timeout_seconds=self.timeout_seconds,
        )
        raw_response = result["response"]
        response, response_normalization = normalize_response(raw_response)
        if response_normalization:
            write_json(round_dir / "raw_response.json", raw_response)
        validate_response(
            response,
            allowed_recent_ids=set(recent_paths),
            allowed_recent_front_ids=set(front_shapes),
        )
        rendered_subtask, rendered_points = render_grounded_subtask(response, front_shapes)

        historical_keyframe_ids = [value.frame_id for value in self.keyframes]
        added_keyframes: list[dict[str, Any]] = []
        for selection in response["new_keyframes"]:
            source_id = selection["frame_id"]
            match = re.fullmatch(r"r_([fw])(\d+)", source_id)
            assert match is not None
            view_short, numeric = match.groups()
            view = "front" if view_short == "f" else "wrist"
            keyframe_id = f"k_{view_short}{numeric}"
            target = run_dir / "keyframes" / f"{keyframe_id}.png"
            if any(value.frame_id == keyframe_id for value in self.keyframes):
                raise ValueError(f"Planner selected an already retained keyframe: {keyframe_id}")
            shutil.copy2(recent_paths[source_id], target)
            keyframe = Keyframe(
                frame_id=keyframe_id,
                source_frame_id=source_id,
                rollout_frame=int(numeric),
                view=view,
                path=target,
                description=selection["description"].strip(),
                reason=selection["reason"].strip(),
                selected_round=round_index,
            )
            self.keyframes.append(keyframe)
            added_keyframes.append(
                {
                    "frame_id": keyframe_id,
                    "source_frame_id": source_id,
                    "rollout_frame": keyframe.rollout_frame,
                    "view": view,
                    "file": str(target.relative_to(run_dir)),
                    "description": keyframe.description,
                    "reason": keyframe.reason,
                }
            )
        self._write_keyframe_index()

        record = {
            "round_index": round_index,
            "recent_rollout_frame_ids": frame_ids,
            "historical_keyframe_ids": historical_keyframe_ids,
            "response": response,
            "response_normalization": response_normalization,
            "rendered_points": rendered_points,
            "grounded_subtask_for_vla": rendered_subtask,
            "added_keyframes": added_keyframes,
            "retained_keyframe_ids_after_round": [value.frame_id for value in self.keyframes],
            "codex_thread_id": result["thread_id"],
            "codex_usage": result["usage"],
            "codex_started_at": result["started_at"],
            "codex_wall_seconds": result["wall_seconds"],
            "initial_demo_images_sent": False,
        }
        write_json(round_dir / "response.json", response)
        write_json(round_dir / "record.json", record)
        self.round_index += 1
        if not self.keep_recent_inputs:
            shutil.rmtree(inputs_dir)
        return rendered_subtask, record
