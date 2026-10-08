#!/usr/bin/env python3
"""RoboMME planner whose episode memory is one persistent Codex CLI session.

This intentionally small ablation keeps only frozen R0 skill retrieval and a
bounded one-shot reinforcement lock outside Codex.  Task progress, stage
tracking, and visual history are left to the resumed Codex conversation.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from planner import (
    CAPACITY_RETRY_DELAY_SECONDS,
    CAPACITY_RETRY_LIMIT,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    get_codex_version,
    normalize_response,
    parse_codex_events,
    render_grounded_subtask,
    require_chatgpt_login,
    sanitized_codex_environment,
    utc_now,
    write_json,
    write_text,
)
from skill_retrieval import ExecutorSkillRetriever
from text_progress_planner import (
    ACTION_GATES_V8_4_6_CRITERIA_PATH,
    action_instance_signature,
    load_action_success_criteria,
    match_action_success_criterion,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AGENTS = SCRIPT_DIR / "prompts" / "codex_workflow_v1_AGENTS.md"
DEFAULT_USER_PROMPT = SCRIPT_DIR / "prompts" / "planner_user_codex_workflow_v1.txt"
DEFAULT_OUTPUT_SCHEMA = SCRIPT_DIR / "schemas" / "planner_codex_workflow_v1.schema.json"
DEFAULT_VOCABULARY = SCRIPT_DIR / "knowledge" / "robomme_executor_vocabulary_v1.json"
WORKFLOW_PROFILE = "codex_workflow_v1"
HARNESS_VERSION = "codex-workflow-v1-r0-reinforcement-lock"
RECENT_STATUSES = {
    "initial",
    "progressing",
    "completed",
    "uncertain",
    "stagnant",
    "regressed",
}
OUTPUT_VALIDATION_RETRY_LIMIT = 2
REINFORCEMENT_LOCK = {
    "button_unmask_press_local_event": 1,
    "unmask_pickup_identity_and_separation": 1,
    "unmask_put_down_stable_release": 1,
    "pickhighlight_put_down_stable_release": 1,
}


def _validate_response(
    response: Any,
    *,
    allowed_skills: dict[str, dict[str, Any]],
    allowed_front_ids: set[str],
    round_index: int,
) -> tuple[dict[str, Any], bool]:
    if not isinstance(response, dict):
        raise ValueError("Planner response must be a JSON object")
    expected = {
        "recent_status",
        "recent_outcome",
        "skill_key",
        "current_subtask_template",
        "points",
    }
    if set(response) != expected:
        raise ValueError(f"Planner response fields must be exactly {sorted(expected)}")
    status = response["recent_status"]
    if status not in RECENT_STATUSES:
        raise ValueError(f"invalid recent_status: {status!r}")
    if round_index == 0 and status != "initial":
        raise ValueError("recent_status must be initial on round 0")
    if round_index > 0 and status == "initial":
        raise ValueError("recent_status may only be initial on round 0")
    outcome = response["recent_outcome"]
    if not isinstance(outcome, str) or not outcome.strip() or len(outcome) > 320:
        raise ValueError("recent_outcome must contain 1-320 characters")
    skill_key = response["skill_key"]
    if skill_key not in allowed_skills:
        raise ValueError(
            f"skill_key {skill_key!r} is outside current candidates "
            f"{sorted(allowed_skills)}"
        )
    canonical = allowed_skills[skill_key]["canonical_template"]
    canonicalized = response["current_subtask_template"] != canonical
    response = dict(response)
    response["current_subtask_template"] = canonical
    response, _ = normalize_response(response)
    points = response["points"]
    if not isinstance(points, list) or len(points) > 2:
        raise ValueError("points must be a list containing at most two entries")
    point_ids: set[str] = set()
    fields = {"id", "object_description", "source_frame_id", "x_norm", "y_norm"}
    for index, point in enumerate(points):
        if not isinstance(point, dict) or set(point) != fields:
            raise ValueError(f"points[{index}] has invalid fields")
        point_id = point["id"]
        if point_id not in {"point_1", "point_2"} or point_id in point_ids:
            raise ValueError(f"points[{index}].id is invalid or duplicated")
        point_ids.add(point_id)
        if point["source_frame_id"] not in allowed_front_ids:
            raise ValueError(
                f"points[{index}].source_frame_id is not a current front frame"
            )
        if not isinstance(point["object_description"], str) or not point[
            "object_description"
        ].strip():
            raise ValueError(f"points[{index}].object_description must be non-empty")
        for coordinate in ("x_norm", "y_norm"):
            value = point[coordinate]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 999:
                raise ValueError(f"points[{index}].{coordinate} is invalid")
    placeholders = set(re.findall(r"<(point_[0-9]+)>", canonical))
    if placeholders != point_ids:
        raise ValueError(
            "canonical template placeholders must match points exactly: "
            f"template={sorted(placeholders)}, points={sorted(point_ids)}"
        )
    return response, canonicalized


class CodexWorkflowPlanner:
    """Use one exact Codex session per RoboMME episode."""

    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        system_prompt_path: Path = DEFAULT_AGENTS,
        user_prompt_path: Path = DEFAULT_USER_PROMPT,
        output_schema_path: Path = DEFAULT_OUTPUT_SCHEMA,
        timeout_seconds: float = 900.0,
        keep_recent_inputs: bool = False,
        verify_login: bool = True,
        retrieval_scheme: str = "R0",
        retrieval_top_k: int = 5,
    ) -> None:
        if retrieval_scheme != "R0":
            raise ValueError("codex_workflow_v1 intentionally supports only R0")
        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be positive")
        self.codex_bin = codex_bin
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.system_prompt_path = system_prompt_path.resolve()
        self.user_prompt_path = user_prompt_path.resolve()
        self.output_schema_path = output_schema_path.resolve()
        self.user_prompt_template = self.user_prompt_path.read_text(encoding="utf-8")
        self.timeout_seconds = timeout_seconds
        self.keep_recent_inputs = keep_recent_inputs
        self.environment = sanitized_codex_environment()
        self.codex_version = get_codex_version(codex_bin, self.environment)
        self.provider_version = self.codex_version
        self.inference_provider = "codex"
        self.login_status = (
            require_chatgpt_login(codex_bin, self.environment)
            if verify_login
            else "not checked"
        )
        self.prompt_profile = WORKFLOW_PROFILE
        self.harness_version = HARNESS_VERSION
        self.retrieval_scheme = retrieval_scheme
        self.retrieval_top_k = retrieval_top_k
        self.rethinking_threshold = None
        self.exploration_patience = None
        self.candidate_stagnant_limit = None
        self.max_exploration_candidates = None
        self.thinking_budget_tokens = None
        self.thinking_finalizer_max_output_tokens = None
        self.skill_retriever = ExecutorSkillRetriever(DEFAULT_VOCABULARY)
        self.lock_catalog = load_action_success_criteria(
            ACTION_GATES_V8_4_6_CRITERIA_PATH
        )
        self.run_dir: Path | None = None
        self.session_workspace: Path | None = None
        self.codex_session_id: str | None = None
        self.detailed_task_description = ""
        self.planner_guidance: dict[str, Any] = {}
        self.task = ""
        self.episode = -1
        self.round_index = 0
        self.previous_status = "initial"
        self.previous_subtask = "None. These are the initial rollout observations."
        self.previous_subtask_template: str | None = None
        self.previous_skill_key: str | None = None
        self.previous_points: list[dict[str, Any]] = []
        self.control_feedback = "No executor command has run yet."
        self.action_instance_sequence = 0
        self.active_action_signature: str | None = None
        self.active_action_instance_id: str | None = None
        self.lock_states: dict[str, dict[str, Any]] = {}

    def start_episode(
        self,
        *,
        run_dir: Path,
        detailed_task_description: str,
        task: str,
        episode: int,
        init_case_id: str,
        planner_guidance: dict[str, Any] | None = None,
    ) -> None:
        description = detailed_task_description.strip()
        if not description:
            raise ValueError("detailed_task_description must be non-empty")
        run_dir = run_dir.resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty planner run directory: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "rounds").mkdir()
        workspace = run_dir / "session_workspace"
        workspace.mkdir()
        shutil.copy2(self.system_prompt_path, workspace / "AGENTS.md")
        self.run_dir = run_dir
        self.session_workspace = workspace
        self.detailed_task_description = description
        self.planner_guidance = planner_guidance or {}
        self.task = task
        self.episode = episode
        self.round_index = 0
        self.codex_session_id = None
        self.previous_status = "initial"
        self.previous_subtask = "None. These are the initial rollout observations."
        self.previous_subtask_template = None
        self.previous_skill_key = None
        self.previous_points = []
        self.control_feedback = "No executor command has run yet."
        self.action_instance_sequence = 0
        self.active_action_signature = None
        self.active_action_instance_id = None
        self.lock_states = {}
        write_json(
            run_dir / "episode_config.json",
            {
                "harness_version": self.harness_version,
                "prompt_profile": self.prompt_profile,
                "memory_mode": "codex_persistent_session",
                "task": task,
                "episode": episode,
                "init_case_id": init_case_id,
                "detailed_task_description": description,
                "planner_guidance": self.planner_guidance,
                "retrieval_scheme": self.retrieval_scheme,
                "retrieval_top_k": self.retrieval_top_k,
                "rethinking_enabled": False,
                "episode_lessons_enabled": False,
                "completion_refine_gate_enabled": False,
                "reinforcement_lock": REINFORCEMENT_LOCK,
            },
        )

    def _require_started(self) -> tuple[Path, Path]:
        if self.run_dir is None or self.session_workspace is None:
            raise RuntimeError("start_episode must be called before plan")
        return self.run_dir, self.session_workspace

    def _run_codex(
        self, *, prompt: str, image_paths: list[Path], destination: Path
    ) -> dict[str, Any]:
        if not image_paths:
            raise ValueError("workflow planning requires attached images")
        _, workspace = self._require_started()
        destination.mkdir(parents=True, exist_ok=True)
        events_path = destination / "codex_events.jsonl"
        stderr_path = destination / "codex_stderr.log"
        response_path = destination / "response.txt"
        common = [
            "--ignore-user-config",
            "--ignore-rules",
            "--model",
            self.model,
            "--config",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "--output-schema",
            str(self.output_schema_path),
            "--json",
            "--output-last-message",
            str(response_path),
        ]
        image_args: list[str] = []
        for path in image_paths:
            image_args.extend(("--image", str(path.resolve())))
        started_at = utc_now()
        started = time.monotonic()
        capacity_retries = 0
        while True:
            # A failed initial attempt may still create a session before reporting
            # provider capacity. Rebuild the command for each retry so the next
            # attempt resumes that exact session instead of creating another.
            if self.codex_session_id is None:
                command = [
                    self.codex_bin,
                    "exec",
                    *common,
                    "--cd",
                    str(workspace),
                    "--skip-git-repo-check",
                    "--sandbox",
                    "read-only",
                    "--color",
                    "never",
                    *image_args,
                    "--",
                    "-",
                ]
            else:
                command = [
                    self.codex_bin,
                    "exec",
                    "resume",
                    *common,
                    *image_args,
                    self.codex_session_id,
                    "-",
                ]
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
                        env=self.environment,
                        timeout=self.timeout_seconds,
                    )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"Codex timed out after {self.timeout_seconds:.1f}s"
                ) from exc
            thread_id, usage = parse_codex_events(events_path)
            if thread_id is not None:
                if self.codex_session_id is not None and thread_id != self.codex_session_id:
                    raise RuntimeError(
                        "Codex resumed the wrong session: "
                        f"expected={self.codex_session_id}, actual={thread_id}"
                    )
                self.codex_session_id = thread_id
            if result.returncode == 0:
                break
            failure_text = "\n".join(
                (
                    events_path.read_text(encoding="utf-8", errors="replace"),
                    stderr_path.read_text(encoding="utf-8", errors="replace"),
                )
            )
            if (
                "selected model is at capacity" not in failure_text.lower()
                or capacity_retries >= CAPACITY_RETRY_LIMIT
            ):
                raise RuntimeError(
                    f"Codex exited with status {result.returncode}; see {destination}"
                )
            capacity_retries += 1
            shutil.copy2(
                events_path,
                destination / f"codex_events.capacity_attempt_{capacity_retries:02d}.jsonl",
            )
            shutil.copy2(
                stderr_path,
                destination / f"codex_stderr.capacity_attempt_{capacity_retries:02d}.log",
            )
            time.sleep(CAPACITY_RETRY_DELAY_SECONDS)
        if self.codex_session_id is None:
            raise RuntimeError("Codex completed without a thread.started session id")
        if not response_path.is_file():
            raise RuntimeError("Codex completed without response.txt")
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            # Preserve the marker used by the rollout driver's bounded JSON
            # fallback policy.
            raise ValueError(
                f"JSONDecodeError: Codex response is not valid JSON: {response_path}"
            ) from exc
        return {
            "response": response,
            "thread_id": self.codex_session_id,
            "usage": usage,
            "started_at": started_at,
            "wall_seconds": time.monotonic() - started,
            "exit_code": result.returncode,
            "capacity_retry_count": capacity_retries,
        }

    def _lock_for_previous_action(self) -> dict[str, Any] | None:
        criterion = match_action_success_criterion(
            self.previous_subtask, self.lock_catalog, task=self.task
        )
        criterion_id = str(criterion["criterion_id"]) if criterion else None
        if criterion_id not in REINFORCEMENT_LOCK:
            self.active_action_signature = None
            self.active_action_instance_id = None
            return None
        signature = action_instance_signature(
            criterion_id,
            self.previous_skill_key,
            self.previous_subtask_template,
            self.previous_subtask,
        )
        if signature != self.active_action_signature:
            self.action_instance_sequence += 1
            self.active_action_signature = signature
            self.active_action_instance_id = (
                f"workflow_action_{self.action_instance_sequence:03d}::{signature}"
            )
            self.lock_states[self.active_action_instance_id] = {
                "action_instance_id": self.active_action_instance_id,
                "criterion_id": criterion_id,
                "status": "available",
                "extra_command_budget": REINFORCEMENT_LOCK[criterion_id],
                "extra_commands_consumed": 0,
            }
        return self.lock_states[self.active_action_instance_id]

    @staticmethod
    def _archive_failed_round(run_dir: Path, round_dir: Path, round_index: int) -> Path:
        failed_root = run_dir / "failed_inference_rounds"
        failed_root.mkdir(parents=True, exist_ok=True)
        suffix = 1
        failed_round = failed_root / f"round_{round_index:04d}_attempt_{suffix:03d}"
        while failed_round.exists():
            suffix += 1
            failed_round = failed_root / (
                f"round_{round_index:04d}_attempt_{suffix:03d}"
            )
        shutil.move(str(round_dir), str(failed_round))
        return failed_round

    def plan(
        self,
        *,
        front_frames: list[np.ndarray],
        wrist_frames: list[np.ndarray],
        frame_ids: list[int],
    ) -> tuple[str, dict[str, Any]]:
        run_dir, _ = self._require_started()
        if (
            not front_frames
            or len(front_frames) != len(wrist_frames)
            or len(front_frames) != len(frame_ids)
        ):
            raise ValueError("front, wrist, and frame_ids must have equal non-zero length")
        if frame_ids != sorted(frame_ids) or len(set(frame_ids)) != len(frame_ids):
            raise ValueError("frame_ids must be unique and chronological")
        round_index = self.round_index
        round_dir = run_dir / "rounds" / f"round_{round_index:04d}"
        inputs_dir = round_dir / "recent_inputs"
        inputs_dir.mkdir(parents=True)
        image_paths: list[Path] = []
        manifest: list[str] = []
        attachments: list[dict[str, Any]] = []
        front_shapes: dict[str, tuple[int, int]] = {}
        for rollout_frame, front, wrist in zip(
            frame_ids, front_frames, wrist_frames, strict=True
        ):
            for view, short, value in (("front", "f", front), ("wrist", "w", wrist)):
                array = np.asarray(value, dtype=np.uint8)
                if array.ndim != 3 or array.shape[2] != 3:
                    raise ValueError(f"{view} frame {rollout_frame} is not RGB")
                frame_id = f"r_{short}{rollout_frame:04d}"
                path = inputs_dir / f"{frame_id}.png"
                Image.fromarray(array).save(path)
                image_paths.append(path)
                manifest.append(
                    f"{frame_id}: recent {view} observation at rollout frame {rollout_frame}"
                )
                attachments.append(
                    {
                        "frame_id": frame_id,
                        "view": view,
                        "rollout_frame": rollout_frame,
                        "file": str(path.relative_to(run_dir)),
                    }
                )
                if view == "front":
                    front_shapes[frame_id] = (int(array.shape[0]), int(array.shape[1]))

        retrieval_query = (
            f"Fixed task description: {self.detailed_task_description}\n"
            f"Previous executor skill: {self.previous_subtask}\n"
            f"Previous status: {self.previous_status}"
        )
        candidates = self.skill_retriever.retrieve(
            query=retrieval_query,
            task=self.task,
            previous_template_id=self.previous_skill_key,
            recent_status=self.previous_status,
            top_k=self.retrieval_top_k,
            scheme=self.retrieval_scheme,
        )
        allowed = {candidate["template_id"]: candidate for candidate in candidates}
        prompt = self.user_prompt_template.format(
            round_index=round_index,
            detailed_task_description=self.detailed_task_description,
            planner_guidance=json.dumps(self.planner_guidance, ensure_ascii=False, indent=2),
            previous_subtask=self.previous_subtask,
            control_feedback=self.control_feedback,
            allowed_executor_skills=json.dumps(
                {
                    "retrieval_scheme": self.retrieval_scheme,
                    "skills": [
                        {
                            "rank": candidate["rank"],
                            "template_id": candidate["template_id"],
                            "canonical_template": candidate["canonical_template"],
                            "point_arity": candidate["point_arity"],
                            "simple_labels": candidate["simple_labels"],
                            "training_frame_count": candidate["training_frame_count"],
                        }
                        for candidate in candidates
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            image_manifest="\n".join(manifest),
        )
        write_text(round_dir / "prompt.txt", prompt)
        write_json(
            round_dir / "input_manifest.json",
            {
                "round_index": round_index,
                "codex_session_id_before": self.codex_session_id,
                "previous_subtask": self.previous_subtask,
                "previous_skill_key": self.previous_skill_key,
                "control_feedback": self.control_feedback,
                "recent_rollout_frame_ids": frame_ids,
                "historical_image_count_in_explicit_prompt": 0,
                "attachments": attachments,
                "retrieval_candidates": [
                    {
                        "rank": candidate["rank"],
                        "template_id": candidate["template_id"],
                        "canonical_template": candidate["canonical_template"],
                    }
                    for candidate in candidates
                ],
            },
        )

        inference_results: list[dict[str, Any]] = []
        validation_errors: list[str] = []
        response: dict[str, Any] | None = None
        canonicalized = False
        retry_prompt = prompt
        for attempt in range(OUTPUT_VALIDATION_RETRY_LIMIT + 1):
            destination = round_dir if attempt == 0 else round_dir / f"validation_retry_{attempt:02d}"
            try:
                result = self._run_codex(
                    prompt=retry_prompt,
                    image_paths=image_paths,
                    destination=destination,
                )
            except Exception as exc:
                # This round was not accepted, so planner state and round_index
                # remain unchanged. Archive all inputs/provider artifacts so a
                # later call can retry the same logical round without a collision.
                write_json(
                    round_dir / "inference_error.json",
                    {
                        "round_index": round_index,
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                self._archive_failed_round(run_dir, round_dir, round_index)
                raise
            inference_results.append(result)
            try:
                response, canonicalized = _validate_response(
                    result["response"],
                    allowed_skills=allowed,
                    allowed_front_ids=set(front_shapes),
                    round_index=round_index,
                )
                break
            except (KeyError, ValueError) as exc:
                validation_errors.append(f"{type(exc).__name__}: {exc}")
                write_json(destination / "invalid_response.json", result["response"])
                if attempt >= OUTPUT_VALIDATION_RETRY_LIMIT:
                    write_json(
                        round_dir / "inference_error.json",
                        {
                            "round_index": round_index,
                            "attempt": attempt,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    self._archive_failed_round(run_dir, round_dir, round_index)
                    raise
                retry_prompt = (
                    "Your previous response for this same planner round was rejected "
                    "without advancing the environment. Correct the complete JSON "
                    f"using the same attached observations. Harness error: {exc}"
                )
        assert response is not None
        proposed_template = inference_results[-1]["response"].get(
            "current_subtask_template"
        )
        proposed_subtask, rendered_points = render_grounded_subtask(response, front_shapes)
        proposed_skill_key = response["skill_key"]
        effective_subtask = proposed_subtask
        effective_template = response["current_subtask_template"]
        effective_skill_key = proposed_skill_key
        effective_points = rendered_points
        lock_state = self._lock_for_previous_action()
        lock_event = {
            "applicable": lock_state is not None,
            "action_instance_id": self.active_action_instance_id,
            "state_before": dict(lock_state) if lock_state is not None else None,
            "triggered": False,
            "active_before": bool(
                lock_state and lock_state["status"] == "pending_extra_execution"
            ),
            "released": False,
            "overrode_proposal": False,
            "state_after": None,
        }
        if lock_state is not None:
            if lock_state["status"] == "pending_extra_execution":
                lock_state["extra_commands_consumed"] += 1
                lock_state["status"] = "exhausted"
                lock_event["released"] = True
            elif (
                lock_state["status"] == "available"
                and response["recent_status"] == "completed"
                and proposed_skill_key != self.previous_skill_key
                and proposed_subtask != self.previous_subtask
            ):
                lock_state["status"] = "pending_extra_execution"
                lock_state["blocked_proposal"] = {
                    "skill_key": proposed_skill_key,
                    "subtask": proposed_subtask,
                    "template": effective_template,
                }
                effective_subtask = self.previous_subtask
                effective_template = self.previous_subtask_template or effective_template
                effective_skill_key = self.previous_skill_key or effective_skill_key
                effective_points = list(self.previous_points)
                lock_event["triggered"] = True
                lock_event["overrode_proposal"] = True
        lock_event["state_after"] = dict(lock_state) if lock_state is not None else None

        if lock_event["overrode_proposal"]:
            next_control_feedback = (
                "The reinforcement lock overrode your proposed transition. The skill "
                f"actually executed was: {effective_subtask}. Your proposed successor "
                "was not executed. Reassess from the newest observations."
            )
        else:
            next_control_feedback = (
                f"The skill actually executed was: {effective_subtask}. No proposal "
                "was replaced by the control harness."
            )
        self.previous_status = response["recent_status"]
        self.previous_subtask = effective_subtask
        self.previous_subtask_template = effective_template
        self.previous_skill_key = effective_skill_key
        self.previous_points = effective_points
        self.control_feedback = next_control_feedback
        self.round_index += 1

        usage: dict[str, int] | None = None
        for item in inference_results:
            if item.get("usage") is None:
                continue
            if usage is None:
                usage = {}
            for key, value in item["usage"].items():
                usage[key] = usage.get(key, 0) + int(value)
        record = {
            "round_index": round_index,
            "planner_memory_mode": "codex_persistent_session",
            "prompt_profile": self.prompt_profile,
            "codex_session_id": self.codex_session_id,
            "recent_rollout_frame_ids": frame_ids,
            "historical_image_count": 0,
            "response": response,
            "planner_proposed_subtask_template": proposed_template,
            "planner_proposed_grounded_subtask": proposed_subtask,
            "template_canonicalized_by_harness": canonicalized,
            "grounded_subtask_for_vla": effective_subtask,
            "skill_key": effective_skill_key,
            "instruction_strategy": "workflow_session",
            "planner_mode": "codex_workflow",
            "rethinking_required_this_turn": False,
            "episode_local_lesson_count": 0,
            "retrieval_scheme": self.retrieval_scheme,
            "retrieval_candidates": [
                {
                    "rank": candidate["rank"],
                    "template_id": candidate["template_id"],
                    "canonical_template": candidate["canonical_template"],
                    "retrieval_score": candidate["retrieval_score"],
                }
                for candidate in candidates
            ],
            "reinforcement_lock": lock_event,
            "completion_gate": {
                "applicable": lock_state is not None,
                "semantic_switch_blocked": lock_event["overrode_proposal"],
                "completed_status_blocked": False,
            },
            "codex_usage": usage,
            "codex_wall_seconds": sum(
                float(item["wall_seconds"]) for item in inference_results
            ),
            "codex_started_at": inference_results[0]["started_at"],
            "codex_capacity_retry_count": sum(
                int(item.get("capacity_retry_count", 0)) for item in inference_results
            ),
            "validation_retry_count": len(validation_errors),
            "validation_errors": validation_errors,
            "added_keyframes": [],
        }
        write_json(round_dir / "record.json", record)
        write_json(
            run_dir / "session_state.json",
            {
                "codex_session_id": self.codex_session_id,
                "rounds_completed": self.round_index,
                "previous_skill_key": self.previous_skill_key,
                "previous_subtask": self.previous_subtask,
                "control_feedback": self.control_feedback,
                "active_action_instance_id": self.active_action_instance_id,
                "reinforcement_lock_states": self.lock_states,
            },
        )
        if not self.keep_recent_inputs:
            shutil.rmtree(inputs_dir)
        return effective_subtask, record
