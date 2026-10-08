#!/usr/bin/env python3
"""Reproducible Codex harness for RoboMME initial-demonstration videos.

The harness intentionally uses the locally installed ``codex`` CLI and its
existing ChatGPT login. It never reads, copies, or serializes authentication
material.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any


# This identifies the frame/inference harness implementation. Prompt content is
# versioned independently through PROMPT_PROFILES so v2 frame caches remain reusable.
HARNESS_VERSION = "codex-init-video-v2"
SAMPLING_VERSION = "timestamp-threshold-v1"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_TARGET_FPS = 10.0
DEFAULT_JPEG_QUALITY = 95
AUTH_ENV_VARS_TO_REMOVE = ("CODEX_API_KEY", "OPENAI_API_KEY")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT = SCRIPT_DIR / "prompts" / "system_appendix_v2.txt"
DEFAULT_USER_PROMPT = SCRIPT_DIR / "prompts" / "user_appendix_v2.txt"
DEFAULT_OUTPUT_SCHEMA = SCRIPT_DIR / "schemas" / "response_v2.schema.json"
PLANNER_GUIDANCE_SYSTEM_PROMPT = SCRIPT_DIR / "prompts" / "system_planner_guidance_v3.txt"
PLANNER_GUIDANCE_OUTPUT_SCHEMA = (
    SCRIPT_DIR / "schemas" / "response_planner_guidance_v3.schema.json"
)
DEFAULT_TASK_SPECIFICATIONS = (
    SCRIPT_DIR / "task_specifications" / "robomme_appendix_e_v4.json"
)
DEFAULT_PROMPT_PROFILE = "appendix_v2"
PROMPT_PROFILES = {
    "appendix_v2": {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_prompt": DEFAULT_USER_PROMPT,
        "output_schema": DEFAULT_OUTPUT_SCHEMA,
    },
    "planner_guidance_v3": {
        "system_prompt": PLANNER_GUIDANCE_SYSTEM_PROMPT,
        "user_prompt": DEFAULT_USER_PROMPT,
        "output_schema": PLANNER_GUIDANCE_OUTPUT_SCHEMA,
    },
}
PLANNER_GUIDANCE_FIELDS = (
    "fixed_facts",
    "state_to_track",
    "progress_rules",
    "execution_constraints",
)
EXPECTED_ROBOMME_TASKS = {
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "ButtonUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
}


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    task: str
    episode: int
    task_description: str
    video_path: Path | None


@dataclass(frozen=True)
class PreparedFrames:
    cache_key: str
    cache_dir: Path
    source_sha256: str
    source_fps: float
    source_frame_count: int
    target_fps: float
    frames: list[dict[str, Any]]

    @property
    def image_paths(self) -> list[Path]:
        return [self.cache_dir / frame["file"] for frame in self.frames]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_task_specifications(path: Path) -> dict[str, str]:
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), dict):
        raise ValueError(f"Task specification file must contain a tasks object: {path}")
    specifications = payload["tasks"]
    actual = set(specifications)
    missing = sorted(EXPECTED_ROBOMME_TASKS - actual)
    extra = sorted(actual - EXPECTED_ROBOMME_TASKS)
    if missing or extra:
        raise ValueError(
            f"Task specification coverage mismatch in {path}; missing={missing}, extra={extra}"
        )
    for task, specification in specifications.items():
        if not isinstance(specification, str) or not specification.strip():
            raise ValueError(f"Empty task specification for {task}: {path}")
    return {task: specification.strip() for task, specification in specifications.items()}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def portable_relative_path(path: Path, anchor: Path) -> str:
    return os.path.relpath(path.resolve(), start=anchor.resolve())


def resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_cases(manifest_path: Path) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    seen: set[str] = set()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {manifest_path}:{line_number}: {exc}") from exc
        required = {"case_id", "task", "episode", "task_description", "video_path"}
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"Missing fields on {manifest_path}:{line_number}: {missing}")
        case_id = str(row["case_id"])
        if case_id in seen:
            raise ValueError(f"Duplicate case_id in {manifest_path}: {case_id}")
        seen.add(case_id)
        video_path = (
            resolve_manifest_path(str(row["video_path"]), manifest_path)
            if row["video_path"] is not None
            else None
        )
        case = CaseSpec(
            case_id=case_id,
            task=str(row["task"]),
            episode=int(row["episode"]),
            task_description=str(row["task_description"]).strip(),
            video_path=video_path,
        )
        if not case.task_description:
            raise ValueError(f"Empty task_description for {case_id}")
        if case.video_path is not None and not case.video_path.is_file():
            raise FileNotFoundError(f"Video for {case_id} does not exist: {case.video_path}")
        cases.append(case)
    if not cases:
        raise ValueError(f"No cases found in {manifest_path}")
    return cases


def build_manifest(memer_roots: list[Path], output_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in memer_roots:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"MemER run directory does not exist: {root}")
        for summary_path in sorted(root.glob("*/episode_*/summary.json")):
            episode_dir = summary_path.parent
            video_path = episode_dir / "memer" / "initial_demo.mp4"
            summary = read_json(summary_path)
            task = str(summary.get("task") or episode_dir.parent.name)
            episode = int(summary.get("episode", episode_dir.name.removeprefix("episode_")))
            case_id = f"{task}_episode_{episode:02d}"
            if case_id in seen:
                raise ValueError(
                    f"Duplicate {case_id} across --memer-run-dir inputs; provide non-overlapping roots"
                )
            task_description = str(summary.get("task_goal", "")).strip()
            if not task_description:
                raise ValueError(f"Missing task_goal in {summary_path}")
            seen.add(case_id)
            rows.append(
                {
                    "case_id": case_id,
                    "task": task,
                    "episode": episode,
                    "task_description": task_description,
                    "video_path": (
                        portable_relative_path(video_path, output_path.parent)
                        if video_path.is_file()
                        else None
                    ),
                }
            )
    rows.sort(key=lambda row: (row["task"], row["episode"]))
    if not rows:
        raise ValueError("No completed MemER episode summary was found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    write_text(output_path, content)
    return rows


@contextlib.contextmanager
def cache_lock(lock_path: Path, timeout_seconds: float = 600.0):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    file_descriptor: int | None = None
    while file_descriptor is None:
        try:
            file_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(file_descriptor, f"pid={os.getpid()} created={utc_now()}\n".encode())
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for cache lock: {lock_path}")
            time.sleep(0.25)
    try:
        yield
    finally:
        os.close(file_descriptor)
        lock_path.unlink(missing_ok=True)


def frame_cache_key(source_sha256: str, target_fps: float, jpeg_quality: int) -> str:
    return canonical_hash(
        {
            "kind": "frames",
            "harness_version": HARNESS_VERSION,
            "sampling_version": SAMPLING_VERSION,
            "source_sha256": source_sha256,
            "target_fps": target_fps,
            "jpeg_quality": jpeg_quality,
        }
    )


def load_prepared_frames(cache_dir: Path) -> PreparedFrames:
    manifest = read_json(cache_dir / "manifest.json")
    return PreparedFrames(
        cache_key=str(manifest["cache_key"]),
        cache_dir=cache_dir,
        source_sha256=str(manifest["source_sha256"]),
        source_fps=float(manifest["source_fps"]),
        source_frame_count=int(manifest["source_frame_count"]),
        target_fps=float(manifest["target_fps"]),
        frames=list(manifest["frames"]),
    )


def prepare_frames(
    video_path: Path,
    frame_cache_root: Path,
    target_fps: float,
    jpeg_quality: int,
) -> tuple[PreparedFrames, bool]:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Video decoding requires the project environment with imageio installed"
        ) from exc
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    source_sha256 = sha256_file(video_path)
    key = frame_cache_key(source_sha256, target_fps, jpeg_quality)
    cache_dir = frame_cache_root / key
    complete_path = cache_dir / ".complete"
    if complete_path.is_file():
        return load_prepared_frames(cache_dir), True

    lock_path = frame_cache_root / ".locks" / f"{key}.lock"
    with cache_lock(lock_path):
        if complete_path.is_file():
            return load_prepared_frames(cache_dir), True
        frame_cache_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}.tmp-", dir=frame_cache_root))
        try:
            frames_dir = temporary / "frames"
            frames_dir.mkdir()
            reader = imageio.get_reader(video_path)
            metadata = reader.get_meta_data()
            source_fps = float(metadata["fps"])
            if source_fps <= 0:
                raise ValueError(f"Invalid source FPS {source_fps}: {video_path}")
            if target_fps > source_fps + 1e-6:
                raise ValueError(
                    f"target_fps ({target_fps}) exceeds source FPS ({source_fps}): {video_path}"
                )

            selected: list[dict[str, Any]] = []
            source_frame_count = 0
            next_sample_time = 0.0
            try:
                for frame_index, frame in enumerate(reader):
                    source_frame_count += 1
                    timestamp = frame_index / source_fps
                    if timestamp + 1e-9 < next_sample_time:
                        continue
                    frame_id = f"i_f{frame_index:04d}"
                    relative_file = f"frames/{frame_id}.jpg"
                    imageio.imwrite(temporary / relative_file, frame, quality=jpeg_quality)
                    selected.append(
                        {
                            "frame_id": frame_id,
                            "source_frame_index": frame_index,
                            "timestamp_seconds": round(timestamp, 6),
                            "file": relative_file,
                        }
                    )
                    next_sample_time += 1.0 / target_fps
            finally:
                reader.close()
            if not selected:
                raise ValueError(f"No frames decoded from {video_path}")

            manifest = {
                "cache_key": key,
                "harness_version": HARNESS_VERSION,
                "sampling_version": SAMPLING_VERSION,
                "source_sha256": source_sha256,
                "source_fps": source_fps,
                "source_frame_count": source_frame_count,
                "target_fps": target_fps,
                "jpeg_quality": jpeg_quality,
                "frames": selected,
            }
            write_json(temporary / "manifest.json", manifest)
            write_text(temporary / ".complete", utc_now() + "\n")
            try:
                os.replace(temporary, cache_dir)
            except OSError:
                if not complete_path.is_file():
                    raise
                shutil.rmtree(temporary)
            return load_prepared_frames(cache_dir), False
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def render_prompt(
    system_prompt: str,
    user_prompt_template: str,
    task_name: str,
    task_specification: str,
    task_description: str,
    frames: list[dict[str, Any]],
) -> str:
    if frames:
        video_context = (
            "The attached frames are the complete initial demonstration video "
            "downsampled from 30 FPS to 10 FPS. They are provided in chronological "
            "order and labeled with their original frame identifiers."
        )
    else:
        video_context = (
            "No initial demonstration video is provided for this task. Use the episode "
            "instruction and official task specification without inventing visual evidence."
        )
    try:
        user_prompt = user_prompt_template.format(
            task_name=task_name,
            task_specification=task_specification,
            task_description=task_description,
            video_context=video_context,
        )
    except KeyError as exc:
        raise ValueError(f"Unsupported placeholder in user prompt: {exc}") from exc
    prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}\n"
    if frames:
        frame_manifest = "\n".join(
            f'{frame["frame_id"]}: {float(frame["timestamp_seconds"]):.2f} s'
            for frame in frames
        )
        prompt += f"\nFrame manifest in attachment order:\n{frame_manifest}\n"
    return prompt


def get_codex_version(codex_bin: str, environment: dict[str, str]) -> str:
    result = subprocess.run(
        [codex_bin, "--version"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to run {codex_bin} --version: {result.stderr.strip()}")
    return result.stdout.strip()


def require_chatgpt_login(codex_bin: str, environment: dict[str, str]) -> str:
    result = subprocess.run(
        [codex_bin, "login", "status"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    status = "\n".join(value for value in (result.stdout.strip(), result.stderr.strip()) if value)
    if result.returncode != 0 or "chatgpt" not in status.lower():
        raise RuntimeError(
            "Codex is not logged in with ChatGPT. Run `codex login` interactively first. "
            f"Reported status: {status or '<empty>'}"
        )
    return status


def sanitized_codex_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in AUTH_ENV_VARS_TO_REMOVE:
        environment.pop(variable, None)
    return environment


def inference_cache_key(
    *,
    case: CaseSpec,
    prepared: PreparedFrames,
    prompt: str,
    system_prompt: str,
    user_prompt_template: str,
    output_schema_text: str,
    model: str,
    reasoning_effort: str,
    codex_version: str,
) -> str:
    return canonical_hash(
        {
            "kind": "inference",
            "harness_version": HARNESS_VERSION,
            "case": {
                "case_id": case.case_id,
                "task": case.task,
                "episode": case.episode,
                "task_description": case.task_description,
            },
            "frame_cache_key": prepared.cache_key,
            "prompt_sha256": sha256_text(prompt),
            "system_prompt_sha256": sha256_text(system_prompt),
            "user_prompt_template_sha256": sha256_text(user_prompt_template),
            "output_schema_sha256": sha256_text(output_schema_text),
            "model": model,
            "reasoning_effort": reasoning_effort,
            "codex_version": codex_version,
            "cli_mode": {
                "ephemeral": True,
                "ignore_user_config": True,
                "ignore_rules": True,
                "sandbox": "read-only",
            },
        }
    )


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


def validate_response(response: Any, allowed_frame_ids: set[str] | None = None) -> None:
    if not isinstance(response, dict):
        raise ValueError("Response must be a JSON object")
    expected_v2 = {"video_description", "detailed_task_description"}
    expected_v1 = expected_v2 | {"keyframes"}
    expected_v3 = expected_v2 | {"planner_guidance"}
    if set(response) not in (expected_v2, expected_v1, expected_v3):
        raise ValueError(
            "Response fields must match v2, planner-guidance v3, or legacy v1; "
            f"got {sorted(response)}"
        )
    for key in ("video_description", "detailed_task_description"):
        if not isinstance(response[key], str) or not response[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if "keyframes" in response:
        if not isinstance(response["keyframes"], list):
            raise ValueError("keyframes must be a list")
        if allowed_frame_ids is None:
            raise ValueError("Legacy keyframe validation requires attached frame identifiers")
        for index, keyframe in enumerate(response["keyframes"]):
            if not isinstance(keyframe, dict) or set(keyframe) != {"frame_id", "reason"}:
                raise ValueError(f"keyframes[{index}] must contain exactly frame_id and reason")
            if keyframe["frame_id"] not in allowed_frame_ids:
                raise ValueError(
                    f"keyframes[{index}].frame_id is not an attached frame: {keyframe['frame_id']}"
                )
            if not isinstance(keyframe["reason"], str) or not keyframe["reason"].strip():
                raise ValueError(f"keyframes[{index}].reason must be a non-empty string")

    if "planner_guidance" in response:
        guidance = response["planner_guidance"]
        if not isinstance(guidance, dict):
            raise ValueError("planner_guidance must be an object")
        if set(guidance) != set(PLANNER_GUIDANCE_FIELDS):
            raise ValueError(
                "planner_guidance must contain exactly "
                f"{list(PLANNER_GUIDANCE_FIELDS)}; got {sorted(guidance)}"
            )
        for field_name in PLANNER_GUIDANCE_FIELDS:
            items = guidance[field_name]
            if not isinstance(items, list):
                raise ValueError(f"planner_guidance.{field_name} must be an array")
            if len(items) > 12:
                raise ValueError(f"planner_guidance.{field_name} may contain at most 12 items")
            for index, item in enumerate(items):
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(
                        f"planner_guidance.{field_name}[{index}] must be a non-empty string"
                    )
                if len(item) > 500:
                    raise ValueError(
                        f"planner_guidance.{field_name}[{index}] may contain at most 500 characters"
                    )


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
    ]
    if image_paths:
        command.extend(["--image", *(str(path.resolve()) for path in image_paths)])
    command.extend(["--", "-"])
    started_at = utc_now()
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="codex_init_video_") as isolated_dir:
        command[command.index("__ISOLATED_WORKDIR__")] = isolated_dir
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
                    check=False,
                    env=environment,
                    timeout=timeout_seconds,
                )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Codex timed out after {timeout_seconds:.1f}s") from exc
    elapsed = time.monotonic() - start
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
        "wall_seconds": elapsed,
        "exit_code": result.returncode,
    }


def publish_inference_cache(temporary: Path, cache_dir: Path) -> None:
    write_text(temporary / ".complete", utc_now() + "\n")
    try:
        os.replace(temporary, cache_dir)
    except OSError:
        if not (cache_dir / ".complete").is_file():
            raise
        shutil.rmtree(temporary)


def relative_symlink(target: Path, link_path: Path) -> None:
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Refusing to replace existing path: {link_path}")
    link_path.symlink_to(os.path.relpath(target.resolve(), start=link_path.parent.resolve()), target_is_directory=True)


def materialize_case(
    *,
    case: CaseSpec,
    prepared: PreparedFrames,
    inference_dir: Path,
    output_dir: Path,
    cache_hit: bool,
    frame_cache_hit: bool | None,
) -> dict[str, Any]:
    case_dir = output_dir / "cases" / case.case_id
    if case_dir.exists():
        previous_summary = case_dir / "case_summary.json"
        if previous_summary.is_file():
            previous = read_json(previous_summary)
            if previous.get("inference_cache_key") == inference_dir.name and previous.get("complete") is True:
                return previous
        raise FileExistsError(
            f"Refusing to overwrite incomplete or incompatible case directory: {case_dir}"
        )
    case_dir.mkdir(parents=True)
    for filename in (
        "response.json",
        "response.txt",
        "codex_events.jsonl",
        "codex_stderr.log",
        "prompt.txt",
        "input_manifest.json",
        "run_summary.json",
    ):
        shutil.copy2(inference_dir / filename, case_dir / filename)
    frames_link = None
    frame_cache_key = None
    if prepared.frames:
        relative_symlink(prepared.cache_dir / "frames", case_dir / "frames_10fps")
        frames_link = "frames_10fps"
        frame_cache_key = prepared.cache_key
    run_summary = read_json(inference_dir / "run_summary.json")
    summary = {
        "complete": True,
        "case_id": case.case_id,
        "task": case.task,
        "episode": case.episode,
        "inference_cache_key": inference_dir.name,
        "frame_cache_key": frame_cache_key,
        "inference_cache_hit": cache_hit,
        "frame_cache_hit": frame_cache_hit,
        "response_file": "response.json",
        "input_manifest_file": "input_manifest.json",
        "frames_link": frames_link,
        "usage": run_summary.get("usage"),
        "wall_seconds": run_summary.get("wall_seconds"),
        "materialized_at": utc_now(),
    }
    write_json(case_dir / "case_summary.json", summary)
    return summary


def process_case(
    *,
    case: CaseSpec,
    output_dir: Path,
    cache_root: Path,
    system_prompt: str,
    user_prompt_template: str,
    task_specifications: dict[str, str],
    output_schema: Path,
    output_schema_text: str,
    codex_bin: str,
    codex_version: str,
    model: str,
    reasoning_effort: str,
    target_fps: float,
    jpeg_quality: int,
    environment: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    if case.task not in task_specifications:
        raise ValueError(f"No official task specification is defined for {case.task}")
    task_specification = task_specifications[case.task]
    if case.video_path is not None:
        prepared, frame_cache_hit = prepare_frames(
            case.video_path,
            cache_root / "frames",
            target_fps,
            jpeg_quality,
        )
    else:
        prepared = PreparedFrames(
            cache_key="no-initial-video-v2",
            cache_dir=cache_root / "frames",
            source_sha256="",
            source_fps=0.0,
            source_frame_count=0,
            target_fps=target_fps,
            frames=[],
        )
        frame_cache_hit = None
    prompt = render_prompt(
        system_prompt,
        user_prompt_template,
        case.task,
        task_specification,
        case.task_description,
        prepared.frames,
    )
    key = inference_cache_key(
        case=case,
        prepared=prepared,
        prompt=prompt,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        output_schema_text=output_schema_text,
        model=model,
        reasoning_effort=reasoning_effort,
        codex_version=codex_version,
    )
    inference_root = cache_root / "inference"
    inference_dir = inference_root / key
    inference_cache_hit = (inference_dir / ".complete").is_file()

    if not inference_cache_hit:
        lock_path = inference_root / ".locks" / f"{key}.lock"
        with cache_lock(lock_path):
            inference_cache_hit = (inference_dir / ".complete").is_file()
            if not inference_cache_hit:
                inference_root.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix=f".{key}.tmp-", dir=inference_root))
                try:
                    write_text(temporary / "prompt.txt", prompt)
                    input_manifest = {
                        "harness_version": HARNESS_VERSION,
                        "case_id": case.case_id,
                        "task": case.task,
                        "episode": case.episode,
                        "task_description": case.task_description,
                        "official_task_specification": task_specification,
                        "official_task_specification_sha256": sha256_text(task_specification),
                        "source_video": str(case.video_path) if case.video_path is not None else None,
                        "source_video_sha256": prepared.source_sha256 or None,
                        "source_fps": prepared.source_fps or None,
                        "source_frame_count": prepared.source_frame_count,
                        "target_fps": prepared.target_fps if prepared.frames else None,
                        "attached_frame_count": len(prepared.frames),
                        "frame_cache_key": prepared.cache_key if prepared.frames else None,
                        "frames": [
                            {
                                **frame,
                                "case_file": f"frames_10fps/{Path(frame['file']).name}",
                            }
                            for frame in prepared.frames
                        ],
                        "model": model,
                        "reasoning_effort": reasoning_effort,
                        "codex_version": codex_version,
                        "prompt_sha256": sha256_text(prompt),
                        "output_schema_sha256": sha256_text(output_schema_text),
                    }
                    write_json(temporary / "input_manifest.json", input_manifest)
                    result = run_codex_inference(
                        codex_bin=codex_bin,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        prompt=prompt,
                        output_schema=output_schema,
                        image_paths=prepared.image_paths,
                        destination=temporary,
                        environment=environment,
                        timeout_seconds=timeout_seconds,
                    )
                    validate_response(
                        result["response"],
                        {str(frame["frame_id"]) for frame in prepared.frames},
                    )
                    write_json(temporary / "response.json", result["response"])
                    write_json(
                        temporary / "run_summary.json",
                        {
                            "complete": True,
                            "inference_cache_key": key,
                            "model": model,
                            "reasoning_effort": reasoning_effort,
                            "codex_version": codex_version,
                            "ephemeral": True,
                            "isolated_process": True,
                            "authentication": "local ChatGPT login",
                            "thread_id": result["thread_id"],
                            "started_at": result["started_at"],
                            "wall_seconds": result["wall_seconds"],
                            "exit_code": result["exit_code"],
                            "usage": result["usage"],
                        },
                    )
                    publish_inference_cache(temporary, inference_dir)
                except Exception:
                    failure_dir = output_dir / "failures" / case.case_id / uuid.uuid4().hex
                    failure_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(temporary), failure_dir)
                    raise

    return materialize_case(
        case=case,
        prepared=prepared,
        inference_dir=inference_dir,
        output_dir=output_dir,
        cache_hit=inference_cache_hit,
        frame_cache_hit=frame_cache_hit,
    )


def rebuild_batch_outputs(output_dir: Path) -> list[dict[str, Any]]:
    summaries = [read_json(path) for path in sorted((output_dir / "cases").glob("*/case_summary.json"))]
    results_path = output_dir / "results.jsonl"
    results_content = "".join(
        json.dumps(
            {
                **summary,
                "response": read_json(
                    output_dir / "cases" / summary["case_id"] / summary["response_file"]
                ),
            },
            ensure_ascii=False,
        )
        + "\n"
        for summary in summaries
    )
    write_text(results_path, results_content)
    usage_fields = [
        "case_id",
        "task",
        "episode",
        "inference_cache_hit",
        "frame_cache_hit",
        "wall_seconds",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    ]
    usage_path = output_dir / "usage.csv"
    temporary = usage_path.with_name(f".{usage_path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=usage_fields)
        writer.writeheader()
        for summary in summaries:
            usage = summary.get("usage") or {}
            writer.writerow(
                {
                    "case_id": summary["case_id"],
                    "task": summary["task"],
                    "episode": summary["episode"],
                    "inference_cache_hit": summary["inference_cache_hit"],
                    "frame_cache_hit": summary["frame_cache_hit"],
                    "wall_seconds": summary.get("wall_seconds"),
                    "input_tokens": usage.get("input_tokens"),
                    "cached_input_tokens": usage.get("cached_input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                }
            )
    os.replace(temporary, usage_path)
    write_json(
        output_dir / "batch_summary.json",
        {
            "complete_cases": len(summaries),
            "generated_at": utc_now(),
            "results_jsonl": "results.jsonl",
            "usage_csv": "usage.csv",
        },
    )
    return summaries


def select_cases(
    cases: list[CaseSpec],
    case_ids: list[str],
    tasks: list[str],
    start_index: int,
    stride: int,
    limit: int | None,
) -> list[CaseSpec]:
    if case_ids and tasks:
        raise ValueError("Use either --case-id or --task filters, not both")
    if case_ids:
        requested = set(case_ids)
        selected = [case for case in cases if case.case_id in requested]
        missing = sorted(requested.difference(case.case_id for case in selected))
        if missing:
            raise ValueError(f"Requested case_id values are absent from the manifest: {missing}")
    elif tasks:
        requested_tasks = set(tasks)
        selected = [case for case in cases if case.task in requested_tasks]
        missing_tasks = sorted(requested_tasks.difference(case.task for case in selected))
        if missing_tasks:
            raise ValueError(f"Requested tasks are absent from the manifest: {missing_tasks}")
    else:
        selected = cases
    selected = selected[start_index::stride]
    if limit is not None:
        selected = selected[:limit]
    return selected


def run_batch(args: argparse.Namespace) -> None:
    manifest_path = args.input_jsonl.resolve()
    cases = select_cases(
        load_cases(manifest_path),
        args.case_id,
        args.task,
        args.start_index,
        args.stride,
        args.limit,
    )
    if not cases:
        raise ValueError("Case selection is empty")
    print(f"Selected {len(cases)} case(s)")
    for case in cases:
        print(f"  {case.case_id}: {case.video_path}")
    if args.dry_run:
        print("Dry run: no frames were extracted and Codex was not called")
        return

    output_dir = args.output_dir.resolve()
    cache_root = args.cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    system_prompt = args.system_prompt.read_text(encoding="utf-8")
    user_prompt_template = args.user_prompt.read_text(encoding="utf-8")
    output_schema_text = args.output_schema.read_text(encoding="utf-8")
    task_specifications_text = args.task_specifications.read_text(encoding="utf-8")
    task_specifications = load_task_specifications(args.task_specifications)
    read_json(args.output_schema)

    environment = sanitized_codex_environment()
    login_status = require_chatgpt_login(args.codex_bin, environment)
    codex_version = get_codex_version(args.codex_bin, environment)
    print(f"Codex: {codex_version}; auth: {login_status}")
    write_json(
        output_dir / "run_config.json",
        {
            "harness_version": HARNESS_VERSION,
            "prompt_profile": args.prompt_profile,
            "manifest": portable_relative_path(manifest_path, output_dir),
            "cache_dir": portable_relative_path(cache_root, output_dir),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "target_fps": args.target_fps,
            "jpeg_quality": args.jpeg_quality,
            "codex_version": codex_version,
            "authentication": "local ChatGPT login",
            "system_prompt_sha256": sha256_text(system_prompt),
            "user_prompt_sha256": sha256_text(user_prompt_template),
            "output_schema_sha256": sha256_text(output_schema_text),
            "task_specifications": portable_relative_path(args.task_specifications, output_dir),
            "task_specifications_sha256": sha256_text(task_specifications_text),
            "selected_case_ids": [case.case_id for case in cases],
            "created_at": utc_now(),
        },
    )

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        summary = process_case(
            case=case,
            output_dir=output_dir,
            cache_root=cache_root,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            task_specifications=task_specifications,
            output_schema=args.output_schema,
            output_schema_text=output_schema_text,
            codex_bin=args.codex_bin,
            codex_version=codex_version,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            target_fps=args.target_fps,
            jpeg_quality=args.jpeg_quality,
            environment=environment,
            timeout_seconds=args.timeout_seconds,
        )
        print(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "inference_cache_hit": summary["inference_cache_hit"],
                    "usage": summary.get("usage"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        rebuild_batch_outputs(output_dir)
    print(f"Complete: {len(rebuild_batch_outputs(output_dir))} materialized case(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Build a portable JSONL input manifest from completed MemER eval directories",
    )
    manifest_parser.add_argument("--memer-run-dir", type=Path, action="append", required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)

    run_parser = subparsers.add_parser("run", help="Process manifest cases with local Codex")
    run_parser.add_argument("--input-jsonl", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--cache-dir", type=Path, required=True)
    run_parser.add_argument(
        "--prompt-profile",
        choices=sorted(PROMPT_PROFILES),
        default=DEFAULT_PROMPT_PROFILE,
        help="Versioned prompt/schema bundle; individual path flags may override it.",
    )
    run_parser.add_argument("--system-prompt", type=Path)
    run_parser.add_argument("--user-prompt", type=Path)
    run_parser.add_argument("--output-schema", type=Path)
    run_parser.add_argument(
        "--task-specifications",
        type=Path,
        default=DEFAULT_TASK_SPECIFICATIONS,
    )
    run_parser.add_argument("--model", default=DEFAULT_MODEL)
    run_parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default=DEFAULT_REASONING_EFFORT,
    )
    run_parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    run_parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    run_parser.add_argument("--timeout-seconds", type=float, default=900.0)
    run_parser.add_argument("--codex-bin", default="codex")
    run_parser.add_argument("--case-id", action="append", default=[])
    run_parser.add_argument("--task", action="append", default=[])
    run_parser.add_argument("--start-index", type=int, default=0)
    run_parser.add_argument("--stride", type=int, default=1)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.command == "run":
        profile = PROMPT_PROFILES[args.prompt_profile]
        for field_name in ("system_prompt", "user_prompt", "output_schema"):
            if getattr(args, field_name) is None:
                setattr(args, field_name, profile[field_name])
        if args.start_index < 0:
            parser.error("--start-index must be nonnegative")
        if args.stride < 1:
            parser.error("--stride must be positive")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if not 1 <= args.jpeg_quality <= 100:
            parser.error("--jpeg-quality must be in [1, 100]")
        for path_name in (
            "system_prompt",
            "user_prompt",
            "output_schema",
            "task_specifications",
        ):
            path = getattr(args, path_name)
            if not path.is_file():
                parser.error(f"--{path_name.replace('_', '-')} does not exist: {path}")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        rows = build_manifest(args.memer_run_dir, args.output.resolve())
        print(f"Wrote {len(rows)} cases to {args.output.resolve()}")
    else:
        run_batch(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
