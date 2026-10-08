#!/usr/bin/env python3
"""Loopback HTTP service exposing independently stateful Codex planner sessions."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import secrets
import shutil
import tarfile
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from planner import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_SCHEMA,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    CodexRolloutPlanner,
)
from text_progress_planner import (
    DEFAULT_PROMPT_PROFILE as DEFAULT_TEXT_PROGRESS_PROMPT_PROFILE,
    DEFAULT_OUTPUT_SCHEMA as DEFAULT_TEXT_PROGRESS_OUTPUT_SCHEMA,
    DEFAULT_SYSTEM_PROMPT as DEFAULT_TEXT_PROGRESS_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT as DEFAULT_TEXT_PROGRESS_USER_PROMPT,
    PROMPT_PROFILES as TEXT_PROGRESS_PROMPT_PROFILES,
    CodexTextProgressPlanner,
)
from workflow_planner import (
    DEFAULT_AGENTS as DEFAULT_WORKFLOW_AGENTS,
    DEFAULT_OUTPUT_SCHEMA as DEFAULT_WORKFLOW_OUTPUT_SCHEMA,
    DEFAULT_USER_PROMPT as DEFAULT_WORKFLOW_USER_PROMPT,
    CodexWorkflowPlanner,
)


MAX_REQUEST_BYTES = 256 * 1024 * 1024


def decode_png(value: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"))


class Session:
    def __init__(self, planner: Any) -> None:
        self.planner = planner
        self.lock = threading.Lock()


class PlannerService:
    def __init__(
        self,
        planner_class: type[Any],
        planner_kind: str,
        planner_kwargs: dict[str, Any],
        result_root_override: Path | None = None,
    ) -> None:
        self.planner_class = planner_class
        self.planner_kind = planner_kind
        self.planner_kwargs = planner_kwargs
        self.sessions: dict[str, Session] = {}
        self.sessions_lock = threading.Lock()
        probe = planner_class(**planner_kwargs)
        self.codex_version = probe.codex_version
        self.inference_provider = getattr(probe, "inference_provider", "codex")
        self.provider_version = getattr(
            probe, "provider_version", self.codex_version
        )
        self.model = probe.model
        self.reasoning_effort = probe.reasoning_effort
        self.prompt_profile = getattr(probe, "prompt_profile", None)
        self.rollout_behavior_id = getattr(probe, "rollout_behavior_id", None)
        intervention_plan = getattr(probe, "runtime_intervention_plan", None)
        self.runtime_intervention_bundle = (
            {
                "bundle_id": intervention_plan.bundle["bundle_id"],
                "revision": intervention_plan.bundle["revision"],
                "bundle_sha256": intervention_plan.bundle["bundle_sha256"],
            }
            if intervention_plan is not None
            else None
        )
        self.rethinking_threshold = getattr(probe, "rethinking_threshold", None)
        self.exploration_patience = getattr(probe, "exploration_patience", None)
        self.candidate_stagnant_limit = getattr(
            probe, "candidate_stagnant_limit", None
        )
        self.max_exploration_candidates = getattr(
            probe, "max_exploration_candidates", None
        )
        self.retrieval_scheme = getattr(probe, "retrieval_scheme", None)
        self.retrieval_top_k = getattr(probe, "retrieval_top_k", None)
        self.thinking_budget_tokens = getattr(
            probe, "thinking_budget_tokens", None
        )
        self.thinking_finalizer_max_output_tokens = getattr(
            probe, "thinking_finalizer_max_output_tokens", None
        )
        self.result_root_override = (
            result_root_override.resolve() if result_root_override is not None else None
        )
        if self.result_root_override is not None:
            self.result_root_override.mkdir(parents=True, exist_ok=True)

    def episode_dir(self, task: str, episode: int) -> Path:
        if self.result_root_override is None:
            raise RuntimeError("episode upload requires --result-root-override")
        if not task or any(value in task for value in ("/", "\\", "..")):
            raise ValueError(f"invalid task name: {task!r}")
        if episode < 0:
            raise ValueError("episode must be nonnegative")
        return self.result_root_override / task / f"episode_{episode:02d}"

    def archive_incomplete_episode(self, episode_dir: Path, session_id: str) -> None:
        if not episode_dir.exists():
            return
        summary_path = episode_dir / "summary.json"
        if summary_path.is_file():
            try:
                if json.loads(summary_path.read_text(encoding="utf-8")).get("complete") is True:
                    raise FileExistsError(f"refusing to overwrite complete episode: {episode_dir}")
            except json.JSONDecodeError:
                pass
        failed_root = self.result_root_override / "_failed_attempts"
        failed_root.mkdir(parents=True, exist_ok=True)
        destination = failed_root / f"{episode_dir.parent.name}_{episode_dir.name}_{session_id}"
        suffix = 1
        while destination.exists():
            destination = failed_root / (
                f"{episode_dir.parent.name}_{episode_dir.name}_{session_id}_{suffix:03d}"
            )
            suffix += 1
        shutil.move(str(episode_dir), str(destination))

    def start_episode(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = str(request["session_id"]).strip()
        if not session_id:
            raise ValueError("session_id must be non-empty")
        planner = self.planner_class(**self.planner_kwargs)
        task = str(request["task"])
        episode = int(request["episode"])
        run_dir = Path(request["run_dir"])
        if self.result_root_override is not None:
            episode_dir = self.episode_dir(task, episode)
            self.archive_incomplete_episode(episode_dir, session_id)
            run_dir = episode_dir / "codex_planner"
        start_kwargs = {
            "run_dir": run_dir,
            "detailed_task_description": str(request["detailed_task_description"]),
            "task": task,
        }
        if self.prompt_profile not in {
            "retrieval_v8_5_self_evolve_v1",
            "retrieval_v8_5_self_evolve_v2",
            "retrieval_v8_5_self_evolve_v3",
            "retrieval_v8_5_self_evolve_v4",
            "retrieval_v8_5_self_evolve_v5",
            "retrieval_v8_5_self_evolve_v6",
            "self_evolve_v2_train_active",
            "self_evolve_v3_train30_val20",
            "self_evolve_v3_multiproblem_auto_hard20",
            "self_evolve_v4_fullstack_trainval",
        }:
            start_kwargs.update(
                {
                    "episode": episode,
                    "init_case_id": str(request["init_case_id"]),
                }
            )
        if self.planner_kind in {"text-progress", "codex-workflow"}:
            guidance = request.get("planner_guidance")
            start_kwargs["planner_guidance"] = guidance if guidance is not None else {}
        planner.start_episode(
            **start_kwargs,
        )
        with self.sessions_lock:
            if session_id in self.sessions:
                raise ValueError(f"Planner session already exists: {session_id}")
            self.sessions[session_id] = Session(planner)
        return {"ok": True, "session_id": session_id}

    def upload_episode(self, request: dict[str, Any]) -> dict[str, Any]:
        task = str(request["task"])
        episode = int(request["episode"])
        payload = base64.b64decode(request["archive_b64"], validate=True)
        episode_dir = self.episode_dir(task, episode)
        episode_dir.mkdir(parents=True, exist_ok=True)
        upload_root = self.result_root_override / "_upload_tmp"
        upload_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="episode_", dir=upload_root) as temporary:
            temporary_dir = Path(temporary)
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                for member in archive.getmembers():
                    member_path = Path(member.name)
                    if (
                        member_path.is_absolute()
                        or ".." in member_path.parts
                        or member.issym()
                        or member.islnk()
                        or member.isdev()
                    ):
                        raise ValueError(f"unsafe archive member: {member.name!r}")
                archive.extractall(temporary_dir, filter="data")
            summary_source = temporary_dir / "summary.json"
            error_source = temporary_dir / "error.json"
            marker_source = summary_source if summary_source.is_file() else error_source
            if not marker_source.is_file():
                raise ValueError("episode archive contains neither summary.json nor error.json")
            for source in temporary_dir.iterdir():
                if source == marker_source:
                    continue
                destination = episode_dir / source.name
                if source.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)
            os.replace(marker_source, episode_dir / marker_source.name)
        return {
            "ok": True,
            "task": task,
            "episode": episode,
            "complete": (episode_dir / "summary.json").is_file(),
        }

    def episode_status(self, request: dict[str, Any]) -> dict[str, Any]:
        task = str(request["task"])
        episode = int(request["episode"])
        summary_path = self.episode_dir(task, episode) / "summary.json"
        summary = None
        if summary_path.is_file():
            try:
                candidate = json.loads(summary_path.read_text(encoding="utf-8"))
                if candidate.get("complete") is True:
                    summary = candidate
            except (OSError, json.JSONDecodeError):
                summary = None
        return {
            "ok": True,
            "task": task,
            "episode": episode,
            "complete": summary is not None,
            "summary": summary,
        }

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = str(request["session_id"])
        with self.sessions_lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown planner session: {session_id}")
        front_png = request["front_png"]
        wrist_png = request["wrist_png"]
        frame_ids = [int(value) for value in request["frame_ids"]]
        if len(front_png) != len(wrist_png) or len(front_png) != len(frame_ids):
            raise ValueError("front_png, wrist_png, and frame_ids must have equal length")
        with session.lock:
            subtask, record = session.planner.plan(
                front_frames=[decode_png(value) for value in front_png],
                wrist_frames=[decode_png(value) for value in wrist_png],
                frame_ids=frame_ids,
            )
        return {"ok": True, "grounded_subtask": subtask, "record": record}

    def end_episode(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = str(request["session_id"])
        with self.sessions_lock:
            existed = self.sessions.pop(session_id, None) is not None
        return {"ok": True, "session_id": session_id, "existed": existed}


class PlannerHTTPServer(ThreadingHTTPServer):
    service: PlannerService
    access_token: str
    denied_clients: set[str]


class Handler(BaseHTTPRequestHandler):
    server: PlannerHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[codex-planner] {self.address_string()} {fmt % args}", flush=True)

    def send_json(self, status: int, value: dict[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def authorized(self) -> bool:
        return self.headers.get("Authorization", "") == f"Bearer {self.server.access_token}"

    def client_denied(self) -> bool:
        return self.client_address[0] in getattr(self.server, "denied_clients", set())

    def do_GET(self) -> None:  # noqa: N802
        if self.client_denied():
            self.send_json(503, {"ok": False, "error": "client temporarily denied"})
            return
        if not self.authorized():
            self.send_json(401, {"ok": False, "error": "unauthorized"})
            return
        if self.path == "/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "model": self.server.service.model,
                    "reasoning_effort": self.server.service.reasoning_effort,
                    "codex_version": self.server.service.codex_version,
                    "inference_provider": self.server.service.inference_provider,
                    "provider_version": self.server.service.provider_version,
                    "planner_kind": self.server.service.planner_kind,
                    "prompt_profile": self.server.service.prompt_profile,
                    "rollout_behavior_id": self.server.service.rollout_behavior_id,
                    "runtime_intervention_bundle": (
                        self.server.service.runtime_intervention_bundle
                    ),
                    "rethinking_threshold": self.server.service.rethinking_threshold,
                    "exploration_patience": self.server.service.exploration_patience,
                    "candidate_stagnant_limit": (
                        self.server.service.candidate_stagnant_limit
                    ),
                    "max_exploration_candidates": (
                        self.server.service.max_exploration_candidates
                    ),
                    "retrieval_scheme": self.server.service.retrieval_scheme,
                    "retrieval_top_k": self.server.service.retrieval_top_k,
                    "active_sessions": len(self.server.service.sessions),
                },
            )
        else:
            self.send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.client_denied():
                self.send_json(503, {"ok": False, "error": "client temporarily denied"})
                return
            if not self.authorized():
                self.send_json(401, {"ok": False, "error": "unauthorized"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError(f"Invalid Content-Length: {length}")
            request = json.loads(self.rfile.read(length))
            if self.path == "/start_episode":
                result = self.server.service.start_episode(request)
            elif self.path == "/plan":
                result = self.server.service.plan(request)
            elif self.path == "/end_episode":
                result = self.server.service.end_episode(request)
            elif self.path == "/upload_episode":
                result = self.server.service.upload_episode(request)
            elif self.path == "/episode_status":
                result = self.server.service.episode_status(request)
            else:
                self.send_json(404, {"ok": False, "error": "not found"})
                return
            self.send_json(200, result)
        except Exception as exc:
            traceback.print_exc()
            self.send_json(
                500,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18229)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument(
        "--inference-provider",
        choices=("codex", "local_openai"),
        default="codex",
        help="Keep the planner/harness fixed and select only the model-call backend.",
    )
    parser.add_argument(
        "--openai-base-url",
        help="OpenAI-compatible server root, for example http://127.0.0.1:8000/v1.",
    )
    parser.add_argument(
        "--openai-api-key-env",
        default="LOCAL_VLM_API_KEY",
        help="Environment variable containing an optional local-server bearer token.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--thinking-budget-tokens",
        type=int,
        default=0,
        help=(
            "For local thinking models, cap the first reasoning pass at this many "
            "tokens and use a nonthinking schema finalizer if it has not emitted "
            "valid JSON. Zero keeps the original single-pass behavior."
        ),
    )
    parser.add_argument(
        "--thinking-finalizer-max-output-tokens",
        type=int,
        default=4096,
        help="Maximum JSON tokens for the nonthinking finalization pass.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=false to a compatible local model chat template.",
    )
    parser.add_argument(
        "--planner-kind",
        choices=("keyframes", "text-progress", "codex-workflow"),
        default="keyframes",
    )
    parser.add_argument(
        "--text-progress-profile",
        choices=sorted(TEXT_PROGRESS_PROMPT_PROFILES),
        default=DEFAULT_TEXT_PROGRESS_PROMPT_PROFILE,
    )
    parser.add_argument("--system-prompt", type=Path)
    parser.add_argument("--user-prompt", type=Path)
    parser.add_argument("--output-schema", type=Path)
    parser.add_argument(
        "--runtime-experience-memory",
        type=Path,
        help="Explicit candidate/frozen runtime memory for train-active self-evolution.",
    )
    parser.add_argument(
        "--runtime-lesson-memory",
        type=Path,
        help="Task-scoped structured prompt experience for self-evolve v2.",
    )
    parser.add_argument(
        "--refine-check-criteria",
        type=Path,
        help="Explicit criterion catalog paired with --runtime-experience-memory.",
    )
    parser.add_argument(
        "--runtime-intervention-bundle",
        type=Path,
        help=(
            "Typed v4 prompt/gate/ledger/retrieval/harness intervention bundle; "
            "requires the isolated v4 text-progress profile."
        ),
    )
    parser.add_argument("--rethinking-threshold", type=int, default=3)
    parser.add_argument("--exploration-patience", type=int, default=2)
    parser.add_argument("--candidate-stagnant-limit", type=int, default=2)
    parser.add_argument("--max-exploration-candidates", type=int, default=3)
    parser.add_argument("--retrieval-scheme", choices=("R0", "R1", "R2", "R3", "R4"), default="R0")
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--keep-recent-inputs", action="store_true")
    parser.add_argument(
        "--deny-client",
        action="append",
        default=[],
        help="Temporarily reject requests from this exact client IP; repeat as needed.",
    )
    parser.add_argument(
        "--result-root-override",
        type=Path,
        help="Write planner records and authenticated episode uploads below this root.",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        required=True,
        help="0600 file containing the bearer token; generated when absent",
    )
    args = parser.parse_args()
    if args.rethinking_threshold < 1:
        parser.error("--rethinking-threshold must be positive")
    if args.exploration_patience < 1:
        parser.error("--exploration-patience must be positive")
    if args.candidate_stagnant_limit < 1:
        parser.error("--candidate-stagnant-limit must be positive")
    if args.max_exploration_candidates < 1:
        parser.error("--max-exploration-candidates must be positive")
    if args.retrieval_top_k < 1:
        parser.error("--retrieval-top-k must be positive")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    if args.thinking_budget_tokens < 0:
        parser.error("--thinking-budget-tokens must be nonnegative")
    if args.thinking_finalizer_max_output_tokens < 1:
        parser.error("--thinking-finalizer-max-output-tokens must be positive")
    if args.inference_provider == "local_openai":
        if args.planner_kind != "text-progress":
            parser.error("local_openai currently supports --planner-kind text-progress")
        if not args.openai_base_url:
            parser.error("--openai-base-url is required for local_openai")
    if args.runtime_experience_memory is not None and args.planner_kind != "text-progress":
        parser.error("--runtime-experience-memory requires --planner-kind text-progress")
    if args.runtime_lesson_memory is not None and args.planner_kind != "text-progress":
        parser.error("--runtime-lesson-memory requires --planner-kind text-progress")
    if args.refine_check_criteria is not None and args.planner_kind != "text-progress":
        parser.error("--refine-check-criteria requires --planner-kind text-progress")
    if args.runtime_intervention_bundle is not None and args.planner_kind != "text-progress":
        parser.error("--runtime-intervention-bundle requires --planner-kind text-progress")

    if args.planner_kind == "text-progress":
        planner_class = CodexTextProgressPlanner
        profile = TEXT_PROGRESS_PROMPT_PROFILES[args.text_progress_profile]
        system_prompt = args.system_prompt or profile["system_prompt"]
        user_prompt = args.user_prompt or profile["user_prompt"]
        output_schema = args.output_schema or profile["output_schema"]
    elif args.planner_kind == "codex-workflow":
        if args.inference_provider != "codex":
            parser.error("codex-workflow requires --inference-provider codex")
        planner_class = CodexWorkflowPlanner
        system_prompt = args.system_prompt or DEFAULT_WORKFLOW_AGENTS
        user_prompt = args.user_prompt or DEFAULT_WORKFLOW_USER_PROMPT
        output_schema = args.output_schema or DEFAULT_WORKFLOW_OUTPUT_SCHEMA
    else:
        planner_class = CodexRolloutPlanner
        system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
        user_prompt = args.user_prompt or DEFAULT_USER_PROMPT
        output_schema = args.output_schema or DEFAULT_OUTPUT_SCHEMA

    kwargs = {
        "codex_bin": args.codex_bin,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "system_prompt_path": system_prompt,
        "user_prompt_path": user_prompt,
        "output_schema_path": output_schema,
        "timeout_seconds": args.timeout_seconds,
        "keep_recent_inputs": args.keep_recent_inputs,
        "verify_login": args.inference_provider == "codex",
    }
    if args.planner_kind == "text-progress":
        kwargs["inference_provider"] = args.inference_provider
        kwargs["openai_base_url"] = args.openai_base_url
        kwargs["openai_api_key"] = (
            os.environ.get(args.openai_api_key_env)
            if args.inference_provider == "local_openai"
            else None
        )
        kwargs["max_output_tokens"] = args.max_output_tokens
        kwargs["temperature"] = args.temperature
        kwargs["enable_thinking"] = not args.disable_thinking
        kwargs["thinking_budget_tokens"] = (
            args.thinking_budget_tokens or None
        )
        kwargs["thinking_finalizer_max_output_tokens"] = (
            args.thinking_finalizer_max_output_tokens
        )
        kwargs["prompt_profile"] = args.text_progress_profile
        kwargs["runtime_experience_memory_path"] = (
            args.runtime_experience_memory.resolve()
            if args.runtime_experience_memory is not None
            else None
        )
        kwargs["runtime_lesson_memory_path"] = (
            args.runtime_lesson_memory.resolve()
            if args.runtime_lesson_memory is not None
            else None
        )
        kwargs["refine_check_criteria_path"] = (
            args.refine_check_criteria.resolve()
            if args.refine_check_criteria is not None
            else None
        )
        kwargs["runtime_intervention_bundle_path"] = (
            args.runtime_intervention_bundle.resolve()
            if args.runtime_intervention_bundle is not None
            else None
        )
        kwargs["rethinking_threshold"] = args.rethinking_threshold
        kwargs["exploration_patience"] = args.exploration_patience
        kwargs["candidate_stagnant_limit"] = args.candidate_stagnant_limit
        kwargs["max_exploration_candidates"] = args.max_exploration_candidates
        kwargs["retrieval_scheme"] = args.retrieval_scheme
        kwargs["retrieval_top_k"] = args.retrieval_top_k
    elif args.planner_kind == "codex-workflow":
        kwargs["retrieval_scheme"] = args.retrieval_scheme
        kwargs["retrieval_top_k"] = args.retrieval_top_k
    server = PlannerHTTPServer((args.host, args.port), Handler)
    server.service = PlannerService(
        planner_class,
        args.planner_kind,
        kwargs,
        result_root_override=args.result_root_override,
    )
    token_file = args.token_file.resolve()
    token_file.parent.mkdir(parents=True, exist_ok=True)
    if token_file.exists():
        access_token = token_file.read_text(encoding="utf-8").strip()
        if not access_token:
            parser.error(f"empty --token-file: {token_file}")
    else:
        access_token = secrets.token_hex(32)
        descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(access_token + "\n")
    os.chmod(token_file, 0o600)
    server.access_token = access_token
    server.denied_clients = set(args.deny_client)
    print(
        json.dumps(
            {
                "event": "ready",
                "host": args.host,
                "port": args.port,
                "model": server.service.model,
                "reasoning_effort": server.service.reasoning_effort,
                "codex_version": server.service.codex_version,
                "inference_provider": server.service.inference_provider,
                "provider_version": server.service.provider_version,
                "planner_kind": server.service.planner_kind,
                "prompt_profile": server.service.prompt_profile,
                "rollout_behavior_id": server.service.rollout_behavior_id,
                "rethinking_threshold": server.service.rethinking_threshold,
                "exploration_patience": server.service.exploration_patience,
                "candidate_stagnant_limit": server.service.candidate_stagnant_limit,
                "max_exploration_candidates": (
                    server.service.max_exploration_candidates
                ),
                "retrieval_scheme": server.service.retrieval_scheme,
                "retrieval_top_k": server.service.retrieval_top_k,
                "thinking_budget_tokens": server.service.thinking_budget_tokens,
                "thinking_finalizer_max_output_tokens": (
                    server.service.thinking_finalizer_max_output_tokens
                ),
                "result_root_override": (
                    str(server.service.result_root_override)
                    if server.service.result_root_override is not None
                    else None
                ),
                "authentication": (
                    "local ChatGPT login"
                    if server.service.inference_provider == "codex"
                    else "local OpenAI-compatible endpoint"
                ),
                "http_authentication": "bearer token",
                "denied_clients": sorted(server.denied_clients),
                "token_file": str(token_file),
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
