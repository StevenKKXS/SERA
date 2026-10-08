#!/usr/bin/env python3
"""HTTP client used by a remote RoboMME rollout to call the local Codex planner."""

from __future__ import annotations

import base64
import io
import json
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def planner_http_error(status_code: int, detail: str) -> RuntimeError:
    return RuntimeError(f"Codex planner HTTP {status_code}: {detail}")


def encode_png(frame: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class RemoteCodexPlanner:
    def __init__(
        self, url: str, timeout_seconds: float = 1200.0, token_file: Path | None = None
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        if token_file is None:
            raise ValueError("A planner bearer-token file is required")
        self.access_token = token_file.read_text(encoding="utf-8").strip()
        if not self.access_token:
            raise ValueError(f"Planner token file is empty: {token_file}")
        self.session_id: str | None = None

    def post(self, endpoint: str, value: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.url}/{endpoint}",
            data=json.dumps(value).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.access_token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise planner_http_error(exc.code, detail) from exc
        if result.get("ok") is not True:
            raise RuntimeError(result.get("error", "Codex planner server returned an error"))
        return result

    def start_episode(
        self,
        *,
        session_id: str,
        run_dir: Path,
        detailed_task_description: str,
        task: str,
        episode: int,
        init_case_id: str,
        planner_guidance: dict[str, Any] | None = None,
    ) -> None:
        if self.session_id is not None:
            raise RuntimeError(f"Planner session is already active: {self.session_id}")
        self.post(
            "start_episode",
            {
                "session_id": session_id,
                "run_dir": str(run_dir.resolve()),
                "detailed_task_description": detailed_task_description,
                "task": task,
                "episode": episode,
                "init_case_id": init_case_id,
                "planner_guidance": planner_guidance or {},
            },
        )
        self.session_id = session_id

    def plan(
        self,
        *,
        front_frames: list[np.ndarray],
        wrist_frames: list[np.ndarray],
        frame_ids: list[int],
    ) -> tuple[str, dict[str, Any]]:
        if self.session_id is None:
            raise RuntimeError("start_episode must be called before plan")
        result = self.post(
            "plan",
            {
                "session_id": self.session_id,
                "front_png": [encode_png(frame) for frame in front_frames],
                "wrist_png": [encode_png(frame) for frame in wrist_frames],
                "frame_ids": frame_ids,
            },
        )
        return result["grounded_subtask"], result["record"]

    def end_episode(self) -> None:
        if self.session_id is None:
            return
        session_id = self.session_id
        self.session_id = None
        self.post("end_episode", {"session_id": session_id})

    def upload_episode(
        self, *, episode_dir: Path, task: str, episode: int, attempts: int = 5
    ) -> dict[str, Any]:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for path in sorted(episode_dir.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=str(path.relative_to(episode_dir)))
        request = {
            "task": task,
            "episode": episode,
            "archive_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self.post("upload_episode", request)
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(min(30.0, 2.0**attempt))
        raise RuntimeError(f"episode upload failed after {attempts} attempts") from last_error

    def get_episode_summary(self, *, task: str, episode: int) -> dict[str, Any] | None:
        result = self.post("episode_status", {"task": task, "episode": episode})
        summary = result.get("summary")
        return summary if isinstance(summary, dict) and summary.get("complete") is True else None
