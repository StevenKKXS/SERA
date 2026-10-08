#!/usr/bin/env python3
"""Model-call backends shared by the stateful rollout planner.

Prompts, planner state, skill retrieval, validation, and the deterministic
harness stay outside this module. A backend only maps one rendered prompt and
an ordered image list to one structured JSON response.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _endpoint(base_url: str, suffix: str) -> str:
    root = base_url.rstrip("/")
    return root + suffix if root.endswith("/v1") else root + "/v1" + suffix


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _request_json(
    *,
    url: str,
    payload: dict[str, Any] | None,
    api_key: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers=_headers(api_key),
        method="GET" if payload is None else "POST",
    )
    # Local planner traffic must never be routed through the workstation or
    # cluster HTTP proxy. This also makes loopback behavior deterministic.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI-compatible planner HTTP {exc.code} from {url}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Unable to reach OpenAI-compatible planner at {url}: {exc.reason}"
        ) from exc
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return result


def get_openai_compatible_version(
    *, base_url: str, api_key: str | None, timeout_seconds: float = 30.0
) -> str:
    response = _request_json(
        url=_endpoint(base_url, "/models"),
        payload=None,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    model_ids = [
        str(row.get("id"))
        for row in response.get("data", [])
        if isinstance(row, dict) and row.get("id")
    ]
    return "openai-compatible:" + (",".join(model_ids) or "unknown-model")


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _assistant_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    raise ValueError("OpenAI-compatible response has no textual assistant content")


def _parse_json_content(content: str) -> dict[str, Any]:
    value = content.strip()
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[1].strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    elif value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise
        result = json.loads(value[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("Planner response must be a JSON object")
    return result


def _normalize_usage(raw_usage: Any) -> dict[str, int] | None:
    if not isinstance(raw_usage, dict):
        return None
    usage: dict[str, int] = {}
    for source, destination in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = raw_usage.get(source)
        if isinstance(value, int):
            usage[destination] = value
    details = raw_usage.get("prompt_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
        usage["cached_input_tokens"] = details["cached_tokens"]
    return usage or None


def _sum_usage(*values: dict[str, int] | None) -> dict[str, int] | None:
    total: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        for key, amount in value.items():
            total[key] = total.get(key, 0) + amount
    return total or None


def _finish_reason(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    value = choices[0].get("finish_reason")
    return str(value) if value is not None else None


def run_openai_compatible_inference(
    *,
    model: str,
    prompt: str,
    output_schema: Path,
    image_paths: list[Path],
    destination: Path,
    timeout_seconds: float,
    base_url: str,
    api_key: str | None,
    max_output_tokens: int = 4096,
    temperature: float = 0.0,
    enable_thinking: bool = True,
    thinking_budget_tokens: int | None = None,
    thinking_finalizer_max_output_tokens: int = 4096,
    **_: Any,
) -> dict[str, Any]:
    """Call a vLLM-style endpoint using the existing planner call contract."""
    if not image_paths:
        raise ValueError("Rollout planning requires at least one attached image")
    schema = json.loads(output_schema.read_text(encoding="utf-8"))
    image_content = [
        {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
        for path in image_paths
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "planner_response",
            "schema": schema,
            "strict": True,
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    bounded_thinking = bool(enable_thinking and thinking_budget_tokens is not None)
    first_stage_max_tokens = (
        min(max_output_tokens, int(thinking_budget_tokens))
        if bounded_thinking
        else max_output_tokens
    )
    request_record = {
        "base_url": base_url,
        "model": model,
        "image_paths": [str(path.resolve()) for path in image_paths],
        "image_count": len(image_paths),
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "bounded_thinking": bounded_thinking,
        "thinking_budget_tokens": (
            first_stage_max_tokens if bounded_thinking else None
        ),
        "thinking_finalizer_max_output_tokens": (
            thinking_finalizer_max_output_tokens if bounded_thinking else None
        ),
        "output_schema": str(output_schema.resolve()),
        "authentication": "bearer" if api_key else "none",
    }
    (destination / "provider_request.json").write_text(
        json.dumps(request_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    started_at = utc_now()
    started = time.monotonic()
    def run_stage(
        *,
        stage_dir: Path,
        stage_prompt: str,
        stage_enable_thinking: bool,
        stage_max_tokens: int,
        stage_name: str,
    ) -> tuple[dict[str, Any], str, str | None, dict[str, int] | None]:
        stage_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": stage_prompt},
                        *image_content,
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": stage_max_tokens,
            "response_format": response_format,
            "chat_template_kwargs": {"enable_thinking": stage_enable_thinking},
        }
        (stage_dir / "provider_request.json").write_text(
            json.dumps(
                {
                    **request_record,
                    "stage": stage_name,
                    "max_output_tokens": stage_max_tokens,
                    "enable_thinking": stage_enable_thinking,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        response_payload = _request_json(
            url=_endpoint(base_url, "/chat/completions"),
            payload=payload,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        (stage_dir / "provider_response.json").write_text(
            json.dumps(response_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        choices = response_payload.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            raise ValueError("OpenAI-compatible response has no choices[0]")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("OpenAI-compatible response has no choices[0].message")
        content = _assistant_text(message)
        (stage_dir / "response.txt").write_text(content, encoding="utf-8")
        reasoning = message.get("reasoning_content", message.get("reasoning"))
        if isinstance(reasoning, str) and reasoning:
            (stage_dir / "reasoning.txt").write_text(reasoning, encoding="utf-8")
        return (
            response_payload,
            content,
            reasoning if isinstance(reasoning, str) and reasoning else None,
            _normalize_usage(response_payload.get("usage")),
        )

    first_stage_dir = destination / "thinking_stage" if bounded_thinking else destination
    response_payload, content, reasoning, first_usage = run_stage(
        stage_dir=first_stage_dir,
        stage_prompt=prompt,
        stage_enable_thinking=enable_thinking,
        stage_max_tokens=first_stage_max_tokens,
        stage_name="bounded_thinking" if bounded_thinking else "single_pass",
    )
    finalizer_used = False
    finalizer_usage: dict[str, int] | None = None
    first_finish_reason = _finish_reason(response_payload)
    try:
        parsed_response = _parse_json_content(content)
        final_stage_dir = first_stage_dir
    except (json.JSONDecodeError, ValueError) as first_parse_error:
        if not bounded_thinking:
            raise
        finalizer_used = True
        draft = reasoning or content
        finalizer_prompt = (
            prompt
            + "\n\nA bounded thinking draft from an earlier pass follows. Treat it "
            "as advisory and correct it against the attached images and the fixed "
            "task facts. Do not repeat or discuss the draft. Return only the complete "
            "JSON object required by the schema.\n\n<bounded_thinking_draft>\n"
            + draft
            + "\n</bounded_thinking_draft>"
        )
        final_stage_dir = destination / "finalization_stage"
        response_payload, content, _, finalizer_usage = run_stage(
            stage_dir=final_stage_dir,
            stage_prompt=finalizer_prompt,
            stage_enable_thinking=False,
            stage_max_tokens=thinking_finalizer_max_output_tokens,
            stage_name="nonthinking_json_finalizer",
        )
        try:
            parsed_response = _parse_json_content(content)
        except (json.JSONDecodeError, ValueError) as final_parse_error:
            raise ValueError(
                "Bounded thinking and nonthinking finalizer both failed to produce "
                f"planner JSON: first={type(first_parse_error).__name__}: "
                f"{first_parse_error}; final={type(final_parse_error).__name__}: "
                f"{final_parse_error}"
            ) from final_parse_error

    # Keep the established root-level artifact contract: these files always
    # describe the response accepted by the backend. Stage directories retain
    # the bounded draft and formatter call for complete auditing.
    if final_stage_dir != destination:
        for name in ("provider_response.json", "response.txt"):
            shutil.copy2(final_stage_dir / name, destination / name)
        if (first_stage_dir / "reasoning.txt").exists():
            shutil.copy2(first_stage_dir / "reasoning.txt", destination / "reasoning.txt")
        elif bounded_thinking:
            (destination / "reasoning.txt").write_text(
                reasoning or (first_stage_dir / "response.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
    wall_seconds = time.monotonic() - started
    usage = _sum_usage(first_usage, finalizer_usage)
    return {
        "response": parsed_response,
        "thread_id": response_payload.get("id"),
        "usage": usage,
        "started_at": started_at,
        "wall_seconds": wall_seconds,
        "exit_code": 0,
        "capacity_retry_count": 0,
        "provider": "local_openai",
        "thinking_finalizer_used": finalizer_used,
        "thinking_stage_finish_reason": first_finish_reason,
        "thinking_stage_usage": first_usage,
        "finalization_stage_usage": finalizer_usage,
    }
