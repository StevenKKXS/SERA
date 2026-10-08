"""Validate and compile immutable self-evolve v4 intervention bundles.

The v4 controller treats an experience as a typed set of runtime interventions,
not only as prompt text.  Built-in component types deliberately compile to the
already-tested v1-v6 planner mechanisms.  This restores their behavior without
changing the mature v8.5 profile or dynamically applying source patches during
an episode.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .registry import HOOK_ORDER, PLUGIN_REGISTRY


CANDIDATE_SCHEMA_VERSION = "self-evolve-v4-candidate-intervention-v1"
BUNDLE_SCHEMA_VERSION = "self-evolve-runtime-intervention-bundle-v1"
BASE_PROFILE = "retrieval_v8_5_refine_then_lock"

BUILTIN_COMPONENT_TYPES = frozenset(PLUGIN_REGISTRY)

FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "episode",
        "episode_id",
        "init_case_id",
        "seed",
        "source_episode_refs",
        "supporting_episode_refs",
        "evaluator_state",
        "future_frames",
        "absolute_coordinates",
        "simulator_state",
        "ground_truth_state",
        "test_outcome",
    }
)
FORBIDDEN_IDENTITY_PATTERNS = (
    re.compile(
        r"(?i)(?:^|[^a-z0-9])(?:episode(?:_id)?|ep)[\s:=/_-]*0*[0-9]+(?:[^0-9]|$)"
    ),
    re.compile(r"(?i)\binit_case_id\b"),
    re.compile(r"(?i)\b(?:environment_)?seed\s*[:=]\s*[0-9]+\b"),
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_task_scope(value: Any, field_name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(task, str) or not task for task in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{field_name} must contain unique non-empty task names")
    return list(value)


def assert_runtime_identity_isolated(value: Any, *, source: str) -> None:
    """Reject episode-specific and privileged state from runtime artifacts."""
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_RUNTIME_KEYS:
                raise ValueError(f"{source} contains forbidden runtime key {key!r}")
            assert_runtime_identity_isolated(nested, source=f"{source}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_runtime_identity_isolated(nested, source=f"{source}[{index}]")
    elif isinstance(value, str):
        for pattern in FORBIDDEN_IDENTITY_PATTERNS:
            match = pattern.search(value)
            if match is not None:
                raise ValueError(
                    f"{source} contains episode-specific text {match.group(0)!r}"
                )


def _validate_relative_artifact(
    artifact: Any, *, component_id: str, artifact_name: str, bundle_dir: Path | None
) -> dict[str, str]:
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise ValueError(
            f"{component_id}.{artifact_name} must contain exactly path and sha256"
        )
    relative = Path(_require_string(artifact.get("path"), "artifact.path"))
    expected = _require_string(artifact.get("sha256"), "artifact.sha256")
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Artifact path must be bundle-relative: {relative}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Invalid SHA256 for artifact {relative}")
    if bundle_dir is not None:
        resolved = (bundle_dir / relative).resolve()
        try:
            resolved.relative_to(bundle_dir.resolve())
        except ValueError as error:
            raise ValueError(
                f"Artifact escapes bundle directory: {relative}"
            ) from error
        if not resolved.is_file():
            raise ValueError(f"Missing intervention artifact: {resolved}")
        actual = sha256_file(resolved)
        if actual != expected:
            raise ValueError(
                f"Artifact SHA256 mismatch for {relative}: expected {expected}, got {actual}"
            )
    return {"path": relative.as_posix(), "sha256": expected}


def _validate_lesson(component: dict[str, Any]) -> None:
    experience = component["config"].get("experience")
    required = {
        "experience_id",
        "revision",
        "condition",
        "lesson",
        "recommended_skill_or_policy",
        "success_evidence",
        "forbidden_behavior",
    }
    if not isinstance(experience, dict) or set(experience) != required:
        raise ValueError(
            f"{component['component_id']} prompt_lesson experience fields mismatch"
        )
    _require_string(experience["experience_id"], "experience_id")
    if not isinstance(experience["revision"], int) or experience["revision"] < 1:
        raise ValueError("Prompt lesson revision must be positive")
    for field_name in ("condition", "lesson", "recommended_skill_or_policy"):
        _require_string(experience[field_name], field_name)
    for field_name in ("success_evidence", "forbidden_behavior"):
        if not isinstance(experience[field_name], list) or any(
            not isinstance(item, str) for item in experience[field_name]
        ):
            raise ValueError(f"{field_name} must be a list of strings")


def _validate_component(
    component: Any, *, candidate_tasks: set[str], bundle_dir: Path | None
) -> dict[str, Any]:
    required = {
        "component_id",
        "type",
        "task_scope",
        "priority",
        "state_namespace",
        "claims",
        "config",
        "artifacts",
    }
    if not isinstance(component, dict) or set(component) != required:
        raise ValueError(
            f"Intervention component fields mismatch: expected {sorted(required)}"
        )
    component_id = _require_string(component["component_id"], "component_id")
    component_type = _require_string(component["type"], "component.type")
    if component_type not in BUILTIN_COMPONENT_TYPES:
        raise ValueError(f"Unsupported intervention component type {component_type!r}")
    task_scope = set(
        _require_task_scope(component["task_scope"], "component.task_scope")
    )
    if not task_scope.issubset(candidate_tasks):
        raise ValueError(f"{component_id} task scope escapes candidate task scope")
    if not isinstance(component["priority"], int):
        raise ValueError(f"{component_id}.priority must be an integer")
    namespace = component["state_namespace"]
    if namespace is not None:
        _require_string(namespace, f"{component_id}.state_namespace")
    claims = component["claims"]
    if (
        not isinstance(claims, list)
        or any(not isinstance(claim, str) or not claim for claim in claims)
        or len(set(claims)) != len(claims)
    ):
        raise ValueError(f"{component_id}.claims must contain unique strings")
    if not isinstance(component["config"], dict):
        raise ValueError(f"{component_id}.config must be an object")
    artifacts = component["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError(f"{component_id}.artifacts must be an object")
    normalized_artifacts = {
        str(name): _validate_relative_artifact(
            artifact,
            component_id=component_id,
            artifact_name=str(name),
            bundle_dir=bundle_dir,
        )
        for name, artifact in artifacts.items()
    }

    config = component["config"]
    if component_type == "prompt_lesson":
        if artifacts:
            raise ValueError("prompt_lesson is inline and must not declare artifacts")
        _validate_lesson(component)
    elif component_type == "criterion_overlay":
        if set(artifacts) != {"catalog"} or config:
            raise ValueError("criterion_overlay requires exactly a catalog artifact")
    elif component_type == "bounded_transition_gate":
        expected = {
            "experience_id",
            "revision",
            "situation_class",
            "criterion_id",
            "max_blocked_transitions",
        }
        if set(config) != expected or artifacts:
            raise ValueError("Invalid bounded_transition_gate configuration")
        for field_name in ("experience_id", "situation_class", "criterion_id"):
            _require_string(config[field_name], field_name)
        if not isinstance(config["revision"], int) or config["revision"] < 1:
            raise ValueError("Gate revision must be positive")
        if (
            not isinstance(config["max_blocked_transitions"], int)
            or config["max_blocked_transitions"] < 1
        ):
            raise ValueError("Gate block budget must be positive")
    elif component_type == "swing_visual_binding_gate":
        if set(config) != {"experience_id", "revision"} or artifacts:
            raise ValueError("Invalid swing_visual_binding_gate configuration")
        if task_scope != {"SwingXtimes"}:
            raise ValueError("Swing visual binding gate must be scoped to SwingXtimes")
        _require_string(config["experience_id"], "experience_id")
        if not isinstance(config["revision"], int) or config["revision"] < 1:
            raise ValueError("Swing gate revision must be positive")
    elif component_type == "counting_event_ledger":
        if (
            set(config) != {"explicit_first_pickup", "exact_pickup_ordinal"}
            or artifacts
        ):
            raise ValueError("Invalid counting_event_ledger configuration")
        if any(not isinstance(config[key], bool) for key in config):
            raise ValueError("Counting ledger options must be boolean")
        if not task_scope.issubset({"PickXtimes", "BinFill"}):
            raise ValueError("Counting ledger is scoped only to PickXtimes/BinFill")
    elif component_type == "video_repick_event_ledger":
        if config or artifacts:
            raise ValueError("VideoRepick event ledger has no authoring options")
        if task_scope != {"VideoRepick"}:
            raise ValueError("VideoRepick event ledger must be scoped to VideoRepick")
    elif component_type == "video_repick_grounding_recovery_gate":
        expected = {
            "trigger_after_failed_pickups",
            "minimum_point_delta_norm",
            "max_forced_regrounds_per_ordinal",
        }
        if set(config) != expected or artifacts:
            raise ValueError("Invalid VideoRepick grounding recovery configuration")
        if task_scope != {"VideoRepick"}:
            raise ValueError("VideoRepick grounding recovery must be task-scoped")
        for key in expected:
            if isinstance(config[key], bool) or not isinstance(config[key], int):
                raise ValueError(f"{key} must be an integer")
        if config["trigger_after_failed_pickups"] < 2:
            raise ValueError("Grounding recovery must preserve at least one retry")
        if not 1 <= config["minimum_point_delta_norm"] <= 250:
            raise ValueError("Grounding recovery point delta must be in [1, 250]")
        if not 1 <= config["max_forced_regrounds_per_ordinal"] <= 3:
            raise ValueError("Grounding recovery budget must be in [1, 3]")
    elif component_type == "unmask_pickup_grounding_recovery_gate":
        expected = {
            "trigger_after_failed_pickups",
            "minimum_point_delta_norm",
            "max_forced_regrounds_per_target",
        }
        if set(config) != expected or artifacts:
            raise ValueError("Invalid unmask pickup grounding recovery configuration")
        if not task_scope.issubset({"ButtonUnmask", "ButtonUnmaskSwap"}):
            raise ValueError("Unmask pickup grounding recovery must be task-scoped")
        for key in expected:
            if isinstance(config[key], bool) or not isinstance(config[key], int):
                raise ValueError(f"{key} must be an integer")
        if config["trigger_after_failed_pickups"] < 2:
            raise ValueError("Unmask recovery must preserve at least one retry")
        if not 1 <= config["minimum_point_delta_norm"] <= 250:
            raise ValueError("Unmask recovery point delta must be in [1, 250]")
        if not 1 <= config["max_forced_regrounds_per_target"] <= 3:
            raise ValueError("Unmask recovery budget must be in [1, 3]")
    elif component_type == "button_swap_press_grounding_recovery_gate":
        expected = {
            "trigger_after_failed_presses",
            "minimum_point_delta_norm",
            "minimum_completed_button_delta_norm",
            "max_forced_regrounds_per_button",
        }
        if set(config) != expected or artifacts:
            raise ValueError("Invalid ButtonUnmaskSwap press recovery configuration")
        if task_scope != {"ButtonUnmaskSwap"}:
            raise ValueError(
                "ButtonUnmaskSwap press recovery must be scoped only to "
                "ButtonUnmaskSwap"
            )
        for key in expected:
            if isinstance(config[key], bool) or not isinstance(config[key], int):
                raise ValueError(f"{key} must be an integer")
        if config["trigger_after_failed_presses"] < 2:
            raise ValueError("Button press recovery must preserve at least one retry")
        for key in (
            "minimum_point_delta_norm",
            "minimum_completed_button_delta_norm",
        ):
            if not 1 <= config[key] <= 500:
                raise ValueError(f"{key} must be in [1, 500]")
        if not 1 <= config["max_forced_regrounds_per_button"] <= 3:
            raise ValueError("Button press recovery budget must be in [1, 3]")
    elif component_type == "pickhighlight_pickup_grounding_recovery_gate":
        expected = {
            "trigger_after_failed_pickups",
            "minimum_point_delta_norm",
            "max_forced_regrounds_per_target",
        }
        if set(config) != expected or artifacts:
            raise ValueError("Invalid PickHighlight pickup grounding recovery configuration")
        if task_scope != {"PickHighlight"}:
            raise ValueError("PickHighlight pickup grounding recovery must be task-scoped")
        for key in expected:
            if isinstance(config[key], bool) or not isinstance(config[key], int):
                raise ValueError(f"{key} must be an integer")
        if config["trigger_after_failed_pickups"] < 2:
            raise ValueError("PickHighlight recovery must preserve at least one retry")
        if not 1 <= config["minimum_point_delta_norm"] <= 250:
            raise ValueError("PickHighlight recovery point delta must be in [1, 250]")
        if not 1 <= config["max_forced_regrounds_per_target"] <= 3:
            raise ValueError("PickHighlight recovery budget must be in [1, 3]")
    elif component_type == "retrieval_overlay":
        if (
            set(config) != {"policy"}
            or config["policy"]
            not in {
                "pickxtimes_explicit_first_pickup",
                "counting_live_ordinal",
                "patternlock_all_directions",
                "pickhighlight_all_pickup_templates",
            }
            or artifacts
        ):
            raise ValueError("Unsupported retrieval_overlay policy")
        if (
            config["policy"] == "patternlock_all_directions"
            and task_scope != {"PatternLock"}
        ):
            raise ValueError(
                "PatternLock direction retrieval must be scoped only to PatternLock"
            )
        if (
            config["policy"] == "pickhighlight_all_pickup_templates"
            and task_scope != {"PickHighlight"}
        ):
            raise ValueError(
                "PickHighlight pickup retrieval must be scoped only to PickHighlight"
            )
    elif component_type == "action_canonicalizer":
        if (
            set(config) != {"policy"}
            or config["policy"] != "counting_live_ordinal"
            or artifacts
        ):
            raise ValueError("Unsupported action_canonicalizer policy")
    elif component_type == "reinforcement_overlay":
        counts = config.get("dispatch_counts")
        if (
            set(config) != {"dispatch_counts"}
            or not isinstance(counts, dict)
            or artifacts
        ):
            raise ValueError("Invalid reinforcement_overlay configuration")
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, int)
            or value < 1
            for key, value in counts.items()
        ):
            raise ValueError("Reinforcement dispatch counts must be positive integers")
    elif component_type == "python_plugin":
        if set(artifacts) != {"module"}:
            raise ValueError("python_plugin requires exactly a module artifact")
        if set(config) != {"entrypoint", "hooks"}:
            raise ValueError("python_plugin requires entrypoint and hooks")
        _require_string(config["entrypoint"], "python_plugin.entrypoint")
        hooks = config["hooks"]
        if (
            not isinstance(hooks, list)
            or not hooks
            or any(hook not in HOOK_ORDER for hook in hooks)
            or len(set(hooks)) != len(hooks)
        ):
            raise ValueError("python_plugin hooks must be unique registered hook names")
    elif component_type == "core_harness_patch":
        if set(artifacts) != {"patch"}:
            raise ValueError("core_harness_patch requires exactly a patch artifact")
        expected = {"base_commit", "materialized_commit", "touched_symbols"}
        if set(config) != expected:
            raise ValueError("Invalid core_harness_patch configuration")
        _require_string(config["base_commit"], "base_commit")
        _require_string(config["materialized_commit"], "materialized_commit")
        if (
            not isinstance(config["touched_symbols"], list)
            or not config["touched_symbols"]
            or any(
                not isinstance(symbol, str) or not symbol
                for symbol in config["touched_symbols"]
            )
            or len(set(config["touched_symbols"])) != len(config["touched_symbols"])
        ):
            raise ValueError("core_harness_patch must declare touched_symbols")

    normalized = dict(component)
    normalized["artifacts"] = normalized_artifacts
    assert_runtime_identity_isolated(normalized, source=component_id)
    return normalized


def validate_candidate_manifest(
    value: Any, *, bundle_dir: Path | None = None
) -> dict[str, Any]:
    required = {
        "schema_version",
        "candidate_id",
        "revision",
        "status",
        "problem_id",
        "task_scope",
        "base_profile",
        "diagnosis",
        "components",
        "authoring",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            f"Candidate manifest fields mismatch: expected {sorted(required)}"
        )
    if value["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("Unsupported v4 candidate schema")
    _require_string(value["candidate_id"], "candidate_id")
    _require_string(value["problem_id"], "problem_id")
    if not isinstance(value["revision"], int) or value["revision"] < 1:
        raise ValueError("Candidate revision must be positive")
    if value["status"] not in {"candidate", "approved"}:
        raise ValueError("Candidate status must be candidate or approved")
    tasks = set(_require_task_scope(value["task_scope"], "task_scope"))
    if len(tasks) != 1:
        raise ValueError("A v4 candidate must be scoped to exactly one task")
    if value["base_profile"] != BASE_PROFILE:
        raise ValueError(f"v4 candidates must extend {BASE_PROFILE}")
    diagnosis = value["diagnosis"]
    if not isinstance(diagnosis, dict) or set(diagnosis) != {
        "first_divergence",
        "responsible_module",
        "runtime_observable",
        "repair_surfaces",
    }:
        raise ValueError("Invalid v4 diagnosis record")
    _require_string(diagnosis["first_divergence"], "diagnosis.first_divergence")
    _require_string(diagnosis["responsible_module"], "diagnosis.responsible_module")
    if diagnosis["runtime_observable"] is not True:
        raise ValueError("Runtime interventions require an observable trigger")
    if (
        not isinstance(diagnosis["repair_surfaces"], list)
        or not diagnosis["repair_surfaces"]
    ):
        raise ValueError("diagnosis.repair_surfaces must be non-empty")
    authoring = value["authoring"]
    if not isinstance(authoring, dict) or set(authoring) != {
        "skill_name",
        "source",
        "analysis_inputs",
        "runtime_uses_episode_identity",
        "offline_acceptance_verifier_modified",
    }:
        raise ValueError("authoring must record the v4 skill and offline provenance")
    if authoring["skill_name"] != "self-evolve-intervention-author":
        raise ValueError(
            "V4 candidates must be authored through the intervention skill"
        )
    _require_string(authoring["source"], "authoring.source")
    if not isinstance(authoring["analysis_inputs"], list):
        raise ValueError("authoring.analysis_inputs must be a list")
    for analysis_input in authoring["analysis_inputs"]:
        if (
            not isinstance(analysis_input, dict)
            or set(analysis_input) != {"dataset_split", "episode_id"}
            or analysis_input["dataset_split"] not in {"train", "val"}
            or not isinstance(analysis_input["episode_id"], int)
        ):
            raise ValueError(
                "V4 authoring inputs must be split-qualified train/val episodes"
            )
    if authoring["runtime_uses_episode_identity"] is not False:
        raise ValueError("Runtime intervention may not use analysis episode identity")
    if authoring["offline_acceptance_verifier_modified"] is not False:
        raise ValueError("Candidate may not modify its offline acceptance verifier")
    components = value["components"]
    if not isinstance(components, list) or not components:
        raise ValueError("A candidate requires at least one intervention component")
    normalized_components = [
        _validate_component(component, candidate_tasks=tasks, bundle_dir=bundle_dir)
        for component in components
    ]
    component_ids = [component["component_id"] for component in normalized_components]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("Candidate contains duplicate component IDs")
    normalized = dict(value)
    normalized["components"] = normalized_components
    return normalized


def _check_component_conflicts(components: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    namespaces: set[str] = set()
    claims: dict[str, str] = {}
    for component in components:
        component_id = component["component_id"]
        if component_id in ids:
            raise ValueError(f"Duplicate intervention component ID {component_id}")
        ids.add(component_id)
        namespace = component.get("state_namespace")
        if namespace is not None:
            if namespace in namespaces:
                raise ValueError(f"Duplicate intervention state namespace {namespace}")
            namespaces.add(namespace)
        for claim in component.get("claims", []):
            if claim in claims:
                raise ValueError(
                    f"Intervention claim {claim!r} conflicts between "
                    f"{claims[claim]} and {component_id}"
                )
            claims[claim] = component_id


def compose_candidate_manifests(
    candidates: list[dict[str, Any]], *, bundle_id: str, revision: int
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("Cannot compose an empty intervention bundle")
    normalized = [validate_candidate_manifest(candidate) for candidate in candidates]
    candidate_ids = [candidate["candidate_id"] for candidate in normalized]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("A runtime bundle cannot contain duplicate candidate IDs")
    tasks = {tuple(candidate["task_scope"]) for candidate in normalized}
    if len(tasks) != 1:
        raise ValueError("A task working bundle cannot mix task scopes")
    components = [
        component for candidate in normalized for component in candidate["components"]
    ]
    _check_component_conflicts(components)
    runtime = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_id": _require_string(bundle_id, "bundle_id"),
        "revision": revision,
        "base_profile": BASE_PROFILE,
        "task_scope": list(next(iter(tasks))),
        "candidate_ids": candidate_ids,
        "components": sorted(
            components, key=lambda row: (int(row["priority"]), row["component_id"])
        ),
    }
    runtime["bundle_sha256"] = canonical_hash(runtime)
    return runtime


@dataclass
class RuntimeInterventionPlan:
    bundle: dict[str, Any]
    bundle_path: Path
    criterion_overlay_paths: list[Path] = field(default_factory=list)
    prompt_experiences: list[dict[str, Any]] = field(default_factory=list)
    transition_gates: dict[str, dict[str, Any]] = field(default_factory=dict)
    swing_gate_experience: dict[str, Any] | None = None
    counting_ledger: bool = False
    counting_explicit_first_pickup: bool = False
    counting_exact_pickup_ordinal: bool = False
    patternlock_all_directions: bool = False
    pickhighlight_all_pickup_templates: bool = False
    video_repick_ledger: bool = False
    video_repick_grounding_recovery: dict[str, int] | None = None
    unmask_pickup_grounding_recovery: dict[str, int] | None = None
    button_swap_press_grounding_recovery: dict[str, int] | None = None
    pickhighlight_pickup_grounding_recovery: dict[str, int] | None = None
    reinforcement_dispatch_counts: dict[str, int] = field(default_factory=dict)
    python_plugins: list[dict[str, Any]] = field(default_factory=list)
    core_patch_records: list[dict[str, Any]] = field(default_factory=list)


def load_runtime_intervention_bundle(path: Path) -> RuntimeInterventionPlan:
    path = path.resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "bundle_id",
        "revision",
        "base_profile",
        "task_scope",
        "candidate_ids",
        "components",
        "bundle_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Invalid runtime intervention bundle fields")
    if value["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError("Unsupported runtime intervention bundle schema")
    expected_hash = value["bundle_sha256"]
    unhashed = {key: nested for key, nested in value.items() if key != "bundle_sha256"}
    if canonical_hash(unhashed) != expected_hash:
        raise ValueError("Runtime intervention bundle hash mismatch")
    _require_string(value["bundle_id"], "bundle_id")
    if not isinstance(value["revision"], int) or value["revision"] < 1:
        raise ValueError("Runtime intervention bundle revision must be positive")
    if value["base_profile"] != BASE_PROFILE:
        raise ValueError("Runtime intervention bundle base profile mismatch")
    tasks = set(_require_task_scope(value["task_scope"], "task_scope"))
    components = [
        _validate_component(component, candidate_tasks=tasks, bundle_dir=path.parent)
        for component in value["components"]
    ]
    _check_component_conflicts(components)
    plan = RuntimeInterventionPlan(bundle=value, bundle_path=path)
    for component in components:
        component_type = component["type"]
        config = component["config"]
        if component_type == "criterion_overlay":
            plan.criterion_overlay_paths.append(
                path.parent / component["artifacts"]["catalog"]["path"]
            )
        elif component_type == "prompt_lesson":
            plan.prompt_experiences.append(
                {
                    **config["experience"],
                    "task_scope": list(component["task_scope"]),
                    "status": "approved",
                }
            )
        elif component_type == "bounded_transition_gate":
            criterion_id = config["criterion_id"]
            if criterion_id in plan.transition_gates:
                raise ValueError(f"Multiple gates target criterion {criterion_id}")
            plan.transition_gates[criterion_id] = {
                "experience_id": config["experience_id"],
                "revision": config["revision"],
                "task_scope": list(component["task_scope"]),
                "situation_class": config["situation_class"],
                "max_blocked_transitions": config["max_blocked_transitions"],
            }
        elif component_type == "swing_visual_binding_gate":
            if plan.swing_gate_experience is not None:
                raise ValueError("Only one Swing visual binding gate may be active")
            plan.swing_gate_experience = {
                "experience_id": config["experience_id"],
                "revision": config["revision"],
            }
        elif component_type == "counting_event_ledger":
            plan.counting_ledger = True
            plan.counting_explicit_first_pickup |= config["explicit_first_pickup"]
            plan.counting_exact_pickup_ordinal |= config["exact_pickup_ordinal"]
        elif component_type == "video_repick_event_ledger":
            plan.video_repick_ledger = True
        elif component_type == "video_repick_grounding_recovery_gate":
            if plan.video_repick_grounding_recovery is not None:
                raise ValueError("Only one VideoRepick grounding gate may be active")
            plan.video_repick_grounding_recovery = {
                key: int(value) for key, value in config.items()
            }
        elif component_type == "unmask_pickup_grounding_recovery_gate":
            if plan.unmask_pickup_grounding_recovery is not None:
                raise ValueError("Only one unmask pickup grounding gate may be active")
            plan.unmask_pickup_grounding_recovery = {
                key: int(value) for key, value in config.items()
            }
        elif component_type == "button_swap_press_grounding_recovery_gate":
            if plan.button_swap_press_grounding_recovery is not None:
                raise ValueError(
                    "Only one ButtonUnmaskSwap press grounding gate may be active"
                )
            plan.button_swap_press_grounding_recovery = {
                key: int(value) for key, value in config.items()
            }
        elif component_type == "pickhighlight_pickup_grounding_recovery_gate":
            if plan.pickhighlight_pickup_grounding_recovery is not None:
                raise ValueError("Only one PickHighlight pickup grounding gate may be active")
            plan.pickhighlight_pickup_grounding_recovery = {
                key: int(value) for key, value in config.items()
            }
        elif component_type == "retrieval_overlay":
            if config["policy"] == "pickxtimes_explicit_first_pickup":
                plan.counting_explicit_first_pickup = True
            elif config["policy"] == "counting_live_ordinal":
                plan.counting_exact_pickup_ordinal = True
            elif config["policy"] == "patternlock_all_directions":
                plan.patternlock_all_directions = True
            elif config["policy"] == "pickhighlight_all_pickup_templates":
                plan.pickhighlight_all_pickup_templates = True
        elif component_type == "action_canonicalizer":
            plan.counting_exact_pickup_ordinal = True
        elif component_type == "reinforcement_overlay":
            for key, count in config["dispatch_counts"].items():
                if key in plan.reinforcement_dispatch_counts:
                    raise ValueError(f"Duplicate reinforcement key {key}")
                plan.reinforcement_dispatch_counts[key] = int(count)
        elif component_type == "python_plugin":
            plan.python_plugins.append(component)
        elif component_type == "core_harness_patch":
            plan.core_patch_records.append(component)
    if (
        plan.counting_explicit_first_pickup or plan.counting_exact_pickup_ordinal
    ) and not plan.counting_ledger:
        raise ValueError("Counting retrieval/canonicalization requires an event ledger")
    return plan
